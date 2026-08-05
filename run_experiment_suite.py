from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from repo_runtime import configure_repo_runtime
from pipeline_utils import (
    DEFAULT_MODEL,
    ROOT,
    create_ultralytics_model,
    ensure_results_csv,
    load_json,
    load_yaml,
    normalize_path,
    resolve_device,
)

configure_repo_runtime(Path(__file__).resolve().parent)

from custom_yolo import (
    apply_local_ultralytics_patches,
    clear_experiment_context,
    component_status,
    set_experiment_context,
)

RESULTS_DIR = ROOT / "results"
SUITE_CSV = RESULTS_DIR / "aircraft_skin_defects_new_mainline.csv"
TARGET_CLASS_NAMES = ["crack", "dent", "missing-head", "paint-off", "scratch"]
CLASS_NAME_TO_SLUG = {
    "crack": "crack",
    "dent": "dent",
    "missing-head": "missing_head",
    "paint-off": "paint_off",
    "scratch": "scratch",
}
EXPERIMENT_META_COLUMNS = [
    "suite_id",
    "experiment_group",
    "experiment_id",
    "experiment_title",
    "dataset_id",
    "dataset_role",
    "dataset_yaml",
    "model",
    "project",
    "name",
    "augment_profile",
    "feature_variant",
    "backbone_variant",
    "neck_variant",
    "loss_variant",
    "assigner_variant",
    "notes",
]
CLASS_METRIC_COLUMNS = [
    metric
    for class_name in TARGET_CLASS_NAMES
    for metric in (
        f"{CLASS_NAME_TO_SLUG[class_name]}_ap50",
        f"{CLASS_NAME_TO_SLUG[class_name]}_ap50_95",
    )
]
CSV_COLUMNS = [
    "run_status",
    "run_id",
    "blocked_reason",
    "weights_path",
    "run_dir",
    "resolved_device",
    "imgsz",
    "epochs",
    "batch",
    "workers",
    "seed",
    "optimizer",
    "amp",
    "precision",
    "recall",
    "map50",
    "map50_95",
    *CLASS_METRIC_COLUMNS,
    "class_metrics_json",
    "model_layers",
    "model_parameters",
    "model_gradients",
    "model_flops_g",
    "speed_preprocess_ms",
    "speed_inference_ms",
    "speed_postprocess_ms",
    "export_requested",
    "export_format",
    "export_path",
    "artifact_size_mb",
    *EXPERIMENT_META_COLUMNS,
]


def slugify_class_name(class_name: str) -> str:
    return class_name.strip().replace("-", "_").replace(" ", "_").replace("/", "_")


def configure_suite_schema(manifest: dict[str, Any], manifest_path: Path) -> None:
    global SUITE_CSV, TARGET_CLASS_NAMES, CLASS_NAME_TO_SLUG, CLASS_METRIC_COLUMNS, CSV_COLUMNS

    suite_id = manifest.get("suite_id", manifest_path.stem)
    results_csv = resolve_manifest_path(manifest_path, manifest.get("results_csv"))
    SUITE_CSV = results_csv if results_csv else RESULTS_DIR / f"{suite_id}.csv"

    target_class_names = manifest.get("target_class_names")
    if not target_class_names:
        dataset_yaml = resolve_manifest_path(manifest_path, manifest.get("dataset_yaml"))
        if dataset_yaml and dataset_yaml.exists():
            target_class_names = class_names_from_yaml(dataset_yaml)
        else:
            target_class_names = list(TARGET_CLASS_NAMES)

    TARGET_CLASS_NAMES = list(target_class_names)
    CLASS_NAME_TO_SLUG = {class_name: slugify_class_name(class_name) for class_name in TARGET_CLASS_NAMES}
    CLASS_METRIC_COLUMNS = [
        metric
        for class_name in TARGET_CLASS_NAMES
        for metric in (
            f"{CLASS_NAME_TO_SLUG[class_name]}_ap50",
            f"{CLASS_NAME_TO_SLUG[class_name]}_ap50_95",
        )
    ]
    CSV_COLUMNS = [
        "run_status",
        "run_id",
        "blocked_reason",
        "weights_path",
        "run_dir",
        "resolved_device",
        "imgsz",
        "epochs",
        "batch",
        "workers",
        "seed",
        "optimizer",
        "amp",
        "precision",
        "recall",
        "map50",
        "map50_95",
        *CLASS_METRIC_COLUMNS,
        "class_metrics_json",
        "model_layers",
        "model_parameters",
        "model_gradients",
        "model_flops_g",
        "speed_preprocess_ms",
        "speed_inference_ms",
        "speed_postprocess_ms",
        "export_requested",
        "export_format",
        "export_path",
        "artifact_size_mb",
        *EXPERIMENT_META_COLUMNS,
    ]


def ensure_suite_csv() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not SUITE_CSV.exists():
        with SUITE_CSV.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def append_suite_row(row: dict[str, Any]) -> None:
    ensure_suite_csv()
    normalized = {column: row.get(column, "") for column in CSV_COLUMNS}
    with SUITE_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writerow(normalized)


def successful_run_ids() -> set[str]:
    if not SUITE_CSV.exists():
        return set()
    with SUITE_CSV.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return {
            str(row.get("run_id", ""))
            for row in reader
            if row.get("run_status") == "success" and row.get("run_id")
        }


def float_or_blank(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def read_manifest(path: Path) -> dict[str, Any]:
    return load_json(path)


def resolve_manifest_path(manifest_path: Path, value: str | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (manifest_path.parent.parent / candidate).resolve()


def local_yolo_config_dir() -> Path:
    return (ROOT / ".yolo-config").resolve()


def ensure_local_yolo_config() -> None:
    configure_repo_runtime(ROOT)


def class_names_from_yaml(dataset_yaml: Path) -> list[str]:
    data = load_yaml(dataset_yaml)
    names = data.get("names", [])
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def extract_overall_metrics(result: Any) -> dict[str, str]:
    results_dict = getattr(result, "results_dict", None) or {}
    return {
        "precision": float_or_blank(results_dict.get("metrics/precision(B)")),
        "recall": float_or_blank(results_dict.get("metrics/recall(B)")),
        "map50": float_or_blank(results_dict.get("metrics/mAP50(B)")),
        "map50_95": float_or_blank(results_dict.get("metrics/mAP50-95(B)")),
    }


def extract_class_metrics(result: Any, class_names: list[str]) -> dict[str, Any]:
    rows: dict[str, dict[str, str]] = {
        class_name: {"ap50": "", "ap50_95": ""} for class_name in TARGET_CLASS_NAMES
    }
    ap_class_index = getattr(result, "ap_class_index", [])
    if ap_class_index is None:
        ap_class_index = []
    else:
        ap_class_index = list(ap_class_index)

    for idx, class_idx in enumerate(ap_class_index):
        if class_idx >= len(class_names):
            continue
        class_name = class_names[class_idx]
        if class_name not in rows:
            continue
        class_result = list(result.class_result(idx))
        if len(class_result) >= 4:
            rows[class_name]["ap50"] = float_or_blank(class_result[2])
            rows[class_name]["ap50_95"] = float_or_blank(class_result[3])

    flat: dict[str, str] = {}
    for class_name in TARGET_CLASS_NAMES:
        slug = CLASS_NAME_TO_SLUG[class_name]
        flat[f"{slug}_ap50"] = rows[class_name]["ap50"]
        flat[f"{slug}_ap50_95"] = rows[class_name]["ap50_95"]
    flat["class_metrics_json"] = json.dumps(rows, ensure_ascii=False)
    return flat


def extract_speed_metrics(result: Any) -> dict[str, str]:
    speed = getattr(result, "speed", None) or {}
    return {
        "speed_preprocess_ms": float_or_blank(speed.get("preprocess")),
        "speed_inference_ms": float_or_blank(speed.get("inference")),
        "speed_postprocess_ms": float_or_blank(speed.get("postprocess")),
    }


def extract_model_profile(model_obj: Any, imgsz: int) -> dict[str, str]:
    profile = {
        "model_layers": "",
        "model_parameters": "",
        "model_gradients": "",
        "model_flops_g": "",
    }
    try:
        from ultralytics.utils.torch_utils import get_flops, get_num_gradients, get_num_params
        model = model_obj.model
        profile["model_layers"] = str(len(list(model.modules())))
        profile["model_parameters"] = str(get_num_params(model))
        profile["model_gradients"] = str(get_num_gradients(model))
        profile["model_flops_g"] = float_or_blank(get_flops(model, imgsz))
    except Exception:
        return profile
    return profile


def resolve_experiment_model(manifest_path: Path, experiment: dict[str, Any], default_model: str) -> str:
    model_value = experiment.get("model") or default_model
    resolved = resolve_manifest_path(manifest_path, model_value)
    if resolved and resolved.exists():
        return str(resolved)
    if any(separator in str(model_value) for separator in ("/", "\\")):
        return str(resolved) if resolved else model_value
    return model_value


def resolve_pretrained(manifest_path: Path, experiment: dict[str, Any]) -> str | None:
    value = experiment.get("pretrained")
    resolved = resolve_manifest_path(manifest_path, value)
    return str(resolved) if resolved else None


def resolve_experiment_dataset_yaml(
    manifest_path: Path, experiment: dict[str, Any], default_dataset_yaml: Path
) -> Path:
    value = experiment.get("dataset_yaml")
    resolved = resolve_manifest_path(manifest_path, value)
    return resolved if resolved else default_dataset_yaml


def build_runtime_context(metadata: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {
        "enable_nwd_loss": metadata.get("loss_variant") == "nwd_loss",
        "enable_focaler_iou": metadata.get("loss_variant") == "focaler_iou",
        "enable_nwd_assigner": metadata.get("assigner_variant") == "nwd_assigner",
        "nwd_loss_ratio": 0.5,
        "nwd_assigner_ratio": 0.5,
        "nwd_constant": 12.8,
        "focaler_lower": 0.0,
        "focaler_upper": 0.95,
    }
    return context


def experiment_requires_local_patches(metadata: dict[str, Any], experiment_model: str) -> bool:
    model_text = str(experiment_model).lower()
    if "custom_yolo" in model_text:
        return True
    if metadata.get("feature_variant") not in ("", "default"):
        return True
    if metadata.get("neck_variant") not in ("", "default"):
        return True
    if metadata.get("loss_variant") not in ("", "default"):
        return True
    if metadata.get("assigner_variant") not in ("", "default"):
        return True
    return False


def determine_readiness(experiment: dict[str, Any], metadata: dict[str, Any], repo_root: Path) -> tuple[bool, str]:
    status = component_status(repo_root)
    missing: list[str] = []
    if metadata.get("feature_variant") == "dwt_down":
        if not status["dwt_module"]:
            missing.append("DWTDown patch module")
        if not status["dwt_yaml"]:
            missing.append("DWT model yaml")
    if metadata.get("feature_variant") == "wtconv_neck_refine" and not status["neck_wt_yaml"]:
        missing.append("Neck WT refine model yaml")
    if metadata.get("feature_variant") == "wtconv_residual_backbone" and not status["residual_wt_yaml"]:
        missing.append("Residual WT backbone model yaml")
    if metadata.get("feature_variant") == "p2_head" and not status["p2_yaml"]:
        missing.append("P2 head model yaml")
    if metadata.get("feature_variant") == "p2_head" and metadata.get("neck_variant") == "eca_attention":
        if not status["p2_eca_yaml"]:
            missing.append("P2 + ECA model yaml")
    if metadata.get("feature_variant") == "spd_stem" and not status["spd_yaml"]:
        missing.append("SPD stem model yaml")
    if metadata.get("feature_variant") == "spd_defect_p2" and not status["spd_defect_p2_yaml"]:
        missing.append("SPD + defect-aware P2 model yaml")
    if metadata.get("feature_variant") == "spd_defect_p2" and metadata.get("neck_variant") == "dynamic_fusion":
        if not status["spd_defect_p2_dyfuse_yaml"]:
            missing.append("SPD + defect-aware P2 + dynamic fusion model yaml")
    if metadata.get("loss_variant") == "nwd_loss" and not status["nwd_patch"]:
        missing.append("NWD loss patch")
    if metadata.get("loss_variant") == "focaler_iou" and not status["nwd_patch"]:
        missing.append("Focaler-IoU bbox loss patch")
    if metadata.get("assigner_variant") == "nwd_assigner" and not status["nwd_patch"]:
        missing.append("NWD assigner patch")
    if metadata.get("neck_variant") == "rep_neck" and not status["repneck_yaml"]:
        missing.append("Rep neck model yaml")
    if missing:
        return False, "; ".join(missing) + " missing"
    if experiment.get("ready", False):
        return True, ""
    if not experiment.get("auto_unlock", False):
        return False, experiment.get("blocked_reason", "Not ready.")
    return True, ""


def run_experiment(
    *,
    manifest_path: Path,
    suite_id: str,
    experiment: dict[str, Any],
    metadata_defaults: dict[str, Any],
    train_defaults: dict[str, Any],
    dataset_yaml: Path,
    default_model: str,
    project: Path,
    export_format: str,
    enable_export: bool,
) -> dict[str, Any]:
    ensure_local_yolo_config()
    class_names = class_names_from_yaml(dataset_yaml)
    train_kwargs = dict(train_defaults)
    train_kwargs.update(experiment.get("overrides", {}))

    run_name = experiment["name"]
    resolved_device = resolve_device(str(train_kwargs.get("device", "auto")))
    train_kwargs["device"] = resolved_device
    train_kwargs["data"] = str(dataset_yaml)
    train_kwargs["project"] = str(project)
    train_kwargs["name"] = run_name

    metadata = dict(metadata_defaults)
    metadata.update(experiment.get("metadata", {}))
    runtime_context = build_runtime_context(metadata)
    runtime_context.update(experiment.get("runtime", {}))
    set_experiment_context(runtime_context)

    experiment_model = resolve_experiment_model(manifest_path, experiment, default_model)
    if experiment_requires_local_patches(metadata, experiment_model):
        apply_local_ultralytics_patches()
    pretrained = resolve_pretrained(manifest_path, experiment)
    if pretrained:
        train_kwargs["pretrained"] = pretrained

    meta_row = {
        "suite_id": suite_id,
        "experiment_group": metadata.get("experiment_group", suite_id),
        "experiment_id": experiment["experiment_id"],
        "experiment_title": experiment["title"],
        "dataset_id": dataset_yaml.parent.name,
        "dataset_role": metadata.get("dataset_role", ""),
        "dataset_yaml": normalize_path(dataset_yaml),
        "model": experiment_model,
        "project": normalize_path(project),
        "name": run_name,
        "augment_profile": metadata.get("augment_profile", ""),
        "feature_variant": metadata.get("feature_variant", ""),
        "backbone_variant": metadata.get("backbone_variant", ""),
        "neck_variant": metadata.get("neck_variant", ""),
        "loss_variant": metadata.get("loss_variant", ""),
        "assigner_variant": metadata.get("assigner_variant", ""),
        "notes": metadata.get("notes", ""),
    }

    try:
        model_obj = create_ultralytics_model(experiment_model)
        profile = extract_model_profile(model_obj, int(train_kwargs["imgsz"]))
        result = model_obj.train(**train_kwargs)
        run_dir = Path(getattr(model_obj.trainer, "save_dir", ""))
        weights_path = run_dir / "weights" / "best.pt"
        overall_metrics = extract_overall_metrics(result)
        class_metrics = extract_class_metrics(result, class_names)
        speed_metrics = extract_speed_metrics(result)

        export_path = ""
        artifact_size_mb = ""
        if enable_export:
            export_path = str(model_obj.export(format=export_format))
            if export_path and Path(export_path).exists():
                artifact_size_mb = float_or_blank(Path(export_path).stat().st_size / (1024 * 1024))
        elif weights_path.exists():
            artifact_size_mb = float_or_blank(weights_path.stat().st_size / (1024 * 1024))

        row = {
            "run_status": "success",
            "run_id": experiment["experiment_id"],
            "blocked_reason": "",
            "weights_path": normalize_path(weights_path),
            "run_dir": normalize_path(run_dir),
            "resolved_device": resolved_device,
            "imgsz": str(train_kwargs.get("imgsz", "")),
            "epochs": str(train_kwargs.get("epochs", "")),
            "batch": str(train_kwargs.get("batch", "")),
            "workers": str(train_kwargs.get("workers", "")),
            "seed": str(train_kwargs.get("seed", "")),
            "optimizer": str(train_kwargs.get("optimizer", "")),
            "amp": str(train_kwargs.get("amp", "")),
            **overall_metrics,
            **class_metrics,
            **profile,
            **speed_metrics,
            "export_requested": str(enable_export),
            "export_format": export_format if enable_export else "",
            "export_path": normalize_path(export_path),
            "artifact_size_mb": artifact_size_mb,
            **meta_row,
        }
        append_suite_row(row)
        return row
    finally:
        clear_experiment_context()


def blocked_row(
    *,
    manifest_path: Path,
    suite_id: str,
    experiment: dict[str, Any],
    metadata_defaults: dict[str, Any],
    dataset_yaml: Path,
    default_model: str,
    project: Path,
    blocked_reason: str,
) -> dict[str, Any]:
    metadata = dict(metadata_defaults)
    metadata.update(experiment.get("metadata", {}))
    experiment_model = resolve_experiment_model(manifest_path, experiment, default_model)
    row = {
        "run_status": "blocked",
        "run_id": experiment["experiment_id"],
        "blocked_reason": blocked_reason,
        "weights_path": "",
        "run_dir": "",
        "resolved_device": "",
        "imgsz": "",
        "epochs": "",
        "batch": "",
        "workers": "",
        "seed": "",
        "optimizer": "",
        "amp": "",
        "precision": "",
        "recall": "",
        "map50": "",
        "map50_95": "",
        "class_metrics_json": "",
        "model_layers": "",
        "model_parameters": "",
        "model_gradients": "",
        "model_flops_g": "",
        "speed_preprocess_ms": "",
        "speed_inference_ms": "",
        "speed_postprocess_ms": "",
        "export_requested": "",
        "export_format": "",
        "export_path": "",
        "artifact_size_mb": "",
        "suite_id": suite_id,
        "experiment_group": metadata.get("experiment_group", suite_id),
        "experiment_id": experiment["experiment_id"],
        "experiment_title": experiment["title"],
        "dataset_id": dataset_yaml.parent.name,
        "dataset_role": metadata.get("dataset_role", ""),
        "dataset_yaml": normalize_path(dataset_yaml),
        "model": experiment_model,
        "project": normalize_path(project),
        "name": experiment["name"],
        "augment_profile": metadata.get("augment_profile", ""),
        "feature_variant": metadata.get("feature_variant", ""),
        "backbone_variant": metadata.get("backbone_variant", ""),
        "neck_variant": metadata.get("neck_variant", ""),
        "loss_variant": metadata.get("loss_variant", ""),
        "assigner_variant": metadata.get("assigner_variant", ""),
        "notes": metadata.get("notes", ""),
    }
    for column in CLASS_METRIC_COLUMNS:
        row[column] = ""
    append_suite_row(row)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the aircraft_skin_defects_new mainline experiment suite.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiments" / "aircraft_skin_defects_new_mainline.json",
        help="Path to a suite manifest JSON file.",
    )
    parser.add_argument("--list", action="store_true", help="List experiments and exit.")
    parser.add_argument("--only", nargs="*", help="Run only the given experiment ids, e.g. B0 B1.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Default YOLO model checkpoint or model yaml.")
    parser.add_argument("--device", help="Override device from the manifest.")
    parser.add_argument("--imgsz", type=int, help="Override image size from the manifest.")
    parser.add_argument("--epochs", type=int, help="Override epochs from the manifest.")
    parser.add_argument("--batch", type=int, help="Override batch size from the manifest.")
    parser.add_argument("--workers", type=int, help="Override workers from the manifest.")
    parser.add_argument("--project", type=Path, help="Override project directory from the manifest.")
    parser.add_argument("--export-format", default="onnx", help="Export format to use when export is enabled.")
    parser.add_argument("--export", action="store_true", help="Export artifacts after training.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip experiment ids already logged as success.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without starting training.")
    return parser.parse_args()


def print_experiment_table(manifest_path: Path, experiments: list[dict[str, Any]], metadata_defaults: dict[str, Any]) -> None:
    print("experiment_id\tstatus\tname\ttitle\tblocked_reason")
    for exp in experiments:
        metadata = dict(metadata_defaults)
        metadata.update(exp.get("metadata", {}))
        ready, reason = determine_readiness(exp, metadata, ROOT)
        status = "ready" if ready else "blocked"
        print(f"{exp['experiment_id']}\t{status}\t{exp['name']}\t{exp['title']}\t{reason or exp.get('blocked_reason', '')}")


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = read_manifest(manifest_path)
    configure_suite_schema(manifest, manifest_path)
    dataset_yaml = resolve_manifest_path(manifest_path, manifest["dataset_yaml"])
    project = (ROOT / manifest.get("project", "runs/aircraft_new")).resolve()
    if args.project:
        project = args.project.resolve()

    train_defaults = dict(manifest.get("train_defaults", {}))
    if args.device:
        train_defaults["device"] = args.device
    if args.imgsz is not None:
        train_defaults["imgsz"] = args.imgsz
    if args.epochs is not None:
        train_defaults["epochs"] = args.epochs
    if args.batch is not None:
        train_defaults["batch"] = args.batch
    if args.workers is not None:
        train_defaults["workers"] = args.workers

    experiments = manifest.get("experiments", [])
    if args.only:
        selected = set(args.only)
        experiments = [exp for exp in experiments if exp["experiment_id"] in selected]

    metadata_defaults = dict(manifest.get("metadata_defaults", {}))
    if args.list:
        print_experiment_table(manifest_path, experiments, metadata_defaults)
        print(f"\nmanifest={manifest_path}")
        print(f"csv={SUITE_CSV}")
        return 0

    if not dataset_yaml or not dataset_yaml.exists():
        raise SystemExit(f"Dataset YAML not found: {dataset_yaml}")

    ensure_suite_csv()
    ensure_results_csv()
    completed = successful_run_ids() if args.skip_existing else set()

    suite_id = manifest.get("suite_id", args.manifest.stem)

    for experiment in experiments:
        experiment_dataset_yaml = resolve_experiment_dataset_yaml(manifest_path, experiment, dataset_yaml)
        metadata = dict(metadata_defaults)
        metadata.update(experiment.get("metadata", {}))
        ready, readiness_reason = determine_readiness(experiment, metadata, ROOT)
        status = "ready" if ready else "blocked"
        print(f"[{suite_id}] {experiment['experiment_id']} {experiment['name']} status={status}")
        if experiment["experiment_id"] in completed:
            print("  skipped existing successful run")
            continue
        if not experiment_dataset_yaml.exists():
            row = blocked_row(
                manifest_path=manifest_path,
                suite_id=suite_id,
                experiment=experiment,
                metadata_defaults=metadata_defaults,
                dataset_yaml=experiment_dataset_yaml,
                default_model=args.model,
                project=project,
                blocked_reason=f"Dataset YAML not found: {experiment_dataset_yaml}",
            )
            print(f"  blocked_reason={row['blocked_reason']}")
            continue
        if not ready:
            row = blocked_row(
                manifest_path=manifest_path,
                suite_id=suite_id,
                experiment=experiment,
                metadata_defaults=metadata_defaults,
                dataset_yaml=experiment_dataset_yaml,
                default_model=args.model,
                project=project,
                blocked_reason=readiness_reason or experiment.get("blocked_reason", "Not ready."),
            )
            print(f"  blocked_reason={row['blocked_reason']}")
            continue

        if args.dry_run:
            experiment_model = resolve_experiment_model(manifest_path, experiment, args.model)
            print(f"  dry_run model={experiment_model}")
            print(f"  dry_run dataset={experiment_dataset_yaml}")
            print(f"  dry_run train_defaults={json.dumps(train_defaults, ensure_ascii=False)}")
            print(f"  dry_run overrides={json.dumps(experiment.get('overrides', {}), ensure_ascii=False)}")
            continue

        row = run_experiment(
            manifest_path=manifest_path,
            suite_id=suite_id,
            experiment=experiment,
            metadata_defaults=metadata_defaults,
            train_defaults=train_defaults,
            dataset_yaml=experiment_dataset_yaml,
            default_model=args.model,
            project=project,
            export_format=args.export_format,
            enable_export=args.export,
        )
        class_summary = []
        try:
            class_metrics = json.loads(row.get("class_metrics_json", "") or "{}")
            for class_name, metrics in class_metrics.items():
                ap5095 = metrics.get("ap50_95", "")
                if ap5095 not in ("", None):
                    class_summary.append(f"{class_name}_ap50_95={ap5095}")
        except json.JSONDecodeError:
            class_summary = []
        print(
            f"  success map50={row['map50']} map50_95={row['map50_95']} "
            + " ".join(class_summary[:3])
        )
        print(f"  run_dir={row['run_dir']}")
        print(f"  weights={row['weights_path']}")

    print(f"suite_csv={SUITE_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



