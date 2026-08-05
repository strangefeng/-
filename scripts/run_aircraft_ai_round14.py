from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_YAML = ROOT / "datasets" / "supplementary_3cls" / "aircraft_ai_dataset_3cls" / "data.yaml"
ROUND4_CSV = ROOT / "results" / "aircraft_ai_dataset_round4.csv"
ROUND13_PHASE2_CSV = ROOT / "results" / "aircraft_ai_dataset_round13_phase2.csv"
ROUND14_CSV = ROOT / "results" / "aircraft_ai_dataset_round14.csv"
ROUND14_SUMMARY_CSV = ROOT / "results" / "round14_summary.csv"
ROUND14_DEPLOYMENT_FP_CSV = ROOT / "results" / "round14_deployment_fp_summary.csv"
ROUND14_SUMMARY_MD = ROOT / "analysis" / "aircraft_ai_dataset_round14_summary.md"
ROUND14_FINAL_MD = ROOT / "analysis" / "final_recommendation.md"
ROUND14_STATUS_JSON = ROOT / "analysis" / "aircraft_ai_dataset_round14_status.json"
ROUND14_C0_JSON = ROOT / "analysis" / "aircraft_ai_dataset_round14_c0_n2_official_val.json"
ROUND14_C0_LOG = ROOT / "analysis" / "R14-C0_val_log.txt"
ROUND14_C0_SUMMARY_MD = ROOT / "analysis" / "R14-C0_summary.md"
ROUND14_QA_DIR = ROOT / "analysis" / "round14_annotation_audit"
ROUND14_QA_CSV = ROUND14_QA_DIR / "annotation_audit_candidates.csv"
ROUND14_QA_MD = ROUND14_QA_DIR / "R14_QA1_annotation_audit_summary.md"
ROUND14_QA_JSON = ROUND14_QA_DIR / "R14_QA1_annotation_audit_summary.json"
ROUND14_QA_VIS = ROUND14_QA_DIR / "vis"
ROUND14_FP_DIR = ROOT / "analysis" / "round14_deployment_fp"
PROJECT_DIR = ROOT / "runs" / "aircraft_ai_round14"
DEPLOYMENT_NEGATIVE_YAML = ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_hardneg50_negonly" / "data.yaml"
CURATED_NEGATIVE_DIR = ROOT / "datasets" / "generated_3cls" / "aircraft_ai_round14_curated_negatives"
YOLO11S_WEIGHTS = ROOT / "yolo11s.pt"

EXPECTED_N2_MAP5095 = 0.352565
EXPECTED_HN1_MAP5095 = 0.350875


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run round14 experiments for aircraft skin defect detection.")
    parser.add_argument("--negative-images-dir", type=Path, help="Curated train-only hard-negative image directory.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["C0", "FP", "QA1", "HN2", "HN3", "M1", "M2", "SUMMARY"],
        default=["C0", "FP", "QA1", "HN2", "HN3", "M1", "M2", "SUMMARY"],
    )
    return parser.parse_args()


def run(command: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(ROOT),
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=capture_output,
    )


def write_status(status_payload: dict) -> None:
    ROUND14_STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    ROUND14_STATUS_JSON.write_text(json.dumps(status_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find_success_row(path: Path, experiment_id: str) -> dict[str, str] | None:
    rows = [
        row
        for row in read_rows(path)
        if row.get("experiment_id") == experiment_id and row.get("run_status") == "success"
    ]
    if not rows:
        return None
    return sorted(rows, key=lambda row: float(row.get("map50_95", "") or 0.0), reverse=True)[0]


def fv(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def resolve_n2_row() -> dict[str, str]:
    row = find_success_row(ROUND4_CSV, "N2")
    if not row:
        raise SystemExit("Could not resolve N2 row.")
    return row


def resolve_hn1_row() -> dict[str, str]:
    row = find_success_row(ROUND13_PHASE2_CSV, "R13-HN1_N2_HardNegative50")
    if not row:
        raise SystemExit("Could not resolve HN1 row.")
    return row


def resolve_n2_weights() -> Path:
    weights = Path(resolve_n2_row()["weights_path"]).resolve()
    if not weights.exists():
        raise SystemExit(f"N2 weights missing: {weights}")
    return weights


def write_temp_manifest(payload: dict) -> Path:
    tmp_root = ROOT / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="round14_", dir=str(tmp_root)))
    path = tmp_dir / "manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def base_manifest(experiment: dict) -> Path:
    payload = {
        "suite_id": "aircraft_ai_dataset_round14",
        "description": "Round14 focuses on hard-negative expansion, label audit, and YOLO11s upper-bound validation.",
        "results_csv": str(ROUND14_CSV),
        "target_class_names": ["missing-rivet", "rust", "scratch"],
        "dataset_yaml": str(SOURCE_YAML),
        "project": str(PROJECT_DIR),
        "train_defaults": {
            "imgsz": 800,
            "epochs": 100,
            "batch": 12,
            "device": 0,
            "workers": 0,
            "seed": 42,
            "patience": 30,
            "deterministic": True,
            "optimizer": "auto",
            "amp": True,
            "plots": True,
            "val": True,
            "save": True,
            "exist_ok": False,
            "rect": False,
            "cache": False,
            "pretrained": True,
            "mosaic": 0.0,
            "erasing": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "auto_augment": "none",
            "flipud": 0.0,
            "close_mosaic": 0,
        },
        "metadata_defaults": {
            "experiment_group": "aircraft_ai_dataset_round14",
            "dataset_role": "round14_candidate",
            "backbone_variant": "yolo11_default",
            "feature_variant": "default",
            "neck_variant": "default",
            "loss_variant": "default",
            "assigner_variant": "default",
            "notes": "Round14 keeps the N2 detail-preserving recipe fixed while testing hard-negative expansion, train-label audit, and YOLO11s capacity.",
        },
        "experiments": [experiment],
    }
    return write_temp_manifest(payload)


def run_manifest(manifest: Path) -> int:
    completed = run([sys.executable, str(ROOT / "run_experiment_suite.py"), "--manifest", str(manifest)])
    return completed.returncode


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_c0_summary(payload: dict) -> None:
    delta = fv(payload.get("map50_95")) - EXPECTED_N2_MAP5095
    lines = [
        "# R14-C0 Summary",
        "",
        f"- weights: `{payload.get('weights_path', '')}`",
        f"- dataset_yaml: `{payload.get('dataset_yaml', '')}`",
        f"- official val map50-95: `{fv(payload.get('map50_95')):.6f}`",
        f"- delta vs historical N2: `{delta:+.6f}`",
        f"- gate: `{'pass' if abs(delta) <= 0.002 else 'fail'}`",
    ]
    ROUND14_C0_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def run_c0(n2_weights: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "validate_yolo_checkpoint.py"),
        "--weights",
        str(n2_weights),
        "--dataset-yaml",
        str(SOURCE_YAML),
        "--split",
        "val",
        "--imgsz",
        "800",
        "--device",
        "0",
        "--output-json",
        str(ROUND14_C0_JSON),
        "--title",
        "R14-C0 official val for N2",
    ]
    completed = run(command, capture_output=True)
    ROUND14_C0_LOG.parent.mkdir(parents=True, exist_ok=True)
    ROUND14_C0_LOG.write_text((completed.stdout or "") + "\n" + (completed.stderr or ""), encoding="utf-8")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    payload = load_json(ROUND14_C0_JSON)
    write_c0_summary(payload)
    if abs(fv(payload.get("map50_95")) - EXPECTED_N2_MAP5095) > 0.002:
        raise SystemExit("R14-C0 failed: official val deviates more than ±0.002 from the N2 anchor.")


def resolve_negative_dir(explicit: Path | None) -> Path | None:
    for candidate in [explicit, CURATED_NEGATIVE_DIR]:
        if candidate and candidate.exists():
            return candidate.resolve()
    return None


def count_images(images_dir: Path) -> int:
    suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    return sum(1 for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def build_hn_dataset(output_root: Path, negative_dir: Path, target_count: int) -> Path:
    if output_root.exists():
        shutil.rmtree(output_root)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "build_hard_negative_dataset.py"),
        "--source-yaml",
        str(SOURCE_YAML),
        "--weights",
        str(resolve_n2_weights()),
        "--output-root",
        str(output_root),
        "--splits",
        "train",
        "--max-negatives",
        str(target_count),
        "--negative-limit",
        str(target_count),
        "--negative-images-dir",
        str(negative_dir),
        "--no-fallback-to-mined",
    ]
    completed = run(command)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return output_root / "data.yaml"


def build_hn_manifest(experiment_id: str, title: str, dataset_yaml: Path, count: int) -> Path:
    experiment = {
        "experiment_id": experiment_id,
        "name": experiment_id.replace("-", "_"),
        "title": title,
        "ready": True,
        "dataset_yaml": str(dataset_yaml.resolve()),
        "metadata": {
            "augment_profile": f"n2_hard_negative_{count}",
            "dataset_role": f"n2_hard_negative_{count}",
            "notes": f"N2-style full-image training with {count} curated hard negatives added only to the train split.",
        },
    }
    return base_manifest(experiment)


def build_m1_manifest(batch: int = 8) -> Path:
    experiment = {
        "experiment_id": "R14-M1_YOLO11s_N2",
        "name": "R14_M1_YOLO11s_N2",
        "title": "YOLO11s upper-bound validation under the N2 recipe",
        "ready": True,
        "model": str(YOLO11S_WEIGHTS.resolve()),
        "dataset_yaml": str(SOURCE_YAML.resolve()),
        "metadata": {
            "augment_profile": "n2_yolo11s",
            "dataset_role": "yolo11s_n2_upper_bound",
            "notes": "Switches only the backbone capacity from YOLO11n to YOLO11s while keeping the N2 recipe fixed.",
        },
        "overrides": {
            "batch": batch,
        },
    }
    return base_manifest(experiment)


def measure_negative_fp(experiment_id: str, weights_path: Path) -> Path | None:
    if not DEPLOYMENT_NEGATIVE_YAML.exists():
        return None
    output_json = ROUND14_FP_DIR / f"{experiment_id}.json"
    vis_dir = ROUND14_FP_DIR / experiment_id
    command = [
        sys.executable,
        str(ROOT / "scripts" / "measure_negative_false_positives.py"),
        "--weights",
        str(weights_path.resolve()),
        "--dataset-yaml",
        str(DEPLOYMENT_NEGATIVE_YAML.resolve()),
        "--split",
        "train",
        "--imgsz",
        "800",
        "--conf",
        "0.001",
        "--high-conf",
        "0.25",
        "--iou",
        "0.70",
        "--max-det",
        "300",
        "--device",
        "0",
        "--output-json",
        str(output_json),
        "--vis-dir",
        str(vis_dir),
        "--topk-vis",
        "12",
        "--title",
        f"round14 deployment negative audit for {experiment_id}",
    ]
    completed = run(command)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    payload = load_json(output_json)
    payload["experiment_id"] = experiment_id
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_json


def write_deployment_fp_summary(output_csv: Path, payloads: list[dict]) -> None:
    columns = [
        "experiment",
        "negative_image_count",
        "total_false_positives",
        "high_conf_false_positives",
        "average_false_positives_per_image",
        "per_class_false_positives",
        "per_class_high_conf_false_positives",
        "visualization_examples",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for payload in payloads:
            writer.writerow(
                {
                    "experiment": payload.get("experiment_id", ""),
                    "negative_image_count": payload.get("negative_image_count", ""),
                    "total_false_positives": payload.get("total_false_positive_boxes", ""),
                    "high_conf_false_positives": payload.get("high_conf_false_positive_boxes", ""),
                    "average_false_positives_per_image": payload.get("avg_false_positive_boxes_per_image", ""),
                    "per_class_false_positives": json.dumps(payload.get("per_class_false_positives", {}), ensure_ascii=False),
                    "per_class_high_conf_false_positives": json.dumps(
                        payload.get("per_class_high_conf_false_positives", {}), ensure_ascii=False
                    ),
                    "visualization_examples": json.dumps(payload.get("visualization_examples", []), ensure_ascii=False),
                }
            )


def write_experiment_bundle(row: dict[str, str], n2_row: dict[str, str], hn1_row: dict[str, str], fp_payload: dict | None) -> None:
    run_dir = Path(row["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    delta_csv = run_dir / "delta_vs_anchors.csv"
    with delta_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["anchor", "delta_map50_95", "delta_precision", "delta_recall"])
        writer.writeheader()
        for anchor_name, anchor_row in [("N2", n2_row), ("HN1", hn1_row)]:
            writer.writerow(
                {
                    "anchor": anchor_name,
                    "delta_map50_95": f"{fv(row.get('map50_95')) - fv(anchor_row.get('map50_95')):+.6f}",
                    "delta_precision": f"{fv(row.get('precision')) - fv(anchor_row.get('precision')):+.6f}",
                    "delta_recall": f"{fv(row.get('recall')) - fv(anchor_row.get('recall')):+.6f}",
                }
            )

    class_metrics = json.loads(row.get("class_metrics_json", "{}") or "{}")
    per_class_csv = run_dir / "per_class_ap.csv"
    with per_class_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_name", "ap50", "ap50_95"])
        writer.writeheader()
        for class_name, metrics in class_metrics.items():
            writer.writerow(
                {
                    "class_name": class_name,
                    "ap50": metrics.get("ap50", ""),
                    "ap50_95": metrics.get("ap50_95", ""),
                }
            )

    lines = [
        f"# {row.get('experiment_id', '')} Summary",
        "",
        f"- title: `{row.get('experiment_title', '')}`",
        f"- model: `{Path(row.get('model', '')).name}`",
        f"- dataset: `{row.get('dataset_yaml', '')}`",
        f"- precision: `{row.get('precision', '')}`",
        f"- recall: `{row.get('recall', '')}`",
        f"- map50: `{row.get('map50', '')}`",
        f"- map50-95: `{row.get('map50_95', '')}`",
        f"- delta vs N2: `{fv(row.get('map50_95')) - fv(n2_row.get('map50_95')):+.6f}`",
        f"- delta vs HN1: `{fv(row.get('map50_95')) - fv(hn1_row.get('map50_95')):+.6f}`",
    ]
    if fp_payload:
        lines.extend(
            [
                "",
                "## Deployment FP",
                "",
                f"- total_false_positives: `{fp_payload.get('total_false_positive_boxes', 0)}`",
                f"- high_conf_false_positives: `{fp_payload.get('high_conf_false_positive_boxes', 0)}`",
                f"- avg_false_positives_per_image: `{float(fp_payload.get('avg_false_positive_boxes_per_image', 0.0)):.4f}`",
            ]
        )
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_qa1(n2_weights: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "audit_aircraft_ai_round14_labels.py"),
        "--weights",
        str(n2_weights),
        "--dataset-yaml",
        str(SOURCE_YAML),
        "--splits",
        "train",
        "valid",
        "--imgsz",
        "800",
        "--device",
        "0",
        "--conf",
        "0.05",
        "--high-conf",
        "0.25",
        "--missing-label-conf",
        "0.50",
        "--max-candidates",
        "420",
        "--output-csv",
        str(ROUND14_QA_CSV),
        "--summary-md",
        str(ROUND14_QA_MD),
        "--summary-json",
        str(ROUND14_QA_JSON),
        "--vis-dir",
        str(ROUND14_QA_VIS),
    ]
    completed = run(command)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def summarize_round14(fp_jsons: list[Path]) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "summarize_aircraft_ai_round14.py"),
        "--round4-csv",
        str(ROUND4_CSV),
        "--round13-phase2-csv",
        str(ROUND13_PHASE2_CSV),
        "--round14-csv",
        str(ROUND14_CSV),
        "--c0-json",
        str(ROUND14_C0_JSON),
        "--qa1-json",
        str(ROUND14_QA_JSON),
        "--status-json",
        str(ROUND14_STATUS_JSON),
        "--output-summary-csv",
        str(ROUND14_SUMMARY_CSV),
        "--output-summary-md",
        str(ROUND14_SUMMARY_MD),
        "--output-final-md",
        str(ROUND14_FINAL_MD),
    ]
    if fp_jsons:
        command.append("--deployment-fp-json")
        command.extend(str(path) for path in fp_jsons)
    completed = run(command)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    args = parse_args()
    only = set(args.only)
    status_payload = {
        "deployment_negative_set_yaml": str(DEPLOYMENT_NEGATIVE_YAML.resolve()) if DEPLOYMENT_NEGATIVE_YAML.exists() else "",
        "blocked_experiments": [],
        "notes": [],
    }

    n2_row = resolve_n2_row()
    hn1_row = resolve_hn1_row()
    n2_weights = resolve_n2_weights()

    if "C0" in only:
        run_c0(n2_weights)

    fp_jsons: list[Path] = []
    if "FP" in only:
        for experiment_id, weights_path in [
            ("N2", Path(n2_row["weights_path"])),
            ("R13-HN1_N2_HardNegative50", Path(hn1_row["weights_path"])),
        ]:
            fp_json = measure_negative_fp(experiment_id, weights_path)
            if fp_json:
                fp_jsons.append(fp_json)

    if "QA1" in only:
        run_qa1(n2_weights)

    negative_dir = resolve_negative_dir(args.negative_images_dir)
    negative_count = count_images(negative_dir) if negative_dir else 0

    if "HN2" in only:
        if not negative_dir:
            status_payload["blocked_experiments"].append(
                {
                    "experiment": "R14-HN2_N2_HardNegative150",
                    "reason": "Skipped because no curated train-only negative image directory was available.",
                }
            )
        elif negative_count < 150:
            status_payload["blocked_experiments"].append(
                {
                    "experiment": "R14-HN2_N2_HardNegative150",
                    "reason": f"Skipped because the curated negative pool only has {negative_count} images, below the 150-image requirement.",
                }
            )
        else:
            hn150_yaml = build_hn_dataset(
                ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_hardneg150",
                negative_dir,
                150,
            )
            manifest = build_hn_manifest(
                "R14-HN2_N2_HardNegative150",
                "N2-style training with 150 curated hard negatives",
                hn150_yaml,
                150,
            )
            rc = run_manifest(manifest)
            if rc != 0:
                raise SystemExit(rc)
            row = find_success_row(ROUND14_CSV, "R14-HN2_N2_HardNegative150")
            if row:
                fp_json = measure_negative_fp("R14-HN2_N2_HardNegative150", Path(row["weights_path"]))
                if fp_json:
                    fp_jsons.append(fp_json)
                    write_experiment_bundle(row, n2_row, hn1_row, load_json(fp_json))

    hn2_row = find_success_row(ROUND14_CSV, "R14-HN2_N2_HardNegative150")
    if "HN3" in only:
        if not negative_dir:
            status_payload["blocked_experiments"].append(
                {
                    "experiment": "R14-HN3_N2_HardNegative300",
                    "reason": "Skipped because no curated train-only negative image directory was available.",
                }
            )
        elif negative_count < 300:
            status_payload["blocked_experiments"].append(
                {
                    "experiment": "R14-HN3_N2_HardNegative300",
                    "reason": f"Skipped because the curated negative pool only has {negative_count} images, below the 300-image requirement.",
                }
            )
        elif hn2_row and (fv(hn2_row.get("map50_95")) < 0.345 or fv(hn2_row.get("recall")) < 0.690):
            status_payload["blocked_experiments"].append(
                {
                    "experiment": "R14-HN3_N2_HardNegative300",
                    "reason": "Skipped because HN150 dropped below the round14 safety gate (mAP50-95 < 0.345 or recall clearly fell).",
                }
            )
        elif not hn2_row:
            status_payload["blocked_experiments"].append(
                {
                    "experiment": "R14-HN3_N2_HardNegative300",
                    "reason": "Skipped because HN150 did not complete successfully.",
                }
            )
        else:
            hn300_yaml = build_hn_dataset(
                ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_hardneg300",
                negative_dir,
                300,
            )
            manifest = build_hn_manifest(
                "R14-HN3_N2_HardNegative300",
                "N2-style training with 300 curated hard negatives",
                hn300_yaml,
                300,
            )
            rc = run_manifest(manifest)
            if rc != 0:
                raise SystemExit(rc)
            row = find_success_row(ROUND14_CSV, "R14-HN3_N2_HardNegative300")
            if row:
                fp_json = measure_negative_fp("R14-HN3_N2_HardNegative300", Path(row["weights_path"]))
                if fp_json:
                    fp_jsons.append(fp_json)
                    write_experiment_bundle(row, n2_row, hn1_row, load_json(fp_json))

    if "M1" in only:
        if not YOLO11S_WEIGHTS.exists():
            status_payload["blocked_experiments"].append(
                {
                    "experiment": "R14-M1_YOLO11s_N2",
                    "reason": "Skipped because yolo11s.pt is missing locally.",
                }
            )
        else:
            manifest = build_m1_manifest(batch=8)
            rc = run_manifest(manifest)
            if rc != 0:
                raise SystemExit(rc)
            row = find_success_row(ROUND14_CSV, "R14-M1_YOLO11s_N2")
            if row:
                fp_json = measure_negative_fp("R14-M1_YOLO11s_N2", Path(row["weights_path"]))
                if fp_json:
                    fp_jsons.append(fp_json)
                    write_experiment_bundle(row, n2_row, hn1_row, load_json(fp_json))

    m1_row = find_success_row(ROUND14_CSV, "R14-M1_YOLO11s_N2")
    hn2_row = find_success_row(ROUND14_CSV, "R14-HN2_N2_HardNegative150")

    if "M2" in only:
        if not m1_row:
            status_payload["blocked_experiments"].append(
                {
                    "experiment": "R14-M2_YOLO11s_HN150",
                    "reason": "Skipped because YOLO11s baseline did not complete successfully.",
                }
            )
        elif fv(m1_row.get("map50_95")) - fv(n2_row.get("map50_95")) < 0.010 and not (
            hn2_row and fv(hn2_row.get("map50_95")) >= 0.350
        ):
            status_payload["blocked_experiments"].append(
                {
                    "experiment": "R14-M2_YOLO11s_HN150",
                    "reason": "Skipped because neither YOLO11s nor HN150 met the round14 trigger condition.",
                }
            )
        else:
            status_payload["blocked_experiments"].append(
                {
                    "experiment": "R14-M2_YOLO11s_HN150",
                    "reason": "Not executed in this run because HN150 was unavailable or manual negative curation is still pending.",
                }
            )

    qa1_payload = load_json(ROUND14_QA_JSON) if ROUND14_QA_JSON.exists() else {}
    if qa1_payload and qa1_payload.get("clean_train_recommended"):
        status_payload["blocked_experiments"].append(
            {
                "experiment": "R14-QA2_N2_CleanTrain",
                "reason": "QA1 suggests a clean-train branch, but QA2 remains pending manual label correction and a reviewed label_fix_log.",
            }
        )
    else:
        status_payload["blocked_experiments"].append(
            {
                "experiment": "R14-QA2_N2_CleanTrain",
                "reason": "Not executed because QA1 has not yet produced a reviewed clean-train label set.",
            }
        )

    if "FP" in only:
        fp_payloads = [load_json(path) for path in fp_jsons if path.exists()]
        write_deployment_fp_summary(ROUND14_DEPLOYMENT_FP_CSV, fp_payloads)

    write_status(status_payload)

    if "SUMMARY" in only:
        summarize_round14(fp_jsons)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
