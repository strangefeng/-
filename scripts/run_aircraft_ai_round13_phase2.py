from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_YAML = ROOT / "datasets" / "supplementary_3cls" / "aircraft_ai_dataset_3cls" / "data.yaml"
ROUND4_CSV = ROOT / "results" / "aircraft_ai_dataset_round4.csv"
ROUND12_CSV = ROOT / "results" / "aircraft_ai_dataset_round12_automation.csv"
PHASE2_CSV = ROOT / "results" / "aircraft_ai_dataset_round13_phase2.csv"
PROJECT_DIR = ROOT / "runs" / "aircraft_ai_round13_phase2"

EXPECTED_N2_MAP5095 = 0.352565
EXPECTED_S2_MAP5095 = 0.344840
EXPECTED_N2_PRECISION = 0.782191


def run(command: list[str]) -> int:
    return subprocess.call(command, cwd=ROOT)


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


def resolve_n2_weights() -> Path:
    row = find_success_row(ROUND4_CSV, "N2")
    if not row:
        raise SystemExit("Could not resolve N2 weights.")
    return Path(row["weights_path"]).resolve()


def resolve_s2_row() -> dict[str, str]:
    row = find_success_row(ROUND12_CSV, "S2")
    if not row:
        raise SystemExit("Could not resolve S2 row.")
    return row


def resolve_s2_weights() -> Path:
    return Path(resolve_s2_row()["weights_path"]).resolve()


def write_temp_manifest(payload: dict) -> Path:
    tmp_root = ROOT / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="round13_phase2_", dir=str(tmp_root)))
    path = tmp_dir / "manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def base_manifest(experiment: dict) -> Path:
    payload = {
        "suite_id": "aircraft_ai_dataset_round13_phase2",
        "description": "Round13 phase2 loss and hard-negative experiments.",
        "results_csv": str(PHASE2_CSV),
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
            "mosaic": 0.0,
            "erasing": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "auto_augment": "none",
            "close_mosaic": 0,
        },
        "metadata_defaults": {
            "experiment_group": "aircraft_ai_dataset_round13_phase2",
            "dataset_role": "round13_phase2",
            "backbone_variant": "yolo11_default",
            "feature_variant": "default",
            "neck_variant": "default",
            "loss_variant": "default",
            "assigner_variant": "default",
            "notes": "Round13 phase2 keeps the N2 training policy fixed while testing localization loss and hard-negative suppression.",
        },
        "experiments": [experiment],
    }
    return write_temp_manifest(payload)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def float_or(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def step0_consistency(n2_weights: Path, s2_weights: Path) -> tuple[Path, Path]:
    n2_json = ROOT / "analysis" / "aircraft_ai_dataset_round13_c0_n2_official_val.json"
    s2_json = ROOT / "analysis" / "aircraft_ai_dataset_round13_c0_s2_official_val.json"
    commands = [
        [
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
            str(n2_json),
            "--title",
            "R13-C0 official val for N2",
        ],
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_yolo_checkpoint.py"),
            "--weights",
            str(s2_weights),
            "--dataset-yaml",
            str(SOURCE_YAML),
            "--split",
            "val",
            "--imgsz",
            "800",
            "--device",
            "0",
            "--output-json",
            str(s2_json),
            "--title",
            "R13-C0 official val for S2",
        ],
    ]
    for command in commands:
        rc = run(command)
        if rc != 0:
            raise SystemExit(rc)

    n2_payload = read_json(n2_json)
    delta = abs(float_or(n2_payload.get("map50_95")) - EXPECTED_N2_MAP5095)
    if delta > 0.002:
        raise SystemExit(
            "R13-C0 failed: N2 official val deviates more than ±0.002. "
            "Check data.yaml, class order, valid path, weights path, and evaluation script."
        )
    return n2_json, s2_json


def build_l1_manifest() -> Path:
    experiment = {
        "experiment_id": "R13-L1_N2_FocalerIoU",
        "name": "R13_L1_N2_FocalerIoU",
        "title": "N2-style full-image training with Focaler-IoU",
        "ready": True,
        "dataset_yaml": str(SOURCE_YAML.resolve()),
        "metadata": {
            "augment_profile": "n2_focaler_iou",
            "dataset_role": "n2_fullimage_focaler_iou",
            "loss_variant": "focaler_iou",
            "notes": "Keeps the N2 full-image recipe fixed and only replaces the bbox regression loss with Focaler-IoU.",
        },
    }
    return base_manifest(experiment)


def build_l2_manifest(s2_dataset_yaml: str) -> Path:
    experiment = {
        "experiment_id": "R13-L2_S2_FocalerIoU",
        "name": "R13_L2_S2_FocalerIoU",
        "title": "S2 slice512 training with Focaler-IoU",
        "ready": True,
        "dataset_yaml": str(Path(s2_dataset_yaml).resolve()),
        "metadata": {
            "augment_profile": "s2_focaler_iou",
            "dataset_role": "s2_slice512_focaler_iou",
            "loss_variant": "focaler_iou",
            "notes": "Keeps the S2 sliced-training data fixed and only replaces bbox regression with Focaler-IoU.",
        },
    }
    return base_manifest(experiment)


def prepare_hn_datasets(n2_weights: Path, negative_images_dir: Path | None) -> tuple[Path, Path]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_aircraft_ai_round13.py"),
        "--weights",
        str(n2_weights),
        "--stage",
        "HN1_BASE",
        "HN1_NEG_ONLY",
    ]
    if negative_images_dir:
        command.extend(["--negative-images-dir", str(negative_images_dir.resolve())])
    rc = run(command)
    if rc != 0:
        raise SystemExit(rc)
    return (
        ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_hardneg50" / "data.yaml",
        ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_hardneg50_negonly" / "data.yaml",
    )


def build_hn1_manifest(hn_dataset_yaml: Path) -> Path:
    experiment = {
        "experiment_id": "R13-HN1_N2_HardNegative50",
        "name": "R13_HN1_N2_HardNegative50",
        "title": "N2-style full-image training with 50 hard negatives",
        "ready": True,
        "dataset_yaml": str(hn_dataset_yaml.resolve()),
        "metadata": {
            "augment_profile": "n2_hard_negative_50",
            "dataset_role": "n2_hard_negative_50",
            "notes": "Adds 50 hard-negative empty-label images into the N2 train split to reduce false positives on rivets, seams, reflections, shadows, and texture patterns.",
        },
    }
    return base_manifest(experiment)


def build_combined_slice_hn_dataset(s2_dataset_yaml: str, negative_only_yaml: Path) -> Path:
    s2_root = Path(s2_dataset_yaml).resolve().parent
    output_root = ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_slice512_o20_plus_hn50"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "build_combined_yolo_dataset.py"),
        "--reference-yaml",
        str(SOURCE_YAML),
        "--output-root",
        str(output_root),
        "--source-datasets",
        str(s2_root),
        str(negative_only_yaml.resolve().parent),
    ]
    rc = run(command)
    if rc != 0:
        raise SystemExit(rc)
    return output_root / "data.yaml"


def build_final_f1_manifest(hn_dataset_yaml: Path) -> Path:
    experiment = {
        "experiment_id": "R13-F1_N2_FocalerIoU_HN50",
        "name": "R13_F1_N2_FocalerIoU_HN50",
        "title": "N2-style training with Focaler-IoU and HN50",
        "ready": True,
        "dataset_yaml": str(hn_dataset_yaml.resolve()),
        "metadata": {
            "augment_profile": "n2_focaler_iou_hn50",
            "dataset_role": "n2_focaler_iou_hn50",
            "loss_variant": "focaler_iou",
            "notes": "Combines the N2-style focaler-IoU regression loss with 50 hard negatives.",
        },
    }
    return base_manifest(experiment)


def build_final_f2_manifest(slice_hn_dataset_yaml: Path) -> Path:
    experiment = {
        "experiment_id": "R13-F2_S2_FocalerIoU_HN50",
        "name": "R13_F2_S2_FocalerIoU_HN50",
        "title": "S2 slice512 training with Focaler-IoU and HN50",
        "ready": True,
        "dataset_yaml": str(slice_hn_dataset_yaml.resolve()),
        "metadata": {
            "augment_profile": "s2_focaler_iou_hn50",
            "dataset_role": "s2_focaler_iou_hn50",
            "loss_variant": "focaler_iou",
            "notes": "Combines the S2 sliced-training data with Focaler-IoU and 50 hard negatives.",
        },
    }
    return base_manifest(experiment)


def build_seed_rerun_manifest(base_row: dict[str, str], seed: int) -> Path:
    experiment = {
        "experiment_id": f"{base_row['experiment_id']}_seed{seed}",
        "name": f"{base_row['experiment_id']}_seed{seed}",
        "title": f"{base_row['experiment_title']} (seed {seed})",
        "ready": True,
        "dataset_yaml": str(Path(base_row["dataset_yaml"]).resolve()),
        "metadata": {
            "augment_profile": base_row.get("augment_profile", ""),
            "dataset_role": base_row.get("dataset_role", ""),
            "feature_variant": base_row.get("feature_variant", "default"),
            "backbone_variant": base_row.get("backbone_variant", "yolo11_default"),
            "neck_variant": base_row.get("neck_variant", "default"),
            "loss_variant": base_row.get("loss_variant", "default"),
            "assigner_variant": base_row.get("assigner_variant", "default"),
            "notes": f"Phase2 seed rerun for {base_row['experiment_id']}.",
        },
        "overrides": {"seed": seed},
    }
    return base_manifest(experiment)


def run_manifest(manifest: Path) -> None:
    rc = run([sys.executable, str(ROOT / "run_experiment_suite.py"), "--manifest", str(manifest)])
    if rc != 0:
        raise SystemExit(rc)


def measure_negative_fp(weights: Path, dataset_yaml: Path, output_json: Path, title: str) -> None:
    rc = run(
        [
            sys.executable,
            str(ROOT / "scripts" / "measure_negative_false_positives.py"),
            "--weights",
            str(weights),
            "--dataset-yaml",
            str(dataset_yaml),
            "--split",
            "train",
            "--imgsz",
            "800",
            "--conf",
            "0.001",
            "--high-conf",
            "0.25",
            "--device",
            "0",
            "--output-json",
            str(output_json),
            "--title",
            title,
        ]
    )
    if rc != 0:
        raise SystemExit(rc)


def is_l1_effective(row: dict[str, str] | None) -> bool:
    if not row:
        return False
    return float_or(row.get("map50_95")) > EXPECTED_N2_MAP5095


def is_l2_effective(row: dict[str, str] | None) -> bool:
    if not row:
        return False
    return float_or(row.get("map50_95")) > EXPECTED_S2_MAP5095 and float_or(row.get("recall")) >= 0.72


def is_hn1_effective(row: dict[str, str] | None, n2_neg_fp: dict | None, hn1_neg_fp: dict | None) -> bool:
    if not row or not n2_neg_fp or not hn1_neg_fp:
        return False
    precision_gain = float_or(row.get("precision")) > EXPECTED_N2_PRECISION
    fp_drop = float_or(hn1_neg_fp.get("high_conf_false_positive_boxes")) < float_or(n2_neg_fp.get("high_conf_false_positive_boxes"))
    map_floor = float_or(row.get("map50_95")) >= EXPECTED_N2_MAP5095 - 0.010
    return precision_gain and fp_drop and map_floor


def main() -> int:
    parser = argparse.ArgumentParser(description="Run round13 phase2 experiments in strict sequence.")
    parser.add_argument("--negative-images-dir", type=Path, help="Optional directory of standalone normal-surface hard negatives.")
    parser.add_argument("--enable-wiseiou", action="store_true", help="Reserved flag; Wise-IoU is optional and skipped by default.")
    args = parser.parse_args()

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    n2_weights = resolve_n2_weights()
    s2_row = resolve_s2_row()
    s2_weights = resolve_s2_weights()

    c0_n2_json, c0_s2_json = step0_consistency(n2_weights, s2_weights)

    run_manifest(build_l1_manifest())
    l1_row = find_success_row(PHASE2_CSV, "R13-L1_N2_FocalerIoU")

    run_manifest(build_l2_manifest(s2_row["dataset_yaml"]))
    l2_row = find_success_row(PHASE2_CSV, "R13-L2_S2_FocalerIoU")

    hn_dataset_yaml, negative_only_yaml = prepare_hn_datasets(n2_weights, args.negative_images_dir)
    run_manifest(build_hn1_manifest(hn_dataset_yaml))
    hn1_row = find_success_row(PHASE2_CSV, "R13-HN1_N2_HardNegative50")

    n2_neg_fp_json = ROOT / "analysis" / "aircraft_ai_dataset_round13_phase2_n2_negative_fp.json"
    hn1_neg_fp_json = ROOT / "analysis" / "aircraft_ai_dataset_round13_phase2_hn1_negative_fp.json"
    measure_negative_fp(n2_weights, hn_dataset_yaml, n2_neg_fp_json, "N2 negative-image false positives")
    if hn1_row:
        measure_negative_fp(Path(hn1_row["weights_path"]).resolve(), hn_dataset_yaml, hn1_neg_fp_json, "HN1 negative-image false positives")

    n2_neg_fp = read_json(n2_neg_fp_json) if n2_neg_fp_json.exists() else None
    hn1_neg_fp = read_json(hn1_neg_fp_json) if hn1_neg_fp_json.exists() else None

    if args.enable_wiseiou:
        print("[round13-phase2] Wise-IoU v3 requested, but it remains skipped in this run because no stable local implementation is wired.")

    final_candidates: list[dict[str, str]] = []
    if is_l1_effective(l1_row) and is_hn1_effective(hn1_row, n2_neg_fp, hn1_neg_fp):
        run_manifest(build_final_f1_manifest(hn_dataset_yaml))
        row = find_success_row(PHASE2_CSV, "R13-F1_N2_FocalerIoU_HN50")
        if row:
            final_candidates.append(row)

    if is_l2_effective(l2_row) and is_hn1_effective(hn1_row, n2_neg_fp, hn1_neg_fp):
        slice_hn_yaml = build_combined_slice_hn_dataset(s2_row["dataset_yaml"], negative_only_yaml)
        run_manifest(build_final_f2_manifest(slice_hn_yaml))
        row = find_success_row(PHASE2_CSV, "R13-F2_S2_FocalerIoU_HN50")
        if row:
            final_candidates.append(row)

    rerun_targets: list[dict[str, str]] = []
    for row in [l1_row, l2_row, hn1_row, *final_candidates]:
        if row and float_or(row.get("map50_95")) > EXPECTED_N2_MAP5095:
            rerun_targets.append(row)

    for row in rerun_targets:
        run_manifest(build_seed_rerun_manifest(row, 2026))

    rc = run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_aircraft_ai_round13_phase2.py"),
            "--round4-csv",
            str(ROUND4_CSV),
            "--round12-csv",
            str(ROUND12_CSV),
            "--phase2-csv",
            str(PHASE2_CSV),
            "--c0-n2-json",
            str(c0_n2_json),
            "--c0-s2-json",
            str(c0_s2_json),
            "--n2-neg-fp-json",
            str(n2_neg_fp_json),
            "--hn1-neg-fp-json",
            str(hn1_neg_fp_json),
            "--output-md",
            str(ROOT / "analysis" / "aircraft_ai_dataset_round13_phase2_summary.md"),
            "--output-csv",
            str(ROOT / "results" / "round13_phase2_summary.csv"),
        ]
    )
    if rc != 0:
        return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
