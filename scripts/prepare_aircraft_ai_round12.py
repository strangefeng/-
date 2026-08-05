from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_YAML = ROOT / "datasets" / "supplementary_3cls" / "aircraft_ai_dataset_3cls" / "data.yaml"
ROUND4_CSV = ROOT / "results" / "aircraft_ai_dataset_round4.csv"


STAGES = {
    "S2": {
        "builder": "build_sliced_yolo_dataset.py",
        "args": [
            "--source-yaml",
            str(SOURCE_YAML),
            "--output-root",
            str(ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_slice512_o20"),
            "--mode",
            "sliced",
            "--slice-size",
            "512",
            "--overlap",
            "0.20",
            "--max-empty-tiles-per-image",
            "1",
        ],
    },
    "S3": {
        "builder": "build_sliced_yolo_dataset.py",
        "args": [
            "--source-yaml",
            str(SOURCE_YAML),
            "--output-root",
            str(ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_mixedslice512_o20"),
            "--mode",
            "mixed",
            "--slice-size",
            "512",
            "--overlap",
            "0.20",
            "--max-empty-tiles-per-image",
            "1",
        ],
    },
    "P1": {
        "builder": "build_defect_patch_dataset.py",
        "args": [
            "--source-yaml",
            str(SOURCE_YAML),
            "--output-root",
            str(ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_patch_mr"),
            "--target-classes",
            "missing-rivet",
            "--copies-per-image",
            "1",
            "--pastes-per-copy",
            "2",
            "--prefer-bright",
        ],
    },
    "P2": {
        "builder": "build_defect_patch_dataset.py",
        "args": [
            "--source-yaml",
            str(SOURCE_YAML),
            "--output-root",
            str(ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_patch_mr_scratch"),
            "--target-classes",
            "missing-rivet",
            "scratch",
            "--copies-per-image",
            "1",
            "--pastes-per-copy",
            "2",
            "--prefer-bright",
        ],
    },
    "P3": {
        "builder": "build_defect_patch_dataset.py",
        "args": [
            "--source-yaml",
            str(SOURCE_YAML),
            "--output-root",
            str(ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_patch_mr_scratch_photo"),
            "--target-classes",
            "missing-rivet",
            "scratch",
            "--copies-per-image",
            "1",
            "--pastes-per-copy",
            "2",
            "--brightness-jitter",
            "0.12",
            "--contrast-jitter",
            "0.10",
            "--blur-prob",
            "0.20",
            "--noise-std",
            "4.0",
            "--prefer-bright",
        ],
    },
    "HN1": {
        "builder": "build_hard_negative_dataset.py",
        "needs_weights": True,
        "args": [
            "--source-yaml",
            str(SOURCE_YAML),
            "--output-root",
            str(ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_hardneg50"),
            "--splits",
            "valid",
            "test",
            "--imgsz",
            "800",
            "--conf",
            "0.25",
            "--max-negatives",
            "50",
            "--min-crop-edge",
            "160",
            "--max-crop-edge",
            "320",
        ],
    },
    "HN2": {
        "builder": "build_hard_negative_dataset.py",
        "needs_weights": True,
        "args": [
            "--source-yaml",
            str(SOURCE_YAML),
            "--output-root",
            str(ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_hardneg100"),
            "--splits",
            "valid",
            "test",
            "--imgsz",
            "800",
            "--conf",
            "0.25",
            "--max-negatives",
            "100",
            "--min-crop-edge",
            "160",
            "--max-crop-edge",
            "320",
        ],
    },
}
DEFAULT_STAGE_ORDER = ["S2", "S3", "P1", "P2", "P3", "HN1", "HN2"]


def run_step(command: list[str]) -> None:
    print(f"[round12-prepare] running: {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(ROOT), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def resolve_n2_weights(explicit: Path | None) -> Path:
    if explicit:
        weights = explicit.resolve()
        if not weights.exists():
            raise SystemExit(f"N2 weights not found: {weights}")
        return weights
    if not ROUND4_CSV.exists():
        raise SystemExit(f"Round-4 CSV not found: {ROUND4_CSV}")

    best_row: dict[str, str] | None = None
    with ROUND4_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("experiment_id") != "N2" or row.get("run_status") != "success":
                continue
            if best_row is None or float(row.get("map50_95", "0") or 0.0) > float(best_row.get("map50_95", "0") or 0.0):
                best_row = row
    if not best_row:
        raise SystemExit("Could not find a successful N2 row in round-4 results.")
    weights = Path(best_row.get("weights_path", "")).resolve()
    if not weights.exists():
        raise SystemExit(f"N2 weights missing on disk: {weights}")
    return weights


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare round-12 automation datasets.")
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=sorted(STAGES),
        default=DEFAULT_STAGE_ORDER,
        help="Which round-12 dataset builders to run.",
    )
    parser.add_argument("--weights", type=Path, help="Optional explicit N2 weights for hard-negative mining.")
    args = parser.parse_args()

    py = sys.executable
    n2_weights: Path | None = None

    for stage in args.stage:
        config = STAGES[stage]
        command = [py, str(ROOT / "scripts" / config["builder"]), *config["args"]]
        if config.get("needs_weights"):
            if n2_weights is None:
                n2_weights = resolve_n2_weights(args.weights)
            command.extend(["--weights", str(n2_weights)])
        run_step(command)

    print("[round12-prepare] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
