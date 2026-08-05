from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_YAML = ROOT / "datasets" / "supplementary_3cls" / "aircraft_ai_dataset_3cls" / "data.yaml"
ROUND4_CSV = ROOT / "results" / "aircraft_ai_dataset_round4.csv"


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


def run_step(command: list[str]) -> None:
    print(f"[round13-prepare] running: {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(ROOT), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def stage_commands(n2_weights: Path, negative_images_dir: Path | None) -> dict[str, list[str]]:
    py = sys.executable
    commands: dict[str, list[str]] = {}

    commands["S2a"] = [
        py,
        str(ROOT / "scripts" / "build_sliced_yolo_dataset.py"),
        "--source-yaml",
        str(SOURCE_YAML),
        "--output-root",
        str(ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_slice640_o20"),
        "--mode",
        "sliced",
        "--slice-size",
        "640",
        "--overlap",
        "0.20",
        "--max-empty-tiles-per-image",
        "1",
    ]

    commands["S2b"] = [
        py,
        str(ROOT / "scripts" / "build_combined_yolo_dataset.py"),
        "--reference-yaml",
        str(SOURCE_YAML),
        "--output-root",
        str(ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_slice512_640_mix"),
        "--source-datasets",
        str(ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_slice512_o20"),
        str(ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_slice640_o20"),
    ]

    hn_base = [
        py,
        str(ROOT / "scripts" / "build_hard_negative_dataset.py"),
        "--source-yaml",
        str(SOURCE_YAML),
        "--weights",
        str(n2_weights),
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
        "--negative-limit",
        "50",
        "--min-crop-edge",
        "160",
        "--max-crop-edge",
        "320",
    ]
    if negative_images_dir:
        hn_base.extend(["--negative-images-dir", str(negative_images_dir.resolve())])
    commands["HN1_BASE"] = hn_base

    hn_neg_only = [
        py,
        str(ROOT / "scripts" / "build_hard_negative_dataset.py"),
        "--source-yaml",
        str(SOURCE_YAML),
        "--weights",
        str(n2_weights),
        "--output-root",
        str(ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_hardneg50_negonly"),
        "--splits",
        "valid",
        "test",
        "--imgsz",
        "800",
        "--conf",
        "0.25",
        "--max-negatives",
        "50",
        "--negative-limit",
        "50",
        "--min-crop-edge",
        "160",
        "--max-crop-edge",
        "320",
        "--no-copy-source-train",
    ]
    if negative_images_dir:
        hn_neg_only.extend(["--negative-images-dir", str(negative_images_dir.resolve())])
    commands["HN1_NEG_ONLY"] = hn_neg_only

    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare round-13 automation datasets.")
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=["S2a", "S2b", "HN1_BASE", "HN1_NEG_ONLY"],
        default=["S2a", "S2b", "HN1_BASE", "HN1_NEG_ONLY"],
    )
    parser.add_argument("--weights", type=Path, help="Optional explicit N2 weights.")
    parser.add_argument("--negative-images-dir", type=Path, help="Optional standalone normal-surface negative image directory.")
    args = parser.parse_args()

    n2_weights = resolve_n2_weights(args.weights)
    commands = stage_commands(n2_weights, args.negative_images_dir)
    for stage in args.stage:
        run_step(commands[stage])

    print("[round13-prepare] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
