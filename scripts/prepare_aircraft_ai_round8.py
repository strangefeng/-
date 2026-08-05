from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STAGES = {
    "H1": {
        "review_output": "aircraft_ai_round8_targeted72_review",
        "review_pack": "aircraft_ai_round8_targeted72_review_pack",
        "dataset_root": "aircraft_ai_dataset_3cls_manual_clean_v2_targeted72",
        "limit": "150",
        "area_small": "0.0020",
        "dim_small": "0.030",
        "boxes_dense": "20",
        "small_ratio_suspicious": "0.30",
    },
    "H2": {
        "review_output": "aircraft_ai_round8_targeted90_review",
        "review_pack": "aircraft_ai_round8_targeted90_review_pack",
        "dataset_root": "aircraft_ai_dataset_3cls_manual_clean_v3_targeted90",
        "limit": "150",
        "area_small": "0.0020",
        "dim_small": "0.030",
        "boxes_dense": "18",
        "small_ratio_suspicious": "0.25",
    },
}


def run_step(command: list[str], cwd: Path) -> None:
    print(f"[round8-prepare] running: {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def prepare_stage(stage: str, repo: Path, python_exe: Path) -> None:
    cfg = STAGES[stage]
    source_v1 = repo / "datasets" / "supplementary_3cls" / "aircraft_ai_dataset_3cls" / "data.yaml"
    review_output = repo.parent / "review_outputs" / cfg["review_output"]
    review_pack = repo.parent / "review_outputs" / cfg["review_pack"]
    generated_root = repo / "datasets" / "generated_3cls" / cfg["dataset_root"]

    run_step(
        [
            str(python_exe),
            str(repo.parent / "review_tools" / "batch_visualize_yolo_review.py"),
            "--dataset",
            str(repo.parent / "Aircraft AI Dataset.v1i.yolov11"),
            "--output",
            str(review_output),
            "--mode",
            "suspicious",
            "--limit",
            cfg["limit"],
            "--area-small",
            cfg["area_small"],
            "--dim-small",
            cfg["dim_small"],
            "--boxes-dense",
            cfg["boxes_dense"],
            "--small-ratio-suspicious",
            cfg["small_ratio_suspicious"],
        ],
        repo.parent,
    )
    run_step(
        [
            str(python_exe),
            str(repo.parent / "review_tools" / "collect_suspicious_review_set.py"),
            "--summary",
            str(review_output / "summary.json"),
            "--output",
            str(review_pack),
            "--limit",
            cfg["limit"],
            "--flat",
        ],
        repo.parent,
    )
    run_step(
        [
            str(python_exe),
            str(repo / "scripts" / "prepare_targeted_cleanup_policy.py"),
            "--review-summary",
            str(review_output / "summary.json"),
            "--review-manifest",
            str(review_pack / "manifest.json"),
            "--output-manifest",
            str(review_pack / "train_only_manifest.json"),
            "--report",
            str(review_pack / "targeted_cleanup_report.json"),
            "--scope",
            "train_only",
        ],
        repo,
    )
    run_step(
        [
            str(python_exe),
            str(repo / "scripts" / "build_manual_clean_dataset.py"),
            "--source-yaml",
            str(source_v1),
            "--review-manifest",
            str(review_pack / "train_only_manifest.json"),
            "--output-root",
            str(generated_root),
        ],
        repo,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare expanded targeted-cleanup datasets for Aircraft AI round 8.")
    parser.add_argument("--stage", choices=sorted(STAGES), nargs="+", default=["H1"], help="Which round-8 cleanup stages to prepare.")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    python_exe = Path(sys.executable).resolve()

    for stage in args.stage:
        prepare_stage(stage, repo, python_exe)

    print("[round8-prepare] done")


if __name__ == "__main__":
    main()
