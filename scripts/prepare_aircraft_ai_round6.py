from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(command: list[str], cwd: Path) -> None:
    print(f"[round6-prepare] running: {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare generated datasets and review packs for Aircraft AI round 6.")
    parser.add_argument("--skip-review-pack", action="store_true", help="Skip suspicious-label review pack generation.")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    python_exe = Path(sys.executable).resolve()
    source_v1 = repo / "datasets" / "supplementary_3cls" / "aircraft_ai_dataset_3cls" / "data.yaml"
    source_v2 = repo / "datasets" / "supplementary_3cls" / "aircraft_ai_dataset_3cls_v2_fit" / "data.yaml"

    generated_root = repo / "datasets" / "generated_3cls"
    manual_clean = generated_root / "aircraft_ai_dataset_3cls_manual_clean_v1"
    v1_focus = generated_root / "aircraft_ai_dataset_3cls_focuscrop_v1"
    v2_focus = generated_root / "aircraft_ai_dataset_3cls_focuscrop_v2_fit"
    manual_clean_focus = generated_root / "aircraft_ai_dataset_3cls_focuscrop_v1_manualclean"
    review_output = repo.parent / "review_outputs" / "aircraft_ai_round6_label_review"
    review_pack = repo.parent / "review_outputs" / "aircraft_ai_round6_label_review_pack"

    run_step(
        [
            str(python_exe),
            str(repo / "scripts" / "build_focus_crop_dataset.py"),
            "--source-yaml",
            str(source_v1),
            "--output-root",
            str(v1_focus),
            "--target-classes",
            "missing-rivet",
            "--small-area-threshold",
            "0.015",
            "--small-dim-threshold",
            "0.08",
            "--min-crop-edge",
            "256",
            "--max-crop-edge",
            "640",
            "--context-scale",
            "7.0",
            "--max-crops-per-image",
            "2",
        ],
        repo,
    )

    run_step(
        [
            str(python_exe),
            str(repo / "scripts" / "build_focus_crop_dataset.py"),
            "--source-yaml",
            str(source_v2),
            "--output-root",
            str(v2_focus),
            "--target-classes",
            "missing-rivet",
            "--small-area-threshold",
            "0.015",
            "--small-dim-threshold",
            "0.08",
            "--min-crop-edge",
            "256",
            "--max-crop-edge",
            "640",
            "--context-scale",
            "7.0",
            "--max-crops-per-image",
            "2",
        ],
        repo,
    )

    if not args.skip_review_pack:
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
                "120",
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
                "--flat",
            ],
            repo.parent,
        )

    run_step(
        [
            str(python_exe),
            str(repo / "scripts" / "build_manual_clean_dataset.py"),
            "--source-yaml",
            str(source_v1),
            "--review-manifest",
            str(review_pack / "manifest.json"),
            "--output-root",
            str(manual_clean),
        ],
        repo,
    )

    run_step(
        [
            str(python_exe),
            str(repo / "scripts" / "build_focus_crop_dataset.py"),
            "--source-yaml",
            str(manual_clean / "data.yaml"),
            "--output-root",
            str(manual_clean_focus),
            "--target-classes",
            "missing-rivet",
            "--small-area-threshold",
            "0.015",
            "--small-dim-threshold",
            "0.08",
            "--min-crop-edge",
            "256",
            "--max-crop-edge",
            "640",
            "--context-scale",
            "7.0",
            "--max-crops-per-image",
            "2",
        ],
        repo,
    )

    print("[round6-prepare] done")


if __name__ == "__main__":
    main()
