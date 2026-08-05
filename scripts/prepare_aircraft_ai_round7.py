from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(command: list[str], cwd: Path) -> None:
    print(f"[round7-prepare] running: {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cleaned datasets and targeted review packs for Aircraft AI round 7.")
    parser.add_argument("--skip-review-pack", action="store_true", help="Skip targeted review pack generation.")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    python_exe = Path(sys.executable).resolve()

    source_v1 = repo / "datasets" / "supplementary_3cls" / "aircraft_ai_dataset_3cls" / "data.yaml"
    reviewed_manifest = repo.parent / "review_outputs" / "aircraft_ai_round6_label_review_pack" / "manifest.json"
    generated_root = repo / "datasets" / "generated_3cls"
    manual_clean_pilot = generated_root / "aircraft_ai_dataset_3cls_manual_clean_v1_pilot"
    manual_clean_targeted = generated_root / "aircraft_ai_dataset_3cls_manual_clean_v1_targeted"

    run_step(
        [
            str(python_exe),
            str(repo / "scripts" / "build_manual_clean_dataset.py"),
            "--source-yaml",
            str(source_v1),
            "--review-manifest",
            str(reviewed_manifest),
            "--output-root",
            str(manual_clean_pilot),
        ],
        repo,
    )

    if not args.skip_review_pack:
        review_output = repo.parent / "review_outputs" / "aircraft_ai_round7_targeted_review"
        review_pack = repo.parent / "review_outputs" / "aircraft_ai_round7_targeted_review_pack"
        targeted_manifest = review_pack / "train_only_manifest.json"
        targeted_report = review_pack / "targeted_cleanup_report.json"
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
                "--area-small",
                "0.0020",
                "--dim-small",
                "0.030",
                "--boxes-dense",
                "20",
                "--small-ratio-suspicious",
                "0.40",
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
                "120",
                "--flat",
            ],
            repo.parent,
        )
        run_step(
            [
                str(python_exe),
                str(repo / "scripts" / "prepare_aircraft_ai_round7_targeted_cleanup.py"),
                "--review-summary",
                str(review_output / "summary.json"),
                "--review-manifest",
                str(review_pack / "manifest.json"),
                "--output-manifest",
                str(targeted_manifest),
                "--report",
                str(targeted_report),
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
                str(targeted_manifest),
                "--output-root",
                str(manual_clean_targeted),
            ],
            repo,
        )

    print("[round7-prepare] done")


if __name__ == "__main__":
    main()
