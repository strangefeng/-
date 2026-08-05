from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> int:
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    py = sys.executable
    builder = ROOT / "scripts" / "build_missing_rivet_oversample_dataset.py"
    jobs = [
        (
            ROOT / "datasets" / "supplementary_3cls" / "aircraft_ai_dataset_3cls" / "data.yaml",
            ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_mrdup_x2_public",
            "2",
        ),
        (
            ROOT / "datasets" / "supplementary_3cls" / "aircraft_ai_dataset_3cls" / "data.yaml",
            ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_mrdup_x3_public",
            "3",
        ),
        (
            ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_manual_clean_v1_targeted" / "data.yaml",
            ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_manual_clean_v1_targeted_mrdup_x2",
            "2",
        ),
    ]
    for source_yaml, output_dir, factor in jobs:
        rc = run(
            [
                py,
                str(builder),
                "--source-yaml",
                str(source_yaml),
                "--output-dir",
                str(output_dir),
                "--target-class",
                "0",
                "--duplicate-factor",
                factor,
            ]
        )
        if rc != 0:
            return rc

    return run(
        [
            py,
            str(ROOT / "run_experiment_suite.py"),
            "--manifest",
            str(ROOT / "experiments" / "aircraft_ai_dataset_round11_finalline.json"),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
