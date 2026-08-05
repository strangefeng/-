from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_step(command: list[str], cwd: Path) -> None:
    print(f"[r1_rect] running: {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the R1_rect validation experiment and summarize results.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    python_exe = Path(sys.executable).resolve()
    manifest = repo / "experiments" / "aircraft_ai_dataset_r1_rect.json"
    suite_csv = repo / "results" / "aircraft_ai_dataset_r1_rect.csv"
    summary_md = repo / "analysis" / "aircraft_ai_dataset_r1_rect_summary.md"
    summary_script = repo / "scripts" / "summarize_aircraft_ai_r1_rect.py"
    mainline_csv = repo / "results" / "aircraft_ai_dataset_mainline.csv"
    round2_csv = repo / "results" / "aircraft_ai_dataset_round2.csv"

    os.environ["YOLO_CONFIG_DIR"] = str((repo / ".yolo-config").resolve())

    run_cmd = [
        str(python_exe),
        str(repo / "run_experiment_suite.py"),
        "--manifest",
        str(manifest),
        "--export-format",
        "none",
    ]
    if args.dry_run:
        run_cmd.append("--dry-run")
    run_step(run_cmd, repo)

    if not args.dry_run:
        summary_cmd = [
            str(python_exe),
            str(summary_script),
            "--mainline-csv",
            str(mainline_csv),
            "--round2-csv",
            str(round2_csv),
            "--rect-csv",
            str(suite_csv),
            "--output",
            str(summary_md),
        ]
        run_step(summary_cmd, repo)

    print("[r1_rect] done")


if __name__ == "__main__":
    main()
