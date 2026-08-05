from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_step(command: list[str], cwd: Path) -> None:
    print(f"[suite] running: {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Aircraft AI full suite and summarize results.")
    parser.add_argument("--dry-run", action="store_true", help="Run manifest dry-run only.")
    parser.add_argument("--only", nargs="*", help="Run only selected experiment ids.")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    python_exe = Path(sys.executable).resolve()
    manifest = repo / "experiments" / "aircraft_ai_dataset_mainline.json"
    suite_csv = repo / "results" / "aircraft_ai_dataset_mainline.csv"
    summary_md = repo / "analysis" / "aircraft_ai_dataset_mainline_summary.md"
    summary_script = repo / "scripts" / "summarize_aircraft_ai_suite.py"

    os.environ["YOLO_CONFIG_DIR"] = str((repo / ".yolo-config").resolve())

    print(f"[suite] repo={repo}")
    print(f"[suite] manifest={manifest}")
    print(f"[suite] csv={suite_csv}")
    print(f"[suite] summary={summary_md}")

    run_cmd = [
        str(python_exe),
        str(repo / "run_experiment_suite.py"),
        "--manifest",
        str(manifest),
        "--export-format",
        "none",
    ]
    if args.only:
        run_cmd.append("--only")
        run_cmd.extend(args.only)
    if args.dry_run:
        run_cmd.append("--dry-run")

    run_step(run_cmd, repo)

    if not args.dry_run:
        summary_cmd = [
            str(python_exe),
            str(summary_script),
            "--csv",
            str(suite_csv),
            "--output",
            str(summary_md),
            "--suite-id",
            "aircraft_ai_dataset_mainline",
        ]
        run_step(summary_cmd, repo)

    print("[suite] done")


if __name__ == "__main__":
    main()
