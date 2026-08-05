from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_step(command: list[str], cwd: Path) -> None:
    print(f"[round2] running: {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aircraft AI round-2 experiments and summarize results.")
    parser.add_argument("--dry-run", action="store_true", help="Run manifest dry-run only.")
    parser.add_argument("--only", nargs="*", help="Run only selected experiment ids.")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    python_exe = Path(sys.executable).resolve()
    manifest = repo / "experiments" / "aircraft_ai_dataset_round2.json"
    suite_csv = repo / "results" / "aircraft_ai_dataset_round2.csv"
    summary_md = repo / "analysis" / "aircraft_ai_dataset_round2_summary.md"
    summary_script = repo / "scripts" / "summarize_aircraft_ai_suite.py"

    os.environ["YOLO_CONFIG_DIR"] = str((repo / ".yolo-config").resolve())

    print(f"[round2] repo={repo}")
    print(f"[round2] manifest={manifest}")
    print(f"[round2] csv={suite_csv}")
    print(f"[round2] summary={summary_md}")

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
            "aircraft_ai_dataset_round2",
        ]
        run_step(summary_cmd, repo)

    print("[round2] done")


if __name__ == "__main__":
    main()
