from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_step(command: list[str], cwd: Path) -> None:
    print(f"[round8] running: {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def stage_to_experiment_ids(stages: list[str] | None) -> list[str] | None:
    if not stages:
        return None
    return stages


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aircraft AI round-8 expanded targeted-cleaning experiments and summarize results.")
    parser.add_argument("--dry-run", action="store_true", help="Run manifest dry-run only.")
    parser.add_argument("--only", nargs="*", help="Run only selected experiment ids.")
    parser.add_argument("--stage", nargs="*", choices=["H1", "H2"], help="Prepare the matching cleanup stages before running.")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip round-8 dataset preparation.")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    python_exe = Path(sys.executable).resolve()
    manifest_arg = Path("experiments") / "aircraft_ai_dataset_round8.json"
    suite_csv = repo / "results" / "aircraft_ai_dataset_round8.csv"
    summary_md = repo / "analysis" / "aircraft_ai_dataset_round8_summary.md"
    summary_script = Path("scripts") / "summarize_aircraft_ai_round8.py"
    round4_csv = Path("results") / "aircraft_ai_dataset_round4.csv"
    round6_csv = Path("results") / "aircraft_ai_dataset_round6.csv"
    round7_csv = Path("results") / "aircraft_ai_dataset_round7.csv"

    os.environ["YOLO_CONFIG_DIR"] = str((repo / ".yolo-config").resolve())

    stage_selection = args.stage or args.only
    if not args.skip_prepare:
        prepare_cmd = [str(python_exe), str(repo / "scripts" / "prepare_aircraft_ai_round8.py")]
        if stage_selection:
            prepare_cmd.extend(["--stage", *stage_selection])
        run_step(prepare_cmd, repo)

    if suite_csv.exists() and not args.only and not args.dry_run:
        suite_csv.unlink()

    run_cmd = [
        str(python_exe),
        str(Path("run_experiment_suite.py")),
        "--manifest",
        str(manifest_arg),
        "--export-format",
        "none",
    ]
    selected_ids = args.only or stage_to_experiment_ids(args.stage)
    if selected_ids:
        run_cmd.append("--only")
        run_cmd.extend(selected_ids)
    if args.dry_run:
        run_cmd.append("--dry-run")

    run_step(run_cmd, repo)

    if not args.dry_run:
        summary_cmd = [
            str(python_exe),
            str(summary_script),
            "--round4-csv",
            str(round4_csv),
            "--round6-csv",
            str(round6_csv),
            "--round7-csv",
            str(round7_csv),
            "--round8-csv",
            str(suite_csv),
            "--output",
            str(summary_md),
        ]
        run_step(summary_cmd, repo)

    print("[round8] done")


if __name__ == "__main__":
    main()
