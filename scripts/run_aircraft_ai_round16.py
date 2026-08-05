from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "aircraft_ai_dataset_round16_aerodefect.json"
RESULTS_CSV = ROOT / "results" / "aircraft_ai_dataset_round16_aerodefect.csv"
SUMMARY_SCRIPT = ROOT / "scripts" / "summarize_aircraft_ai_round16.py"
EXPERIMENT_ORDER = ["A1", "A2", "A3", "A4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Round16 AeroDefect-YOLO experiments with resume support.")
    parser.add_argument("--only", nargs="+", choices=EXPERIMENT_ORDER)
    parser.add_argument("--max-new", type=int, default=0, help="Maximum new experiments to run; 0 means no limit.")
    parser.add_argument("--epochs", type=int, help="Override epochs for smoke or continuation runs.")
    parser.add_argument("--batch", type=int, help="Override batch size.")
    parser.add_argument("--device", help="Override device.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def read_completed() -> set[str]:
    if not RESULTS_CSV.exists():
        return set()
    with RESULTS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row.get("experiment_id", "")
            for row in csv.DictReader(handle)
            if row.get("run_status") == "success"
        }


def run(command: list[str]) -> int:
    completed = subprocess.run(command, cwd=str(ROOT), check=False)
    return completed.returncode


def summarize() -> int:
    return run([sys.executable, str(SUMMARY_SCRIPT)])


def main() -> int:
    args = parse_args()
    selected = args.only or EXPERIMENT_ORDER

    list_cmd = [sys.executable, str(ROOT / "run_experiment_suite.py"), "--manifest", str(MANIFEST), "--list"]
    if args.list:
        return run(list_cmd)

    completed = read_completed()
    pending = [exp_id for exp_id in selected if exp_id not in completed]
    if args.max_new > 0:
        pending = pending[: args.max_new]

    if args.dry_run:
        print(f"completed={sorted(completed)}")
        print(f"pending={pending}")
        return 0

    if not pending:
        return summarize()

    for exp_id in pending:
        command = [
            sys.executable,
            str(ROOT / "run_experiment_suite.py"),
            "--manifest",
            str(MANIFEST),
            "--only",
            exp_id,
        ]
        if args.epochs is not None:
            command.extend(["--epochs", str(args.epochs)])
        if args.batch is not None:
            command.extend(["--batch", str(args.batch)])
        if args.device:
            command.extend(["--device", args.device])
        rc = run(command)
        if rc != 0:
            summarize()
            return rc
        summarize()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
