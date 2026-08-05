from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def run_step(command: list[str], cwd: Path) -> None:
    print(f"[round3] running: {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_success_row(path: Path, experiment_id: str) -> dict[str, str] | None:
    for row in read_csv_rows(path):
        if row.get("experiment_id") == experiment_id and row.get("run_status") == "success":
            return row
    return None


def value(row: dict[str, str] | None, key: str) -> float:
    if not row:
        return float("-inf")
    try:
        return float(row.get(key, "") or "")
    except (TypeError, ValueError):
        return float("-inf")


def should_run_t2(r1_row: dict[str, str], t1_row: dict[str, str]) -> bool:
    r1_map = value(r1_row, "map50_95")
    t1_map = value(t1_row, "map50_95")
    r1_missing = value(r1_row, "missing_rivet_ap50_95")
    t1_missing = value(t1_row, "missing_rivet_ap50_95")
    return (t1_map >= r1_map) or (t1_missing > r1_missing and t1_map >= r1_map - 0.005)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aircraft AI round-3 experiments and summarize results.")
    parser.add_argument("--dry-run", action="store_true", help="Run manifest dry-run only.")
    parser.add_argument("--force-t2", action="store_true", help="Always run T2 instead of using the T1 gate.")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    python_exe = Path(sys.executable).resolve()
    manifest = repo / "experiments" / "aircraft_ai_dataset_round3.json"
    suite_csv = repo / "results" / "aircraft_ai_dataset_round3.csv"
    summary_md = repo / "analysis" / "aircraft_ai_dataset_round3_summary.md"
    summary_script = repo / "scripts" / "summarize_aircraft_ai_round3.py"
    mainline_csv = repo / "results" / "aircraft_ai_dataset_mainline.csv"
    round2_csv = repo / "results" / "aircraft_ai_dataset_round2.csv"

    os.environ["YOLO_CONFIG_DIR"] = str((repo / ".yolo-config").resolve())

    print(f"[round3] repo={repo}")
    print(f"[round3] manifest={manifest}")
    print(f"[round3] csv={suite_csv}")
    print(f"[round3] summary={summary_md}")

    if args.dry_run:
        run_step(
            [
                str(python_exe),
                str(repo / "run_experiment_suite.py"),
                "--manifest",
                str(manifest),
                "--export-format",
                "none",
                "--dry-run",
            ],
            repo,
        )
        print("[round3] dry-run done")
        return

    if suite_csv.exists():
        suite_csv.unlink()

    run_step(
        [
            str(python_exe),
            str(repo / "run_experiment_suite.py"),
            "--manifest",
            str(manifest),
            "--export-format",
            "none",
            "--only",
            "T1",
        ],
        repo,
    )

    r1_row = find_success_row(round2_csv, "R1")
    t1_row = find_success_row(suite_csv, "T1")
    run_t2 = args.force_t2
    if not run_t2 and r1_row and t1_row:
        run_t2 = should_run_t2(r1_row, t1_row)

    if run_t2:
        print("[round3] T1 passed the gate; continuing with T2.")
        run_step(
            [
                str(python_exe),
                str(repo / "run_experiment_suite.py"),
                "--manifest",
                str(manifest),
                "--export-format",
                "none",
                "--only",
                "T2",
            ],
            repo,
        )
    else:
        print("[round3] T1 did not pass the gate; T2 skipped by design.")

    run_step(
        [
            str(python_exe),
            str(summary_script),
            "--mainline-csv",
            str(mainline_csv),
            "--round2-csv",
            str(round2_csv),
            "--round3-csv",
            str(suite_csv),
            "--output",
            str(summary_md),
        ],
        repo,
    )

    print("[round3] done")


if __name__ == "__main__":
    main()
