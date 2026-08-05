from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


STAGES = ["H1", "H2"]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find_success_row(path: Path, experiment_id: str) -> dict[str, str] | None:
    candidates = [
        row
        for row in read_rows(path)
        if row.get("experiment_id") == experiment_id and row.get("run_status") == "success"
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            int(row.get("epochs", "0") or 0),
            float(row.get("map50_95", "") or 0.0),
        ),
        reverse=True,
    )[0]


def fv(value: str | None) -> float:
    try:
        return float(value or "")
    except (TypeError, ValueError):
        return float("-inf")


def class_ap95(row: dict[str, str], class_name: str) -> float:
    raw = row.get("class_metrics_json", "")
    if not raw:
        return float("-inf")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return float("-inf")
    try:
        return float(payload.get(class_name, {}).get("ap50_95", ""))
    except (TypeError, ValueError):
        return float("-inf")


def run_step(command: list[str], cwd: Path) -> None:
    print(f"[autoloop] running: {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def is_actual_improvement(candidate: dict[str, str], best_clean: dict[str, str]) -> tuple[bool, dict[str, float]]:
    delta_map = fv(candidate.get("map50_95")) - fv(best_clean.get("map50_95"))
    delta_missing = class_ap95(candidate, "missing-rivet") - class_ap95(best_clean, "missing-rivet")
    delta_scratch = class_ap95(candidate, "scratch") - class_ap95(best_clean, "scratch")
    improved = delta_map >= 0.001 or (delta_map >= 0 and delta_missing >= 0.002 and delta_scratch > -0.010)
    return improved, {
        "delta_map50_95": delta_map,
        "delta_missing_rivet_ap50_95": delta_missing,
        "delta_scratch_ap50_95": delta_scratch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bounded cleanup autoloop until a meaningful gain appears or stages are exhausted.")
    parser.add_argument("--stages", nargs="*", choices=STAGES, default=STAGES, help="Cleanup stages to evaluate in order.")
    parser.add_argument("--reset", action="store_true", help="Delete the round-8 CSV before starting a fresh autoloop.")
    parser.add_argument("--evaluate-only", action="store_true", help="Do not launch training; only evaluate the requested stages from the existing round-8 CSV.")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    python_exe = Path(sys.executable).resolve()
    round4_csv = repo / "results" / "aircraft_ai_dataset_round4.csv"
    round7_csv = repo / "results" / "aircraft_ai_dataset_round7.csv"
    round8_csv = repo / "results" / "aircraft_ai_dataset_round8.csv"
    autoloop_report = repo / "analysis" / "aircraft_ai_dataset_round8_autoloop_report.json"

    n2 = find_success_row(round4_csv, "N2")
    best_clean = find_success_row(round7_csv, "G2")
    if not n2 or not best_clean:
        raise SystemExit("Could not locate N2 or G2 anchors for autoloop.")

    if args.reset and round8_csv.exists():
        round8_csv.unlink()

    history: list[dict[str, object]] = []

    for stage in args.stages:
        if not args.evaluate_only:
            run_cmd = [
                str(python_exe),
                str(repo / "scripts" / "run_aircraft_ai_round8.py"),
                "--stage",
                stage,
                "--only",
                stage,
            ]
            run_step(run_cmd, repo)
        candidate = find_success_row(round8_csv, stage)
        if not candidate:
            raise SystemExit(f"Stage {stage} did not produce a successful row.")

        improved, deltas = is_actual_improvement(candidate, best_clean)
        beats_n2 = fv(candidate.get("map50_95")) > fv(n2.get("map50_95"))
        history.append(
            {
                "stage": stage,
                "map50_95": fv(candidate.get("map50_95")),
                "missing_rivet_ap50_95": class_ap95(candidate, "missing-rivet"),
                "scratch_ap50_95": class_ap95(candidate, "scratch"),
                "deltas_vs_best_clean": deltas,
                "actual_improvement": improved,
                "beats_n2": beats_n2,
            }
        )

        if improved or beats_n2:
            best_clean = candidate
            break

    report = {
        "starting_anchor": "G2",
        "stages_requested": args.stages,
        "history": history,
        "stopped": True,
        "stop_reason": "actual_improvement" if history and history[-1]["actual_improvement"] else ("beat_n2" if history and history[-1]["beats_n2"] else "stages_exhausted"),
    }
    autoloop_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"autoloop_report={autoloop_report}")
    print(f"stop_reason={report['stop_reason']}")


if __name__ == "__main__":
    main()
