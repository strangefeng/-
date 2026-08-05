from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


N2_MAP50_95 = 0.352565
TARGET_DELTA_LOW = 0.010000
TARGET_DELTA_HIGH = 0.020000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the Round16 AeroDefect-YOLO experiment matrix.")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--round16-csv", type=Path, default=root / "results" / "aircraft_ai_dataset_round16_aerodefect.csv")
    parser.add_argument("--output-md", type=Path, default=root / "analysis" / "aircraft_ai_dataset_round16_aerodefect_summary.md")
    parser.add_argument("--output-json", type=Path, default=root / "analysis" / "aircraft_ai_dataset_round16_aerodefect_summary.json")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fv(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def latest_success_by_experiment(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("run_status") != "success":
            continue
        latest[row.get("experiment_id", "")] = row
    return latest


def class_ap(row: dict[str, str], class_name: str) -> float:
    try:
        payload = json.loads(row.get("class_metrics_json", "{}") or "{}")
        return float(payload.get(class_name, {}).get("ap50_95", 0.0) or 0.0)
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0


def main() -> int:
    args = parse_args()
    rows = latest_success_by_experiment(read_rows(args.round16_csv))
    ordered_ids = ["A1", "A2", "A3", "A4"]
    completed = [rows[exp_id] for exp_id in ordered_ids if exp_id in rows]
    best = max(completed, key=lambda row: fv(row, "map50_95"), default=None)

    summary_rows = []
    for exp_id in ordered_ids:
        row = rows.get(exp_id)
        if not row:
            summary_rows.append({"experiment_id": exp_id, "status": "pending"})
            continue
        summary_rows.append(
            {
                "experiment_id": exp_id,
                "title": row.get("experiment_title", ""),
                "map50": fv(row, "map50"),
                "map50_95": fv(row, "map50_95"),
                "delta_vs_n2": fv(row, "map50_95") - N2_MAP50_95,
                "precision": fv(row, "precision"),
                "recall": fv(row, "recall"),
                "missing_rivet": class_ap(row, "missing-rivet"),
                "rust": class_ap(row, "rust"),
                "scratch": class_ap(row, "scratch"),
                "weights_path": row.get("weights_path", ""),
                "run_dir": row.get("run_dir", ""),
                "status": "success",
            }
        )

    best_delta = fv(best, "map50_95") - N2_MAP50_95 if best else 0.0
    target_status = "not_reached"
    if best_delta >= TARGET_DELTA_HIGH:
        target_status = "reached_2_point_gain"
    elif best_delta >= TARGET_DELTA_LOW:
        target_status = "reached_1_point_gain"

    payload = {
        "n2_map50_95": N2_MAP50_95,
        "target_delta_low": TARGET_DELTA_LOW,
        "target_delta_high": TARGET_DELTA_HIGH,
        "target_status": target_status,
        "best_experiment_id": best.get("experiment_id", "") if best else "",
        "best_map50_95": fv(best, "map50_95") if best else 0.0,
        "best_delta_vs_n2": best_delta,
        "completed_count": len(completed),
        "rows": summary_rows,
    }

    lines = [
        "# aircraft_ai_dataset_round16_aerodefect Summary",
        "",
        f"- N2 standard validation mAP50-95: `{N2_MAP50_95:.6f}`",
        f"- 1-point target: `{N2_MAP50_95 + TARGET_DELTA_LOW:.6f}`",
        f"- 2-point target: `{N2_MAP50_95 + TARGET_DELTA_HIGH:.6f}`",
        f"- current target status: `{target_status}`",
        "",
        "## Main Matrix",
        "",
        "| exp | title | mAP50 | mAP50-95 | delta vs N2 | precision | recall | missing-rivet | rust | scratch |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary_rows:
        if item["status"] != "success":
            lines.append(f"| {item['experiment_id']} | pending |  |  |  |  |  |  |  |  |")
            continue
        lines.append(
            "| {experiment_id} | {title} | {map50:.6f} | {map50_95:.6f} | {delta_vs_n2:+.6f} | "
            "{precision:.6f} | {recall:.6f} | {missing_rivet:.6f} | {rust:.6f} | {scratch:.6f} |".format(**item)
        )

    lines.extend(["", "## Recommendation", ""])
    if not best:
        lines.append("- No Round16 experiment has completed yet.")
    elif best_delta >= TARGET_DELTA_LOW:
        lines.append(
            f"- Promote `{best.get('experiment_id')}` as the current SCI candidate and validate it with TTA/test-split transfer."
        )
    else:
        lines.append(
            "- Continue the matrix. If A4 still misses the +0.01 target, the next branch should tune the defect-aware P2 gate or add a clean-train/hard-negative combination rather than broad architecture churn."
        )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
