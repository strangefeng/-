from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_row(rows: list[dict[str, str]], experiment_id: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("experiment_id") == experiment_id and row.get("run_status") == "success":
            return row
    return None


def f(value: str | None) -> float | None:
    try:
        return float(value or "")
    except (TypeError, ValueError):
        return None


def metric_from_payload(row: dict[str, str], class_name: str, key: str) -> str:
    raw = row.get("class_metrics_json", "")
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    return str(payload.get(class_name, {}).get(key, ""))


def compare_delta(base: dict[str, str], target: dict[str, str], key: str) -> str:
    base_v = f(base.get(key))
    target_v = f(target.get(key))
    if base_v is None or target_v is None:
        return ""
    return f"{target_v - base_v:+.6f}"


def build_report(anchor_b0: dict[str, str], anchor_r1: dict[str, str], round3_rows: list[dict[str, str]]) -> str:
    ordered: list[dict[str, str]] = [anchor_b0, anchor_r1]
    ordered.extend(row for row in round3_rows if row.get("run_status") == "success")

    lines: list[str] = []
    lines.append("# aircraft_ai_dataset_round3 Comparison Summary")
    lines.append("")
    lines.append("## Included Models")
    lines.append("")
    lines.append("- `B0`: round-1 official baseline anchor")
    lines.append("- `R1`: round-2 high-resolution anchor")
    lines.append("- `T1/T2`: round-3 newly trained experiments")
    lines.append("")
    lines.append("## Main Comparison")
    lines.append("")
    headers = [
        "exp",
        "title",
        "map50",
        "map50-95",
        "precision",
        "recall",
        "missing-rivet",
        "rust",
        "scratch",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in ordered:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("experiment_id", ""),
                    row.get("experiment_title", ""),
                    row.get("map50", ""),
                    row.get("map50_95", ""),
                    row.get("precision", ""),
                    row.get("recall", ""),
                    metric_from_payload(row, "missing-rivet", "ap50_95"),
                    metric_from_payload(row, "rust", "ap50_95"),
                    metric_from_payload(row, "scratch", "ap50_95"),
                ]
            )
            + " |"
        )
    lines.append("")

    round3_success = [row for row in round3_rows if row.get("run_status") == "success"]
    if round3_success:
        best_round3 = sorted(round3_success, key=lambda r: f(r.get("map50_95")) or float("-inf"), reverse=True)[0]
        lines.append("## Round-3 Best vs Anchors")
        lines.append("")
        lines.append(f"- Best round-3 candidate: `{best_round3.get('experiment_id', '')}` - {best_round3.get('experiment_title', '')}")
        lines.append(f"- Delta vs `R1` on `mAP50-95`: `{compare_delta(anchor_r1, best_round3, 'map50_95')}`")
        lines.append(
            f"- Delta vs `R1` on `missing-rivet AP50-95`: "
            f"`{(f(metric_from_payload(best_round3, 'missing-rivet', 'ap50_95')) or 0) - (f(metric_from_payload(anchor_r1, 'missing-rivet', 'ap50_95')) or 0):+.6f}`"
        )
        lines.append(f"- Delta vs `B0` on `mAP50-95`: `{compare_delta(anchor_b0, best_round3, 'map50_95')}`")
        lines.append("")

        lines.append("## Recommendation")
        lines.append("")
        best_map = f(best_round3.get("map50_95")) or float("-inf")
        b0_map = f(anchor_b0.get("map50_95")) or float("-inf")
        r1_map = f(anchor_r1.get("map50_95")) or float("-inf")
        best_missing = f(metric_from_payload(best_round3, "missing-rivet", "ap50_95")) or float("-inf")
        r1_missing = f(metric_from_payload(anchor_r1, "missing-rivet", "ap50_95")) or float("-inf")
        if best_map > r1_map and best_map >= b0_map:
            lines.append("- Round 3 produced a candidate that is stronger than both anchors on `mAP50-95`, so it can be promoted as the new main model.")
        elif best_missing > r1_missing:
            lines.append("- Round 3 improved the key small-object class `missing-rivet`, but the global gain is still limited, so it should be discussed as a targeted improvement rather than a full replacement.")
        else:
            lines.append("- Round 3 did not surpass the anchors clearly enough; the current evidence still favors `B0/R1` as the most stable references.")
        lines.append("")

    blocked = [row for row in round3_rows if row.get("run_status") == "blocked"]
    if blocked:
        lines.append("## Blocked Runs")
        lines.append("")
        for row in blocked:
            lines.append(f"- `{row.get('experiment_id', '')}`: {row.get('blocked_reason', '')}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize round-3 results against B0 and R1 anchors.")
    parser.add_argument("--mainline-csv", type=Path, required=True)
    parser.add_argument("--round2-csv", type=Path, required=True)
    parser.add_argument("--round3-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    b0 = find_row(read_rows(args.mainline_csv), "B0")
    r1 = find_row(read_rows(args.round2_csv), "R1")
    if not b0 or not r1:
        raise SystemExit("Could not locate B0 or R1 anchors.")
    round3_rows = read_rows(args.round3_csv)
    report = build_report(b0, r1, round3_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"summary_md={args.output}")


if __name__ == "__main__":
    main()
