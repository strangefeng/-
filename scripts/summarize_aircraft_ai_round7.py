from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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


def class_ap95(row: dict[str, str], class_name: str) -> str:
    raw = row.get("class_metrics_json", "")
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    return str(payload.get(class_name, {}).get("ap50_95", ""))


def build_report(n2: dict[str, str], c1: dict[str, str], round7_rows: list[dict[str, str]]) -> str:
    success_rows = [row for row in round7_rows if row.get("run_status") == "success"]
    ordered = [n2, c1] + sorted(success_rows, key=lambda row: fv(row.get("map50_95")), reverse=True)

    lines: list[str] = []
    lines.append("# aircraft_ai_dataset_round7 Summary")
    lines.append("")
    lines.append("## Main Comparison")
    lines.append("")
    headers = ["exp", "title", "map50", "map50-95", "precision", "recall", "missing-rivet", "rust", "scratch"]
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
                    class_ap95(row, "missing-rivet"),
                    class_ap95(row, "rust"),
                    class_ap95(row, "scratch"),
                ]
            )
            + " |"
        )
    lines.append("")

    if success_rows:
        best = sorted(success_rows, key=lambda row: fv(row.get("map50_95")), reverse=True)[0]
        lines.append("## Best New Candidate")
        lines.append("")
        lines.append(f"- Best round-7 experiment: `{best.get('experiment_id', '')}` - {best.get('experiment_title', '')}")
        lines.append(f"- Delta vs `N2` on `mAP50-95`: `{fv(best.get('map50_95')) - fv(n2.get('map50_95')):+.6f}`")
        lines.append(
            f"- Delta vs `N2` on `missing-rivet AP50-95`: "
            f"`{fv(class_ap95(best, 'missing-rivet')) - fv(class_ap95(n2, 'missing-rivet')):+.6f}`"
        )
        lines.append(f"- Delta vs `C1` on `mAP50-95`: `{fv(best.get('map50_95')) - fv(c1.get('map50_95')):+.6f}`")
        lines.append("")
        lines.append("## Recommendation")
        lines.append("")
        if fv(best.get("map50_95")) > fv(n2.get("map50_95")):
            lines.append("- Round 7 finally showed that label cleaning alone can beat N2, so the next step should be a larger targeted-review pass rather than any new training trick.")
        elif fv(best.get("map50_95")) > fv(c1.get("map50_95")):
            lines.append("- Round 7 did not beat N2 yet, but cleaning alone already outperformed the round-6 cleaned focus-crop branch, so the next investment should be a larger 50-150 image targeted cleanup.")
        else:
            lines.append("- The round-7 pilot cleanup is still below both N2 and C1, so do not add new training ablations; expand the manual review pool first and only then rerun.")
        lines.append("")

    blocked = [row for row in round7_rows if row.get("run_status") == "blocked"]
    if blocked:
        lines.append("## Blocked Runs")
        lines.append("")
        for row in blocked:
            lines.append(f"- `{row.get('experiment_id', '')}`: {row.get('blocked_reason', '')}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize round-7 results against the N2 and C1 anchors.")
    parser.add_argument("--round4-csv", type=Path, required=True)
    parser.add_argument("--round6-csv", type=Path, required=True)
    parser.add_argument("--round7-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    n2 = find_success_row(args.round4_csv, "N2")
    c1 = find_success_row(args.round6_csv, "C1")
    if not n2 or not c1:
        raise SystemExit("Could not locate N2 or C1 anchors.")

    round7_rows = read_rows(args.round7_csv)
    report = build_report(n2, c1, round7_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"summary_md={args.output}")


if __name__ == "__main__":
    main()
