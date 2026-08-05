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


def build_report(
    b0: dict[str, str],
    r1: dict[str, str],
    n2: dict[str, str],
    round5_rows: list[dict[str, str]],
) -> str:
    success_rows = [row for row in round5_rows if row.get("run_status") == "success"]
    ordered = [b0, r1, n2] + sorted(success_rows, key=lambda row: fv(row.get("map50_95")), reverse=True)

    lines: list[str] = []
    lines.append("# aircraft_ai_dataset_round5 Summary")
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
        lines.append(f"- Best round-5 experiment: `{best.get('experiment_id', '')}` - {best.get('experiment_title', '')}")
        lines.append(f"- Delta vs `N2` on `mAP50-95`: `{fv(best.get('map50_95')) - fv(n2.get('map50_95')):+.6f}`")
        lines.append(
            f"- Delta vs `N2` on `missing-rivet AP50-95`: "
            f"`{fv(class_ap95(best, 'missing-rivet')) - fv(class_ap95(n2, 'missing-rivet')):+.6f}`"
        )
        lines.append(f"- Delta vs `R1` on `mAP50-95`: `{fv(best.get('map50_95')) - fv(r1.get('map50_95')):+.6f}`")
        lines.append(f"- Delta vs `B0` on `mAP50-95`: `{fv(best.get('map50_95')) - fv(b0.get('map50_95')):+.6f}`")
        lines.append("")
        lines.append("## Recommendation")
        lines.append("")
        if fv(best.get("map50_95")) > fv(n2.get("map50_95")):
            lines.append("- The fit-within export produced a new best candidate beyond the stretch-export anchor `N2`, so it should become the preferred dataset variant for follow-up work.")
        elif fv(class_ap95(best, "missing-rivet")) > fv(class_ap95(n2, "missing-rivet")) and fv(best.get("map50_95")) >= fv(n2.get("map50_95")) - 0.005:
            lines.append("- The fit-within export mainly improved the key small-object class while keeping global accuracy close, so it is worth keeping as a task-oriented data preprocessing choice.")
        else:
            lines.append("- The fit-within export did not beat the stretch-export anchor clearly enough, so `N2` on the old export should remain the safer reference for now.")
        lines.append("")

    blocked = [row for row in round5_rows if row.get("run_status") == "blocked"]
    if blocked:
        lines.append("## Blocked Runs")
        lines.append("")
        for row in blocked:
            lines.append(f"- `{row.get('experiment_id', '')}`: {row.get('blocked_reason', '')}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize round-5 results against B0, R1 and N2 anchors.")
    parser.add_argument("--mainline-csv", type=Path, required=True)
    parser.add_argument("--round2-csv", type=Path, required=True)
    parser.add_argument("--round4-csv", type=Path, required=True)
    parser.add_argument("--round5-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    b0 = find_success_row(args.mainline_csv, "B0")
    r1 = find_success_row(args.round2_csv, "R1")
    n2 = find_success_row(args.round4_csv, "N2")
    if not b0 or not r1 or not n2:
        raise SystemExit("Could not locate B0, R1 or N2 anchors.")

    round5_rows = read_rows(args.round5_csv)
    report = build_report(b0, r1, n2, round5_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"summary_md={args.output}")


if __name__ == "__main__":
    main()
