from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def float_value(value: str | None) -> float:
    try:
        return float(value or "")
    except (TypeError, ValueError):
        return float("-inf")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def discover_class_names(rows: list[dict[str, str]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        raw = row.get("class_metrics_json", "")
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for class_name in payload.keys():
            if class_name not in names:
                names.append(class_name)
    return names


def best_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    success_rows = [row for row in rows if row.get("run_status") == "success"]
    if not success_rows:
        return None
    return sorted(success_rows, key=lambda row: float_value(row.get("map50_95")), reverse=True)[0]


def build_markdown(rows: list[dict[str, str]], suite_id: str) -> str:
    success_rows = [row for row in rows if row.get("run_status") == "success"]
    blocked_rows = [row for row in rows if row.get("run_status") == "blocked"]
    classes = discover_class_names(rows)
    top = best_row(rows)

    lines: list[str] = []
    lines.append(f"# {suite_id} Experiment Summary")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Total rows: `{len(rows)}`")
    lines.append(f"- Successful runs: `{len(success_rows)}`")
    lines.append(f"- Blocked runs: `{len(blocked_rows)}`")
    lines.append("")

    if top:
        lines.append("## Best Current Model")
        lines.append("")
        lines.append(f"- Experiment: `{top.get('experiment_id', '')}` - {top.get('experiment_title', '')}")
        lines.append(f"- mAP50: `{top.get('map50', '')}`")
        lines.append(f"- mAP50-95: `{top.get('map50_95', '')}`")
        lines.append(f"- Precision: `{top.get('precision', '')}`")
        lines.append(f"- Recall: `{top.get('recall', '')}`")
        lines.append(f"- Run dir: `{top.get('run_dir', '')}`")
        lines.append("")

    if success_rows:
        lines.append("## Result Table")
        lines.append("")
        headers = ["exp", "title", "map50", "map50-95", "precision", "recall", *classes]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in sorted(success_rows, key=lambda r: float_value(r.get("map50_95")), reverse=True):
            payload = {}
            raw = row.get("class_metrics_json", "")
            if raw:
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {}
            class_cells = []
            for class_name in classes:
                class_metrics = payload.get(class_name, {})
                class_cells.append(class_metrics.get("ap50_95", ""))
            values = [
                row.get("experiment_id", ""),
                row.get("experiment_title", ""),
                row.get("map50", ""),
                row.get("map50_95", ""),
                row.get("precision", ""),
                row.get("recall", ""),
                *class_cells,
            ]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    if top:
        lines.append("## Recommendation")
        lines.append("")
        lines.append(
            f"- Recommended baseline reference: `{success_rows[0].get('experiment_id', '') if success_rows else ''}` should remain the baseline anchor if it is the official default run."
        )
        lines.append(
            f"- Recommended best model candidate: `{top.get('experiment_id', '')}` because it currently has the highest `mAP50-95`."
        )
        lines.append(
            "- Use the gap between the official baseline and the best candidate as the main evidence in the ablation section."
        )
        lines.append("")

    if blocked_rows:
        lines.append("## Blocked Runs")
        lines.append("")
        for row in blocked_rows:
            lines.append(f"- `{row.get('experiment_id', '')}`: {row.get('blocked_reason', '')}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Aircraft AI suite results into a markdown report.")
    parser.add_argument("--csv", type=Path, required=True, help="Suite CSV path.")
    parser.add_argument("--output", type=Path, required=True, help="Markdown output path.")
    parser.add_argument("--suite-id", default="aircraft_ai_dataset_mainline", help="Suite identifier.")
    args = parser.parse_args()

    rows = read_rows(args.csv)
    report = build_markdown(rows, args.suite_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"summary_md={args.output}")


if __name__ == "__main__":
    main()
