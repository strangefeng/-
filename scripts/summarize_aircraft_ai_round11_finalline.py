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


def build_report(anchors: list[dict[str, str]], round11_rows: list[dict[str, str]]) -> str:
    success_rows = [row for row in round11_rows if row.get("run_status") == "success"]
    ordered = anchors + sorted(success_rows, key=lambda row: fv(row.get("map50_95")), reverse=True)
    n2 = next(row for row in anchors if row.get("experiment_id") == "N2")

    lines: list[str] = []
    lines.append("# aircraft_ai_dataset_round11_finalline Summary")
    lines.append("")
    lines.append("## Main Comparison")
    lines.append("")
    headers = ["exp", "title", "dataset_role", "map50", "map50-95", "precision", "recall", "missing-rivet", "rust", "scratch"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in ordered:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("experiment_id", ""),
                    row.get("experiment_title", ""),
                    row.get("dataset_role", ""),
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
        lines.append(f"- Best round-11 experiment: `{best.get('experiment_id', '')}` - {best.get('experiment_title', '')}")
        lines.append(f"- Delta vs `N2` on `mAP50-95`: `{fv(best.get('map50_95')) - fv(n2.get('map50_95')):+.6f}`")
        lines.append(
            f"- Delta vs `N2` on `missing-rivet AP50-95`: "
            f"`{fv(class_ap95(best, 'missing-rivet')) - fv(class_ap95(n2, 'missing-rivet')):+.6f}`"
        )
        lines.append(
            f"- Delta vs `N2` on `rust AP50-95`: "
            f"`{fv(class_ap95(best, 'rust')) - fv(class_ap95(n2, 'rust')):+.6f}`"
        )
        lines.append(
            f"- Delta vs `N2` on `scratch AP50-95`: "
            f"`{fv(class_ap95(best, 'scratch')) - fv(class_ap95(n2, 'scratch')):+.6f}`"
        )
        lines.append("")
        lines.append("## Recommendation")
        lines.append("")
        if fv(best.get("map50_95")) > fv(n2.get("map50_95")):
            lines.append("- The data-distribution repair line beat the N2 anchor, so this branch should become the final mainline candidate.")
        elif fv(class_ap95(best, "missing-rivet")) > fv(class_ap95(n2, "missing-rivet")) and fv(best.get("map50_95")) >= fv(n2.get("map50_95")) - 0.005:
            lines.append("- The new branch mainly improved the key missing-rivet class while keeping overall performance close, so it is a viable final-task-oriented alternative.")
        else:
            lines.append("- Missing-rivet oversampling did not beat the N2 anchor clearly enough, so N2 should remain the final mainline reference.")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize round-11 final-line results.")
    parser.add_argument("--round4-csv", type=Path, required=True)
    parser.add_argument("--round7-csv", type=Path, required=True)
    parser.add_argument("--round11-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    n2 = find_success_row(args.round4_csv, "N2")
    g2 = find_success_row(args.round7_csv, "G2")
    if not n2 or not g2:
        raise SystemExit("Could not locate one or more anchor rows (N2, G2).")

    report = build_report([n2, g2], read_rows(args.round11_csv))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"summary_md={args.output}")


if __name__ == "__main__":
    main()
