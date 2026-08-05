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


def fv(value: str | float | int | None) -> float:
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


def read_json(path: Path | None) -> dict | None:
    if not path or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary_csv(output_csv: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "experiment_id",
        "experiment_title",
        "dataset_role",
        "map50",
        "map50_95",
        "precision",
        "recall",
        "missing_rivet_ap50_95",
        "rust_ap50_95",
        "scratch_ap50_95",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "experiment_id": row.get("experiment_id", ""),
                    "experiment_title": row.get("experiment_title", ""),
                    "dataset_role": row.get("dataset_role", ""),
                    "map50": row.get("map50", ""),
                    "map50_95": row.get("map50_95", ""),
                    "precision": row.get("precision", ""),
                    "recall": row.get("recall", ""),
                    "missing_rivet_ap50_95": class_ap95(row, "missing-rivet"),
                    "rust_ap50_95": class_ap95(row, "rust"),
                    "scratch_ap50_95": class_ap95(row, "scratch"),
                }
            )


def build_report(
    anchor_rows: list[dict[str, str]],
    round13_rows: list[dict[str, str]],
    e1_json: dict | None,
    e2_json: dict | None,
) -> str:
    success_rows = [row for row in round13_rows if row.get("run_status") == "success"]
    combined = anchor_rows + sorted(success_rows, key=lambda row: fv(row.get("map50_95")), reverse=True)
    n2 = next(row for row in anchor_rows if row.get("experiment_id") == "N2")
    s2 = next(row for row in anchor_rows if row.get("experiment_id") == "S2")

    lines: list[str] = []
    lines.append("# aircraft_ai_dataset_round13_automation Summary")
    lines.append("")
    lines.append("## Main Comparison")
    lines.append("")
    headers = ["exp", "title", "dataset_role", "map50", "map50-95", "precision", "recall", "missing-rivet", "rust", "scratch"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in combined:
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

    if e1_json:
        best = e1_json.get("best_run", {})
        lines.append("## E1 Best Run")
        lines.append("")
        lines.append(
            f"- Best sliced-inference threshold search: conf=`{best.get('conf', '')}`, merge_iou=`{best.get('merge_iou', '')}`"
        )
        lines.append(
            f"- Metrics: precision=`{fv(best.get('precision')):.6f}`, recall=`{fv(best.get('recall')):.6f}`, "
            f"map50=`{fv(best.get('map50')):.6f}`, map50_95=`{fv(best.get('map50_95')):.6f}`"
        )
        lines.append("")

    if e2_json:
        lines.append("## E2 Comparison")
        lines.append("")
        lines.append("| exp | model | mode | map50-95 | precision | recall |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in e2_json.get("results", []):
            lines.append(
                f"| {row.get('experiment_id', '')} | {row.get('model_id', '')} | {row.get('mode', '')} | "
                f"{fv(row.get('map50_95')):.6f} | {fv(row.get('precision')):.6f} | {fv(row.get('recall')):.6f} |"
            )
        lines.append("")

    if success_rows:
        best = sorted(success_rows, key=lambda row: fv(row.get("map50_95")), reverse=True)[0]
        lines.append("## Best Round13 Candidate")
        lines.append("")
        lines.append(f"- Best round-13 candidate: `{best.get('experiment_id', '')}` - {best.get('experiment_title', '')}")
        lines.append(f"- Delta vs `N2` on `mAP50-95`: `{fv(best.get('map50_95')) - fv(n2.get('map50_95')):+.6f}`")
        lines.append(f"- Delta vs `S2` on `mAP50-95`: `{fv(best.get('map50_95')) - fv(s2.get('map50_95')):+.6f}`")
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
            lines.append("- Round 13 produced a new candidate above N2, so the mainline should move from the N2 anchor to this refinement branch.")
        elif fv(best.get("map50_95")) > fv(s2.get("map50_95")):
            lines.append("- Round 13 improved on S2 but still did not beat N2, so it should be retained as a stronger auxiliary branch rather than the final mainline.")
        else:
            lines.append("- Round 13 did not beat either N2 or S2 clearly enough, so N2 remains the final mainline reference.")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize round-13 automation results.")
    parser.add_argument("--round4-csv", type=Path, required=True)
    parser.add_argument("--round12-csv", type=Path, required=True)
    parser.add_argument("--round13-csv", type=Path, required=True)
    parser.add_argument("--e1-json", type=Path)
    parser.add_argument("--e2-json", type=Path)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    n2 = find_success_row(args.round4_csv, "N2")
    s2 = find_success_row(args.round12_csv, "S2")
    if not n2 or not s2:
        raise SystemExit("Could not locate the N2 or S2 anchor row.")

    round13_rows = read_rows(args.round13_csv)
    summary_rows = [n2, s2] + [row for row in round13_rows if row.get("run_status") == "success"]
    write_summary_csv(args.output_csv, summary_rows)

    report = build_report(
        [n2, s2],
        round13_rows,
        read_json(args.e1_json),
        read_json(args.e2_json),
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    print(f"summary_md={args.output_md}")
    print(f"summary_csv={args.output_csv}")


if __name__ == "__main__":
    main()
