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
    rows = [
        row
        for row in read_rows(path)
        if row.get("experiment_id") == experiment_id and row.get("run_status") == "success"
    ]
    if not rows:
        return None
    return sorted(rows, key=lambda row: float(row.get("map50_95", "") or 0.0), reverse=True)[0]


def read_json(path: Path | None) -> dict | None:
    if not path or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fv(value: object) -> float:
    try:
        return float(value)
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


def output_summary_csv(output_csv: Path, rows: list[dict[str, str]], n2: dict[str, str], s2: dict[str, str]) -> None:
    columns = [
        "experiment_id",
        "experiment_title",
        "map50",
        "map50_95",
        "precision",
        "recall",
        "missing_rivet_ap50_95",
        "rust_ap50_95",
        "scratch_ap50_95",
        "delta_vs_n2_map50_95",
        "delta_vs_s2_map50_95",
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
                    "map50": row.get("map50", ""),
                    "map50_95": row.get("map50_95", ""),
                    "precision": row.get("precision", ""),
                    "recall": row.get("recall", ""),
                    "missing_rivet_ap50_95": class_ap95(row, "missing-rivet"),
                    "rust_ap50_95": class_ap95(row, "rust"),
                    "scratch_ap50_95": class_ap95(row, "scratch"),
                    "delta_vs_n2_map50_95": f"{fv(row.get('map50_95')) - fv(n2.get('map50_95')):+.6f}",
                    "delta_vs_s2_map50_95": f"{fv(row.get('map50_95')) - fv(s2.get('map50_95')):+.6f}",
                }
            )


def build_report(
    n2: dict[str, str],
    s2: dict[str, str],
    rows: list[dict[str, str]],
    c0_n2: dict | None,
    c0_s2: dict | None,
    n2_neg_fp: dict | None,
    hn1_neg_fp: dict | None,
) -> str:
    success_rows = [row for row in rows if row.get("run_status") == "success"]
    combined = [n2, s2] + success_rows

    lines: list[str] = []
    lines.append("# aircraft_ai_dataset_round13_phase2 Summary")
    lines.append("")
    lines.append("## Main Comparison")
    lines.append("")
    lines.append("| exp | title | map50 | map50-95 | precision | recall | missing-rivet | rust | scratch |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in combined:
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

    if c0_n2 or c0_s2:
        lines.append("## C0 Consistency Check")
        lines.append("")
        if c0_n2:
            lines.append(
                f"- N2 official val: map50_95=`{fv(c0_n2.get('map50_95')):.6f}`, "
                f"delta vs expected=`{fv(c0_n2.get('map50_95')) - fv(n2.get('map50_95')):+.6f}`"
            )
        if c0_s2:
            lines.append(
                f"- S2 official val: map50_95=`{fv(c0_s2.get('map50_95')):.6f}`, "
                f"delta vs expected=`{fv(c0_s2.get('map50_95')) - fv(s2.get('map50_95')):+.6f}`"
            )
        lines.append("")

    if n2_neg_fp or hn1_neg_fp:
        lines.append("## Hard-Negative FP Audit")
        lines.append("")
        if n2_neg_fp:
            lines.append(
                f"- N2 on negative images: total_fp=`{n2_neg_fp.get('total_false_positive_boxes', 0)}`, "
                f"high_conf_fp=`{n2_neg_fp.get('high_conf_false_positive_boxes', 0)}`, "
                f"avg_fp_per_image=`{float(n2_neg_fp.get('avg_false_positive_boxes_per_image', 0.0)):.4f}`"
            )
        if hn1_neg_fp:
            lines.append(
                f"- HN1 on negative images: total_fp=`{hn1_neg_fp.get('total_false_positive_boxes', 0)}`, "
                f"high_conf_fp=`{hn1_neg_fp.get('high_conf_false_positive_boxes', 0)}`, "
                f"avg_fp_per_image=`{float(hn1_neg_fp.get('avg_false_positive_boxes_per_image', 0.0)):.4f}`"
            )
        lines.append("")

    if success_rows:
        best = sorted(success_rows, key=lambda row: fv(row.get("map50_95")), reverse=True)[0]
        lines.append("## Best Candidate")
        lines.append("")
        lines.append(f"- Best phase2 candidate: `{best.get('experiment_id', '')}` - {best.get('experiment_title', '')}")
        lines.append(f"- Delta vs N2 on mAP50-95: `{fv(best.get('map50_95')) - fv(n2.get('map50_95')):+.6f}`")
        lines.append(f"- Delta vs S2 on mAP50-95: `{fv(best.get('map50_95')) - fv(s2.get('map50_95')):+.6f}`")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize round13 phase2 outputs.")
    parser.add_argument("--round4-csv", type=Path, required=True)
    parser.add_argument("--round12-csv", type=Path, required=True)
    parser.add_argument("--phase2-csv", type=Path, required=True)
    parser.add_argument("--c0-n2-json", type=Path)
    parser.add_argument("--c0-s2-json", type=Path)
    parser.add_argument("--n2-neg-fp-json", type=Path)
    parser.add_argument("--hn1-neg-fp-json", type=Path)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    n2 = find_success_row(args.round4_csv, "N2")
    s2 = find_success_row(args.round12_csv, "S2")
    if not n2 or not s2:
        raise SystemExit("Could not locate N2 or S2 anchor rows.")

    phase2_rows = read_rows(args.phase2_csv)
    success_rows = [row for row in phase2_rows if row.get("run_status") == "success"]
    output_summary_csv(args.output_csv, success_rows, n2, s2)

    report = build_report(
        n2,
        s2,
        phase2_rows,
        read_json(args.c0_n2_json),
        read_json(args.c0_s2_json),
        read_json(args.n2_neg_fp_json),
        read_json(args.hn1_neg_fp_json),
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    print(f"summary_md={args.output_md}")
    print(f"summary_csv={args.output_csv}")


if __name__ == "__main__":
    main()
