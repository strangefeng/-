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


def fv(value: str | None) -> float | None:
    try:
        return float(value or "")
    except (TypeError, ValueError):
        return None


def class_ap95(row: dict[str, str], class_name: str) -> str:
    raw = row.get("class_metrics_json", "")
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    return str(payload.get(class_name, {}).get("ap50_95", ""))


def delta(a: dict[str, str], b: dict[str, str], key: str) -> str:
    av = fv(a.get(key))
    bv = fv(b.get(key))
    if av is None or bv is None:
        return ""
    return f"{bv - av:+.6f}"


def build_report(b0: dict[str, str], r1: dict[str, str], r1_rect: dict[str, str]) -> str:
    lines: list[str] = []
    lines.append("# aircraft_ai_dataset_r1_rect Summary")
    lines.append("")
    lines.append("## Main Comparison")
    lines.append("")
    lines.append("| exp | title | map50 | map50-95 | precision | recall | missing-rivet | rust | scratch | rect |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row, rect_value in ((b0, "false"), (r1, "false"), (r1_rect, "true")):
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
                    rect_value,
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Key Deltas")
    lines.append("")
    lines.append(f"- `R1_rect` vs `R1` on `mAP50-95`: `{delta(r1, r1_rect, 'map50_95')}`")
    mv_r1 = fv(class_ap95(r1, "missing-rivet")) or 0.0
    mv_rect = fv(class_ap95(r1_rect, "missing-rivet")) or 0.0
    lines.append(f"- `R1_rect` vs `R1` on `missing-rivet AP50-95`: `{mv_rect - mv_r1:+.6f}`")
    lines.append(f"- `R1_rect` vs `B0` on `mAP50-95`: `{delta(b0, r1_rect, 'map50_95')}`")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    r1_map = fv(r1.get("map50_95")) or float("-inf")
    rect_map = fv(r1_rect.get("map50_95")) or float("-inf")
    r1_missing = fv(class_ap95(r1, "missing-rivet")) or float("-inf")
    rect_missing = fv(class_ap95(r1_rect, "missing-rivet")) or float("-inf")
    if rect_map > r1_map:
        lines.append("- `rect=True` improved the 800-resolution baseline overall, so it is worth keeping as the next anchor setting.")
    elif rect_missing > r1_missing and rect_map >= r1_map - 0.005:
        lines.append("- `rect=True` mainly helped the key small-object class while keeping overall performance close, so it is worth discussing as a targeted adjustment.")
    else:
        lines.append("- `rect=True` did not provide a clear gain over the current `R1`, so the non-rectangular setup should remain the preferred 800-resolution reference.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the R1_rect comparison against B0 and R1.")
    parser.add_argument("--mainline-csv", type=Path, required=True)
    parser.add_argument("--round2-csv", type=Path, required=True)
    parser.add_argument("--rect-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    b0 = find_success_row(args.mainline_csv, "B0")
    r1 = find_success_row(args.round2_csv, "R1")
    r1_rect = find_success_row(args.rect_csv, "R1_rect")
    if not b0 or not r1 or not r1_rect:
        raise SystemExit("Could not locate B0, R1, or R1_rect.")

    report = build_report(b0, r1, r1_rect)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"summary_md={args.output}")


if __name__ == "__main__":
    main()
