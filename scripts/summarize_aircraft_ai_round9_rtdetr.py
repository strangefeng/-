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


def build_report(b0: dict[str, str], r1: dict[str, str], n2: dict[str, str], d1: dict[str, str]) -> str:
    ordered = [b0, r1, n2, d1]
    lines: list[str] = []
    lines.append("# aircraft_ai_dataset_round9_rtdetr Summary")
    lines.append("")
    lines.append("## Main Comparison")
    lines.append("")
    headers = ["exp", "title", "model", "map50", "map50-95", "precision", "recall", "missing-rivet", "rust", "scratch"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in ordered:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("experiment_id", ""),
                    row.get("experiment_title", ""),
                    Path(row.get("model", "")).name or row.get("model", ""),
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
    lines.append("## RT-DETR Verdict")
    lines.append("")
    lines.append(f"- `D1` vs `N2` on `mAP50-95`: `{fv(d1.get('map50_95')) - fv(n2.get('map50_95')):+.6f}`")
    lines.append(f"- `D1` vs `N2` on `mAP50`: `{fv(d1.get('map50')) - fv(n2.get('map50')):+.6f}`")
    lines.append(
        f"- `D1` vs `N2` on `missing-rivet AP50-95`: "
        f"`{fv(class_ap95(d1, 'missing-rivet')) - fv(class_ap95(n2, 'missing-rivet')):+.6f}`"
    )
    lines.append(
        f"- `D1` vs `R1` on `mAP50-95`: "
        f"`{fv(d1.get('map50_95')) - fv(r1.get('map50_95')):+.6f}`"
    )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    if fv(d1.get("map50_95")) > fv(n2.get("map50_95")):
        lines.append("- RT-DETR-L beat the current YOLO11n N2 anchor on the public dataset, so it deserves promotion to the next cross-model baseline line.")
    else:
        lines.append("- RT-DETR-L did not beat the current YOLO11n N2 anchor, so it should be treated as a checked comparison model rather than the new mainline.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RT-DETR round against the existing anchors.")
    parser.add_argument("--mainline-csv", type=Path, required=True)
    parser.add_argument("--round2-csv", type=Path, required=True)
    parser.add_argument("--round4-csv", type=Path, required=True)
    parser.add_argument("--round9-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    b0 = find_success_row(args.mainline_csv, "B0")
    r1 = find_success_row(args.round2_csv, "R1")
    n2 = find_success_row(args.round4_csv, "N2")
    d1 = find_success_row(args.round9_csv, "D1")
    if not b0 or not r1 or not n2 or not d1:
        raise SystemExit("Could not locate one or more anchor rows (B0, R1, N2, D1).")

    report = build_report(b0, r1, n2, d1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"summary_md={args.output}")


if __name__ == "__main__":
    main()
