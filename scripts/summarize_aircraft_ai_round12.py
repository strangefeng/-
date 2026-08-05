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


def eval_row_from_json(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    class_metrics = payload.get("class_metrics", {})
    return {
        "experiment_id": "S1",
        "experiment_title": "N2 weights with sliced inference only",
        "dataset_role": str(payload.get("dataset_role", "sliced_inference_only")),
        "map50": f"{fv(payload.get('map50')):.6f}",
        "map50_95": f"{fv(payload.get('map50_95')):.6f}",
        "precision": f"{fv(payload.get('precision')):.6f}",
        "recall": f"{fv(payload.get('recall')):.6f}",
        "class_metrics_json": json.dumps(
            {
                key: {
                    "ap50": f"{fv(value.get('ap50')):.6f}",
                    "ap50_95": f"{fv(value.get('ap50_95')):.6f}",
                }
                for key, value in class_metrics.items()
            },
            ensure_ascii=False,
        ),
    }


def build_report(anchor_rows: list[dict[str, str]], round12_rows: list[dict[str, str]], s1_row: dict[str, str] | None) -> str:
    success_rows = [row for row in round12_rows if row.get("run_status") == "success"]
    combined = list(anchor_rows)
    if s1_row:
        combined.append(s1_row)
    combined.extend(sorted(success_rows, key=lambda row: fv(row.get("map50_95")), reverse=True))
    n2 = next(row for row in anchor_rows if row.get("experiment_id") == "N2")

    lines: list[str] = []
    lines.append("# aircraft_ai_dataset_round12_automation Summary")
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

    candidates = list(success_rows)
    if s1_row:
        candidates.append(s1_row)
    if candidates:
        best = sorted(candidates, key=lambda row: fv(row.get("map50_95")), reverse=True)[0]
        lines.append("## Best New Candidate")
        lines.append("")
        lines.append(f"- Best round-12 candidate: `{best.get('experiment_id', '')}` - {best.get('experiment_title', '')}")
        lines.append(f"- Delta vs `N2` on `mAP50-95`: `{fv(best.get('map50_95')) - fv(n2.get('map50_95')):+.6f}`")
        lines.append(
            f"- Delta vs `N2` on `missing-rivet AP50-95`: "
            f"`{fv(class_ap95(best, 'missing-rivet')) - fv(class_ap95(n2, 'missing-rivet')):+.6f}`"
        )
        lines.append(
            f"- Delta vs `N2` on `scratch AP50-95`: "
            f"`{fv(class_ap95(best, 'scratch')) - fv(class_ap95(n2, 'scratch')):+.6f}`"
        )
        lines.append("")
        lines.append("## Recommendation")
        lines.append("")
        if fv(best.get("map50_95")) > fv(n2.get("map50_95")):
            if best.get("experiment_id", "").startswith("S"):
                lines.append("- The sliced-information line beat the N2 anchor, so slicing should become the new mainline direction.")
            elif best.get("experiment_id", "").startswith("P"):
                lines.append("- The defect-level patch augmentation line beat the N2 anchor, so donor-based diversity should become the next mainline direction.")
            elif best.get("experiment_id", "").startswith("HN"):
                lines.append("- The hard-negative line beat the N2 anchor, so false-positive suppression should be merged into the mainline recipe.")
            else:
                lines.append("- A round-12 data-information variant beat N2, so the mainline should move to that branch.")
        elif best.get("experiment_id") == "S1" and fv(best.get("map50_95")) >= fv(n2.get("map50_95")) - 0.005:
            lines.append("- Sliced inference alone did not clearly surpass N2, but it stayed close enough to remain a deployment-side recall-oriented option.")
        else:
            lines.append("- None of the round-12 data-information branches beat the N2 anchor clearly enough, so N2 should remain the final mainline reference.")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize round-12 automation results.")
    parser.add_argument("--round4-csv", type=Path, required=True)
    parser.add_argument("--round12-csv", type=Path, required=True)
    parser.add_argument("--s1-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    n2 = find_success_row(args.round4_csv, "N2")
    if not n2:
        raise SystemExit("Could not locate the N2 anchor row.")

    report = build_report([n2], read_rows(args.round12_csv), eval_row_from_json(args.s1_json))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"summary_md={args.output}")


if __name__ == "__main__":
    main()
