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


def fv(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def load_json(path: Path | None) -> dict | None:
    if not path or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def class_ap95(row: dict[str, str], class_name: str) -> str:
    raw = row.get("class_metrics_json", "")
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    return str(payload.get(class_name, {}).get("ap50_95", ""))


def status_blocked_map(status_payload: dict | None) -> dict[str, str]:
    if not status_payload:
        return {}
    blocked = {}
    for item in status_payload.get("blocked_experiments", []):
        blocked[str(item.get("experiment", ""))] = str(item.get("reason", ""))
    return blocked


def build_deployment_fp_map(paths: list[Path]) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for path in paths:
        payload = load_json(path)
        if not payload:
            continue
        experiment = str(payload.get("experiment_id") or path.stem)
        mapping[experiment] = payload
    return mapping


def model_label(row: dict[str, str]) -> str:
    model_path = str(row.get("model", ""))
    name = Path(model_path).name if model_path else ""
    return name or model_path


def write_summary_csv(
    output_csv: Path,
    rows: list[dict[str, str]],
    n2: dict[str, str],
    hn1: dict[str, str],
    fp_map: dict[str, dict],
    blocked: dict[str, str],
) -> None:
    columns = [
        "experiment",
        "model",
        "dataset",
        "imgsz",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "missing_rivet_ap50_95",
        "rust_ap50_95",
        "scratch_ap50_95",
        "delta_vs_N2",
        "delta_vs_HN1",
        "negative_total_fp",
        "negative_high_conf_fp",
        "params",
        "model_size",
        "inference_time",
        "notes",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        expected = ["R14-HN2_N2_HardNegative150", "R14-HN3_N2_HardNegative300", "R14-QA2_N2_CleanTrain", "R14-M1_YOLO11s_N2", "R14-M2_YOLO11s_HN150"]
        row_by_id = {row.get("experiment_id", ""): row for row in rows}
        for experiment in expected:
            row = row_by_id.get(experiment)
            fp_payload = fp_map.get(experiment, {})
            if row:
                notes = row.get("notes", "")
                if experiment in blocked:
                    notes = (notes + " | " + blocked[experiment]).strip(" |")
                writer.writerow(
                    {
                        "experiment": experiment,
                        "model": model_label(row),
                        "dataset": row.get("dataset_id", ""),
                        "imgsz": row.get("imgsz", ""),
                        "precision": row.get("precision", ""),
                        "recall": row.get("recall", ""),
                        "map50": row.get("map50", ""),
                        "map50_95": row.get("map50_95", ""),
                        "missing_rivet_ap50_95": class_ap95(row, "missing-rivet"),
                        "rust_ap50_95": class_ap95(row, "rust"),
                        "scratch_ap50_95": class_ap95(row, "scratch"),
                        "delta_vs_N2": f"{fv(row.get('map50_95')) - fv(n2.get('map50_95')):+.6f}",
                        "delta_vs_HN1": f"{fv(row.get('map50_95')) - fv(hn1.get('map50_95')):+.6f}",
                        "negative_total_fp": str(fp_payload.get("total_false_positive_boxes", "")),
                        "negative_high_conf_fp": str(fp_payload.get("high_conf_false_positive_boxes", "")),
                        "params": row.get("model_parameters", ""),
                        "model_size": row.get("artifact_size_mb", ""),
                        "inference_time": row.get("speed_inference_ms", ""),
                        "notes": notes,
                    }
                )
            else:
                writer.writerow(
                    {
                        "experiment": experiment,
                        "model": "",
                        "dataset": "",
                        "imgsz": "",
                        "precision": "",
                        "recall": "",
                        "map50": "",
                        "map50_95": "",
                        "missing_rivet_ap50_95": "",
                        "rust_ap50_95": "",
                        "scratch_ap50_95": "",
                        "delta_vs_N2": "",
                        "delta_vs_HN1": "",
                        "negative_total_fp": "",
                        "negative_high_conf_fp": "",
                        "params": "",
                        "model_size": "",
                        "inference_time": "",
                        "notes": blocked.get(experiment, "not run"),
                    }
                )


def build_main_report(
    n2: dict[str, str],
    hn1: dict[str, str],
    c0_payload: dict | None,
    qa1_payload: dict | None,
    rows: list[dict[str, str]],
    fp_map: dict[str, dict],
    status_payload: dict | None,
) -> str:
    success_rows = [row for row in rows if row.get("run_status") == "success"]
    blocked = status_blocked_map(status_payload)

    lines: list[str] = []
    lines.append("# aircraft_ai_dataset_round14 Summary")
    lines.append("")
    lines.append("## Main Comparison")
    lines.append("")
    lines.append("| exp | title | map50 | map50-95 | precision | recall | missing-rivet | rust | scratch |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in [n2, hn1, *success_rows]:
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

    if c0_payload:
        lines.append("## C0 Consistency Check")
        lines.append("")
        lines.append(
            f"- N2 official val map50-95: `{fv(c0_payload.get('map50_95')):.6f}` "
            f"(delta vs historical N2 `{fv(c0_payload.get('map50_95')) - fv(n2.get('map50_95')):+.6f}`)"
        )
        lines.append("")

    if qa1_payload:
        lines.append("## QA1 Audit")
        lines.append("")
        lines.append(f"- Audit candidate count: `{qa1_payload.get('candidate_count', 0)}`")
        lines.append(f"- Train fix recommended count: `{qa1_payload.get('train_fix_recommended_count', 0)}`")
        lines.append(f"- Valid record-only count: `{qa1_payload.get('valid_record_only_count', 0)}`")
        lines.append(
            f"- Suggested clean_train branch: `{'yes' if qa1_payload.get('clean_train_recommended') else 'no'}`"
        )
        lines.append("")

    if fp_map:
        lines.append("## Deployment Negative Set")
        lines.append("")
        for experiment, payload in sorted(fp_map.items()):
            lines.append(
                f"- `{experiment}`: total_fp=`{payload.get('total_false_positive_boxes', 0)}`, "
                f"high_conf_fp=`{payload.get('high_conf_false_positive_boxes', 0)}`, "
                f"avg_fp_per_image=`{float(payload.get('avg_false_positive_boxes_per_image', 0.0)):.4f}`"
            )
        lines.append("")

    if blocked:
        lines.append("## Blocked / Skipped")
        lines.append("")
        for experiment, reason in blocked.items():
            lines.append(f"- `{experiment}`: {reason}")
        lines.append("")

    if success_rows:
        best = sorted(success_rows, key=lambda row: fv(row.get("map50_95")), reverse=True)[0]
        lines.append("## Best Round14 Candidate")
        lines.append("")
        lines.append(f"- Best completed candidate: `{best.get('experiment_id', '')}` - {best.get('experiment_title', '')}")
        lines.append(f"- Delta vs N2 on mAP50-95: `{fv(best.get('map50_95')) - fv(n2.get('map50_95')):+.6f}`")
        lines.append(f"- Delta vs HN1 on mAP50-95: `{fv(best.get('map50_95')) - fv(hn1.get('map50_95')):+.6f}`")
        lines.append("")

    return "\n".join(lines)


def build_final_recommendation(
    n2: dict[str, str],
    hn1: dict[str, str],
    rows: list[dict[str, str]],
    qa1_payload: dict | None,
    blocked: dict[str, str],
) -> str:
    success_rows = [row for row in rows if row.get("run_status") == "success"]
    best = sorted(success_rows, key=lambda row: fv(row.get("map50_95")), reverse=True)[0] if success_rows else None
    best_delta = fv(best.get("map50_95")) - fv(n2.get("map50_95")) if best else float("-inf")
    yolo11s_row = next((row for row in success_rows if row.get("experiment_id") == "R14-M1_YOLO11s_N2"), None)
    yolo11s_delta = fv(yolo11s_row.get("map50_95")) - fv(n2.get("map50_95")) if yolo11s_row else float("-inf")
    clean_train_recommended = bool(qa1_payload and qa1_payload.get("clean_train_recommended"))
    hn_candidates = [row for row in success_rows if row.get("experiment_id", "").startswith("R14-HN")]
    best_hn = sorted(hn_candidates, key=lambda row: fv(row.get("map50_95")), reverse=True)[0] if hn_candidates else None

    lines: list[str] = []
    lines.append("# final_recommendation")
    lines.append("")
    lines.append("1. Whether a new model exceeded N2")
    if best and best_delta > 0:
        lines.append(f"- Yes. `{best.get('experiment_id', '')}` exceeded N2 by `{best_delta:+.6f}` mAP50-95.")
    else:
        lines.append("- No. No completed round14 model exceeded the N2 anchor on standard validation.")
    lines.append("")
    lines.append("2. Whether HN150/HN300 are better than HN50")
    if best_hn:
        lines.append(
            f"- `{best_hn.get('experiment_id', '')}` is the best completed hard-negative expansion candidate, "
            f"with delta vs HN1 `{fv(best_hn.get('map50_95')) - fv(hn1.get('map50_95')):+.6f}`."
        )
    elif "R14-HN2_N2_HardNegative150" in blocked or "R14-HN3_N2_HardNegative300" in blocked:
        lines.append("- Not concluded yet. HN150/HN300 were blocked because a curated train-only negative pool was not available.")
    else:
        lines.append("- Not concluded yet.")
    lines.append("")
    lines.append("3. Main bottleneck")
    if clean_train_recommended:
        lines.append("- Label quality and train-side annotation consistency remain a major bottleneck.")
    elif yolo11s_row and yolo11s_delta >= 0.010:
        lines.append("- Model capacity is one of the bottlenecks, although deployment cost must also be considered.")
    else:
        lines.append("- The current bottleneck still looks more data-driven than architecture-driven: label quality and background false positives dominate.")
    lines.append("")
    lines.append("4. Recommended paper main model")
    lines.append("- Keep `N2` as the main paper model unless a later clean-train or hard-negative expansion run clearly exceeds it.")
    lines.append("")
    lines.append("5. Recommended digital-twin deployment model")
    lines.append("- Prefer the strongest hard-negative-suppressed model that keeps mAP close to N2, because deployment visuals are sensitive to false positives.")
    lines.append("")
    lines.append("6. Continue experiments or stop")
    if best and best_delta > 0:
        lines.append("- Continue only for multi-seed confirmation of the new best candidate.")
    elif clean_train_recommended:
        lines.append("- Continue with a manual clean-train branch; that is currently the highest-value next step.")
    else:
        lines.append("- If later HN expansion and clean-train do not beat N2, stop chasing mAP and move into paper writing plus engineering presentation.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize round14 outputs.")
    parser.add_argument("--round4-csv", type=Path, required=True)
    parser.add_argument("--round13-phase2-csv", type=Path, required=True)
    parser.add_argument("--round14-csv", type=Path, required=True)
    parser.add_argument("--c0-json", type=Path)
    parser.add_argument("--qa1-json", type=Path)
    parser.add_argument("--status-json", type=Path)
    parser.add_argument("--deployment-fp-json", nargs="*", type=Path, default=[])
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-summary-md", type=Path, required=True)
    parser.add_argument("--output-final-md", type=Path, required=True)
    args = parser.parse_args()

    n2 = find_success_row(args.round4_csv, "N2")
    hn1 = find_success_row(args.round13_phase2_csv, "R13-HN1_N2_HardNegative50")
    if not n2 or not hn1:
        raise SystemExit("Could not locate N2 or HN1 anchors.")

    rows = read_rows(args.round14_csv)
    fp_map = build_deployment_fp_map(args.deployment_fp_json)
    status_payload = load_json(args.status_json)
    blocked = status_blocked_map(status_payload)

    write_summary_csv(args.output_summary_csv, rows, n2, hn1, fp_map, blocked)

    report = build_main_report(
        n2=n2,
        hn1=hn1,
        c0_payload=load_json(args.c0_json),
        qa1_payload=load_json(args.qa1_json),
        rows=rows,
        fp_map=fp_map,
        status_payload=status_payload,
    )
    args.output_summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_md.write_text(report, encoding="utf-8")

    final_recommendation = build_final_recommendation(
        n2=n2,
        hn1=hn1,
        rows=rows,
        qa1_payload=load_json(args.qa1_json),
        blocked=blocked,
    )
    args.output_final_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_final_md.write_text(final_recommendation, encoding="utf-8")
    print(f"summary_md={args.output_summary_md}")
    print(f"summary_csv={args.output_summary_csv}")
    print(f"final_md={args.output_final_md}")


if __name__ == "__main__":
    main()
