from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ERROR_WEIGHTS = {
    "missing_label": 8.0,
    "wrong_class": 6.0,
    "shifted_box": 4.0,
    "oversized_box": 3.0,
    "undersized_box": 3.0,
    "duplicate_label": 3.0,
    "ambiguous_defect": 2.0,
    "false_positive_background": 1.0,
    "unclear_image": 1.0,
}

ERROR_ACTION = {
    "missing_label": "优先核查是否漏标，若确认是真缺陷则补框并补类别。",
    "wrong_class": "优先核查类别定义，必要时改类而不是重画框。",
    "shifted_box": "优先微调框位置和边界，使框更贴合缺陷主体。",
    "oversized_box": "优先缩框，去掉过多背景，统一边界风格。",
    "undersized_box": "优先检查是否框得过紧，必要时适度放大到缺陷完整区域。",
    "duplicate_label": "优先去重，保留一个最合理的标注框。",
    "ambiguous_defect": "优先人工复核，确认是否属于当前三类定义。",
    "false_positive_background": "优先记录为背景易混区域，必要时纳入负样本池。",
    "unclear_image": "优先确认图像质量问题是否影响标注可靠性。",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean-train execution checklist from round14 QA1 audit candidates.")
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-priority-csv", type=Path, required=True)
    parser.add_argument("--output-class-csv", type=Path, required=True)
    parser.add_argument("--output-fixlog-csv", type=Path, required=True)
    parser.add_argument("--top-images", type=int, default=40)
    return parser.parse_args()


def fv(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dominant(counter: Counter[str]) -> str:
    if not counter:
        return ""
    return counter.most_common(1)[0][0]


def bucket_name(rank: int) -> str:
    if rank <= 10:
        return "A"
    if rank <= 25:
        return "B"
    return "C"


def short_name(path_text: str) -> str:
    return Path(path_text).name


def build_image_priority_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    train_rows = [row for row in rows if row.get("split") == "train"]
    grouped: dict[str, dict[str, object]] = {}

    for row in train_rows:
        image_path = row.get("image_path", "")
        entry = grouped.setdefault(
            image_path,
            {
                "image_path": image_path,
                "image_name": short_name(image_path),
                "issue_count": 0,
                "error_counter": Counter(),
                "class_counter": Counter(),
                "pred_class_counter": Counter(),
                "max_confidence": 0.0,
                "confidence_sum": 0.0,
                "vis_paths": [],
                "candidate_ids": [],
            },
        )
        entry["issue_count"] = int(entry["issue_count"]) + 1
        error_type = row.get("error_type", "")
        class_key = row.get("class_gt") or row.get("class_pred") or "unknown"
        pred_class = row.get("class_pred") or ""
        confidence = fv(row.get("confidence"))
        entry["error_counter"][error_type] += 1
        entry["class_counter"][class_key] += 1
        if pred_class:
            entry["pred_class_counter"][pred_class] += 1
        entry["max_confidence"] = max(fv(entry["max_confidence"]), confidence)
        entry["confidence_sum"] = fv(entry["confidence_sum"]) + confidence
        if row.get("vis_path"):
            entry["vis_paths"].append(row["vis_path"])
        if row.get("candidate_id"):
            entry["candidate_ids"].append(row["candidate_id"])

    priority_rows: list[dict[str, object]] = []
    for image_path, entry in grouped.items():
        issue_count = int(entry["issue_count"])
        avg_conf = fv(entry["confidence_sum"]) / max(1, issue_count)
        error_counter: Counter[str] = entry["error_counter"]  # type: ignore[assignment]
        dominant_error = dominant(error_counter)
        dominant_class = dominant(entry["class_counter"])  # type: ignore[arg-type]
        score = issue_count * 10.0
        for error_type, count in error_counter.items():
            score += ERROR_WEIGHTS.get(error_type, 1.0) * count
        score += avg_conf * 5.0
        action = ERROR_ACTION.get(dominant_error, "人工复核后再决定是否修改。")
        priority_rows.append(
            {
                "image_path": image_path,
                "image_name": entry["image_name"],
                "issue_count": issue_count,
                "priority_score": round(score, 3),
                "dominant_error_type": dominant_error,
                "dominant_class": dominant_class,
                "max_confidence": round(fv(entry["max_confidence"]), 6),
                "avg_confidence": round(avg_conf, 6),
                "error_breakdown_json": json.dumps(dict(error_counter), ensure_ascii=False),
                "class_breakdown_json": json.dumps(dict(entry["class_counter"]), ensure_ascii=False),
                "pred_class_breakdown_json": json.dumps(dict(entry["pred_class_counter"]), ensure_ascii=False),
                "first_action": action,
                "vis_example": entry["vis_paths"][0] if entry["vis_paths"] else "",
                "candidate_ids": ",".join(entry["candidate_ids"][:12]),
            }
        )

    priority_rows.sort(
        key=lambda item: (
            -fv(item["priority_score"]),
            -int(item["issue_count"]),
            -fv(item["max_confidence"]),
            str(item["image_name"]),
        )
    )

    for index, item in enumerate(priority_rows, start=1):
        item["rank"] = index
        item["tier"] = bucket_name(index)
    return priority_rows


def build_class_priority_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    train_rows = [row for row in rows if row.get("split") == "train"]
    counter: Counter[tuple[str, str]] = Counter()
    image_counter: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    max_conf: dict[tuple[str, str], float] = defaultdict(float)

    for row in train_rows:
        class_name = row.get("class_gt") or row.get("class_pred") or "unknown"
        error_type = row.get("error_type", "")
        key = (class_name, error_type)
        counter[key] += 1
        image_counter[key].add(row.get("image_path", ""))
        max_conf[key] = max(max_conf[key], fv(row.get("confidence")))

    class_rows: list[dict[str, object]] = []
    for (class_name, error_type), count in counter.items():
        class_rows.append(
            {
                "class_name": class_name,
                "error_type": error_type,
                "issue_count": count,
                "image_count": len(image_counter[(class_name, error_type)]),
                "max_confidence": round(max_conf[(class_name, error_type)], 6),
                "recommended_action": ERROR_ACTION.get(error_type, "人工复核。"),
                "priority_score": round(count * ERROR_WEIGHTS.get(error_type, 1.0) + max_conf[(class_name, error_type)] * 5.0, 3),
            }
        )

    class_rows.sort(
        key=lambda item: (
            str(item["class_name"]),
            -fv(item["priority_score"]),
            -int(item["issue_count"]),
        )
    )
    return class_rows


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_fixlog_template(path: Path, image_rows: list[dict[str, object]]) -> None:
    columns = [
        "priority_rank",
        "tier",
        "image_path",
        "image_name",
        "dominant_class",
        "dominant_error_type",
        "issue_count",
        "fix_status",
        "fix_action",
        "label_path_before",
        "label_path_after",
        "reviewer",
        "review_date",
        "note",
    ]
    records = [
        {
            "priority_rank": row["rank"],
            "tier": row["tier"],
            "image_path": row["image_path"],
            "image_name": row["image_name"],
            "dominant_class": row["dominant_class"],
            "dominant_error_type": row["dominant_error_type"],
            "issue_count": row["issue_count"],
            "fix_status": "pending",
            "fix_action": "",
            "label_path_before": "",
            "label_path_after": "",
            "reviewer": "",
            "review_date": "",
            "note": "",
        }
        for row in image_rows
    ]
    write_csv(path, records, columns)


def build_markdown(
    summary: dict,
    image_rows: list[dict[str, object]],
    class_rows: list[dict[str, object]],
    top_images: int,
    output_priority_csv: Path,
    output_class_csv: Path,
    output_fixlog_csv: Path,
) -> str:
    train_fix_count = int(summary.get("train_fix_recommended_count", 0))
    missing_rivet_count = int(summary.get("missing_rivet_issue_count", 0))
    rust_count = int(summary.get("rust_issue_count", 0))
    scratch_count = int(summary.get("scratch_issue_count", 0))

    class_sections = {
        class_name: [row for row in class_rows if row["class_name"] == class_name]
        for class_name in ["missing-rivet", "rust", "scratch"]
    }

    lines: list[str] = []
    lines.append("# R14_QA2 Clean Train Execution Checklist")
    lines.append("")
    lines.append("## Main Judgment")
    lines.append("")
    lines.append(f"- `QA1` 在 `train` 中给出 `{train_fix_count}` 条建议修正，已经足够支撑 `clean_train` 分支。")
    lines.append(f"- 当前最优先的问题不是继续换模型，而是先统一 `train` 侧标注边界和漏标。")
    lines.append(f"- 三类问题规模大致是：`missing-rivet={missing_rivet_count}`，`rust={rust_count}`，`scratch={scratch_count}`。")
    lines.append("")
    lines.append("## Fix Order")
    lines.append("")
    lines.append("- 第一优先级：`missing_label`。先补高置信漏标，因为这类问题最可能直接限制召回和真实上限。")
    lines.append("- 第二优先级：`oversized_box`。先统一框边界，减少过大背景框对定位学习的干扰。")
    lines.append("- 第三优先级：`wrong_class` 和 `shifted_box`。这两类数量少于前两者，但会直接污染类别判别和高 IoU 指标。")
    lines.append("- 修订顺序建议先 `train`，不改 `valid`。`valid` 只保留问题记录，不纳入这轮 clean_train 修改。")
    lines.append("")
    lines.append("## Class Priority")
    lines.append("")
    for class_name in ["missing-rivet", "rust", "scratch"]:
        rows = class_sections[class_name]
        if not rows:
            continue
        lines.append(f"- `{class_name}` 先修顺序：")
        for row in rows[:3]:
            lines.append(
                f"  `{row['error_type']}` (`{row['issue_count']}` 条, `{row['image_count']}` 张图) -> {row['recommended_action']}"
            )
    lines.append("")
    lines.append("## Top Images")
    lines.append("")
    lines.append("| rank | tier | image | issue_count | dominant_class | dominant_error | max_conf | first_action | vis |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in image_rows[:top_images]:
        vis = row["vis_example"] or ""
        lines.append(
            f"| {row['rank']} | {row['tier']} | {row['image_name']} | {row['issue_count']} | "
            f"{row['dominant_class']} | {row['dominant_error_type']} | {row['max_confidence']:.3f} | "
            f"{row['first_action']} | {vis} |"
        )
    lines.append("")
    lines.append("## Tier Guide")
    lines.append("")
    lines.append("- `A` 级：前 10 张，优先立刻修。这一批最可能影响 clean_train 首轮收益。")
    lines.append("- `B` 级：第 11 到 25 张，建议在 `A` 级完成后立即跟进。")
    lines.append("- `C` 级：其余图，作为第二轮补修清单。")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Image priority CSV: `{output_priority_csv}`")
    lines.append(f"- Class priority CSV: `{output_class_csv}`")
    lines.append(f"- Label fix log template: `{output_fixlog_csv}`")
    return "\n".join(lines).replace("\n- `missing-rivet` 先修顺序：\n  ", "\n- `missing-rivet` 先修顺序：\n")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.candidates_csv)
    summary = load_json(args.summary_json)

    image_rows = build_image_priority_rows(rows)
    class_rows = build_class_priority_rows(rows)

    write_csv(
        args.output_priority_csv,
        image_rows,
        [
            "rank",
            "tier",
            "image_path",
            "image_name",
            "issue_count",
            "priority_score",
            "dominant_error_type",
            "dominant_class",
            "max_confidence",
            "avg_confidence",
            "error_breakdown_json",
            "class_breakdown_json",
            "pred_class_breakdown_json",
            "first_action",
            "vis_example",
            "candidate_ids",
        ],
    )
    write_csv(
        args.output_class_csv,
        class_rows,
        [
            "class_name",
            "error_type",
            "issue_count",
            "image_count",
            "max_confidence",
            "priority_score",
            "recommended_action",
        ],
    )
    write_fixlog_template(args.output_fixlog_csv, image_rows)

    markdown = build_markdown(
        summary=summary,
        image_rows=image_rows,
        class_rows=class_rows,
        top_images=args.top_images,
        output_priority_csv=args.output_priority_csv,
        output_class_csv=args.output_class_csv,
        output_fixlog_csv=args.output_fixlog_csv,
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(f"output_md={args.output_md}")
    print(f"output_priority_csv={args.output_priority_csv}")
    print(f"output_class_csv={args.output_class_csv}")
    print(f"output_fixlog_csv={args.output_fixlog_csv}")


if __name__ == "__main__":
    main()
