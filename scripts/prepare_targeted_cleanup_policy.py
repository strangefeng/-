from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a train-only targeted cleanup policy to a suspicious review pack and emit a filtered manifest."
    )
    parser.add_argument("--review-summary", type=Path, required=True, help="summary.json from batch_visualize_yolo_review.py")
    parser.add_argument("--review-manifest", type=Path, required=True, help="manifest.json from collect_suspicious_review_set.py")
    parser.add_argument("--output-manifest", type=Path, required=True, help="Where to write the filtered manifest")
    parser.add_argument("--report", type=Path, required=True, help="Where to write a JSON cleanup report")
    parser.add_argument("--scope", choices=["train_only", "all"], default="train_only", help="Which splits are allowed to be edited")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_label_lines(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 5:
            rows.append(parts)
    return rows


def is_small(parts: list[str], area_small: float, dim_small: float) -> bool:
    _, _, _, w, h = parts
    width = float(w)
    height = float(h)
    return width * height < area_small or width < dim_small or height < dim_small


def should_edit(scope: str, split: str) -> bool:
    if scope == "all":
        return True
    return split == "train"


def main() -> None:
    args = parse_args()
    review_summary = load_json(args.review_summary.resolve())
    review_manifest = load_json(args.review_manifest.resolve())

    area_small = float(review_summary["thresholds"]["area_small"])
    dim_small = float(review_summary["thresholds"]["dim_small"])

    filtered_items: list[dict] = []
    decisions: list[dict] = []

    for item in review_manifest.get("items", []):
        split = item.get("split", "")
        editable = should_edit(args.scope, split)
        label_path = Path(item["copied_label_path"])
        rows = parse_label_lines(label_path)
        original_count = len(rows)
        new_rows = rows
        action = "kept"

        if editable and "tiny_box_ratio" in item.get("reasons", []):
            new_rows = [parts for parts in rows if not is_small(parts, area_small, dim_small)]
            action = "removed_small_boxes"
        elif not editable:
            action = "kept_non_edit_split"

        content = "\n".join(" ".join(parts) for parts in new_rows)
        if content:
            content += "\n"
        label_path.write_text(content, encoding="utf-8")

        filtered_items.append(item)
        decisions.append(
            {
                "image_name": item["image_name"],
                "split": split,
                "reasons": item.get("reasons", []),
                "original_boxes": original_count,
                "remaining_boxes": len(new_rows),
                "removed_boxes": original_count - len(new_rows),
                "action": action,
                "label_path": str(label_path),
            }
        )

    output_manifest = {
        "summary_path": review_manifest.get("summary_path", str(args.review_summary)),
        "output_dir": review_manifest.get("output_dir", str(args.review_manifest.parent)),
        "copied_count": len(filtered_items),
        "missing_count": 0,
        "items": filtered_items,
        "missing_files": [],
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    edited_decisions = [item for item in decisions if item["action"] == "removed_small_boxes"]
    report = {
        "review_scope": args.scope,
        "thresholds": {
            "area_small": area_small,
            "dim_small": dim_small
        },
        "reviewed_items": len(decisions),
        "edited_items": len(edited_decisions),
        "removed_box_total": sum(item["removed_boxes"] for item in decisions),
        "emptied_labels": sum(1 for item in decisions if item["remaining_boxes"] == 0),
        "decisions": decisions,
        "notes": "Targeted cleanup policy removes tiny boxes only from allowed splits while keeping dense-box samples as reference-only items."
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"targeted_reviewed_items={report['reviewed_items']}")
    print(f"edited_items={report['edited_items']}")
    print(f"removed_box_total={report['removed_box_total']}")
    print(f"emptied_labels={report['emptied_labels']}")
    print(f"output_manifest={args.output_manifest}")


if __name__ == "__main__":
    main()
