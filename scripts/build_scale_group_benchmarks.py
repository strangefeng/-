from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

GROUPS = {
    "tiny": (0.0, 0.0025),
    "small": (0.0025, 0.01),
    "medium_large": (0.01, float("inf")),
}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def resolve_dataset_root(dataset_yaml: Path) -> Path:
    data = load_yaml(dataset_yaml)
    raw_path = data.get("path")
    if raw_path:
        candidate = Path(raw_path)
        if candidate.exists():
            return candidate
    return dataset_yaml.parent


def group_for_area(area: float) -> str:
    for name, (lower, upper) in GROUPS.items():
        if lower <= area < upper:
            return name
    return "medium_large"


def parse_label_line(line: str) -> tuple[str, float] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        width = float(parts[3])
        height = float(parts[4])
    except ValueError:
        return None
    return line.strip(), width * height


def matching_label_path(labels_dir: Path, image_path: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def build_group_dataset(dataset_yaml: Path, split: str, output_root: Path) -> None:
    source_root = resolve_dataset_root(dataset_yaml)
    source_images_dir = source_root / "images" / split
    source_labels_dir = source_root / "labels" / split
    if not source_images_dir.exists():
        raise SystemExit(f"Missing image split directory: {source_images_dir}")
    if not source_labels_dir.exists():
        raise SystemExit(f"Missing label split directory: {source_labels_dir}")

    source_data = load_yaml(dataset_yaml)
    names = source_data.get("names", {0: "missing-rivet", 1: "rust", 2: "scratch"})
    nc = source_data.get("nc", 3)

    output_root.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(path for path in source_images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)

    for group_name in GROUPS:
        group_root = output_root / group_name
        images_out = group_root / "images" / split
        labels_out = group_root / "labels" / split
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        kept_boxes = 0
        images_with_boxes = 0
        for image_path in image_paths:
            shutil.copy2(image_path, images_out / image_path.name)
            label_path = matching_label_path(source_labels_dir, image_path)
            kept_lines: list[str] = []
            if label_path.exists():
                for raw_line in label_path.read_text(encoding="utf-8-sig").splitlines():
                    parsed = parse_label_line(raw_line)
                    if not parsed:
                        continue
                    line, area = parsed
                    if group_for_area(area) == group_name:
                        kept_lines.append(line)
            if kept_lines:
                images_with_boxes += 1
                kept_boxes += len(kept_lines)
            (labels_out / f"{image_path.stem}.txt").write_text(
                "\n".join(kept_lines) + ("\n" if kept_lines else ""),
                encoding="utf-8",
            )

        yaml_path = output_root / f"{group_name}.yaml"
        yaml_data = {
            "path": str(group_root.resolve()).replace("\\", "/"),
            "train": f"images/{split}",
            "val": f"images/{split}",
            "test": f"images/{split}",
            "nc": nc,
            "names": names,
            "scale_group": group_name,
            "source_dataset_yaml": str(dataset_yaml.resolve()).replace("\\", "/"),
            "source_split": split,
            "box_area_range": list(GROUPS[group_name]),
            "image_count": len(image_paths),
            "images_with_boxes": images_with_boxes,
            "box_count": kept_boxes,
        }
        yaml_path.write_text(yaml.safe_dump(yaml_data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"{group_name}: images={len(image_paths)} images_with_boxes={images_with_boxes} boxes={kept_boxes}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO validation YAMLs split by normalized bbox area.")
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=ROOT / "datasets" / "generated_3cls" / "aircraft_ai_uav_clean_split.yaml",
    )
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "datasets" / "scale_groups" / "aircraft_ai_uav_clean_split_val",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_group_dataset(args.dataset_yaml.resolve(), args.split, args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

