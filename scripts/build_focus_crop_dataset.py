from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_xyxy(self, width: int, height: int) -> tuple[int, int, int, int]:
        x1 = int(round((self.cx - self.w / 2) * width))
        y1 = int(round((self.cy - self.h / 2) * height))
        x2 = int(round((self.cx + self.w / 2) * width))
        y2 = int(round((self.cy + self.h / 2) * height))
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        return x1, y1, x2, y2

    @classmethod
    def from_xyxy(cls, class_id: int, x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> "YoloBox":
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        w = box_w / width
        h = box_h / height
        return cls(class_id=class_id, cx=cx, cy=cy, w=w, h=h)

    def serialize(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a focus-crop dataset variant that augments the train split with box-centered crops for small aircraft defects."
    )
    parser.add_argument("--source-yaml", type=Path, required=True, help="Source dataset YAML.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output dataset root.")
    parser.add_argument(
        "--target-classes",
        nargs="*",
        default=["missing-rivet"],
        help="Class names that should trigger focus crops.",
    )
    parser.add_argument(
        "--small-area-threshold",
        type=float,
        default=0.015,
        help="Create focus crops only for boxes with normalized area below this threshold.",
    )
    parser.add_argument(
        "--small-dim-threshold",
        type=float,
        default=0.08,
        help="Create focus crops when normalized width or height is below this threshold.",
    )
    parser.add_argument(
        "--min-crop-edge",
        type=int,
        default=256,
        help="Minimum focus crop edge in pixels.",
    )
    parser.add_argument(
        "--max-crop-edge",
        type=int,
        default=640,
        help="Maximum focus crop edge in pixels.",
    )
    parser.add_argument(
        "--context-scale",
        type=float,
        default=7.0,
        help="Crop edge multiplier relative to the larger box side.",
    )
    parser.add_argument(
        "--max-crops-per-image",
        type=int,
        default=2,
        help="Maximum number of focus crops generated from one training image.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def resolve_dataset_spec(dataset_yaml: Path) -> dict[str, object]:
    data = load_yaml(dataset_yaml)
    base = Path(data["path"])
    if not base.is_absolute():
        base = (dataset_yaml.parent / base).resolve()

    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    def split_dir(key: str) -> Path:
        value = Path(str(data[key]))
        if value.is_absolute():
            return value.resolve()
        return (base / value).resolve()

    return {
        "base": base,
        "train_images": split_dir("train"),
        "val_images": split_dir("val"),
        "test_images": split_dir("test"),
        "names": list(names),
        "nc": int(data["nc"]),
    }


def list_images(images_dir: Path) -> list[Path]:
    return sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def label_path_for_image(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def parse_label_file(path: Path) -> list[YoloBox]:
    if not path.exists():
        return []
    boxes: list[YoloBox] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id, cx, cy, w, h = parts
        boxes.append(YoloBox(int(class_id), float(cx), float(cy), float(w), float(h)))
    return boxes


def save_label_file(path: Path, boxes: Iterable[YoloBox]) -> None:
    content = "\n".join(box.serialize() for box in boxes)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def clamp_crop(center_x: int, center_y: int, crop_edge: int, width: int, height: int) -> tuple[int, int, int, int]:
    crop_edge = min(crop_edge, width, height)
    half = crop_edge // 2
    x1 = max(0, min(center_x - half, width - crop_edge))
    y1 = max(0, min(center_y - half, height - crop_edge))
    x2 = x1 + crop_edge
    y2 = y1 + crop_edge
    return x1, y1, x2, y2


def crop_boxes(boxes: list[YoloBox], crop_xyxy: tuple[int, int, int, int], width: int, height: int) -> list[YoloBox]:
    x1, y1, x2, y2 = crop_xyxy
    crop_w = x2 - x1
    crop_h = y2 - y1
    cropped: list[YoloBox] = []
    for box in boxes:
        bx1, by1, bx2, by2 = box.to_xyxy(width, height)
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        if not (x1 <= cx < x2 and y1 <= cy < y2):
            continue
        inter_x1 = max(x1, bx1)
        inter_y1 = max(y1, by1)
        inter_x2 = min(x2, bx2)
        inter_y2 = min(y2, by2)
        if inter_x2 - inter_x1 < 2 or inter_y2 - inter_y1 < 2:
            continue
        cropped.append(
            YoloBox.from_xyxy(
                class_id=box.class_id,
                x1=inter_x1 - x1,
                y1=inter_y1 - y1,
                x2=inter_x2 - x1,
                y2=inter_y2 - y1,
                width=crop_w,
                height=crop_h,
            )
        )
    return cropped


def should_focus(box: YoloBox, class_names: list[str], target_classes: set[str], area_thr: float, dim_thr: float) -> bool:
    if not (0 <= box.class_id < len(class_names)):
        return False
    class_name = class_names[box.class_id]
    if class_name not in target_classes:
        return False
    area = box.w * box.h
    return area <= area_thr or box.w <= dim_thr or box.h <= dim_thr


def write_dataset_yaml(output_root: Path, names: list[str], nc: int) -> None:
    data = {
        "path": output_root.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": nc,
        "names": names,
    }
    (output_root / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=False, sort_keys=False), encoding="utf-8")


def build_split_copy(source_images: Path, output_root: Path, split_name: str) -> int:
    images_out = output_root / split_name / "images"
    labels_out = output_root / split_name / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    copied = 0
    for image_path in list_images(source_images):
        label_path = label_path_for_image(image_path)
        shutil.copy2(image_path, images_out / image_path.name)
        if label_path.exists():
            shutil.copy2(label_path, labels_out / label_path.name)
        else:
            (labels_out / label_path.name).write_text("", encoding="utf-8")
        copied += 1
    return copied


def build_focus_train_split(
    source_images: Path,
    output_root: Path,
    class_names: list[str],
    target_classes: set[str],
    area_thr: float,
    dim_thr: float,
    min_crop_edge: int,
    max_crop_edge: int,
    context_scale: float,
    max_crops_per_image: int,
) -> dict[str, int]:
    copied = build_split_copy(source_images, output_root, "train")
    generated = 0
    triggered_images = 0

    images_out = output_root / "train" / "images"
    labels_out = output_root / "train" / "labels"

    for image_path in list_images(source_images):
        label_path = label_path_for_image(image_path)
        boxes = parse_label_file(label_path)
        if not boxes:
            continue

        focus_boxes = [box for box in boxes if should_focus(box, class_names, target_classes, area_thr, dim_thr)]
        if not focus_boxes:
            continue

        triggered_images += 1
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
            width, height = image_rgb.size
            for idx, box in enumerate(focus_boxes[:max_crops_per_image], start=1):
                x1, y1, x2, y2 = box.to_xyxy(width, height)
                box_edge = max(x2 - x1, y2 - y1)
                crop_edge = int(round(box_edge * context_scale))
                crop_edge = max(min_crop_edge, min(max_crop_edge, crop_edge))
                center_x = int(round((x1 + x2) / 2))
                center_y = int(round((y1 + y2) / 2))
                crop_xyxy = clamp_crop(center_x, center_y, crop_edge, width, height)
                crop_boxes_out = crop_boxes(boxes, crop_xyxy, width, height)
                if not crop_boxes_out:
                    continue

                crop = image_rgb.crop(crop_xyxy)
                out_stem = f"{image_path.stem}__focuscrop_{idx}"
                crop_ext = image_path.suffix if image_path.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
                crop_image_path = images_out / f"{out_stem}{crop_ext}"
                crop_label_path = labels_out / f"{out_stem}.txt"
                crop.save(crop_image_path)
                save_label_file(crop_label_path, crop_boxes_out)
                generated += 1

    return {
        "copied_train_images": copied,
        "generated_focus_crops": generated,
        "triggered_train_images": triggered_images,
    }


def main() -> None:
    args = parse_args()
    source_yaml = args.source_yaml.resolve()
    output_root = args.output_root.resolve()
    spec = resolve_dataset_spec(source_yaml)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    names = list(spec["names"])
    write_dataset_yaml(output_root, names, int(spec["nc"]))
    val_count = build_split_copy(Path(spec["val_images"]), output_root, "valid")
    test_count = build_split_copy(Path(spec["test_images"]), output_root, "test")
    train_stats = build_focus_train_split(
        source_images=Path(spec["train_images"]),
        output_root=output_root,
        class_names=names,
        target_classes=set(args.target_classes),
        area_thr=args.small_area_threshold,
        dim_thr=args.small_dim_threshold,
        min_crop_edge=args.min_crop_edge,
        max_crop_edge=args.max_crop_edge,
        context_scale=args.context_scale,
        max_crops_per_image=args.max_crops_per_image,
    )

    meta = {
        "dataset_id": output_root.name,
        "source_dataset_yaml": str(source_yaml),
        "source_dataset": str(spec["base"]),
        "target_classes": list(args.target_classes),
        "small_area_threshold": args.small_area_threshold,
        "small_dim_threshold": args.small_dim_threshold,
        "min_crop_edge": args.min_crop_edge,
        "max_crop_edge": args.max_crop_edge,
        "context_scale": args.context_scale,
        "max_crops_per_image": args.max_crops_per_image,
        "stats": {
            **train_stats,
            "copied_valid_images": val_count,
            "copied_test_images": test_count,
        },
        "notes": "Train split keeps all original images and adds box-centered focus crops for small target classes. Valid/test stay unchanged for fair comparison.",
    }
    (output_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"focus_crop_dataset={output_root}")
    print(json.dumps(meta["stats"], ensure_ascii=False))


if __name__ == "__main__":
    main()
