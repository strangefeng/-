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
        return cls(
            class_id=class_id,
            cx=(x1 + x2) / 2 / width,
            cy=(y1 + y2) / 2 / height,
            w=box_w / width,
            h=box_h / height,
        )

    def serialize(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a sliced YOLO dataset variant for small-defect training."
    )
    parser.add_argument("--source-yaml", type=Path, required=True, help="Source dataset YAML.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output dataset root.")
    parser.add_argument(
        "--mode",
        choices=("sliced", "mixed"),
        default="sliced",
        help="Use only sliced train tiles, or mix original train images with sliced tiles.",
    )
    parser.add_argument("--slice-size", type=int, default=512, help="Tile edge length in pixels.")
    parser.add_argument("--overlap", type=float, default=0.20, help="Tile overlap ratio in [0, 0.8).")
    parser.add_argument(
        "--min-box-size",
        type=int,
        default=2,
        help="Discard clipped boxes smaller than this edge size in pixels.",
    )
    parser.add_argument(
        "--max-empty-tiles-per-image",
        type=int,
        default=1,
        help="Keep at most this many empty tiles per source image. Use -1 to keep all empty tiles.",
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
        "base": base.resolve(),
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


def copy_split_tree(source_images: Path, output_root: Path, split_name: str) -> dict[str, int]:
    source_root = source_images.parent
    dest_root = output_root / split_name
    if dest_root.exists():
        shutil.rmtree(dest_root)
    shutil.copytree(source_root, dest_root)
    return {
        "images": sum(1 for _ in (dest_root / "images").iterdir()),
        "labels": sum(1 for _ in (dest_root / "labels").iterdir()),
    }


def write_dataset_yaml(output_root: Path, names: list[str], nc: int) -> None:
    payload = {
        "path": output_root.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": nc,
        "names": names,
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=False, sort_keys=False),
        encoding="utf-8",
    )


def axis_positions(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    values = list(range(0, max(1, length - window + 1), stride))
    last = length - window
    if values[-1] != last:
        values.append(last)
    return values


def generate_tiles(width: int, height: int, slice_size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    stride = max(1, int(round(slice_size * (1.0 - overlap))))
    xs = axis_positions(width, min(slice_size, width), stride)
    ys = axis_positions(height, min(slice_size, height), stride)
    tile_w = min(slice_size, width)
    tile_h = min(slice_size, height)
    return [(x1, y1, x1 + tile_w, y1 + tile_h) for y1 in ys for x1 in xs]


def crop_boxes_for_tile(
    boxes: list[YoloBox],
    tile_xyxy: tuple[int, int, int, int],
    width: int,
    height: int,
    min_box_size: int,
) -> list[YoloBox]:
    x1, y1, x2, y2 = tile_xyxy
    tile_w = x2 - x1
    tile_h = y2 - y1
    cropped: list[YoloBox] = []
    for box in boxes:
        bx1, by1, bx2, by2 = box.to_xyxy(width, height)
        center_x = (bx1 + bx2) / 2
        center_y = (by1 + by2) / 2
        if not (x1 <= center_x < x2 and y1 <= center_y < y2):
            continue
        inter_x1 = max(x1, bx1)
        inter_y1 = max(y1, by1)
        inter_x2 = min(x2, bx2)
        inter_y2 = min(y2, by2)
        if inter_x2 - inter_x1 < min_box_size or inter_y2 - inter_y1 < min_box_size:
            continue
        cropped.append(
            YoloBox.from_xyxy(
                class_id=box.class_id,
                x1=inter_x1 - x1,
                y1=inter_y1 - y1,
                x2=inter_x2 - x1,
                y2=inter_y2 - y1,
                width=tile_w,
                height=tile_h,
            )
        )
    return cropped


def copy_original_train(train_images: Path, output_root: Path) -> int:
    images_out = output_root / "train" / "images"
    labels_out = output_root / "train" / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    copied = 0
    for image_path in list_images(train_images):
        label_path = label_path_for_image(image_path)
        shutil.copy2(image_path, images_out / image_path.name)
        if label_path.exists():
            shutil.copy2(label_path, labels_out / label_path.name)
        else:
            (labels_out / f"{image_path.stem}.txt").write_text("", encoding="utf-8")
        copied += 1
    return copied


def build_train_tiles(
    train_images: Path,
    output_root: Path,
    slice_size: int,
    overlap: float,
    min_box_size: int,
    max_empty_tiles_per_image: int,
) -> dict[str, int]:
    images_out = output_root / "train" / "images"
    labels_out = output_root / "train" / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    saved_tiles = 0
    saved_non_empty_tiles = 0
    saved_empty_tiles = 0

    for image_path in list_images(train_images):
        label_path = label_path_for_image(image_path)
        boxes = parse_label_file(label_path)
        empty_budget = max_empty_tiles_per_image
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
            width, height = image_rgb.size
            for x1, y1, x2, y2 in generate_tiles(width, height, slice_size, overlap):
                tile_boxes = crop_boxes_for_tile(boxes, (x1, y1, x2, y2), width, height, min_box_size)
                if not tile_boxes:
                    if empty_budget == 0:
                        continue
                    if empty_budget > 0:
                        empty_budget -= 1

                tile = image_rgb.crop((x1, y1, x2, y2))
                tile_stem = f"{image_path.stem}__sl{x1}_{y1}_{x2}_{y2}"
                tile_image_path = images_out / f"{tile_stem}{image_path.suffix}"
                tile_label_path = labels_out / f"{tile_stem}.txt"
                save_format = "PNG" if image_path.suffix.lower() == ".png" else "JPEG"
                save_kwargs = {"quality": 95} if save_format == "JPEG" else {}
                tile.save(tile_image_path, format=save_format, **save_kwargs)
                save_label_file(tile_label_path, tile_boxes)
                saved_tiles += 1
                if tile_boxes:
                    saved_non_empty_tiles += 1
                else:
                    saved_empty_tiles += 1

    return {
        "saved_tiles": saved_tiles,
        "saved_non_empty_tiles": saved_non_empty_tiles,
        "saved_empty_tiles": saved_empty_tiles,
    }


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.overlap < 0.8:
        raise SystemExit("--overlap must be in [0, 0.8).")
    if args.slice_size < 64:
        raise SystemExit("--slice-size must be >= 64.")

    source_yaml = args.source_yaml.resolve()
    output_root = args.output_root.resolve()
    spec = resolve_dataset_spec(source_yaml)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    valid_stats = copy_split_tree(Path(spec["val_images"]), output_root, "valid")
    test_stats = copy_split_tree(Path(spec["test_images"]), output_root, "test")

    original_train_images = 0
    if args.mode == "mixed":
        original_train_images = copy_original_train(Path(spec["train_images"]), output_root)
    else:
        (output_root / "train" / "images").mkdir(parents=True, exist_ok=True)
        (output_root / "train" / "labels").mkdir(parents=True, exist_ok=True)

    train_stats = build_train_tiles(
        train_images=Path(spec["train_images"]),
        output_root=output_root,
        slice_size=args.slice_size,
        overlap=args.overlap,
        min_box_size=args.min_box_size,
        max_empty_tiles_per_image=args.max_empty_tiles_per_image,
    )
    write_dataset_yaml(output_root, list(spec["names"]), int(spec["nc"]))

    meta = {
        "dataset_id": output_root.name,
        "source_dataset_yaml": str(source_yaml),
        "source_dataset": str(spec["base"]),
        "mode": args.mode,
        "slice_size": args.slice_size,
        "overlap": args.overlap,
        "min_box_size": args.min_box_size,
        "max_empty_tiles_per_image": args.max_empty_tiles_per_image,
        "original_train_images_kept": original_train_images,
        "train_tile_stats": train_stats,
        "valid_stats": valid_stats,
        "test_stats": test_stats,
        "notes": "Sliced dataset variant for small-object-heavy aircraft surface defect experiments.",
    }
    (output_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
