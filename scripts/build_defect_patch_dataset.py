from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter


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


@dataclass(frozen=True)
class DonorPatch:
    class_id: int
    class_name: str
    image_path: Path
    patch_array: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a train-only defect patch augmentation dataset for Aircraft AI 3-class experiments."
    )
    parser.add_argument("--source-yaml", type=Path, required=True, help="Source dataset YAML.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output dataset root.")
    parser.add_argument(
        "--target-classes",
        nargs="+",
        default=["missing-rivet"],
        help="Class names to use as donor patches.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--copies-per-image", type=int, default=1, help="Synthetic variants per train image.")
    parser.add_argument("--pastes-per-copy", type=int, default=2, help="How many patches to paste into each copy.")
    parser.add_argument("--min-scale", type=float, default=0.90, help="Minimum donor patch scale.")
    parser.add_argument("--max-scale", type=float, default=1.15, help="Maximum donor patch scale.")
    parser.add_argument("--max-iou", type=float, default=0.05, help="Maximum overlap with existing boxes.")
    parser.add_argument(
        "--brightness-jitter",
        type=float,
        default=0.0,
        help="Symmetric brightness multiplier jitter, e.g. 0.12 gives [0.88, 1.12].",
    )
    parser.add_argument(
        "--contrast-jitter",
        type=float,
        default=0.0,
        help="Symmetric contrast multiplier jitter.",
    )
    parser.add_argument("--blur-prob", type=float, default=0.0, help="Probability of blurring a pasted patch.")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Gaussian noise std on pasted patch RGB values.")
    parser.add_argument("--prefer-bright", action="store_true", help="Prefer brighter placement windows.")
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


def clamp_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def collect_donor_patches(
    image_paths: list[Path],
    target_class_ids: set[int],
    class_names: list[str],
    min_patch_edge: int = 4,
    pad_ratio: float = 0.08,
) -> list[DonorPatch]:
    donors: list[DonorPatch] = []
    for image_path in image_paths:
        boxes = parse_label_file(label_path_for_image(image_path))
        if not boxes:
            continue
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
            width, height = image_rgb.size
            image_array = np.array(image_rgb)
        for box in boxes:
            if box.class_id not in target_class_ids:
                continue
            x1, y1, x2, y2 = box.to_xyxy(width, height)
            pad_x = int(round((x2 - x1) * pad_ratio))
            pad_y = int(round((y2 - y1) * pad_ratio))
            clamped = clamp_box(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, width, height)
            if clamped is None:
                continue
            px1, py1, px2, py2 = clamped
            patch = image_array[py1:py2, px1:px2]
            if min(patch.shape[:2]) < min_patch_edge:
                continue
            donors.append(
                DonorPatch(
                    class_id=box.class_id,
                    class_name=class_names[box.class_id],
                    image_path=image_path,
                    patch_array=patch,
                )
            )
    return donors


def build_soft_mask(size: tuple[int, int], feather: int = 8) -> Image.Image:
    width, height = size
    mask = Image.new("L", (width, height), color=255)
    if feather <= 0:
        return mask
    inner = Image.new("L", (max(1, width - feather * 2), max(1, height - feather * 2)), color=255)
    softened = Image.new("L", (width, height), color=0)
    softened.paste(inner, (min(feather, width // 2), min(feather, height // 2)))
    return softened.filter(ImageFilter.GaussianBlur(radius=max(1, feather // 2)))


def resize_patch(patch: np.ndarray, scale: float) -> np.ndarray:
    patch_image = Image.fromarray(patch)
    new_w = max(2, int(round(patch_image.width * scale)))
    new_h = max(2, int(round(patch_image.height * scale)))
    return np.array(patch_image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR))


def iou_xyxy(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    area_b = max(1, bx2 - bx1) * max(1, by2 - by1)
    return inter_area / float(area_a + area_b - inter_area)


def existing_xyxy_boxes(boxes: Iterable[YoloBox], width: int, height: int) -> list[tuple[int, int, int, int]]:
    return [box.to_xyxy(width, height) for box in boxes]


def choose_paste_position(
    rng: random.Random,
    image_array: np.ndarray,
    patch_size: tuple[int, int],
    existing_boxes_xyxy: list[tuple[int, int, int, int]],
    max_iou: float,
    prefer_bright: bool,
    attempts: int = 48,
) -> tuple[int, int] | None:
    height, width = image_array.shape[:2]
    patch_w, patch_h = patch_size
    if patch_w >= width or patch_h >= height:
        return None

    gray = image_array.mean(axis=2) if prefer_bright else None
    candidates: list[tuple[float, tuple[int, int]]] = []
    for _ in range(attempts):
        x1 = rng.randint(0, width - patch_w)
        y1 = rng.randint(0, height - patch_h)
        candidate = (x1, y1, x1 + patch_w, y1 + patch_h)
        if any(iou_xyxy(candidate, existing) > max_iou for existing in existing_boxes_xyxy):
            continue
        score = float(gray[y1 : y1 + patch_h, x1 : x1 + patch_w].mean()) if gray is not None else 0.0
        candidates.append((score, (x1, y1)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[rng.randrange(min(5, len(candidates)))][1]


def apply_patch_transforms(
    patch: np.ndarray,
    rng: random.Random,
    brightness_jitter: float,
    contrast_jitter: float,
    blur_prob: float,
    noise_std: float,
) -> np.ndarray:
    image = Image.fromarray(patch).convert("RGB")
    if brightness_jitter > 0:
        factor = 1.0 + rng.uniform(-brightness_jitter, brightness_jitter)
        image = ImageEnhance.Brightness(image).enhance(max(0.1, factor))
    if contrast_jitter > 0:
        factor = 1.0 + rng.uniform(-contrast_jitter, contrast_jitter)
        image = ImageEnhance.Contrast(image).enhance(max(0.1, factor))
    if blur_prob > 0 and rng.random() < blur_prob:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.2)))
    array = np.array(image)
    if noise_std > 0:
        np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
        noise = np_rng.normal(0.0, noise_std, size=array.shape)
        array = np.clip(array.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return array


def paste_patch(base_image: Image.Image, patch: np.ndarray, position: tuple[int, int]) -> None:
    patch_image = Image.fromarray(patch).convert("RGBA")
    mask = build_soft_mask(patch_image.size, feather=max(2, min(patch_image.size) // 8))
    patch_image.putalpha(mask)
    base_image.alpha_composite(patch_image, dest=position)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    source_yaml = args.source_yaml.resolve()
    output_root = args.output_root.resolve()
    spec = resolve_dataset_spec(source_yaml)
    class_names = list(spec["names"])
    target_class_ids = {
        class_names.index(class_name)
        for class_name in args.target_classes
        if class_name in class_names
    }
    if len(target_class_ids) != len(args.target_classes):
        missing = sorted(set(args.target_classes) - set(class_names))
        raise SystemExit(f"Unknown target classes: {missing}")

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_stats = copy_split_tree(Path(spec["train_images"]), output_root, "train")
    valid_stats = copy_split_tree(Path(spec["val_images"]), output_root, "valid")
    test_stats = copy_split_tree(Path(spec["test_images"]), output_root, "test")

    train_images = list_images(Path(spec["train_images"]))
    donors = collect_donor_patches(train_images, target_class_ids, class_names)
    if not donors:
        raise SystemExit("No donor patches were found for the requested classes.")

    train_output_images = output_root / "train" / "images"
    train_output_labels = output_root / "train" / "labels"
    generated_count = 0
    pasted_boxes_total = 0
    class_paste_counts = {class_name: 0 for class_name in args.target_classes}

    for image_path in train_images:
        original_boxes = parse_label_file(label_path_for_image(image_path))
        for copy_idx in range(args.copies_per_image):
            with Image.open(image_path) as original_image:
                base_image = original_image.convert("RGBA")
                width, height = base_image.size
                image_array = np.array(base_image.convert("RGB"))

            boxes = list(original_boxes)
            occupied = existing_xyxy_boxes(boxes, width, height)
            pasted_here = 0

            for _ in range(args.pastes_per_copy):
                candidate_donors = [donor for donor in donors if donor.image_path != image_path]
                position = None
                patch = None
                donor = None
                patch_h = 0
                patch_w = 0
                for _attempt in range(8):
                    donor = rng.choice(candidate_donors or donors)
                    scale = rng.uniform(args.min_scale, args.max_scale)
                    patch = resize_patch(donor.patch_array, scale=scale)
                    patch = apply_patch_transforms(
                        patch=patch,
                        rng=rng,
                        brightness_jitter=args.brightness_jitter,
                        contrast_jitter=args.contrast_jitter,
                        blur_prob=args.blur_prob,
                        noise_std=args.noise_std,
                    )
                    patch_h, patch_w = patch.shape[:2]
                    position = choose_paste_position(
                        rng=rng,
                        image_array=image_array,
                        patch_size=(patch_w, patch_h),
                        existing_boxes_xyxy=occupied,
                        max_iou=args.max_iou,
                        prefer_bright=args.prefer_bright,
                    )
                    if position is not None:
                        break
                if position is None or patch is None or donor is None:
                    continue

                x1, y1 = position
                x2, y2 = x1 + patch_w, y1 + patch_h
                paste_patch(base_image, patch, position)
                image_array[y1:y2, x1:x2] = patch[: y2 - y1, : x2 - x1]
                boxes.append(YoloBox.from_xyxy(donor.class_id, x1, y1, x2, y2, width, height))
                occupied.append((x1, y1, x2, y2))
                pasted_here += 1
                pasted_boxes_total += 1
                class_paste_counts[donor.class_name] = class_paste_counts.get(donor.class_name, 0) + 1

            if pasted_here == 0:
                continue

            aug_stem = f"{image_path.stem}__dp{copy_idx + 1}"
            aug_image_path = train_output_images / f"{aug_stem}{image_path.suffix}"
            aug_label_path = train_output_labels / f"{aug_stem}.txt"
            save_format = "PNG" if image_path.suffix.lower() == ".png" else "JPEG"
            save_kwargs = {"quality": 95} if save_format == "JPEG" else {}
            base_image.convert("RGB").save(aug_image_path, format=save_format, **save_kwargs)
            save_label_file(aug_label_path, boxes)
            generated_count += 1

    write_dataset_yaml(output_root, class_names, int(spec["nc"]))
    meta = {
        "dataset_id": output_root.name,
        "source_dataset_yaml": str(source_yaml),
        "source_dataset": str(spec["base"]),
        "target_classes": list(args.target_classes),
        "copies_per_image": args.copies_per_image,
        "pastes_per_copy": args.pastes_per_copy,
        "min_scale": args.min_scale,
        "max_scale": args.max_scale,
        "max_iou": args.max_iou,
        "brightness_jitter": args.brightness_jitter,
        "contrast_jitter": args.contrast_jitter,
        "blur_prob": args.blur_prob,
        "noise_std": args.noise_std,
        "prefer_bright": args.prefer_bright,
        "seed": args.seed,
        "donor_count": len(donors),
        "generated_augmented_images": generated_count,
        "generated_pasted_boxes": pasted_boxes_total,
        "class_paste_counts": class_paste_counts,
        "train_stats": train_stats,
        "valid_stats": valid_stats,
        "test_stats": test_stats,
        "notes": "Train-only defect patch augmentation dataset for missing-rivet and scratch scarcity experiments.",
    }
    (output_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
