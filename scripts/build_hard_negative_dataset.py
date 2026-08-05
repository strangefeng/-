from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repo_runtime import configure_repo_runtime

configure_repo_runtime(ROOT)

from pipeline_utils import create_ultralytics_model, resolve_device


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_xyxy(self, width: int, height: int) -> tuple[float, float, float, float]:
        x1 = (self.cx - self.w / 2) * width
        y1 = (self.cy - self.h / 2) * height
        x2 = (self.cx + self.w / 2) * width
        y2 = (self.cy + self.h / 2) * height
        return x1, y1, x2, y2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine high-confidence false-positive crops and add them as empty-label hard negatives."
    )
    parser.add_argument("--source-yaml", type=Path, required=True, help="Source dataset YAML.")
    parser.add_argument("--weights", type=Path, required=True, help="Weights used to mine false positives.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output dataset root.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["valid", "test"],
        choices=("train", "valid", "test"),
        help="Dataset splits used for mining false positives.",
    )
    parser.add_argument("--imgsz", type=int, default=800, help="Prediction imgsz.")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.70, help="Prediction NMS IoU.")
    parser.add_argument("--max-det", type=int, default=200, help="Max detections per image.")
    parser.add_argument("--max-negatives", type=int, default=50, help="How many negative crops to mine.")
    parser.add_argument(
        "--negative-images-dir",
        type=Path,
        help="Optional directory of standalone negative images to add with empty labels.",
    )
    parser.add_argument(
        "--negative-limit",
        type=int,
        help="Optional cap for external negatives. Defaults to --max-negatives.",
    )
    parser.add_argument(
        "--copy-source-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to copy the original train split before adding hard negatives.",
    )
    parser.add_argument(
        "--fallback-to-mined",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mine false-positive crops if external negatives are absent or insufficient.",
    )
    parser.add_argument(
        "--ignore-gt-iou",
        type=float,
        default=0.10,
        help="A prediction is considered false positive if it overlaps every GT box below this IoU.",
    )
    parser.add_argument("--min-crop-edge", type=int, default=160, help="Minimum crop edge in pixels.")
    parser.add_argument("--max-crop-edge", type=int, default=320, help="Maximum crop edge in pixels.")
    parser.add_argument(
        "--context-scale",
        type=float,
        default=3.0,
        help="Crop edge multiplier relative to the larger prediction side.",
    )
    parser.add_argument("--device", default="auto", help="Inference device.")
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
    return sorted(path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


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


def ensure_empty_split(output_root: Path, split_name: str) -> dict[str, int]:
    dest_root = output_root / split_name
    if dest_root.exists():
        shutil.rmtree(dest_root)
    (dest_root / "images").mkdir(parents=True, exist_ok=True)
    (dest_root / "labels").mkdir(parents=True, exist_ok=True)
    return {"images": 0, "labels": 0}


def box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(1.0, ax2 - ax1) * max(1.0, ay2 - ay1)
    area_b = max(1.0, bx2 - bx1) * max(1.0, by2 - by1)
    return inter_area / max(1e-12, area_a + area_b - inter_area)


def crop_window(
    prediction_xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
    context_scale: float,
    min_crop_edge: int,
    max_crop_edge: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = prediction_xyxy
    side = max(x2 - x1, y2 - y1)
    crop_edge = int(round(side * context_scale))
    crop_edge = max(min_crop_edge, crop_edge)
    crop_edge = min(max_crop_edge, crop_edge, width, height)
    center_x = int(round((x1 + x2) / 2))
    center_y = int(round((y1 + y2) / 2))
    half = crop_edge // 2
    left = max(0, min(center_x - half, width - crop_edge))
    top = max(0, min(center_y - half, height - crop_edge))
    return left, top, left + crop_edge, top + crop_edge


def contains_gt_center(crop_xyxy: tuple[int, int, int, int], gt_boxes: list[tuple[float, float, float, float]]) -> bool:
    x1, y1, x2, y2 = crop_xyxy
    for gx1, gy1, gx2, gy2 in gt_boxes:
        center_x = (gx1 + gx2) / 2
        center_y = (gy1 + gy2) / 2
        if x1 <= center_x < x2 and y1 <= center_y < y2:
            return True
    return False


def copy_external_negative_images(
    negative_images_dir: Path,
    train_images_out: Path,
    train_labels_out: Path,
    limit: int,
) -> tuple[int, list[dict[str, object]]]:
    if not negative_images_dir.exists():
        return 0, []

    copied = 0
    manifest: list[dict[str, object]] = []
    for image_path in list_images(negative_images_dir):
        if copied >= limit:
            break
        stem = f"extneg_{copied + 1:04d}_{image_path.stem}"
        dest_image = train_images_out / f"{stem}{image_path.suffix.lower()}"
        dest_label = train_labels_out / f"{stem}.txt"
        shutil.copy2(image_path, dest_image)
        dest_label.write_text("", encoding="utf-8")
        manifest.append(
            {
                "id": stem,
                "source_type": "external_negative_image",
                "source_image": str(image_path),
            }
        )
        copied += 1
    return copied, manifest


def main() -> int:
    args = parse_args()
    source_yaml = args.source_yaml.resolve()
    output_root = args.output_root.resolve()
    spec = resolve_dataset_spec(source_yaml)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.copy_source_train:
        train_stats = copy_split_tree(Path(spec["train_images"]), output_root, "train")
    else:
        train_stats = ensure_empty_split(output_root, "train")
    valid_stats = copy_split_tree(Path(spec["val_images"]), output_root, "valid")
    test_stats = copy_split_tree(Path(spec["test_images"]), output_root, "test")

    model = create_ultralytics_model(str(args.weights.resolve()))
    resolved_device = resolve_device(args.device)
    train_images_out = output_root / "train" / "images"
    train_labels_out = output_root / "train" / "labels"

    target_negative_count = args.negative_limit if args.negative_limit is not None else args.max_negatives
    added_negatives = 0
    mined = 0
    external_negative_count = 0
    manifest: list[dict[str, object]] = []
    split_path_map = {
        "train": Path(spec["train_images"]),
        "valid": Path(spec["val_images"]),
        "test": Path(spec["test_images"]),
    }

    if args.negative_images_dir:
        copied, ext_manifest = copy_external_negative_images(
            negative_images_dir=args.negative_images_dir.resolve(),
            train_images_out=train_images_out,
            train_labels_out=train_labels_out,
            limit=target_negative_count,
        )
        external_negative_count += copied
        added_negatives += copied
        manifest.extend(ext_manifest)

    remaining_to_mine = max(0, target_negative_count - added_negatives)
    if remaining_to_mine > 0 and args.fallback_to_mined:
        for split_name in args.splits:
            for image_path in list_images(split_path_map[split_name]):
                if mined >= remaining_to_mine:
                    break
                with Image.open(image_path) as image:
                    image_rgb = image.convert("RGB")
                    width, height = image_rgb.size
                gt_boxes = [box.to_xyxy(width, height) for box in parse_label_file(label_path_for_image(image_path))]
                result = model.predict(
                    source=str(image_path),
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    device=resolved_device,
                    verbose=False,
                )[0]
                boxes = getattr(result, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    continue

                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                with Image.open(image_path) as image:
                    source_rgb = image.convert("RGB")
                    for coords, score, class_id in zip(xyxy, confs, classes):
                        if mined >= remaining_to_mine:
                            break
                        pred_box = tuple(float(value) for value in coords.tolist())
                        max_gt_iou = max((box_iou(pred_box, gt_box) for gt_box in gt_boxes), default=0.0)
                        if max_gt_iou >= args.ignore_gt_iou:
                            continue
                        crop_xyxy = crop_window(
                            prediction_xyxy=pred_box,
                            width=width,
                            height=height,
                            context_scale=args.context_scale,
                            min_crop_edge=args.min_crop_edge,
                            max_crop_edge=args.max_crop_edge,
                        )
                        if contains_gt_center(crop_xyxy, gt_boxes):
                            continue
                        x1, y1, x2, y2 = crop_xyxy
                        crop = source_rgb.crop((x1, y1, x2, y2))
                        crop_stem = f"fpneg_{mined + 1:04d}_{split_name}_{image_path.stem}"
                        crop_image_path = train_images_out / f"{crop_stem}.jpg"
                        crop_label_path = train_labels_out / f"{crop_stem}.txt"
                        crop.save(crop_image_path, format="JPEG", quality=95)
                        crop_label_path.write_text("", encoding="utf-8")
                        manifest.append(
                            {
                                "id": crop_stem,
                                "source_type": "mined_false_positive_crop",
                                "source_split": split_name,
                                "source_image": str(image_path),
                                "predicted_class_id": int(class_id),
                                "confidence": float(score),
                                "prediction_xyxy": [round(value, 3) for value in pred_box],
                                "crop_xyxy": [x1, y1, x2, y2],
                            }
                        )
                        mined += 1
            if mined >= remaining_to_mine:
                break

    added_negatives += mined

    write_dataset_yaml(output_root, list(spec["names"]), int(spec["nc"]))
    meta = {
        "dataset_id": output_root.name,
        "source_dataset_yaml": str(source_yaml),
        "source_dataset": str(spec["base"]),
        "weights": str(args.weights.resolve()),
        "splits": list(args.splits),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "max_negatives": args.max_negatives,
        "negative_limit": args.negative_limit,
        "copy_source_train": args.copy_source_train,
        "fallback_to_mined": args.fallback_to_mined,
        "negative_images_dir": str(args.negative_images_dir.resolve()) if args.negative_images_dir else "",
        "ignore_gt_iou": args.ignore_gt_iou,
        "min_crop_edge": args.min_crop_edge,
        "max_crop_edge": args.max_crop_edge,
        "context_scale": args.context_scale,
        "resolved_device": resolved_device,
        "external_negative_images": external_negative_count,
        "mined_negative_crops": mined,
        "added_negative_images_total": added_negatives,
        "manifest": manifest,
        "train_stats": train_stats,
        "valid_stats": valid_stats,
        "test_stats": test_stats,
        "notes": "Adds hard negatives from external normal-surface images when available and falls back to mined false-positive crops if needed.",
    }
    (output_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"mined_negative_crops": mined, "output_root": str(output_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
