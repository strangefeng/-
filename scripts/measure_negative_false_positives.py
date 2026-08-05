from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repo_runtime import configure_repo_runtime

configure_repo_runtime(ROOT)

from pipeline_utils import create_ultralytics_model, normalize_path, resolve_device


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure false positives on empty-label negative images.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--high-conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--vis-dir", type=Path, help="Optional directory for top false-positive visualizations.")
    parser.add_argument("--topk-vis", type=int, default=12, help="How many visualization examples to save.")
    parser.add_argument("--title", default="")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def resolve_dataset_spec(dataset_yaml: Path) -> dict[str, object]:
    data = load_yaml(dataset_yaml)
    base = Path(data["path"])
    if not base.is_absolute():
        base = (dataset_yaml.parent / base).resolve()

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
    }


def list_images(images_dir: Path) -> list[Path]:
    return sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def label_path_for_image(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def is_empty_label(label_path: Path) -> bool:
    return (not label_path.exists()) or (label_path.read_text(encoding="utf-8").strip() == "")


def load_font(size: int = 18) -> ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def class_names_from_model(model: object) -> list[str]:
    names = getattr(getattr(model, "model", None), "names", None) or getattr(model, "names", None) or {}
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names)]
    return [str(name) for name in names]


def draw_prediction_visual(
    image_path: Path,
    boxes_xyxy: list[list[float]],
    box_classes: list[int],
    box_confidences: list[float],
    class_names: list[str],
    save_path: Path,
    title: str,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font(18)
    colors = [
        (235, 87, 87),
        (47, 128, 237),
        (39, 174, 96),
        (242, 153, 74),
        (155, 81, 224),
    ]

    for box, class_id, confidence in zip(boxes_xyxy, box_classes, box_confidences):
        x1, y1, x2, y2 = box
        color = colors[class_id % len(colors)]
        label = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
        caption = f"{label} {confidence:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_bbox = draw.textbbox((x1, y1), caption, font=font)
        tx1, ty1, tx2, ty2 = text_bbox
        draw.rectangle((tx1 - 2, ty1 - 2, tx2 + 2, ty2 + 2), fill=color)
        draw.text((x1, max(0, y1 - (ty2 - ty1) - 2)), caption, fill=(255, 255, 255), font=font)

    if title:
        banner_h = 30
        draw.rectangle((0, 0, image.width, banner_h), fill=(20, 20, 20))
        draw.text((8, 6), title, fill=(255, 255, 255), font=font)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path, format="JPEG", quality=95)


def main() -> int:
    args = parse_args()
    weights = args.weights.resolve()
    dataset_yaml = args.dataset_yaml.resolve()
    spec = resolve_dataset_spec(dataset_yaml)
    split_dir = {
        "train": Path(spec["train_images"]),
        "valid": Path(spec["val_images"]),
        "test": Path(spec["test_images"]),
    }[args.split]

    negative_images = [image_path for image_path in list_images(split_dir) if is_empty_label(label_path_for_image(image_path))]
    device = resolve_device(args.device)
    model = create_ultralytics_model(str(weights))
    class_names = class_names_from_model(model)

    total_detections = 0
    high_conf_detections = 0
    per_class_totals = {name: 0 for name in class_names}
    per_class_high_conf_totals = {name: 0 for name in class_names}
    per_image: list[dict[str, object]] = []

    for image_path in negative_images:
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=device,
            verbose=False,
        )[0]
        boxes = getattr(result, "boxes", None)
        count = int(len(boxes)) if boxes is not None else 0
        high_conf_count = 0
        box_classes: list[int] = []
        box_confidences: list[float] = []
        boxes_xyxy: list[list[float]] = []
        per_class_counts = {name: 0 for name in class_names}
        per_class_high_conf = {name: 0 for name in class_names}
        if boxes is not None and len(boxes):
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            coords = boxes.xyxy.cpu().numpy()
            high_conf_mask = confs >= args.high_conf
            high_conf_count = int(high_conf_mask.sum())
            box_classes = [int(value) for value in classes.tolist()]
            box_confidences = [float(value) for value in confs.tolist()]
            boxes_xyxy = [[float(v) for v in row.tolist()] for row in coords]
            for class_id, confidence in zip(box_classes, box_confidences):
                class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
                per_class_counts.setdefault(class_name, 0)
                per_class_high_conf.setdefault(class_name, 0)
                per_class_counts[class_name] += 1
                per_class_totals.setdefault(class_name, 0)
                per_class_totals[class_name] += 1
                if confidence >= args.high_conf:
                    per_class_high_conf[class_name] += 1
                    per_class_high_conf_totals.setdefault(class_name, 0)
                    per_class_high_conf_totals[class_name] += 1
        total_detections += count
        high_conf_detections += high_conf_count
        per_image.append(
            {
                "image": normalize_path(image_path),
                "detections": count,
                "high_conf_detections": high_conf_count,
                "per_class_false_positives": per_class_counts,
                "per_class_high_conf_false_positives": per_class_high_conf,
                "boxes_xyxy": boxes_xyxy,
                "box_classes": box_classes,
                "box_confidences": box_confidences,
            }
        )

    per_image_sorted = sorted(
        per_image,
        key=lambda item: (
            int(item.get("high_conf_detections", 0)),
            int(item.get("detections", 0)),
        ),
        reverse=True,
    )

    visualization_examples: list[str] = []
    if args.vis_dir:
        vis_dir = args.vis_dir.resolve()
        vis_dir.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(per_image_sorted[: max(0, args.topk_vis)], start=1):
            image_path = Path(str(item["image"]))
            save_path = vis_dir / f"{index:03d}_{image_path.stem}.jpg"
            title = (
                f"fp={item['detections']} "
                f"high_conf={item['high_conf_detections']} "
                f"{image_path.name}"
            )
            draw_prediction_visual(
                image_path=image_path,
                boxes_xyxy=[list(map(float, box)) for box in item.get("boxes_xyxy", [])],
                box_classes=[int(value) for value in item.get("box_classes", [])],
                box_confidences=[float(value) for value in item.get("box_confidences", [])],
                class_names=class_names,
                save_path=save_path,
                title=title,
            )
            visualization_examples.append(normalize_path(save_path))

    payload = {
        "title": args.title or f"negative_fp_{weights.stem}",
        "weights_path": normalize_path(weights),
        "dataset_yaml": normalize_path(dataset_yaml),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "high_conf": args.high_conf,
        "iou": args.iou,
        "device": device,
        "negative_image_count": len(negative_images),
        "total_false_positive_boxes": total_detections,
        "high_conf_false_positive_boxes": high_conf_detections,
        "per_class_false_positives": per_class_totals,
        "per_class_high_conf_false_positives": per_class_high_conf_totals,
        "avg_false_positive_boxes_per_image": (total_detections / len(negative_images)) if negative_images else 0.0,
        "avg_high_conf_false_positive_boxes_per_image": (high_conf_detections / len(negative_images)) if negative_images else 0.0,
        "visualization_examples": visualization_examples,
        "per_image": per_image_sorted,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
