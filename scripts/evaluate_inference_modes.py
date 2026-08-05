from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repo_runtime import configure_repo_runtime

configure_repo_runtime(ROOT)

from pipeline_utils import create_ultralytics_model, normalize_path, resolve_device


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IOU_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


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


@dataclass(frozen=True)
class Detection:
    image_id: str
    class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO full-image, sliced, or fused inference on the original validation/test split."
    )
    parser.add_argument("--mode", choices=("full", "slice", "fused"), default="slice")
    parser.add_argument("--weights", type=Path, help="Fallback weights path used for both full and slice branches.")
    parser.add_argument("--full-weights", type=Path, help="Optional weights path for the full-image branch.")
    parser.add_argument("--slice-weights", type=Path, help="Optional weights path for the sliced branch.")
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--slice-size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--full-imgsz", type=int)
    parser.add_argument("--slice-imgsz", type=int)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--full-conf", type=float)
    parser.add_argument("--slice-conf", type=float)
    parser.add_argument("--final-conf", type=float, default=0.0)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--full-iou", type=float)
    parser.add_argument("--tile-iou", type=float, default=0.70)
    parser.add_argument("--merge-iou", type=float, default=0.50)
    parser.add_argument("--final-merge-iou", type=float)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--title", default="")
    parser.add_argument("--dataset-role", default="")
    parser.add_argument("--output-json", type=Path, required=True)
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
    tile_w = min(slice_size, width)
    tile_h = min(slice_size, height)
    xs = axis_positions(width, tile_w, stride)
    ys = axis_positions(height, tile_h, stride)
    return [(x1, y1, x1 + tile_w, y1 + tile_h) for y1 in ys for x1 in xs]


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


def classwise_nms(detections: list[Detection], iou_thr: float) -> list[Detection]:
    kept: list[Detection] = []
    for class_id in sorted({det.class_id for det in detections}):
        class_dets = sorted(
            (det for det in detections if det.class_id == class_id),
            key=lambda item: item.confidence,
            reverse=True,
        )
        while class_dets:
            current = class_dets.pop(0)
            kept.append(current)
            class_dets = [
                candidate
                for candidate in class_dets
                if box_iou(current.xyxy, candidate.xyxy) < iou_thr
            ]
    return kept


def extract_full_detections_for_image(
    model,
    image_path: Path,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
) -> list[Detection]:
    result = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        verbose=False,
    )[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    image_id = image_path.stem
    return [
        Detection(
            image_id=image_id,
            class_id=int(class_id),
            confidence=float(score),
            xyxy=tuple(float(value) for value in coords.tolist()),
        )
        for coords, score, class_id in zip(xyxy, confs, classes)
    ]


def extract_slice_detections_for_image(
    model,
    image_path: Path,
    slice_size: int,
    overlap: float,
    imgsz: int,
    conf: float,
    tile_iou: float,
    merge_iou: float,
    max_det: int,
    device: str,
) -> list[Detection]:
    with Image.open(image_path) as image:
        image_rgb = image.convert("RGB")
        width, height = image_rgb.size
        image_array = np.array(image_rgb)
    tiles = generate_tiles(width, height, slice_size, overlap)
    crops = [image_array[y1:y2, x1:x2] for x1, y1, x2, y2 in tiles]
    if not crops:
        return []

    results = model.predict(
        source=crops,
        imgsz=imgsz,
        conf=conf,
        iou=tile_iou,
        max_det=max_det,
        device=device,
        verbose=False,
    )

    image_id = image_path.stem
    detections: list[Detection] = []
    for tile_xyxy, result in zip(tiles, results):
        offset_x, offset_y, _, _ = tile_xyxy
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        for coords, score, class_id in zip(xyxy, confs, classes):
            x1, y1, x2, y2 = coords.tolist()
            detections.append(
                Detection(
                    image_id=image_id,
                    class_id=int(class_id),
                    confidence=float(score),
                    xyxy=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                )
            )
    return classwise_nms(detections, merge_iou)


def filter_by_confidence(detections: list[Detection], conf_thr: float) -> list[Detection]:
    if conf_thr <= 0:
        return detections
    return [det for det in detections if det.confidence >= conf_thr]


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    points = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[points + 1] - mrec[points]) * mpre[points + 1]))


def evaluate_class(
    predictions: list[Detection],
    gt_map: dict[str, list[tuple[float, float, float, float]]],
    iou_thr: float,
) -> tuple[float, int, int, int]:
    total_gt = sum(len(items) for items in gt_map.values())
    if total_gt == 0:
        return 0.0, 0, len(predictions), 0
    if not predictions:
        return 0.0, 0, 0, total_gt

    matches = {image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in gt_map.items()}
    predictions_sorted = sorted(predictions, key=lambda item: item.confidence, reverse=True)
    tp = np.zeros(len(predictions_sorted), dtype=np.float32)
    fp = np.zeros(len(predictions_sorted), dtype=np.float32)

    for index, prediction in enumerate(predictions_sorted):
        gt_boxes = gt_map.get(prediction.image_id, [])
        best_iou = 0.0
        best_gt_index = -1
        for gt_index, gt_box in enumerate(gt_boxes):
            if matches[prediction.image_id][gt_index]:
                continue
            iou = box_iou(prediction.xyxy, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index
        if best_gt_index >= 0 and best_iou >= iou_thr:
            matches[prediction.image_id][best_gt_index] = True
            tp[index] = 1.0
        else:
            fp[index] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(total_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    ap = compute_ap(recall, precision)
    return ap, int(tp.sum()), int(fp.sum()), total_gt


def micro_precision_recall(
    predictions_by_class: dict[int, list[Detection]],
    ground_truths: dict[int, dict[str, list[tuple[float, float, float, float]]]],
) -> tuple[float, float]:
    total_tp = 0
    total_fp = 0
    total_gt = 0
    for class_id, gt_map in ground_truths.items():
        _, tp, fp, gt_count = evaluate_class(predictions_by_class.get(class_id, []), gt_map, 0.50)
        total_tp += tp
        total_fp += fp
        total_gt += gt_count
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_gt, 1)
    return precision, recall


def evaluate_mode(
    *,
    mode: str,
    dataset_yaml: Path,
    split: str,
    class_names: list[str],
    full_model,
    slice_model,
    device: str,
    slice_size: int,
    overlap: float,
    full_imgsz: int,
    slice_imgsz: int,
    full_conf: float,
    slice_conf: float,
    final_conf: float,
    full_iou: float,
    tile_iou: float,
    merge_iou: float,
    final_merge_iou: float,
    max_det: int,
) -> dict[str, object]:
    spec = resolve_dataset_spec(dataset_yaml)
    split_images = Path(spec["val_images"]) if split == "valid" else Path(spec["test_images"])

    predictions_by_class: dict[int, list[Detection]] = {idx: [] for idx in range(len(class_names))}
    ground_truths: dict[int, dict[str, list[tuple[float, float, float, float]]]] = {
        idx: {} for idx in range(len(class_names))
    }

    for image_path in list_images(split_images):
        with Image.open(image_path) as image:
            width, height = image.size
        label_boxes = parse_label_file(label_path_for_image(image_path))
        image_id = image_path.stem
        for box in label_boxes:
            ground_truths[box.class_id].setdefault(image_id, []).append(box.to_xyxy(width, height))

        detections: list[Detection]
        if mode == "full":
            detections = extract_full_detections_for_image(
                model=full_model,
                image_path=image_path,
                imgsz=full_imgsz,
                conf=full_conf,
                iou=full_iou,
                max_det=max_det,
                device=device,
            )
            detections = filter_by_confidence(detections, final_conf)
        elif mode == "slice":
            detections = extract_slice_detections_for_image(
                model=slice_model,
                image_path=image_path,
                slice_size=slice_size,
                overlap=overlap,
                imgsz=slice_imgsz,
                conf=slice_conf,
                tile_iou=tile_iou,
                merge_iou=merge_iou,
                max_det=max_det,
                device=device,
            )
            detections = filter_by_confidence(detections, final_conf)
        else:
            full_dets = extract_full_detections_for_image(
                model=full_model,
                image_path=image_path,
                imgsz=full_imgsz,
                conf=full_conf,
                iou=full_iou,
                max_det=max_det,
                device=device,
            )
            slice_dets = extract_slice_detections_for_image(
                model=slice_model,
                image_path=image_path,
                slice_size=slice_size,
                overlap=overlap,
                imgsz=slice_imgsz,
                conf=slice_conf,
                tile_iou=tile_iou,
                merge_iou=merge_iou,
                max_det=max_det,
                device=device,
            )
            detections = classwise_nms(full_dets + slice_dets, final_merge_iou)
            detections = filter_by_confidence(detections, final_conf)

        for detection in detections:
            predictions_by_class[detection.class_id].append(detection)

    class_metrics: dict[str, dict[str, float | int]] = {}
    ap50_values: list[float] = []
    ap5095_values: list[float] = []
    for class_id, class_name in enumerate(class_names):
        ap_thresholds: list[float] = []
        for iou_thr in IOU_THRESHOLDS:
            ap, _, _, _ = evaluate_class(predictions_by_class.get(class_id, []), ground_truths[class_id], iou_thr)
            ap_thresholds.append(ap)
        ap50 = ap_thresholds[0]
        ap5095 = float(np.mean(ap_thresholds)) if ap_thresholds else 0.0
        ap50_values.append(ap50)
        ap5095_values.append(ap5095)
        class_metrics[class_name] = {
            "ap50": ap50,
            "ap50_95": ap5095,
            "predictions": len(predictions_by_class.get(class_id, [])),
            "ground_truths": sum(len(items) for items in ground_truths[class_id].values()),
        }

    precision, recall = micro_precision_recall(predictions_by_class, ground_truths)
    return {
        "precision": precision,
        "recall": recall,
        "map50": float(np.mean(ap50_values)) if ap50_values else 0.0,
        "map50_95": float(np.mean(ap5095_values)) if ap5095_values else 0.0,
        "class_metrics": class_metrics,
    }


def main() -> int:
    args = parse_args()
    dataset_yaml = args.dataset_yaml.resolve()
    spec = resolve_dataset_spec(dataset_yaml)
    class_names = list(spec["names"])
    resolved_device = resolve_device(args.device)

    full_weights = (args.full_weights or args.weights)
    slice_weights = (args.slice_weights or args.weights)
    if args.mode in ("full", "fused") and not full_weights:
        raise SystemExit("Full or fused evaluation requires --weights or --full-weights.")
    if args.mode in ("slice", "fused") and not slice_weights:
        raise SystemExit("Slice or fused evaluation requires --weights or --slice-weights.")

    full_model = create_ultralytics_model(str(full_weights.resolve())) if full_weights and args.mode in ("full", "fused") else None
    slice_model = create_ultralytics_model(str(slice_weights.resolve())) if slice_weights and args.mode in ("slice", "fused") else None

    full_imgsz = args.full_imgsz or args.imgsz
    slice_imgsz = args.slice_imgsz or args.imgsz
    full_conf = args.full_conf if args.full_conf is not None else args.conf
    slice_conf = args.slice_conf if args.slice_conf is not None else args.conf
    full_iou = args.full_iou if args.full_iou is not None else args.iou
    final_merge_iou = args.final_merge_iou if args.final_merge_iou is not None else args.merge_iou

    metrics = evaluate_mode(
        mode=args.mode,
        dataset_yaml=dataset_yaml,
        split=args.split,
        class_names=class_names,
        full_model=full_model,
        slice_model=slice_model,
        device=resolved_device,
        slice_size=args.slice_size,
        overlap=args.overlap,
        full_imgsz=full_imgsz,
        slice_imgsz=slice_imgsz,
        full_conf=full_conf,
        slice_conf=slice_conf,
        final_conf=args.final_conf,
        full_iou=full_iou,
        tile_iou=args.tile_iou,
        merge_iou=args.merge_iou,
        final_merge_iou=final_merge_iou,
        max_det=args.max_det,
    )

    summary = {
        "mode": args.mode,
        "title": args.title or f"{args.mode} inference evaluation",
        "dataset_role": args.dataset_role or f"{args.mode}_inference_eval",
        "full_weights_path": normalize_path(full_weights.resolve()) if full_weights else "",
        "slice_weights_path": normalize_path(slice_weights.resolve()) if slice_weights else "",
        "dataset_yaml": normalize_path(dataset_yaml),
        "split": args.split,
        "slice_size": args.slice_size,
        "overlap": args.overlap,
        "imgsz": args.imgsz,
        "full_imgsz": full_imgsz,
        "slice_imgsz": slice_imgsz,
        "conf": args.conf,
        "full_conf": full_conf,
        "slice_conf": slice_conf,
        "final_conf": args.final_conf,
        "iou": args.iou,
        "full_iou": full_iou,
        "tile_iou": args.tile_iou,
        "merge_iou": args.merge_iou,
        "final_merge_iou": final_merge_iou,
        "max_det": args.max_det,
        "resolved_device": resolved_device,
        **metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
