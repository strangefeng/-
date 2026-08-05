from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
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
ERROR_PRIORITY = {
    "missing_label": 0,
    "wrong_class": 1,
    "shifted_box": 2,
    "oversized_box": 3,
    "undersized_box": 4,
    "duplicate_label": 5,
    "ambiguous_defect": 6,
    "false_positive_background": 7,
    "unclear_image": 8,
}


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_xyxy(self, width: int, height: int) -> tuple[float, float, float, float]:
        x1 = (self.cx - self.w / 2.0) * width
        y1 = (self.cy - self.h / 2.0) * height
        x2 = (self.cx + self.w / 2.0) * width
        y2 = (self.cy + self.h / 2.0) * height
        return x1, y1, x2, y2

    @property
    def area(self) -> float:
        return self.w * self.h


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate round14 annotation audit candidates and review visuals.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "valid", "test"), default=["train", "valid"])
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--high-conf", type=float, default=0.25)
    parser.add_argument("--missing-label-conf", type=float, default=0.50)
    parser.add_argument("--match-iou", type=float, default=0.30)
    parser.add_argument("--shift-iou", type=float, default=0.35)
    parser.add_argument("--background-iou", type=float, default=0.10)
    parser.add_argument("--duplicate-iou", type=float, default=0.65)
    parser.add_argument("--tiny-area", type=float, default=0.00035)
    parser.add_argument("--tiny-dim", type=float, default=0.020)
    parser.add_argument("--large-area", type=float, default=0.080)
    parser.add_argument("--oversize-ratio", type=float, default=1.80)
    parser.add_argument("--undersize-ratio", type=float, default=0.55)
    parser.add_argument("--max-candidates", type=int, default=420)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--vis-dir", type=Path, required=True)
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
        "names": [str(name) for name in names],
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


def area_xyxy(box: tuple[float, float, float, float], width: int, height: int) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, width * height)


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


def clamp_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def class_name(class_names: list[str], class_id: int | None) -> str:
    if class_id is None:
        return ""
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return f"class_{class_id}"


def make_candidate(
    *,
    split: str,
    image_path: Path,
    class_gt: str,
    class_pred: str,
    confidence: float,
    iou: float,
    error_type: str,
    suggested_action: str,
    note: str,
    gt_box: tuple[float, float, float, float] | None = None,
    pred_box: tuple[float, float, float, float] | None = None,
) -> dict[str, object]:
    return {
        "image_path": normalize_path(image_path),
        "split": split,
        "class_gt": class_gt,
        "class_pred": class_pred,
        "confidence": confidence,
        "iou": iou,
        "error_type": error_type,
        "suggested_action": suggested_action,
        "note": note,
        "gt_box": list(gt_box) if gt_box else [],
        "pred_box": list(pred_box) if pred_box else [],
    }


def sort_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    def key(item: dict[str, object]) -> tuple[int, float, float]:
        error_type = str(item.get("error_type", ""))
        confidence = float(item.get("confidence") or 0.0)
        iou = float(item.get("iou") or 0.0)
        return (
            ERROR_PRIORITY.get(error_type, 99),
            -confidence,
            iou,
        )

    return sorted(candidates, key=key)


def dedupe_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str, int, int]] = set()
    for item in sort_candidates(candidates):
        pred_box = item.get("pred_box") or []
        gt_box = item.get("gt_box") or []
        key = (
            str(item.get("image_path", "")),
            str(item.get("error_type", "")),
            str(item.get("class_gt", "")),
            str(item.get("class_pred", "")),
            int(round(float(pred_box[0]) if len(pred_box) == 4 else float(gt_box[0]) if len(gt_box) == 4 else -1)),
            int(round(float(pred_box[1]) if len(pred_box) == 4 else float(gt_box[1]) if len(gt_box) == 4 else -1)),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def render_candidate_visual(
    *,
    image_path: Path,
    candidate: dict[str, object],
    class_names: list[str],
    save_path: Path,
    all_gt: list[tuple[int, tuple[float, float, float, float]]],
    all_pred: list[tuple[int, float, tuple[float, float, float, float]]],
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font(18)

    gt_color = (52, 152, 219)
    pred_color = (231, 76, 60)
    aux_color = (39, 174, 96)

    for class_id, box in all_gt:
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline=gt_color, width=3)
        label = f"GT {class_name(class_names, class_id)}"
        draw.text((x1 + 2, max(0, y1 - 18)), label, fill=gt_color, font=font)

    for class_id, confidence, box in all_pred[:12]:
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline=aux_color, width=2)
        label = f"P {class_name(class_names, class_id)} {confidence:.2f}"
        draw.text((x1 + 2, min(image.height - 18, y2 + 2)), label, fill=aux_color, font=font)

    gt_box = candidate.get("gt_box") or []
    pred_box = candidate.get("pred_box") or []
    if len(gt_box) == 4:
        draw.rectangle(tuple(gt_box), outline=gt_color, width=5)
    if len(pred_box) == 4:
        draw.rectangle(tuple(pred_box), outline=pred_color, width=5)

    banner_lines = [
        f"{candidate.get('error_type', '')} | split={candidate.get('split', '')}",
        (
            f"GT={candidate.get('class_gt', '') or '-'} "
            f"PRED={candidate.get('class_pred', '') or '-'} "
            f"conf={float(candidate.get('confidence') or 0.0):.2f} "
            f"IoU={float(candidate.get('iou') or 0.0):.2f}"
        ),
        str(candidate.get("suggested_action", "")),
    ]
    banner_h = 64
    draw.rectangle((0, 0, image.width, banner_h), fill=(18, 18, 18))
    for idx, text in enumerate(banner_lines):
        draw.text((8, 6 + idx * 18), text, fill=(255, 255, 255), font=font)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path, format="JPEG", quality=95)


def main() -> int:
    args = parse_args()
    dataset_yaml = args.dataset_yaml.resolve()
    weights = args.weights.resolve()
    spec = resolve_dataset_spec(dataset_yaml)
    class_names = list(spec["names"])
    split_dir_map = {
        "train": Path(spec["train_images"]),
        "valid": Path(spec["val_images"]),
        "test": Path(spec["test_images"]),
    }

    model = create_ultralytics_model(str(weights))
    device = resolve_device(args.device)

    candidates: list[dict[str, object]] = []

    for split in args.splits:
        for image_path in list_images(split_dir_map[split]):
            with Image.open(image_path) as image:
                width, height = image.size

            gt_items = parse_label_file(label_path_for_image(image_path))
            gt_xyxy = [clamp_box(item.to_xyxy(width, height), width, height) for item in gt_items]
            gt_pairs = list(zip(gt_items, gt_xyxy))

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
            pred_classes: list[int] = []
            pred_scores: list[float] = []
            pred_xyxy: list[tuple[float, float, float, float]] = []
            if boxes is not None and len(boxes):
                pred_classes = [int(value) for value in boxes.cls.cpu().numpy().astype(int).tolist()]
                pred_scores = [float(value) for value in boxes.conf.cpu().numpy().tolist()]
                pred_xyxy = [
                    clamp_box(tuple(float(v) for v in row.tolist()), width, height)
                    for row in boxes.xyxy.cpu().numpy()
                ]

            matched_pred_indices: set[int] = set()

            for idx_a, (gt_a, box_a) in enumerate(gt_pairs):
                gt_name = class_name(class_names, gt_a.class_id)
                gt_area = gt_a.area

                if gt_area < args.tiny_area or gt_a.w < args.tiny_dim or gt_a.h < args.tiny_dim:
                    candidates.append(
                        make_candidate(
                            split=split,
                            image_path=image_path,
                            class_gt=gt_name,
                            class_pred="",
                            confidence=0.0,
                            iou=0.0,
                            error_type="undersized_box",
                            suggested_action="Check whether the GT box is too small or too tight for this defect.",
                            note=f"GT normalized area={gt_area:.6f}, w={gt_a.w:.4f}, h={gt_a.h:.4f}.",
                            gt_box=box_a,
                        )
                    )
                if gt_area > args.large_area:
                    candidates.append(
                        make_candidate(
                            split=split,
                            image_path=image_path,
                            class_gt=gt_name,
                            class_pred="",
                            confidence=0.0,
                            iou=0.0,
                            error_type="oversized_box",
                            suggested_action="Check whether the GT box covers too much surrounding background.",
                            note=f"GT normalized area={gt_area:.6f}.",
                            gt_box=box_a,
                        )
                    )

                for idx_b, (gt_b, box_b) in enumerate(gt_pairs[idx_a + 1 :], start=idx_a + 1):
                    if gt_a.class_id != gt_b.class_id:
                        continue
                    overlap = box_iou(box_a, box_b)
                    if overlap >= args.duplicate_iou:
                        candidates.append(
                            make_candidate(
                                split=split,
                                image_path=image_path,
                                class_gt=gt_name,
                                class_pred="",
                                confidence=0.0,
                                ioußn­¢G§²ÚîÆ­yÐpred_idx

                if best_any_idx >= 0 and best_any_iou >= args.match_iou:
                    matched_pred_indices.add(best_any_idx)

                if best_any_idx >= 0:
                    pred_name = class_name(class_names, pred_classes[best_any_idx])
                    pred_score = pred_scores[best_any_idx]
                    pred_box = pred_xyxy[best_any_idx]
                    pred_area = area_xyxy(pred_box, width, height)
                    area_ratio = pred_area / max(1e-9, gt_area)

                    if pred_classes[best_any_idx] != gt_a.class_id and best_any_iou >= args.match_iou and pred_score >= args.high_conf:
                        candidates.append(
                            make_candidate(
                                split=split,
                                image_path=image_path,
                                class_gt=gt_name,
                                class_pred=pred_name,
                                confidence=pred_score,
                                iou=best_any_iou,
                                error_type="wrong_class",
                                suggested_action="Review whether the GT class or the visual appearance is ambiguous.",
                                note="High-overlap prediction has a different class from GT.",
                                gt_box=box_a,
                                pred_box=pred_box,
                            )
                        )
                    elif pred_classes[best_any_idx] == gt_a.class_id and pred_score >= args.high_conf:
                        if best_any_iou < args.shift_iou:
                            candidates.append(
                                make_candidate(
                                    split=split,
                                    image_path=image_path,
                                    class_gt=gt_name,
                                    class_pred=pred_name,
                                    confidence=pred_score,
                                    iou=best_any_iou,
                                    error_type="shifted_box",
                                    suggested_action="Review whether the GT boundary is shifted or too loose/tight.",
                                    note="Same-class prediction is confident but misaligned with GT.",
                                    gt_box=box_a,
                                    pred_box=pred_box,
                                )
                            )
                        elif area_ratio >= args.oversize_ratio:
                            candidates.append(
                                make_candidate(
                                    split=split,
                                    image_path=image_path,
                                    class_gt=gt_name,
                                    class_pred=pred_name,
                                    confidence=pred_score,
                                    iou=best_any_iou,
                                    error_type="oversized_box",
                                    suggested_action="Review whether GT is too small or prediction indicates a larger valid defect extent.",
                                    note=f"Prediction/GT area ratio={area_ratio:.2f}.",
                                    gt_box=box_a,
                                    pred_box=pred_box,
                                )
                            )
                        elif area_ratio <= args.undersize_ratio:
                            candidates.append(
                                make_candidate(
                                    split=split,
                                    image_path=image_path,
                                    class_gt=gt_name,
                                    class_pred=pred_name,
                                    confidence=pred_score,
                                    iou=best_any_iou,
                                    error_type="undersized_box",
                                    suggested_action="Review whether GT is too large or prediction better fits the defect core.",
                                    note=f"Prediction/GT area ratio={area_ratio:.2f}.",
                                    gt_box=box_a,
                                    pred_box=pred_box,
                                )
                            )
                    elif pred_score >= args.high_conf and best_any_iou < args.match_iou:
                        candidates.append(
                            make_candidate(
                                split=split,
                                image_path=image_path,
                                class_gt=gt_name,
                                class_pred=pred_name,
                                confidence=pred_score,
                                iou=best_any_iou,
                                error_type="ambiguous_defect",
                                suggested_action="Inspect whether this defect instance is visually ambiguous or difficult to localize.",
                                note="Closest prediction is confident but still has weak overlap with GT.",
                                gt_box=box_a,
                                pred_box=pred_box,
                            )
                        )
                else:
                    candidates.append(
                        make_candidate(
                            split=split,
                            image_path=image_path,
                            class_gt=gt_name,
                            class_pred="",
                            confidence=0.0,
                            iou=0.0,
                            error_type="unclear_image",
                            suggested_action="Inspect image quality and confirm whether the annotated defect is too weak or blurred.",
                            note="No prediction found near this GT instance.",
                            gt_box=box_a,
                        )
                    )

            for pred_idx, pred_box in enumerate(pred_xyxy):
                pred_class_id = pred_classes[pred_idx]
                pred_name = class_name(class_names, pred_class_id)
                pred_score = pred_scores[pred_idx]
                max_iou = max((box_iou(pred_box, gt_box) for _, gt_box in gt_pairs), default=0.0)
                if max_iou >= args.background_iou:
                    continue
                if pred_score >= args.missing_label_conf:
                    error_type = "missing_label"
                    action = "Review whether this confident prediction indicates a missed annotation."
                    note = "High-confidence prediction far from every GT box."
                elif pred_score >= args.high_conf:
                    error_type = "false_positive_background"
                    action = "Review background texture; this may be a systematic false-positive pattern."
                    note = "Confident prediction on a background region without GT overlap."
                else:
                    continue
                candidates.append(
                    make_candidate(
                        split=split,
                        image_path=image_path,
                        class_gt="",
                        class_pred=pred_name,
                        confidence=pred_score,
                        iou=max_iou,
                        error_type=error_type,
                        suggested_action=action,
                        note=note,
                        pred_box=pred_box,
                    )
                )

    candidates = dedupe_candidates(candidates)[: max(1, args.max_candidates)]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.vis_dir.mkdir(parents=True, exist_ok=True)

    csv_columns = [
        "candidate_id",
        "image_path",
        "split",
        "class_gt",
        "class_pred",
        "confidence",
        "iou",
        "error_type",
        "suggested_action",
        "note",
        "vis_path",
    ]

    split_dir_map = {
        "train": Path(spec["train_images"]),
        "valid": Path(spec["val_images"]),
        "test": Path(spec["test_images"]),
    }

    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for index, candidate in enumerate(candidates, start=1):
            image_path = Path(str(candidate["image_path"]))
            label_path = label_path_for_image(image_path)
            with Image.open(image_path) as image:
                width, height = image.size
            gt_items = parse_label_file(label_path)
            gt_pairs = [(item.class_id, clamp_box(item.to_xyxy(width, height), width, height)) for item in gt_items]
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
            pred_triplets: list[tuple[int, float, tuple[float, float, float, float]]] = []
            if boxes is not None and len(boxes):
                pred_triplets = [
                    (
                        int(class_id),
                        float(score),
                        clamp_box(tuple(float(v) for v in row.tolist()), width, height),
                    )
                    for class_id, score, row in zip(
                        boxes.cls.cpu().numpy().astype(int),
                        boxes.conf.cpu().numpy(),
                        boxes.xyxy.cpu().numpy(),
                    )
                ]

            vis_path = args.vis_dir / f"{index:04d}_{candidate['error_type']}_{image_path.stem}.jpg"
            render_candidate_visual(
                image_path=image_path,
                candidate=candidate,
                class_names=class_names,
                save_path=vis_path,
                all_gt=gt_pairs,
                all_pred=pred_triplets,
            )
            candidate["vis_path"] = normalize_path(vis_path)

            writer.writerow(
                {
                    "candidate_id": f"QA1_{index:04d}",
                    "image_path": candidate["image_path"],
                    "split": candidate["split"],
                    "class_gt": candidate["class_gt"],
                    "class_pred": candidate["class_pred"],
                    "confidence": f"{float(candidate['confidence']):.6f}",
                    "iou": f"{float(candidate['iou']):.6f}",
                    "error_type": candidate["error_type"],
                    "suggested_action": candidate["suggested_action"],
                    "note": candidate["note"],
                    "vis_path": candidate["vis_path"],
                }
            )

    error_counter = Counter(str(item.get("error_type", "")) for item in candidates)
    split_counter = Counter(str(item.get("split", "")) for item in candidates)
    gt_class_counter = Counter(str(item.get("class_gt", "")) for item in candidates if item.get("class_gt"))
    pred_class_counter = Counter(str(item.get("class_pred", "")) for item in candidates if item.get("class_pred"))
    train_fix_recommended = sum(
        1
        for item in candidates
        if item.get("split") == "train"
        and str(item.get("error_type")) in {"wrong_class", "missing_label", "oversized_box", "undersized_box", "shifted_box", "duplicate_label"}
    )
    valid_record_only = sum(1 for item in candidates if item.get("split") == "valid")
    clean_train_recommended = train_fix_recommended >= 20

    summary_payload = {
        "weights_path": normalize_path(weights),
        "dataset_yaml": normalize_path(dataset_yaml),
        "splits": list(args.splits),
        "candidate_count": len(candidates),
        "error_type_counts": dict(error_counter),
        "split_counts": dict(split_counter),
        "class_gt_counts": dict(gt_class_counter),
        "class_pred_counts": dict(pred_class_counter),
        "missing_rivet_issue_count": sum(
            1
            for item in candidates
            if item.get("class_gt") == "missing-rivet" or item.get("class_pred") == "missing-rivet"
        ),
        "rust_issue_count": sum(
            1 for item in candidates if item.get("class_gt") == "rust" or item.get("class_pred") == "rust"
        ),
        "scratch_issue_count": sum(
            1 for item in candidates if item.get("class_gt") == "scratch" or item.get("class_pred") == "scratch"
        ),
        "train_fix_recommended_count": train_fix_recommended,
        "valid_record_only_count": valid_record_only,
        "clean_train_recommended": clean_train_recommended,
        "output_csv": normalize_path(args.output_csv),
        "vis_dir": normalize_path(args.vis_dir),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# R14_QA1_annotation_audit_summary")
    lines.append("")
    lines.append(f"- Review candidate count: `{len(candidates)}`")
    lines.append(f"- Train fix recommended count: `{train_fix_recommended}`")
    lines.append(f"- Valid record-only count: `{valid_record_only}`")
    lines.append(f"- Suggested clean_train branch: `{'yes' if clean_train_recommended else 'no'}`")
    lines.append("")
    lines.append("## Error Types")
    lines.append("")
    for error_type, count in sorted(error_counter.items(), key=lambda item: (ERROR_PRIORITY.get(item[0], 99), -item[1])):
        lines.append(f"- `{error_type}`: `{count}`")
    lines.append("")
    lines.append("## Class Focus")
    lines.append("")
    lines.append(f"- `missing-rivet` related issues: `{summary_payload['missing_rivet_issue_count']}`")
    lines.append(f"- `rust` related issues: `{summary_payload['rust_issue_count']}`")
    lines.append(f"- `scratch` related issues: `{summary_payload['scratch_issue_count']}`")
    lines.append("")
    lines.append("## Split Distribution")
    lines.append("")
    for split, count in sorted(split_counter.items()):
        lines.append(f"- `{split}`: `{count}`")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Candidate CSV: `{normalize_path(args.output_csv)}`")
    lines.append(f"- Visualization dir: `{normalize_path(args.vis_dir)}`")
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"candidate_csv={args.output_csv}")
    print(f"summary_json={args.summary_json}")
    print(f"summary_md={args.summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
