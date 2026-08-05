from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_runtime import configure_repo_runtime

configure_repo_runtime(REPO_ROOT)

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from custom_yolo import apply_local_ultralytics_patches


CASE_COLORS = {
    'gt': (0, 255, 0),
    'scratch': (255, 80, 80),
    'other': (255, 180, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Audit the six scratch hard cases across three experiments and render triptychs.')
    parser.add_argument('--suite-csv', type=Path, default=Path('results/aircraft_skin_defects_new_mainline.csv'))
    parser.add_argument('--case-summary', type=Path, default=Path('analysis/scratch_m1_audit/summary.json'))
    parser.add_argument('--experiment-ids', nargs='+', default=['Exp_A', 'Exp_B', 'Exp_C'])
    parser.add_argument('--dataset-root', type=Path, default=Path('../aircraft-skin-defects-new-dataset.v1i.yolov5pytorch'))
    parser.add_argument('--output-dir', type=Path, default=Path('analysis/stage2_scratch_audit'))
    parser.add_argument('--device', default='0')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.001)
    return parser.parse_args()


def latest_success_row(frame: pd.DataFrame, experiment_id: str) -> pd.Series:
    subset = frame[(frame['run_status'] == 'success') & (frame['experiment_id'] == experiment_id)].copy()
    if subset.empty:
        raise KeyError(f'No successful rows found for experiment_id={experiment_id}')
    subset['epochs_num'] = pd.to_numeric(subset['epochs'], errors='coerce').fillna(-1)
    subset = subset.reset_index(names='row_order')
    subset = subset.sort_values(['epochs_num', 'row_order'], ascending=[False, False])
    return subset.iloc[0]


def xyxy_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(1.0, ax2 - ax1) * max(1.0, ay2 - ay1)
    area_b = max(1.0, bx2 - bx1) * max(1.0, by2 - by1)
    return inter_area / (area_a + area_b - inter_area)


def to_list(values: Any) -> list[float]:
    return [float(x) for x in values]


def status_for_match(iou: float, conf: float | None) -> str:
    if conf is None:
        return 'missing'
    if iou >= 0.5 and conf >= 0.25:
        return 'matched'
    if iou >= 0.5:
        return 'low_confidence'
    if iou >= 0.3:
        return 'box_drift'
    return 'missed'


def draw_box(draw: ImageDraw.ImageDraw, box: list[float], color: tuple[int, int, int], text: str, width: int = 4) -> None:
    draw.rectangle(box, outline=color, width=width)
    if text:
        x1, y1, _, _ = box
        font = ImageFont.load_default()
        bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1), text, fill=(0, 0, 0), font=font)


def build_panel(image_path: Path, gt_boxes: list[dict[str, Any]], match_rows: list[dict[str, Any]], title: str) -> Image.Image:
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    for idx, gt in enumerate(gt_boxes, start=1):
        draw_box(draw, gt['gt_box'], CASE_COLORS['gt'], f'GT{idx}', width=4)
    for row in match_rows:
        box = row.get('best_scratch_box')
        if box:
            label = f"{row['case_id']} IoU={row['best_scratch_iou']:.2f} conf={row['best_scratch_conf']:.2f}"
            draw_box(draw, box, CASE_COLORS['scratch'], label, width=3)
        other_box = row.get('best_other_box')
        if other_box:
            other_label = f"{row['best_other_cls']} {row['best_other_conf']:.2f}"
            draw_box(draw, other_box, CASE_COLORS['other'], other_label, width=2)
    canvas = Image.new('RGB', (image.width, image.height + 32), color=(255, 255, 255))
    canvas.paste(image, (0, 32))
    title_draw = ImageDraw.Draw(canvas)
    title_draw.text((12, 8), title, fill=(0, 0, 0), font=ImageFont.load_default())
    return canvas


def stack_triptych(panels: list[Image.Image]) -> Image.Image:
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    triptych = Image.new('RGB', (width, height), color=(255, 255, 255))
    offset = 0
    for panel in panels:
        triptych.paste(panel, (offset, 0))
        offset += panel.width
    return triptych


def main() -> int:
    args = parse_args()
    apply_local_ultralytics_patches()

    suite_csv = args.suite_csv.resolve()
    case_summary = args.case_summary.resolve()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(suite_csv)
    cases_payload = json.loads(case_summary.read_text(encoding='utf-8'))
    case_records = cases_payload['records']

    experiment_rows = {exp_id: latest_success_row(frame, exp_id) for exp_id in args.experiment_ids}
    models = {
        exp_id: YOLO(str(experiment_rows[exp_id]['weights_path']))
        for exp_id in args.experiment_ids
    }

    unique_images: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for idx, record in enumerate(case_records, start=1):
        normalized = {
            'case_id': f'C{idx}',
            'split': record['split'],
            'image': record['image'],
            'gt_box': to_list(record['gt_box']),
        }
        unique_images[(record['split'], record['image'])].append(normalized)

    prediction_cache: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    audit_rows: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, float]] = {exp_id: {'mean_iou': 0.0, 'mean_conf': 0.0, 'matched': 0.0, 'total': 0.0} for exp_id in args.experiment_ids}

    for (split, image_name), gt_boxes in unique_images.items():
        image_path = dataset_root / split / 'images' / image_name
        panels: list[Image.Image] = []
        for exp_id in args.experiment_ids:
            cache_key = (exp_id, split, image_name)
            if cache_key not in prediction_cache:
                result = models[exp_id].predict(source=str(image_path), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
                names = result.names
                boxes = []
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    boxes.append({
                        'cls_id': cls_id,
                        'cls_name': names[cls_id],
                        'conf': float(box.conf.item()),
                        'xyxy': to_list(box.xyxy[0].tolist()),
                    })
                prediction_cache[cache_key] = boxes

            predictions = prediction_cache[cache_key]
            panel_rows: list[dict[str, Any]] = []
            for gt in gt_boxes:
                scratch_candidates = [pred for pred in predictions if pred['cls_name'] == 'scratch']
                best_scratch = None
                best_scratch_iou = 0.0
                for pred in scratch_candidates:
                    iou = xyxy_iou(gt['gt_box'], pred['xyxy'])
                    if best_scratch is None or iou > best_scratch_iou:
                        best_scratch = pred
                        best_scratch_iou = iou

                other_candidates = [pred for pred in predictions if pred['cls_name'] != 'scratch']
                best_other = None
                best_other_iou = 0.0
                for pred in other_candidates:
                    iou = xyxy_iou(gt['gt_box'], pred['xyxy'])
                    if best_other is None or iou > best_other_iou:
                        best_other = pred
                        best_other_iou = iou

                row = {
                    'experiment_id': exp_id,
                    'title': str(experiment_rows[exp_id]['experiment_title']),
                    'split': split,
                    'image': image_name,
                    'case_id': gt['case_id'],
                    'gt_box': gt['gt_box'],
                    'best_scratch_iou': float(best_scratch_iou),
                    'best_scratch_conf': None if best_scratch is None else float(best_scratch['conf']),
                    'best_scratch_box': None if best_scratch is None else best_scratch['xyxy'],
                    'best_other_iou': float(best_other_iou) if best_other is not None else -1.0,
                    'best_other_cls': None if best_other is None else best_other['cls_name'],
                    'best_other_conf': None if best_other is None else float(best_other['conf']),
                    'best_other_box': None if best_other is None else best_other['xyxy'],
                }
                row['status'] = status_for_match(row['best_scratch_iou'], row['best_scratch_conf'])
                audit_rows.append(row)
                panel_rows.append(row)
                aggregate[exp_id]['total'] += 1
                aggregate[exp_id]['mean_iou'] += row['best_scratch_iou']
                aggregate[exp_id]['mean_conf'] += 0.0 if row['best_scratch_conf'] is None else row['best_scratch_conf']
                aggregate[exp_id]['matched'] += 1 if row['status'] == 'matched' else 0

            panel = build_panel(image_path, gt_boxes, panel_rows, f"{exp_id} | {experiment_rows[exp_id]['experiment_title']}")
            panels.append(panel)

        triptych = stack_triptych(panels)
        output_name = f"{split}_{Path(image_name).stem}_triptych.jpg"
        triptych.save(output_dir / output_name, quality=95)

    for exp_id, stats in aggregate.items():
        total = max(1.0, stats['total'])
        stats['mean_iou'] /= total
        stats['mean_conf'] /= total
        stats['match_rate'] = stats['matched'] / total

    summary = {
        'case_source': str(case_summary),
        'dataset_root': str(dataset_root),
        'case_count': len(case_records),
        'image_count': len(unique_images),
        'note': 'The hard-case set contains 6 images in total: 4 from valid and 2 from test. This corrects the common shorthand that calls all 6 images validation cases.',
        'experiments': {
            exp_id: {
                'title': str(experiment_rows[exp_id]['experiment_title']),
                'weights_path': str(experiment_rows[exp_id]['weights_path']),
                **aggregate[exp_id],
            }
            for exp_id in args.experiment_ids
        },
        'records': audit_rows,
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())




