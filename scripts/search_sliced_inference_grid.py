from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_inference_modes import evaluate_mode, resolve_dataset_spec
from pipeline_utils import create_ultralytics_model, normalize_path, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search SAHI sliced-inference thresholds for round13 E1.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--slice-size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--tile-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf-values", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    parser.add_argument("--base-merge-iou", type=float, default=0.50)
    parser.add_argument("--priority-conf", type=float, default=0.20)
    parser.add_argument("--priority-merge-ious", nargs="+", type=float, default=[0.45, 0.50, 0.55])
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def fv(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def main() -> int:
    args = parse_args()
    dataset_yaml = args.dataset_yaml.resolve()
    weights = args.weights.resolve()
    spec = resolve_dataset_spec(dataset_yaml)
    class_names = list(spec["names"])
    device = resolve_device(args.device)
    model = create_ultralytics_model(str(weights))

    runs: list[dict[str, object]] = []
    for conf in args.conf_values:
        merge_ious = args.priority_merge_ious if abs(conf - args.priority_conf) < 1e-9 else [args.base_merge_iou]
        for merge_iou in merge_ious:
            metrics = evaluate_mode(
                mode="slice",
                dataset_yaml=dataset_yaml,
                split=args.split,
                class_names=class_names,
                full_model=None,
                slice_model=model,
                device=device,
                slice_size=args.slice_size,
                overlap=args.overlap,
                full_imgsz=args.imgsz,
                slice_imgsz=args.imgsz,
                full_conf=conf,
                slice_conf=conf,
                final_conf=0.0,
                full_iou=args.tile_iou,
                tile_iou=args.tile_iou,
                merge_iou=merge_iou,
                final_merge_iou=merge_iou,
                max_det=args.max_det,
            )
            runs.append(
                {
                    "mode": "slice",
                    "conf": conf,
                    "merge_iou": merge_iou,
                    "tile_iou": args.tile_iou,
                    **metrics,
                }
            )

    sorted_runs = sorted(runs, key=lambda item: (fv(item.get("map50_95")), fv(item.get("precision"))), reverse=True)
    best = sorted_runs[0] if sorted_runs else {}
    payload = {
        "title": "R13-E1 SAHI threshold search",
        "dataset_role": "round13_sahi_grid_search",
        "weights_path": normalize_path(weights),
        "dataset_yaml": normalize_path(dataset_yaml),
        "split": args.split,
        "slice_size": args.slice_size,
        "overlap": args.overlap,
        "imgsz": args.imgsz,
        "tile_iou": args.tile_iou,
        "base_merge_iou": args.base_merge_iou,
        "priority_conf": args.priority_conf,
        "priority_merge_ious": args.priority_merge_ious,
        "resolved_device": device,
        "runs": sorted_runs,
        "best_run": best,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
