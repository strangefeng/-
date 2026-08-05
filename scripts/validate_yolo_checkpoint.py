from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repo_runtime import configure_repo_runtime

configure_repo_runtime(ROOT)

from pipeline_utils import create_ultralytics_model, normalize_path, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official Ultralytics val on a checkpoint and export metrics.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--augment", action="store_true", help="Enable Ultralytics test-time augmentation.")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--title", default="")
    return parser.parse_args()


def float_or_zero(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> int:
    args = parse_args()
    weights = args.weights.resolve()
    dataset_yaml = args.dataset_yaml.resolve()
    device = resolve_device(args.device)
    model = create_ultralytics_model(str(weights))
    val_kwargs = {
        "data": str(dataset_yaml),
        "split": args.split,
        "imgsz": args.imgsz,
        "device": device,
        "workers": args.workers,
        "iou": args.iou,
        "max_det": args.max_det,
        "augment": args.augment,
        "plots": args.plots,
        "save_json": False,
        "verbose": False,
    }
    if args.conf is not None:
        val_kwargs["conf"] = args.conf
    result = model.val(**val_kwargs)
    names = getattr(result, "names", None) or getattr(model.model, "names", None) or {}
    if isinstance(names, dict):
        class_names = [names[i] for i in sorted(names)]
    else:
        class_names = list(names)

    class_metrics: dict[str, dict[str, float]] = {name: {"ap50": 0.0, "ap50_95": 0.0} for name in class_names}
    ap_class_index = getattr(result, "ap_class_index", [])
    if ap_class_index is None:
        ap_class_index = []
    else:
        ap_class_index = list(ap_class_index)
    for idx, class_idx in enumerate(ap_class_index):
        if class_idx >= len(class_names):
            continue
        class_result = list(result.class_result(idx))
        if len(class_result) >= 4:
            class_metrics[class_names[class_idx]] = {
                "ap50": float_or_zero(class_result[2]),
                "ap50_95": float_or_zero(class_result[3]),
            }

    results_dict = getattr(result, "results_dict", None) or {}
    payload = {
        "title": args.title or f"official_val_{weights.stem}",
        "weights_path": normalize_path(weights),
        "dataset_yaml": normalize_path(dataset_yaml),
        "split": args.split,
        "imgsz": args.imgsz,
        "device": device,
        "workers": args.workers,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "augment": args.augment,
        "precision": float_or_zero(results_dict.get("metrics/precision(B)")),
        "recall": float_or_zero(results_dict.get("metrics/recall(B)")),
        "map50": float_or_zero(results_dict.get("metrics/mAP50(B)")),
        "map50_95": float_or_zero(results_dict.get("metrics/mAP50-95(B)")),
        "class_metrics": class_metrics,
        "save_dir": normalize_path(getattr(result, "save_dir", "")),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
