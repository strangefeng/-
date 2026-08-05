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
    parser = argparse.ArgumentParser(description="Run the round13 E2 full/slice/fused inference comparison suite.")
    parser.add_argument("--n2-weights", type=Path, required=True)
    parser.add_argument("--s2-weights", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--slice-size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--full-conf", type=float, default=0.25)
    parser.add_argument("--slice-conf", type=float, default=0.20)
    parser.add_argument("--final-conf", type=float, default=0.25)
    parser.add_argument("--full-iou", type=float, default=0.70)
    parser.add_argument("--tile-iou", type=float, default=0.70)
    parser.add_argument("--merge-iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_yaml = args.dataset_yaml.resolve()
    n2_weights = args.n2_weights.resolve()
    s2_weights = args.s2_weights.resolve()
    spec = resolve_dataset_spec(dataset_yaml)
    class_names = list(spec["names"])
    device = resolve_device(args.device)

    models = {
        "N2": create_ultralytics_model(str(n2_weights)),
        "S2": create_ultralytics_model(str(s2_weights)),
    }
    evaluations = [
        ("E2-1", "N2", "full", "N2 full-image inference"),
        ("E2-2", "N2", "slice", "N2 sliced inference"),
        ("E2-3", "N2", "fused", "N2 full+sliced fused inference"),
        ("E2-4", "S2", "full", "S2 full-image inference"),
        ("E2-5", "S2", "slice", "S2 sliced inference"),
        ("E2-6", "S2", "fused", "S2 full+sliced fused inference"),
    ]

    results: list[dict[str, object]] = []
    for eval_id, model_key, mode, title in evaluations:
        model = models[model_key]
        metrics = evaluate_mode(
            mode=mode,
            dataset_yaml=dataset_yaml,
            split=args.split,
            class_names=class_names,
            full_model=model,
            slice_model=model,
            device=device,
            slice_size=args.slice_size,
            overlap=args.overlap,
            full_imgsz=args.imgsz,
            slice_imgsz=args.imgsz,
            full_conf=args.full_conf,
            slice_conf=args.slice_conf,
            final_conf=args.final_conf,
            full_iou=args.full_iou,
            tile_iou=args.tile_iou,
            merge_iou=args.merge_iou,
            final_merge_iou=args.merge_iou,
            max_det=args.max_det,
        )
        results.append(
            {
                "experiment_id": eval_id,
                "model_id": model_key,
                "mode": mode,
                "title": title,
                "weights_path": normalize_path(n2_weights if model_key == "N2" else s2_weights),
                **metrics,
            }
        )

    best = sorted(results, key=lambda item: (float(item["map50_95"]), float(item["precision"])), reverse=True)[0]
    payload = {
        "title": "R13-E2 full/slice/fused inference suite",
        "dataset_role": "round13_full_slice_fused_eval",
        "dataset_yaml": normalize_path(dataset_yaml),
        "split": args.split,
        "slice_size": args.slice_size,
        "overlap": args.overlap,
        "imgsz": args.imgsz,
        "full_conf": args.full_conf,
        "slice_conf": args.slice_conf,
        "final_conf": args.final_conf,
        "merge_iou": args.merge_iou,
        "resolved_device": device,
        "n2_weights_path": normalize_path(n2_weights),
        "s2_weights_path": normalize_path(s2_weights),
        "results": results,
        "best_run": best,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
