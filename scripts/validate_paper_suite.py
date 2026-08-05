from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repo_runtime import configure_repo_runtime

configure_repo_runtime(ROOT)

from pipeline_utils import create_ultralytics_model, normalize_path, resolve_device


CSV_COLUMNS = [
    "source_csv",
    "experiment_id",
    "experiment_title",
    "model_name",
    "weights_path",
    "dataset_yaml",
    "benchmark_name",
    "split",
    "imgsz",
    "batch",
    "device",
    "precision",
    "recall",
    "map50",
    "map50_95",
    "class_metrics_json",
    "speed_preprocess_ms",
    "speed_inference_ms",
    "speed_postprocess_ms",
]


def float_or_blank(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def class_names_from_yaml(dataset_yaml: Path) -> list[str]:
    data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8-sig"))
    names = data.get("names", [])
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def extract_overall_metrics(result: Any) -> dict[str, str]:
    results_dict = getattr(result, "results_dict", None) or {}
    return {
        "precision": float_or_blank(results_dict.get("metrics/precision(B)")),
        "recall": float_or_blank(results_dict.get("metrics/recall(B)")),
        "map50": float_or_blank(results_dict.get("metrics/mAP50(B)")),
        "map50_95": float_or_blank(results_dict.get("metrics/mAP50-95(B)")),
    }


def extract_speed_metrics(result: Any) -> dict[str, str]:
    speed = getattr(result, "speed", None) or {}
    return {
        "speed_preprocess_ms": float_or_blank(speed.get("preprocess")),
        "speed_inference_ms": float_or_blank(speed.get("inference")),
        "speed_postprocess_ms": float_or_blank(speed.get("postprocess")),
    }


def extract_class_metrics(result: Any, class_names: list[str]) -> str:
    rows: dict[str, dict[str, str]] = {class_name: {"ap50": "", "ap50_95": ""} for class_name in class_names}
    ap_class_index = getattr(result, "ap_class_index", None)
    if ap_class_index is None:
        ap_class_index = []
    for idx, class_idx in enumerate(list(ap_class_index)):
        if class_idx >= len(class_names):
            continue
        class_result = list(result.class_result(idx))
        if len(class_result) >= 4:
            rows[class_names[class_idx]] = {
                "ap50": float_or_blank(class_result[2]),
                "ap50_95": float_or_blank(class_result[3]),
            }
    return json.dumps(rows, ensure_ascii=False)


def read_success_rows(csv_paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            print(f"missing csv: {csv_path}")
            continue
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("run_status") == "success" and row.get("weights_path"):
                    row["_source_csv"] = str(csv_path)
                    rows.append(row)
    return rows


def existing_keys(output_csv: Path) -> set[tuple[str, str]]:
    if not output_csv.exists():
        return set()
    with output_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        return {
            (row.get("experiment_id", ""), row.get("benchmark_name", ""))
            for row in csv.DictReader(handle)
        }


def append_row(output_csv: Path, row: dict[str, str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    should_header = not output_csv.exists()
    with output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if should_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate trained paper-suite weights on clean/corruption YAMLs.")
    parser.add_argument("--source-csv", nargs="+", type=Path, required=True)
    parser.add_argument("--yaml-dir", type=Path, required=True, help="Directory containing benchmark YAML files.")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "results" / "aircraft_ai_uav_paper_robustness.csv")
    parser.add_argument("--only", nargs="*", help="Optional experiment ids to validate.")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers for validation.")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = set(args.only or [])
    resolved_device = resolve_device(args.device)
    yaml_paths = sorted(args.yaml_dir.glob("*.yaml"))
    if not yaml_paths:
        raise SystemExit(f"No YAML files found in {args.yaml_dir}")

    completed = existing_keys(args.output_csv) if args.skip_existing else set()
    for source_row in read_success_rows([path.resolve() for path in args.source_csv]):
        experiment_id = source_row.get("run_id") or source_row.get("experiment_id", "")
        if selected and experiment_id not in selected:
            continue
        weights_path = Path(source_row["weights_path"])
        if not weights_path.exists():
            print(f"[skip] missing weights for {experiment_id}: {weights_path}")
            continue
        model = create_ultralytics_model(str(weights_path))
        for yaml_path in yaml_paths:
            benchmark_name = yaml_path.stem
            if (experiment_id, benchmark_name) in completed:
                print(f"[skip] {experiment_id} {benchmark_name}")
                continue
            print(f"[val] {experiment_id} on {benchmark_name}")
            result = model.val(
                data=str(yaml_path.resolve()),
                imgsz=args.imgsz,
                batch=args.batch,
                device=resolved_device,
                workers=args.workers,
                split=args.split,
                plots=False,
                verbose=False,
            )
            class_names = class_names_from_yaml(yaml_path)
            out_row = {
                "source_csv": source_row.get("_source_csv", ""),
                "experiment_id": experiment_id,
                "experiment_title": source_row.get("experiment_title", ""),
                "model_name": source_row.get("name", ""),
                "weights_path": normalize_path(weights_path),
                "dataset_yaml": normalize_path(yaml_path),
                "benchmark_name": benchmark_name,
                "split": args.split,
                "imgsz": str(args.imgsz),
                "batch": str(args.batch),
                "device": str(resolved_device),
                **extract_overall_metrics(result),
                "class_metrics_json": extract_class_metrics(result, class_names),
                **extract_speed_metrics(result),
            }
            append_row(args.output_csv, out_row)
    print(f"wrote {args.output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
