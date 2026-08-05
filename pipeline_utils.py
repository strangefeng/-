from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from repo_runtime import configure_repo_runtime


ROOT = Path(__file__).resolve().parent
configure_repo_runtime(ROOT)
RESULTS_DIR = ROOT / "results"
EXPERIMENTS_CSV = RESULTS_DIR / "experiments.csv"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"
MAIN_DATASETS_JSON = ROOT / "datasets" / "main_5cls_datasets.json"
SUPPLEMENTARY_DATASETS_JSON = ROOT / "datasets" / "supplementary_2cls.json"
DEFAULT_MODEL = str((ROOT.parent / "yolo11n.pt").resolve())
DEFAULT_PROJECT = ROOT / "runs"
CSV_COLUMNS = [
    "timestamp",
    "run_id",
    "stage",
    "dataset_id",
    "dataset_yaml",
    "source_dataset",
    "included_in_main",
    "model",
    "imgsz",
    "epochs",
    "batch",
    "device",
    "project",
    "name",
    "run_dir",
    "weights_path",
    "map50",
    "map50_95",
    "precision",
    "recall",
    "export_format",
    "export_path",
    "status",
    "error",
]
METRIC_ALIASES = {
    "precision": ["metrics/precision(B)", "precision", "metrics/precision"],
    "recall": ["metrics/recall(B)", "recall", "metrics/recall"],
    "map50": ["metrics/mAP50(B)", "map50", "metrics/mAP50"],
    "map50_95": ["metrics/mAP50-95(B)", "map50-95", "map", "metrics/mAP50-95"],
}


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-yaml", type=Path, help="Path to a dataset YAML file.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model checkpoint or model name.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device. Use 'auto' to prefer CUDA and fall back to CPU.",
    )
    parser.add_argument(
        "--project",
        default=str(DEFAULT_PROJECT),
        help="Ultralytics project directory for run outputs.",
    )
    parser.add_argument("--name", default="baseline", help="Run name or run name prefix.")
    parser.add_argument(
        "--export-format",
        default="onnx",
        help="Export format after training. Use 'none' to skip export.",
    )
    return parser


def now_timestamp() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def make_run_id() -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def ensure_results_csv() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not EXPERIMENTS_CSV.exists():
        with EXPERIMENTS_CSV.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def append_experiment_row(row: dict[str, Any]) -> None:
    ensure_results_csv()
    normalized = {column: row.get(column, "") for column in CSV_COLUMNS}
    with EXPERIMENTS_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writerow(normalized)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_device(device: str) -> str:
    if device != "auto":
        return device

    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def require_ultralytics():
    configure_repo_runtime(ROOT)
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Install dependencies with "
            "`python -m pip install -r requirements.txt` before running training."
        ) from exc
    return YOLO, RTDETR


def create_ultralytics_model(model: str):
    YOLO, RTDETR = require_ultralytics()
    model_name = str(model).lower()
    if "rtdetr" in model_name or "rt-detr" in model_name:
        return RTDETR(model)
    return YOLO(model)


def dataset_id_from_yaml(dataset_yaml: Path) -> str:
    meta_path = dataset_yaml.with_name("meta.json")
    if meta_path.exists():
        meta = load_json(meta_path)
        if meta.get("dataset_id"):
            return meta["dataset_id"]
    return dataset_yaml.parent.name


def dataset_meta_from_yaml(dataset_yaml: Path) -> dict[str, Any]:
    meta_path = dataset_yaml.with_name("meta.json")
    return load_json(meta_path) if meta_path.exists() else {}


def normalize_path(value: str | Path | None) -> str:
    if value is None:
        return ""
    text = str(value)
    if not text:
        return ""
    return str(Path(text).resolve())


def float_or_blank(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def pick_metric(row: dict[str, Any], key: str) -> str:
    for alias in METRIC_ALIASES[key]:
        if alias in row and row[alias] not in ("", None):
            return float_or_blank(row[alias])
    return ""


def metrics_from_results_csv(run_dir: Path) -> dict[str, str]:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return {"precision": "", "recall": "", "map50": "", "map50_95": ""}

    frame = pd.read_csv(results_csv)
    if frame.empty:
        return {"precision": "", "recall": "", "map50": "", "map50_95": ""}

    row = frame.iloc[-1].to_dict()
    return {
        "precision": pick_metric(row, "precision"),
        "recall": pick_metric(row, "recall"),
        "map50": pick_metric(row, "map50"),
        "map50_95": pick_metric(row, "map50_95"),
    }


def metrics_from_results_obj(result: Any) -> dict[str, str]:
    values = {"precision": "", "recall": "", "map50": "", "map50_95": ""}
    results_dict = getattr(result, "results_dict", None) or {}
    if not results_dict:
        return values

    for key in values:
        values[key] = pick_metric(results_dict, key)
    return values


def combine_metrics(run_dir: Path | None, result: Any) -> dict[str, str]:
    metrics = metrics_from_results_obj(result)
    if run_dir is not None:
        csv_metrics = metrics_from_results_csv(run_dir)
        for key, value in csv_metrics.items():
            if value:
                metrics[key] = value
    return metrics


def build_row_base(
    *,
    run_id: str,
    stage: str,
    dataset_yaml: Path,
    model: str,
    imgsz: int,
    epochs: int,
    batch: int,
    device: str,
    project: str | Path,
    name: str,
) -> dict[str, str]:
    meta = dataset_meta_from_yaml(dataset_yaml)
    return {
        "timestamp": now_timestamp(),
        "run_id": run_id,
        "stage": stage,
        "dataset_id": meta.get("dataset_id", dataset_id_from_yaml(dataset_yaml)),
        "dataset_yaml": normalize_path(dataset_yaml),
        "source_dataset": meta.get("source_dataset", ""),
        "included_in_main": str(meta.get("included_in_main", "")),
        "model": model,
        "imgsz": str(imgsz),
        "epochs": str(epochs),
        "batch": str(batch),
        "device": str(device),
        "project": str(project),
        "name": name,
    }


def run_train(
    *,
    dataset_yaml: Path,
    model: str,
    imgsz: int,
    epochs: int,
    batch: int,
    device: str,
    project: str | Path,
    name: str,
    run_id: str | None = None,
    export_format: str = "onnx",
) -> dict[str, Any]:
    dataset_yaml = dataset_yaml.resolve()
    resolved_device = resolve_device(device)
    run_id = run_id or make_run_id()
    row_base = build_row_base(
        run_id=run_id,
        stage="train",
        dataset_yaml=dataset_yaml,
        model=model,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=resolved_device,
        project=project,
        name=name,
    )

    model_obj = create_ultralytics_model(model)
    try:
        train_result = model_obj.train(
            data=str(dataset_yaml),
            imgsz=imgsz,
            epochs=epochs,
            batch=batch,
            device=resolved_device,
            project=str(project),
            name=name,
        )
        run_dir = Path(getattr(model_obj.trainer, "save_dir", ""))
        weights_path = run_dir / "weights" / "best.pt"
        metrics = combine_metrics(run_dir, train_result)
        append_experiment_row(
            {
                **row_base,
                "run_dir": normalize_path(run_dir),
                "weights_path": normalize_path(weights_path),
                "map50": metrics["map50"],
                "map50_95": metrics["map50_95"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "status": "success",
            }
        )
    except Exception as exc:
        append_experiment_row({**row_base, "status": "failed", "error": str(exc)})
        raise

    export_row_base = build_row_base(
        run_id=run_id,
        stage="export",
        dataset_yaml=dataset_yaml,
        model=model,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=resolved_device,
        project=project,
        name=name,
    )
    export_row_base["run_dir"] = normalize_path(run_dir)
    export_row_base["weights_path"] = normalize_path(weights_path)

    export_path = ""
    if export_format and str(export_format).lower() != "none":
        try:
            export_path = model_obj.export(format=export_format)
            append_experiment_row(
                {
                    **export_row_base,
                    "export_format": export_format,
                    "export_path": normalize_path(export_path),
                    "status": "success",
                }
            )
        except Exception as exc:
            append_experiment_row(
                {
                    **export_row_base,
                    "export_format": export_format,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            raise
    else:
        append_experiment_row(
            {
                **export_row_base,
                "export_format": "none",
                "export_path": "",
                "status": "success",
            }
        )

    return {
        "run_id": run_id,
        "dataset_id": row_base["dataset_id"],
        "run_dir": run_dir,
        "weights_path": weights_path,
        "export_path": Path(export_path) if export_path else None,
        "resolved_device": resolved_device,
    }


def run_validation(
    *,
    dataset_yaml: Path,
    model: str,
    imgsz: int,
    epochs: int,
    batch: int,
    device: str,
    project: str | Path,
    name: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    dataset_yaml = dataset_yaml.resolve()
    resolved_device = resolve_device(device)
    run_id = run_id or make_run_id()
    row_base = build_row_base(
        run_id=run_id,
        stage="val",
        dataset_yaml=dataset_yaml,
        model=model,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=resolved_device,
        project=project,
        name=name,
    )

    model_obj = create_ultralytics_model(model)
    try:
        result = model_obj.val(
            data=str(dataset_yaml),
            imgsz=imgsz,
            batch=batch,
            device=resolved_device,
            project=str(project),
            name=name,
        )
        run_dir = Path(getattr(result, "save_dir", "")) if getattr(result, "save_dir", "") else None
        metrics = combine_metrics(run_dir, result)
        append_experiment_row(
            {
                **row_base,
                "run_dir": normalize_path(run_dir),
                "weights_path": normalize_path(model),
                "map50": metrics["map50"],
                "map50_95": metrics["map50_95"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "status": "success",
            }
        )
    except Exception as exc:
        append_experiment_row({**row_base, "status": "failed", "error": str(exc)})
        raise

    return {
        "run_id": run_id,
        "dataset_id": row_base["dataset_id"],
        "run_dir": run_dir,
        "resolved_device": resolved_device,
    }


def load_main_datasets() -> list[dict[str, Any]]:
    return load_json(MAIN_DATASETS_JSON)


def find_latest_train_record(
    dataset_id: str, project: str | None = None, name_prefix: str | None = None
) -> dict[str, Any] | None:
    ensure_results_csv()
    frame = pd.read_csv(EXPERIMENTS_CSV)
    if frame.empty:
        return None

    filtered = frame[
        (frame["stage"] == "train")
        & (frame["status"] == "success")
        & (frame["dataset_id"] == dataset_id)
    ]
    if project:
        filtered = filtered[filtered["project"] == project]
    if name_prefix:
        filtered = filtered[filtered["name"].astype(str).str.startswith(name_prefix)]
    if filtered.empty:
        return None

    filtered = filtered.sort_values("timestamp", ascending=False)
    return filtered.iloc[0].to_dict()


def summarize_records(project: str | None = None, name_prefix: str | None = None) -> pd.DataFrame:
    ensure_results_csv()
    frame = pd.read_csv(EXPERIMENTS_CSV)
    if frame.empty:
        return pd.DataFrame()

    if project:
        frame = frame[frame["project"] == project]
    if name_prefix:
        frame = frame[frame["name"].astype(str).str.startswith(name_prefix)]
    if frame.empty:
        return frame

    sort_frame = frame.sort_values("timestamp")
    records: list[dict[str, Any]] = []
    for (run_id, dataset_id), group in sort_frame.groupby(["run_id", "dataset_id"], dropna=False):
        row: dict[str, Any] = {"run_id": run_id, "dataset_id": dataset_id}
        latest = group.iloc[-1].to_dict()
        for key in [
            "dataset_yaml",
            "source_dataset",
            "included_in_main",
            "model",
            "imgsz",
            "epochs",
            "batch",
            "device",
            "project",
            "name",
        ]:
            row[key] = latest.get(key, "")

        for stage in ["train", "val", "export"]:
            stage_group = group[group["stage"] == stage]
            if stage_group.empty:
                continue
            stage_row = stage_group.iloc[-1].to_dict()
            row[f"{stage}_status"] = stage_row.get("status", "")
            row[f"{stage}_run_dir"] = stage_row.get("run_dir", "")
            row[f"{stage}_weights_path"] = stage_row.get("weights_path", "")
            row[f"{stage}_map50"] = stage_row.get("map50", "")
            row[f"{stage}_map50_95"] = stage_row.get("map50_95", "")
            row[f"{stage}_precision"] = stage_row.get("precision", "")
            row[f"{stage}_recall"] = stage_row.get("recall", "")
            row[f"{stage}_export_format"] = stage_row.get("export_format", "")
            row[f"{stage}_export_path"] = stage_row.get("export_path", "")
            row[f"{stage}_error"] = stage_row.get("error", "")
        records.append(row)

    return pd.DataFrame(records)



