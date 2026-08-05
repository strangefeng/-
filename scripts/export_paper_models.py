from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repo_runtime import configure_repo_runtime

configure_repo_runtime(ROOT)

from pipeline_utils import create_ultralytics_model, normalize_path


CSV_COLUMNS = [
    "source_csv",
    "experiment_id",
    "model_name",
    "weights_path",
    "format",
    "status",
    "export_path",
    "artifact_size_mb",
    "error",
]


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


def append_row(output_csv: Path, row: dict[str, str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    should_header = not output_csv.exists()
    with output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if should_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def artifact_size(path: str | Path | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    if not candidate.exists():
        return ""
    return f"{candidate.stat().st_size / (1024 * 1024):.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained paper-suite models for edge deployment.")
    parser.add_argument("--source-csv", nargs="+", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", default=["onnx"], help="Ultralytics export formats, e.g. onnx engine")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "results" / "aircraft_ai_uav_paper_exports.csv")
    parser.add_argument("--only", nargs="*", help="Optional experiment ids to export.")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--half", action="store_true", help="Use FP16 export when supported.")
    parser.add_argument("--int8", action="store_true", help="Use INT8 export when supported.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = set(args.only or [])
    for source_row in read_success_rows([path.resolve() for path in args.source_csv]):
        experiment_id = source_row.get("run_id") or source_row.get("experiment_id", "")
        if selected and experiment_id not in selected:
            continue
        weights_path = Path(source_row["weights_path"])
        if not weights_path.exists():
            append_row(
                args.output_csv,
                {
                    "source_csv": source_row.get("_source_csv", ""),
                    "experiment_id": experiment_id,
                    "model_name": source_row.get("name", ""),
                    "weights_path": normalize_path(weights_path),
                    "status": "missing_weights",
                    "error": f"Missing weights: {weights_path}",
                },
            )
            continue
        for fmt in args.formats:
            print(f"[export] {experiment_id} format={fmt}")
            try:
                model = create_ultralytics_model(str(weights_path))
                export_path = model.export(format=fmt, imgsz=args.imgsz, half=args.half, int8=args.int8)
                append_row(
                    args.output_csv,
                    {
                        "source_csv": source_row.get("_source_csv", ""),
                        "experiment_id": experiment_id,
                        "model_name": source_row.get("name", ""),
                        "weights_path": normalize_path(weights_path),
                        "format": fmt,
                        "status": "success",
                        "export_path": normalize_path(export_path),
                        "artifact_size_mb": artifact_size(export_path),
                    },
                )
            except Exception as exc:
                append_row(
                    args.output_csv,
                    {
                        "source_csv": source_row.get("_source_csv", ""),
                        "experiment_id": experiment_id,
                        "model_name": source_row.get("name", ""),
                        "weights_path": normalize_path(weights_path),
                        "format": fmt,
                        "status": "failed",
                        "error": str(exc),
                    },
                )
    print(f"wrote {args.output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
