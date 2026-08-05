from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]

CSV_COLUMNS = [
    "experiment_id",
    "model_name",
    "onnx_path",
    "provider",
    "input_shape",
    "warmup",
    "iterations",
    "latency_mean_ms",
    "latency_median_ms",
    "latency_p95_ms",
    "latency_std_ms",
]


def read_exports(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("status") == "success" and row.get("format") == "onnx" and row.get("export_path")
        ]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def concrete_shape(shape: list[object], imgsz: int) -> tuple[int, ...]:
    fallback = (1, 3, imgsz, imgsz)
    if len(shape) != 4:
        return fallback
    concrete: list[int] = []
    for idx, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            concrete.append(dim)
        else:
            concrete.append(fallback[idx])
    return tuple(concrete)


def benchmark_onnx(onnx_path: Path, imgsz: int, warmup: int, iterations: int) -> dict[str, str]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    input_shape = concrete_shape(list(input_meta.shape), imgsz)
    sample = np.random.default_rng(0).random(input_shape, dtype=np.float32)

    for _ in range(warmup):
        session.run(None, {input_meta.name: sample})

    latencies: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, {input_meta.name: sample})
        latencies.append((time.perf_counter() - start) * 1000.0)

    return {
        "provider": ",".join(session.get_providers()),
        "input_shape": "x".join(str(part) for part in input_shape),
        "latency_mean_ms": f"{statistics.fmean(latencies):.6f}",
        "latency_median_ms": f"{statistics.median(latencies):.6f}",
        "latency_p95_ms": f"{percentile(latencies, 95):.6f}",
        "latency_std_ms": f"{statistics.pstdev(latencies):.6f}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark exported ONNX models with ONNXRuntime CPU.")
    parser.add_argument("--exports-csv", type=Path, default=ROOT / "results" / "aircraft_ai_uav_paper_exports.csv")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "results" / "aircraft_ai_uav_paper_onnx_latency.csv")
    parser.add_argument("--only", nargs="*", help="Optional experiment ids to benchmark.")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = set(args.only or [])
    rows: list[dict[str, str]] = []
    for export_row in read_exports(args.exports_csv.resolve()):
        experiment_id = export_row.get("experiment_id", "")
        if selected and experiment_id not in selected:
            continue
        onnx_path = Path(export_row["export_path"])
        if not onnx_path.exists():
            print(f"[skip] missing ONNX: {onnx_path}")
            continue
        print(f"[bench] {experiment_id}")
        metrics = benchmark_onnx(onnx_path, args.imgsz, args.warmup, args.iterations)
        rows.append(
            {
                "experiment_id": experiment_id,
                "model_name": export_row.get("model_name", ""),
                "onnx_path": str(onnx_path.resolve()),
                "warmup": str(args.warmup),
                "iterations": str(args.iterations),
                **metrics,
            }
        )
    write_rows(args.output_csv.resolve(), rows)
    print(f"wrote {args.output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
