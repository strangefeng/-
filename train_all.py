from __future__ import annotations

from pathlib import Path

from pipeline_utils import build_common_parser, load_main_datasets, run_train


def main() -> None:
    parser = build_common_parser("Train all standardized 5-class datasets.")
    args = parser.parse_args()

    datasets = load_main_datasets()
    failures: list[tuple[str, str]] = []

    for entry in datasets:
        dataset_id = entry["dataset_id"]
        dataset_yaml = Path(entry["yaml"])
        run_name = f"{args.name}_{dataset_id}"
        print(f"[train_all] dataset={dataset_id} yaml={dataset_yaml}")
        try:
            result = run_train(
                dataset_yaml=dataset_yaml,
                model=args.model,
                imgsz=args.imgsz,
                epochs=args.epochs,
                batch=args.batch,
                device=args.device,
                project=args.project,
                name=run_name,
            )
            print(f"[train_all] ok dataset={dataset_id} run_id={result['run_id']}")
        except Exception as exc:
            failures.append((dataset_id, str(exc)))
            print(f"[train_all] failed dataset={dataset_id} error={exc}")

    if failures:
        lines = "\n".join(f"- {dataset_id}: {message}" for dataset_id, message in failures)
        raise SystemExit(f"Some datasets failed:\n{lines}")


if __name__ == "__main__":
    main()
