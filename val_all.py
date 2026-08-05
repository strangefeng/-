from __future__ import annotations

from pathlib import Path

from pipeline_utils import (
    DEFAULT_MODEL,
    build_common_parser,
    find_latest_train_record,
    load_main_datasets,
    run_validation,
)


def main() -> None:
    parser = build_common_parser("Validate all standardized 5-class datasets.")
    args = parser.parse_args()

    datasets = load_main_datasets()
    failures: list[tuple[str, str]] = []

    for entry in datasets:
        dataset_id = entry["dataset_id"]
        dataset_yaml = Path(entry["yaml"])
        run_name = f"{args.name}_{dataset_id}"

        if args.model and args.model != DEFAULT_MODEL:
            weights = args.model
        else:
            latest = find_latest_train_record(
                dataset_id=dataset_id,
                project=args.project,
                name_prefix=args.name,
            )
            if not latest or not latest.get("weights_path"):
                failures.append((dataset_id, "no successful training record found"))
                print(f"[val_all] skip dataset={dataset_id} reason=no successful training record found")
                continue
            weights = latest["weights_path"]

        print(f"[val_all] dataset={dataset_id} yaml={dataset_yaml} weights={weights}")
        try:
            result = run_validation(
                dataset_yaml=dataset_yaml,
                model=weights,
                imgsz=args.imgsz,
                epochs=args.epochs,
                batch=args.batch,
                device=args.device,
                project=args.project,
                name=run_name,
            )
            print(f"[val_all] ok dataset={dataset_id} run_id={result['run_id']}")
        except Exception as exc:
            failures.append((dataset_id, str(exc)))
            print(f"[val_all] failed dataset={dataset_id} error={exc}")

    if failures:
        lines = "\n".join(f"- {dataset_id}: {message}" for dataset_id, message in failures)
        raise SystemExit(f"Some validations failed:\n{lines}")


if __name__ == "__main__":
    main()
