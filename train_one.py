from __future__ import annotations

from pathlib import Path

from pipeline_utils import build_common_parser, run_train


def main() -> None:
    parser = build_common_parser("Train one dataset, export ONNX, and log results.")
    args = parser.parse_args()

    if not args.dataset_yaml:
        parser.error("--dataset-yaml is required for train_one.py")

    dataset_yaml = Path(args.dataset_yaml)
    if not dataset_yaml.exists():
        parser.error(f"Dataset YAML not found: {dataset_yaml}")

    result = run_train(
        dataset_yaml=dataset_yaml,
        model=args.model,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        export_format=args.export_format,
    )
    print(f"train run_id={result['run_id']}")
    print(f"weights={result['weights_path']}")
    print(f"export={result['export_path']}")


if __name__ == "__main__":
    main()
