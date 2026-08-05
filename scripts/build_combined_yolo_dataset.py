from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine train splits from multiple YOLO datasets while keeping one reference valid/test split."
    )
    parser.add_argument("--reference-yaml", type=Path, required=True, help="Reference dataset YAML for valid/test.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output dataset root.")
    parser.add_argument(
        "--source-datasets",
        nargs="+",
        type=Path,
        required=True,
        help="Dataset roots or data.yaml paths whose train splits will be merged.",
    )
    parser.add_argument(
        "--prefix-mode",
        choices=("dataset", "none"),
        default="dataset",
        help="Prefix copied train filenames with the source dataset id to avoid collisions.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def resolve_dataset_root(path_like: Path) -> Path:
    if path_like.is_dir():
        return path_like.resolve()
    if path_like.is_file() and path_like.name.lower() == "data.yaml":
        return path_like.parent.resolve()
    raise SystemExit(f"Unsupported dataset input: {path_like}")


def resolve_dataset_spec(dataset_yaml: Path) -> dict[str, object]:
    data = load_yaml(dataset_yaml)
    base = Path(data["path"])
    if not base.is_absolute():
        base = (dataset_yaml.parent / base).resolve()

    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    def split_dir(key: str) -> Path:
        value = Path(str(data[key]))
        if value.is_absolute():
            return value.resolve()
        return (base / value).resolve()

    return {
        "base": base.resolve(),
        "train_images": split_dir("train"),
        "val_images": split_dir("val"),
        "test_images": split_dir("test"),
        "names": list(names),
        "nc": int(data["nc"]),
    }


def list_images(images_dir: Path) -> list[Path]:
    return sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def label_path_for_image(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def copy_split_tree(source_images: Path, output_root: Path, split_name: str) -> dict[str, int]:
    source_root = source_images.parent
    dest_root = output_root / split_name
    if dest_root.exists():
        shutil.rmtree(dest_root)
    shutil.copytree(source_root, dest_root)
    return {
        "images": sum(1 for _ in (dest_root / "images").iterdir()),
        "labels": sum(1 for _ in (dest_root / "labels").iterdir()),
    }


def write_dataset_yaml(output_root: Path, names: list[str], nc: int) -> None:
    payload = {
        "path": output_root.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": nc,
        "names": names,
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=False, sort_keys=False),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    reference_yaml = args.reference_yaml.resolve()
    output_root = args.output_root.resolve()
    reference_spec = resolve_dataset_spec(reference_yaml)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    valid_stats = copy_split_tree(Path(reference_spec["val_images"]), output_root, "valid")
    test_stats = copy_split_tree(Path(reference_spec["test_images"]), output_root, "test")

    train_images_out = output_root / "train" / "images"
    train_labels_out = output_root / "train" / "labels"
    train_images_out.mkdir(parents=True, exist_ok=True)
    train_labels_out.mkdir(parents=True, exist_ok=True)

    source_summaries: list[dict[str, object]] = []
    total_images = 0

    for source_input in args.source_datasets:
        source_root = resolve_dataset_root(source_input.resolve())
        source_yaml = source_root / "data.yaml"
        if not source_yaml.exists():
            raise SystemExit(f"Dataset yaml not found under source dataset: {source_root}")
        spec = resolve_dataset_spec(source_yaml)
        source_id = source_root.name
        copied = 0
        for image_path in list_images(Path(spec["train_images"])):
            prefix = f"{source_id}__" if args.prefix_mode == "dataset" else ""
            dest_stem = f"{prefix}{image_path.stem}"
            dest_image = train_images_out / f"{dest_stem}{image_path.suffix.lower()}"
            dest_label = train_labels_out / f"{dest_stem}.txt"
            shutil.copy2(image_path, dest_image)
            src_label = label_path_for_image(image_path)
            if src_label.exists():
                shutil.copy2(src_label, dest_label)
            else:
                dest_label.write_text("", encoding="utf-8")
            copied += 1
            total_images += 1
        source_summaries.append(
            {
                "source_dataset_id": source_id,
                "source_dataset_yaml": str(source_yaml),
                "copied_train_images": copied,
            }
        )

    write_dataset_yaml(output_root, list(reference_spec["names"]), int(reference_spec["nc"]))
    meta = {
        "dataset_id": output_root.name,
        "reference_dataset_yaml": str(reference_yaml),
        "reference_dataset": str(reference_spec["base"]),
        "source_datasets": source_summaries,
        "prefix_mode": args.prefix_mode,
        "combined_train_images": total_images,
        "valid_stats": valid_stats,
        "test_stats": test_stats,
        "notes": "Combined train split from multiple prepared YOLO datasets while preserving a single reference valid/test split.",
    }
    (output_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
