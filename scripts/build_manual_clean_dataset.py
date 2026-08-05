from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cleaned dataset copy by applying reviewed label overrides to a YOLO dataset."
    )
    parser.add_argument("--source-yaml", type=Path, required=True, help="Source dataset YAML.")
    parser.add_argument("--review-manifest", type=Path, required=True, help="Review pack manifest JSON.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output dataset root.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def load_review_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
        "base": base,
        "train_images": split_dir("train"),
        "val_images": split_dir("val"),
        "test_images": split_dir("test"),
        "names": list(names),
        "nc": int(data["nc"]),
    }


def copy_split_tree(source_split_images: Path, output_root: Path, split_name: str) -> dict[str, int]:
    source_split_root = source_split_images.parent
    dest_split_root = output_root / split_name
    if dest_split_root.exists():
        shutil.rmtree(dest_split_root)
    shutil.copytree(source_split_root, dest_split_root)
    image_count = sum(1 for _ in (dest_split_root / "images").iterdir())
    label_count = sum(1 for _ in (dest_split_root / "labels").iterdir())
    return {"images": image_count, "labels": label_count}


def write_dataset_yaml(output_root: Path, names: list[str], nc: int) -> None:
    data = {
        "path": output_root.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": nc,
        "names": names,
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=False, sort_keys=False),
        encoding="utf-8",
    )


def apply_review_overrides(review_manifest: dict, output_root: Path) -> list[dict[str, str]]:
    applied: list[dict[str, str]] = []
    split_dir_map = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}

    for item in review_manifest.get("items", []):
        split_name = split_dir_map.get(item.get("split", ""))
        copied_label_path = item.get("copied_label_path")
        stem = item.get("stem")
        if not split_name or not copied_label_path or not stem:
            continue
        reviewed_label = Path(copied_label_path)
        target_label = output_root / split_name / "labels" / f"{stem}.txt"
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reviewed_label, target_label)
        applied.append(
            {
                "split": split_name,
                "stem": stem,
                "target_label": str(target_label),
                "reviewed_label": str(reviewed_label),
            }
        )

    return applied


def main() -> None:
    args = parse_args()
    source_yaml = args.source_yaml.resolve()
    review_manifest_path = args.review_manifest.resolve()
    output_root = args.output_root.resolve()

    spec = resolve_dataset_spec(source_yaml)
    review_manifest = load_review_manifest(review_manifest_path)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_stats = copy_split_tree(Path(spec["train_images"]), output_root, "train")
    val_stats = copy_split_tree(Path(spec["val_images"]), output_root, "valid")
    test_stats = copy_split_tree(Path(spec["test_images"]), output_root, "test")
    write_dataset_yaml(output_root, list(spec["names"]), int(spec["nc"]))
    applied = apply_review_overrides(review_manifest, output_root)

    meta = {
        "dataset_id": output_root.name,
        "source_dataset_yaml": str(source_yaml),
        "source_dataset": str(spec["base"]),
        "review_manifest": str(review_manifest_path),
        "applied_review_items": applied,
        "stats": {
            "train_images": train_stats["images"],
            "train_labels": train_stats["labels"],
            "valid_images": val_stats["images"],
            "valid_labels": val_stats["labels"],
            "test_images": test_stats["images"],
            "test_labels": test_stats["labels"],
            "applied_label_overrides": len(applied),
        },
        "notes": "Dataset copy with reviewed label overrides from the round-6 suspicious review pack applied on top of the source splits.",
    }
    (output_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"manual_clean_dataset={output_root}")
    print(json.dumps(meta["stats"], ensure_ascii=False))


if __name__ == "__main__":
    main()
