from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def has_target_class(label_path: Path, target_class: int) -> bool:
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
        except Exception:
            continue
        if cls == target_class:
            return True
    return False


def copy_split(source_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)


def build_dataset(source_yaml: Path, output_dir: Path, target_class: int, duplicate_factor: int) -> dict:
    cfg = load_yaml(source_yaml)
    source_root = Path(cfg["path"]).resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy full dataset first.
    for split in ["train", "valid", "test"]:
        copy_split(source_root / split, output_dir / split)

    train_img_dir = output_dir / "train" / "images"
    train_lbl_dir = output_dir / "train" / "labels"
    train_labels = sorted(p for p in train_lbl_dir.glob("*.txt") if p.name.lower() != "classes.txt")

    duplicated_images = 0
    source_hit_images = 0
    for label_path in train_labels:
        if not has_target_class(label_path, target_class):
            continue
        source_hit_images += 1
        stem = label_path.stem
        image_candidates = list(train_img_dir.glob(f"{stem}.*"))
        if not image_candidates:
            continue
        image_path = image_candidates[0]
        for dup_idx in range(1, duplicate_factor):
            new_stem = f"{stem}__dupcls{target_class}_{dup_idx}"
            shutil.copy2(image_path, train_img_dir / f"{new_stem}{image_path.suffix}")
            shutil.copy2(label_path, train_lbl_dir / f"{new_stem}.txt")
            duplicated_images += 1

    new_cfg = {
        "path": str(output_dir).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": cfg["nc"],
        "names": cfg["names"],
    }
    write_yaml(output_dir / "data.yaml", new_cfg)

    meta = {
        "dataset_id": output_dir.name,
        "source_dataset_yaml": str(source_yaml.resolve()),
        "source_root": str(source_root),
        "oversample_target_class": target_class,
        "duplicate_factor": duplicate_factor,
        "source_hit_images": source_hit_images,
        "duplicated_train_images": duplicated_images,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a train-only oversampled YOLO dataset for a target class.")
    parser.add_argument("--source-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-class", type=int, default=0)
    parser.add_argument("--duplicate-factor", type=int, required=True, help="Total multiplicative factor for hit images.")
    args = parser.parse_args()

    if args.duplicate_factor < 2:
        raise SystemExit("--duplicate-factor must be >= 2")

    meta = build_dataset(args.source_yaml, args.output_dir, args.target_class, args.duplicate_factor)
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
