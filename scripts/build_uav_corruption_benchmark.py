from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


CLASS_NAMES = ["missing-rivet", "rust", "scratch"]


def motion_blur(image: np.ndarray, level: int) -> np.ndarray:
    sizes = {1: 5, 2: 9, 3: 15}
    k = sizes[level]
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(image, -1, kernel)


def defocus_blur(image: np.ndarray, level: int) -> np.ndarray:
    sizes = {1: 5, 2: 9, 3: 15}
    k = sizes[level]
    return cv2.GaussianBlur(image, (k, k), 0)


def gaussian_noise(image: np.ndarray, level: int) -> np.ndarray:
    sigma = {1: 8, 2: 16, 3: 28}[level]
    noise = np.random.default_rng(42 + level).normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def low_light(image: np.ndarray, level: int) -> np.ndarray:
    gamma = {1: 1.4, 2: 1.8, 3: 2.4}[level]
    table = ((np.arange(256) / 255.0) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(image, table)


def glare(image: np.ndarray, level: int) -> np.ndarray:
    h, w = image.shape[:2]
    radius = int(min(h, w) * {1: 0.12, 2: 0.18, 3: 0.25}[level])
    strength = {1: 0.35, 2: 0.55, 3: 0.75}[level]
    center = (int(w * 0.62), int(h * 0.38))
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    mask = np.clip(1 - dist / radius, 0, 1)[..., None]
    overlay = np.full_like(image, 255)
    return np.clip(image * (1 - mask * strength) + overlay * (mask * strength), 0, 255).astype(np.uint8)


def jpeg_compression(image: np.ndarray, level: int) -> np.ndarray:
    quality = {1: 55, 2: 35, 3: 18}[level]
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def scale_degradation(image: np.ndarray, level: int) -> np.ndarray:
    h, w = image.shape[:2]
    ratio = {1: 0.75, 2: 0.5, 3: 0.35}[level]
    small = cv2.resize(image, (max(1, int(w * ratio)), max(1, int(h * ratio))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


CORRUPTIONS = {
    "motion_blur": motion_blur,
    "defocus_blur": defocus_blur,
    "gaussian_noise": gaussian_noise,
    "low_light": low_light,
    "glare": glare,
    "jpeg": jpeg_compression,
    "scale": scale_degradation,
}


def read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> bool:
    suffix = path.suffix.lower()
    ext = ".jpg" if suffix in {".jpg", ".jpeg"} else suffix
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def write_yaml(path: Path, dataset_root: Path, val_images: str, train_images: str) -> None:
    data = {
        "path": str(dataset_root.resolve()).replace("\\", "/"),
        "train": train_images.replace("\\", "/"),
        "val": val_images.replace("\\", "/"),
        "test": val_images.replace("\\", "/"),
        "nc": len(CLASS_NAMES),
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def find_label(source_root: Path, image_path: Path, split: str) -> Path:
    return source_root / "labels" / split / f"{image_path.stem}.txt"


def build_benchmark(source_root: Path, output_root: Path, source_split: str, levels: list[int]) -> None:
    source_images = source_root / "images" / source_split
    if not source_images.exists():
        raise FileNotFoundError(f"Source image split not found: {source_images}")

    output_root.mkdir(parents=True, exist_ok=True)
    yaml_dir = output_root / "yamls"
    yaml_dir.mkdir(parents=True, exist_ok=True)

    train_rel = str((source_root / "images" / "train").resolve())
    clean_yaml = yaml_dir / "clean.yaml"
    write_yaml(clean_yaml, source_root, str((source_root / "images" / source_split).resolve()), train_rel)

    images = [p for p in source_images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    for corruption_name, fn in CORRUPTIONS.items():
        for level in levels:
            split_name = f"{corruption_name}_s{level}"
            out_images = output_root / "images" / split_name
            out_labels = output_root / "labels" / split_name
            out_images.mkdir(parents=True, exist_ok=True)
            out_labels.mkdir(parents=True, exist_ok=True)

            for image_path in images:
                image = read_image(image_path)
                if image is None:
                    print(f"[warn] could not read image: {image_path}")
                    continue
                corrupted = fn(image, level)
                if not write_image(out_images / image_path.name, corrupted):
                    print(f"[warn] could not write image: {out_images / image_path.name}")
                    continue
                label_path = find_label(source_root, image_path, source_split)
                if label_path.exists():
                    shutil.copy2(label_path, out_labels / label_path.name)
                else:
                    (out_labels / f"{image_path.stem}.txt").write_text("", encoding="utf-8")

            write_yaml(
                yaml_dir / f"{split_name}.yaml",
                output_root,
                f"images/{split_name}",
                train_rel,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build UAV-style corruption benchmark YAMLs and test images.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"D:\桌面\毕设\数据集\Aircraft_AI_relabel_dataset_auto_clean_split"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"D:\桌面\毕设\数据集\yolo-baseline-repro\datasets\robustness\aircraft_ai_uav_clean_split_corruptions"),
    )
    parser.add_argument("--source-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_benchmark(args.source_root.resolve(), args.output_root.resolve(), args.source_split, args.levels)
    print(f"wrote benchmark to {args.output_root.resolve()}")
    print(f"yamls={args.output_root.resolve() / 'yamls'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
