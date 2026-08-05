from __future__ import annotations

import argparse
import importlib.metadata
import json
import traceback
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def resolve_train_dir(dataset_yaml: Path) -> tuple[Path, dict]:
    data = yaml.safe_load(dataset_yaml.read_text(encoding='utf-8-sig'))
    base = Path(data.get('path', ''))
    if not base.is_absolute():
        base = (dataset_yaml.parent / base).resolve()
    train_rel = Path(data['train'])
    train_dir = train_rel if train_rel.is_absolute() else base / train_rel
    return train_dir.resolve(), data


def find_images(train_dir: Path) -> list[Path]:
    return sorted(
        path for path in train_dir.rglob('*') if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_markdown(report: dict) -> str:
    lines = [
        '# Read Check Report',
        '',
        f"- Dataset YAML: `{report['dataset_yaml']}`",
        f"- Train image directory: `{report['train_dir']}`",
        f"- Total images found: `{report['total_images']}`",
        f"- Selected for check: `{report['selected_images']}`",
        f"- Attempted reads: `{report['attempted_reads']}`",
        f"- Successful reads: `{report['successful_reads']}`",
        f"- Failed reads: `{report['failed_reads']}`",
        f"- `cv2` import ok: `{report['cv2_import_ok']}`",
        f"- Installed `opencv-python` version: `{report['opencv_python_version']}`",
    ]

    if report.get('cv2_version'):
        lines.append(f"- Imported `cv2` version: `{report['cv2_version']}`")
    if report.get('cv2_import_error'):
        lines.extend([
            '',
            '## Import Error',
            '',
            '```text',
            report['cv2_import_error'].rstrip(),
            '```',
        ])
    if report.get('failed_files'):
        lines.extend(['', '## Failed Files', ''])
        lines.extend(f"- `{path}`" for path in report['failed_files'])
    elif not report['cv2_import_ok']:
        lines.extend([
            '',
            '## Failed Files',
            '',
            '- File-level `cv2.imread` checks were not attempted because `cv2` failed to import.',
        ])

    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description='Check whether cv2 can read smoke test training images.')
    parser.add_argument('--dataset-yaml', type=Path, required=True, help='Path to the standardized dataset YAML.')
    parser.add_argument('--limit', type=int, default=50, help='Maximum number of images to test.')
    parser.add_argument('--report-md', type=Path, help='Optional markdown report output path.')
    args = parser.parse_args()

    dataset_yaml = args.dataset_yaml.resolve()
    train_dir, _ = resolve_train_dir(dataset_yaml)
    images = find_images(train_dir)
    selected = images[: args.limit]

    report = {
        'dataset_yaml': str(dataset_yaml),
        'train_dir': str(train_dir),
        'total_images': len(images),
        'selected_images': len(selected),
        'attempted_reads': 0,
        'successful_reads': 0,
        'failed_reads': 0,
        'failed_files': [],
        'cv2_import_ok': False,
        'cv2_version': '',
        'opencv_python_version': '',
        'cv2_import_error': '',
    }

    try:
        report['opencv_python_version'] = importlib.metadata.version('opencv-python')
    except importlib.metadata.PackageNotFoundError:
        report['opencv_python_version'] = 'not installed'

    try:
        import cv2
        report['cv2_import_ok'] = True
        report['cv2_version'] = cv2.__version__
    except Exception:
        report['cv2_import_error'] = traceback.format_exc()
        if args.report_md:
            args.report_md.write_text(build_markdown(report), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    for image_path in selected:
        report['attempted_reads'] += 1
        image = cv2.imread(str(image_path))
        if image is None:
            report['failed_reads'] += 1
            report['failed_files'].append(str(image_path))
        else:
            report['successful_reads'] += 1

    if args.report_md:
        args.report_md.write_text(build_markdown(report), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['failed_reads'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
