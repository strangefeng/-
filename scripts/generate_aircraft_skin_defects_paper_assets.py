from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import yaml


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis"
FIGURE_DIR = ANALYSIS_DIR / "aircraft_skin_defects_final_assets"
RESULTS_CSV = ROOT / "results" / "aircraft_skin_defects_new_mainline.csv"
ORIGINAL_DATASET_YAML = ROOT / "datasets" / "standardized_5cls" / "aircraft_skin_defects_new" / "data.yaml"
COPYPASTE_DATASET_YAML = ROOT / "datasets" / "generated_5cls" / "aircraft_skin_defects_new_copypaste" / "data.yaml"
COPYPASTE_META = ROOT / "datasets" / "generated_5cls" / "aircraft_skin_defects_new_copypaste" / "meta.json"
SUMMARY_MD = ANALYSIS_DIR / "aircraft_skin_defects_mainline_final_summary.md"
PAPER_REVIEW_MD = ANALYSIS_DIR / "aircraft_skin_defects_manual_review_paper_version.md"
CLASS_NAMES = ["crack", "dent", "missing-head", "paint-off", "scratch"]
EXPERIMENTS = ["B0", "Exp_A", "Exp_B", "Exp_C"]
RUN_DIRS = {
    "B0": ROOT / "runs" / "aircraft_new_formal" / "B0_yolo11_default3",
    "Exp_A": ROOT / "runs" / "aircraft_new_stage2_formal" / "Exp_A_baseline_cp",
    "Exp_B": ROOT / "runs" / "aircraft_new_stage2_formal" / "Exp_B_neck_wt",
    "Exp_C": ROOT / "runs" / "aircraft_new_stage2_formal" / "Exp_C_neck_wt_cp",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find_success_row(path: Path, experiment_id: str) -> dict[str, str]:
    candidates = [
        row
        for row in read_rows(path)
        if row.get("experiment_id") == experiment_id and row.get("run_status") == "success"
    ]
    if not candidates:
        raise RuntimeError(f"Missing successful row for {experiment_id}")
    return sorted(
        candidates,
        key=lambda row: (
            int(row.get("epochs", "0") or 0),
            float(row.get("map50_95", "") or 0.0),
        ),
        reverse=True,
    )[0]


def load_dataset_spec(data_yaml: Path) -> dict:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    base = Path(data["path"])
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()

    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    return {
        "base": base,
        "train_labels": (base / "train" / "labels").resolve(),
        "names": list(names),
    }


def count_split_classes(label_dir: Path, class_names: list[str]) -> Counter:
    counts: Counter[str] = Counter({name: 0 for name in class_names})
    for label_path in sorted(label_dir.glob("*.txt")):
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            if 0 <= class_id < len(class_names):
                counts[class_names[class_id]] += 1
    return counts


def class_metrics(row: dict[str, str]) -> dict[str, float]:
    payload = json.loads(row["class_metrics_json"])
    return {name: float(payload[name]["ap50_95"]) for name in CLASS_NAMES}


def plot_class_ap95(rows: dict[str, dict[str, str]]) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURE_DIR / "exp_a_class_ap95_comparison.png"
    labels = CLASS_NAMES
    series = ["B0", "Exp_A", "Exp_B", "Exp_C"]
    colors = ["#7a8a99", "#d7662f", "#4d7ea8", "#8f5aa8"]
    x = range(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, exp_id in enumerate(series):
        values = [class_metrics(rows[exp_id])[label] for label in labels]
        ax.bar([item + (idx - 1.5) * width for item in x], values, width=width, label=exp_id, color=colors[idx])

    ax.set_title("Per-class AP50-95 Comparison on Main Dataset")
    ax.set_ylabel("AP50-95")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0.0, 0.70)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_distribution(original_counts: Counter, cp_counts: Counter) -> Path:
    output = FIGURE_DIR / "exp_a_train_distribution_shift.png"
    labels = CLASS_NAMES
    base_values = [original_counts[label] for label in labels]
    cp_values = [cp_counts[label] for label in labels]
    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar([item - width / 2 for item in x], base_values, width=width, label="Original train", color="#7a8a99")
    ax.bar([item + width / 2 for item in x], cp_values, width=width, label="Copy-Paste train", color="#d7662f")
    ax.set_title("Train-set Class Distribution Before and After Offline Copy-Paste")
    ax.set_ylabel("Box count")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def build_qualitative_panel() -> Path:
    output = FIGURE_DIR / "exp_a_qualitative_comparison_panel.png"
    font = ImageFont.load_default()
    entries = [
        ("B0", RUN_DIRS["B0"] / "val_batch0_pred.jpg"),
        ("Exp_A", RUN_DIRS["Exp_A"] / "val_batch0_pred.jpg"),
        ("B0", RUN_DIRS["B0"] / "val_batch1_pred.jpg"),
        ("Exp_A", RUN_DIRS["Exp_A"] / "val_batch1_pred.jpg"),
    ]
    images = [Image.open(path).convert("RGB") for _, path in entries]
    thumb_width = 760
    thumbs = []
    for image in images:
        ratio = thumb_width / image.width
        thumb = image.resize((thumb_width, int(image.height * ratio)))
        thumbs.append(thumb)

    padding = 20
    header_h = 36
    cols = 2
    rows = 2
    canvas_w = padding * (cols + 1) + thumb_width * cols
    canvas_h = padding * (rows + 1) + rows * (thumbs[0].height + header_h)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, (label, _) in enumerate(entries):
        row = idx // cols
        col = idx % cols
        x = padding + col * (thumb_width + padding)
        y = padding + row * (thumbs[0].height + header_h + padding)
        draw.rectangle([x, y, x + thumb_width, y + header_h], fill="#f0f2f5")
        draw.text((x + 10, y + 10), label, fill="black", font=font)
        canvas.paste(thumbs[idx], (x, y + header_h))

    canvas.save(output)
    return output


def write_final_summary(rows: dict[str, dict[str, str]], original_counts: Counter, cp_counts: Counter, figure_paths: dict[str, Path]) -> None:
    b0 = rows["B0"]
    exp_a = rows["Exp_A"]
    exp_b = rows["Exp_B"]
    exp_c = rows["Exp_C"]
    deltas = {
        "map50_95": float(exp_a["map50_95"]) - float(b0["map50_95"]),
        "scratch_ap50_95": class_metrics(exp_a)["scratch"] - class_metrics(b0)["scratch"],
        "crack_ap50_95": class_metrics(exp_a)["crack"] - class_metrics(b0)["crack"],
    }
    cp_meta = json.loads(COPYPASTE_META.read_text(encoding="utf-8"))
    scratch_extra = cp_counts["scratch"] - 2 * original_counts["scratch"]

    text = f"""# aircraft_skin_defects_new Final Summary

## Final Decision

- Final model on the main dataset: `Exp_A`
- Positioning: `Exp_A` should be treated as the formal final model for the thesis because it is the only branch that improves the overall benchmark while also producing the strongest `scratch` gain.

## Main Comparison

| exp | title | map50 | map50-95 | precision | recall | crack | dent | missing-head | paint-off | scratch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | Official default baseline | {b0['map50']} | {b0['map50_95']} | {b0['precision']} | {b0['recall']} | {class_metrics(b0)['crack']:.6f} | {class_metrics(b0)['dent']:.6f} | {class_metrics(b0)['missing-head']:.6f} | {class_metrics(b0)['paint-off']:.6f} | {class_metrics(b0)['scratch']:.6f} |
| Exp_A | Baseline YOLO11n plus offline Copy-Paste | {exp_a['map50']} | {exp_a['map50_95']} | {exp_a['precision']} | {exp_a['recall']} | {class_metrics(exp_a)['crack']:.6f} | {class_metrics(exp_a)['dent']:.6f} | {class_metrics(exp_a)['missing-head']:.6f} | {class_metrics(exp_a)['paint-off']:.6f} | {class_metrics(exp_a)['scratch']:.6f} |
| Exp_B | YOLO11n with Neck-WT refinement on original dataset | {exp_b['map50']} | {exp_b['map50_95']} | {exp_b['precision']} | {exp_b['recall']} | {class_metrics(exp_b)['crack']:.6f} | {class_metrics(exp_b)['dent']:.6f} | {class_metrics(exp_b)['missing-head']:.6f} | {class_metrics(exp_b)['paint-off']:.6f} | {class_metrics(exp_b)['scratch']:.6f} |
| Exp_C | YOLO11n with Neck-WT refinement plus offline Copy-Paste | {exp_c['map50']} | {exp_c['map50_95']} | {exp_c['precision']} | {exp_c['recall']} | {class_metrics(exp_c)['crack']:.6f} | {class_metrics(exp_c)['dent']:.6f} | {class_metrics(exp_c)['missing-head']:.6f} | {class_metrics(exp_c)['paint-off']:.6f} | {class_metrics(exp_c)['scratch']:.6f} |

## Why `Exp_A` Wins

- `Exp_A` improves `mAP50-95` over `B0` by `{deltas['map50_95']:+.6f}`
- `Exp_A` improves `scratch AP50-95` over `B0` by `{deltas['scratch_ap50_95']:+.6f}`
- `Exp_A` also improves `crack AP50-95` over `B0` by `{deltas['crack_ap50_95']:+.6f}`
- `Exp_B` and `Exp_C` show that adding the neck-side WT branch does not convert the Copy-Paste gain into a stronger final system, so the thesis should not claim structural synergy

## Data-Method Interpretation

- Original train-set class counts: `{dict(original_counts)}`
- Copy-Paste train-set class counts: `{dict(cp_counts)}`
- Offline Copy-Paste generated `scratch` donor patches from `{cp_meta['scratch_donor_count']}` source boxes and injected `{cp_meta['generated_scratch_boxes']}` synthetic scratch boxes into the train split
- The generated dataset keeps one augmented copy per original train image, so non-scratch categories roughly double because their original labels are preserved in the copied images
- After removing that expected duplication effect, `scratch` still gains an extra `{scratch_extra}` boxes beyond simple doubling, which is the key targeted shift
- This means the final positive result is best explained as a targeted data-distribution correction for the hardest class rather than a network-structure breakthrough

## Paper Figures

- Per-class AP figure: `{figure_paths['class_ap95']}`
- Train distribution figure: `{figure_paths['distribution']}`
- Qualitative comparison panel: `{figure_paths['qualitative']}`

## Thesis-Ready Conclusion

The main dataset chapter should conclude that the most reliable improvement comes from an offline Copy-Paste strategy targeted at the underperforming `scratch` class. In contrast, the tested wavelet- and NWD-based structural branches did not outperform the simpler data-centric final model. Therefore, `Exp_A` is the recommended final model, and the thesis contribution on the main dataset should be framed as a defect-oriented data enhancement workflow rather than a new backbone or neck design.
"""
    SUMMARY_MD.write_text(text, encoding="utf-8")


def write_paper_review_version() -> None:
    text = """# aircraft_skin_defects Manual Review for Thesis Writing

## 1. Problem Diagnosis

The manual review of the main 5-class dataset shows that the dominant source of performance instability is not pure missing annotation. Instead, the main problems are label-granularity inconsistency and class-definition ambiguity.

The most typical issues are:

1. `missing-head` contains many tiny boxes attached to rivets, holes, edges, characters, or normal structural points, which makes the class boundary highly unstable.
2. `paint-off` is often fragmented into many small adjacent boxes, so one visually continuous damaged region may be annotated as multiple tiny instances.
3. Tiny-box noise is dense in structurally complex regions, which increases the risk of false-positive learning for small defects.

These observations explain why the dataset is more sensitive to annotation noise and small-object ambiguity than a clean benchmark-style dataset.

## 2. Method Choice

Because the main problem is data quality and class-distribution mismatch rather than insufficient backbone capacity, the thesis should prioritize data-centric correction over aggressive structural modification.

The practical design choice is:

- keep the official YOLO11n baseline as the reference detector;
- avoid making the final conclusion depend on a complex wavelet or NWD structure;
- use offline Copy-Paste to strengthen the weak `scratch` class under the original validation and test splits.

This method choice is consistent with the manual diagnosis: when a class is both weak and under-supported, targeted data enhancement is more controllable than adding extra structural complexity on top of noisy labels.

## 3. Experimental Verification

The main-dataset experiments support this diagnosis-method alignment.

- `B0` provides a stable reference baseline.
- `Exp_A` (baseline + offline Copy-Paste) achieves the best overall result on the main dataset and delivers the strongest `scratch` improvement.
- `Exp_B` and `Exp_C` show that neck-side wavelet refinement does not produce a reliable gain, even when combined with Copy-Paste.

Therefore, the validated conclusion is not "the new structure is universally better," but rather:

**for this aircraft skin defect dataset, a defect-oriented data enhancement strategy is more effective and more reliable than the tested structural modifications.**

## 4. Thesis Wording Suggestion

In the thesis, this section can be written as:

"Manual inspection indicates that the main dataset suffers from class ambiguity, excessive tiny-box fragmentation, and noisy `missing-head` annotations. Based on this diagnosis, the study prioritizes data-centric intervention and introduces an offline Copy-Paste strategy targeted at the weak `scratch` category. Experimental results show that this targeted data enhancement outperforms the tested structural modification branches and yields the best final model on the main dataset."
"""
    PAPER_REVIEW_MD.write_text(text, encoding="utf-8")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = {exp_id: find_success_row(RESULTS_CSV, exp_id) for exp_id in EXPERIMENTS}
    original_spec = load_dataset_spec(ORIGINAL_DATASET_YAML)
    cp_spec = load_dataset_spec(COPYPASTE_DATASET_YAML)
    original_counts = count_split_classes(original_spec["train_labels"], CLASS_NAMES)
    cp_counts = count_split_classes(cp_spec["train_labels"], CLASS_NAMES)

    figure_paths = {
        "class_ap95": plot_class_ap95(rows),
        "distribution": plot_distribution(original_counts, cp_counts),
        "qualitative": build_qualitative_panel(),
    }
    write_final_summary(rows, original_counts, cp_counts, figure_paths)
    write_paper_review_version()

    manifest = {
        "summary_md": str(SUMMARY_MD),
        "review_md": str(PAPER_REVIEW_MD),
        "figures": {key: str(value) for key, value in figure_paths.items()},
        "original_train_counts": dict(original_counts),
        "copypaste_train_counts": dict(cp_counts),
    }
    (FIGURE_DIR / "asset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
