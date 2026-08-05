from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROUND4_CSV = ROOT / "results" / "aircraft_ai_dataset_round4.csv"
ROUND12_CSV = ROOT / "results" / "aircraft_ai_dataset_round12_automation.csv"
ROUND13_CSV = ROOT / "results" / "aircraft_ai_dataset_round13_automation.csv"
ROUND13_MANIFEST = ROOT / "experiments" / "aircraft_ai_dataset_round13_automation.json"
SOURCE_YAML = ROOT / "datasets" / "supplementary_3cls" / "aircraft_ai_dataset_3cls" / "data.yaml"


def run(command: list[str]) -> int:
    return subprocess.call(command, cwd=ROOT)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find_success_row(path: Path, experiment_id: str) -> dict[str, str] | None:
    candidates = [
        row
        for row in read_rows(path)
        if row.get("experiment_id") == experiment_id and row.get("run_status") == "success"
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            int(row.get("epochs", "0") or 0),
            float(row.get("map50_95", "") or 0.0),
        ),
        reverse=True,
    )[0]


def resolve_n2_weights(explicit: Path | None) -> Path:
    if explicit:
        weights = explicit.resolve()
        if not weights.exists():
            raise SystemExit(f"N2 weights not found: {weights}")
        return weights
    row = find_success_row(ROUND4_CSV, "N2")
    if not row:
        raise SystemExit("Could not resolve N2 anchor weights from round-4 results.")
    weights = Path(row.get("weights_path", "")).resolve()
    if not weights.exists():
        raise SystemExit(f"Resolved N2 weights are missing: {weights}")
    return weights


def resolve_s2_weights(explicit: Path | None) -> Path:
    if explicit:
        weights = explicit.resolve()
        if not weights.exists():
            raise SystemExit(f"S2 weights not found: {weights}")
        return weights
    row = find_success_row(ROUND12_CSV, "S2")
    if not row:
        raise SystemExit("Could not resolve S2 weights from round-12 results.")
    weights = Path(row.get("weights_path", "")).resolve()
    if not weights.exists():
        raise SystemExit(f"Resolved S2 weights are missing: {weights}")
    return weights


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def write_temp_manifest(payload: dict) -> Path:
    tmp_root = ROOT / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="round13_", dir=str(tmp_root)))
    manifest_path = tmp_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def load_round13_manifest() -> dict:
    return json.loads(ROUND13_MANIFEST.read_text(encoding="utf-8-sig"))


def select_best_slice_row() -> dict[str, str] | None:
    candidates = []
    for exp_id in ("R13-S2a", "R13-S2b"):
        row = find_success_row(ROUND13_CSV, exp_id)
        if row:
            candidates.append(row)
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: float(row.get("map50_95", "") or 0.0), reverse=True)[0]


def build_l1_manifest(best_slice_row: dict[str, str]) -> Path:
    base_manifest = load_round13_manifest()
    best_id = best_slice_row["experiment_id"]
    dataset_yaml = best_slice_row["dataset_yaml"]
    payload = {
        "suite_id": base_manifest["suite_id"],
        "description": base_manifest["description"],
        "results_csv": base_manifest["results_csv"],
        "target_class_names": base_manifest["target_class_names"],
        "dataset_yaml": base_manifest["dataset_yaml"],
        "project": base_manifest["project"],
        "train_defaults": base_manifest["train_defaults"],
        "metadata_defaults": base_manifest["metadata_defaults"],
        "experiments": [
            {
                "experiment_id": "R13-L1",
                "name": f"R13_L1_yolo11n_i800_{slugify(best_id)}_focaler_iou",
                "title": f"Focaler-IoU on {best_id}",
                "ready": True,
                "dataset_yaml": dataset_yaml,
                "metadata": {
                    "augment_profile": f"round13_{slugify(best_id)}_focaler_iou",
                    "dataset_role": f"{best_slice_row.get('dataset_role', 'slice_refinement')}_focaler_iou",
                    "loss_variant": "focaler_iou",
                    "notes": f"Re-runs the best slice candidate {best_id} with Focaler-IoU to improve high-IoU localization quality.",
                },
            }
        ],
    }
    return write_temp_manifest(payload)


def build_hn_slice_dataset(best_slice_row: dict[str, str]) -> Path:
    dataset_root = Path(best_slice_row["dataset_yaml"]).resolve().parent
    output_root = ROOT / "datasets" / "generated_3cls" / f"{dataset_root.name}_plus_hardneg50"
    negative_only_root = ROOT / "datasets" / "generated_3cls" / "aircraft_ai_dataset_3cls_hardneg50_negonly"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "build_combined_yolo_dataset.py"),
        "--reference-yaml",
        str(SOURCE_YAML),
        "--output-root",
        str(output_root),
        "--source-datasets",
        str(dataset_root),
        str(negative_only_root),
    ]
    rc = run(command)
    if rc != 0:
        raise SystemExit(rc)
    return output_root / "data.yaml"


def build_hn_slice_manifest(best_slice_row: dict[str, str]) -> Path:
    dataset_yaml = build_hn_slice_dataset(best_slice_row)
    base_manifest = load_round13_manifest()
    best_id = best_slice_row["experiment_id"]
    payload = {
        "suite_id": base_manifest["suite_id"],
        "description": base_manifest["description"],
        "results_csv": base_manifest["results_csv"],
        "target_class_names": base_manifest["target_class_names"],
        "dataset_yaml": base_manifest["dataset_yaml"],
        "project": base_manifest["project"],
        "train_defaults": base_manifest["train_defaults"],
        "metadata_defaults": base_manifest["metadata_defaults"],
        "experiments": [
            {
                "experiment_id": "R13-HN2",
                "name": f"R13_HN2_yolo11n_i800_{slugify(best_id)}_hardneg50",
                "title": f"Best slice candidate {best_id} plus 50 hard negatives",
                "ready": True,
                "dataset_yaml": str(dataset_yaml),
                "metadata": {
                    "augment_profile": f"round13_{slugify(best_id)}_hardneg50",
                    "dataset_role": f"{best_slice_row.get('dataset_role', 'slice_refinement')}_hard_negative_50",
                    "notes": f"Adds 50 hard negatives to the best slice candidate {best_id} to reduce deployment-time false positives.",
                },
            }
        ],
    }
    return write_temp_manifest(payload)


def select_best_round13_row(exclude_final: bool = True) -> dict[str, str] | None:
    rows = [row for row in read_rows(ROUND13_CSV) if row.get("run_status") == "success"]
    if exclude_final:
        rows = [row for row in rows if not row.get("experiment_id", "").startswith("R13-Final")]
    if not rows:
        return None
    return sorted(rows, key=lambda row: float(row.get("map50_95", "") or 0.0), reverse=True)[0]


def build_final_manifest(best_row: dict[str, str]) -> Path:
    base_manifest = load_round13_manifest()
    dataset_yaml = best_row["dataset_yaml"]
    metadata = {
        "augment_profile": f"round13_final_repro_{slugify(best_row['experiment_id'])}",
        "dataset_role": "round13_final_repro",
        "feature_variant": best_row.get("feature_variant", "default"),
        "backbone_variant": best_row.get("backbone_variant", "yolo11_default"),
        "neck_variant": best_row.get("neck_variant", "default"),
        "loss_variant": best_row.get("loss_variant", "default"),
        "assigner_variant": best_row.get("assigner_variant", "default"),
        "notes": f"Two-seed reproduction of the best completed round13 candidate {best_row['experiment_id']}.",
    }
    experiments = []
    for seed in (42, 2026):
        experiments.append(
            {
                "experiment_id": f"R13-Final-{seed}",
                "name": f"R13_Final_{slugify(best_row['experiment_id'])}_seed{seed}",
                "title": f"Final repro of {best_row['experiment_id']} with seed {seed}",
                "ready": True,
                "dataset_yaml": dataset_yaml,
                "metadata": metadata,
                "overrides": {"seed": seed},
            }
        )
    payload = {
        "suite_id": base_manifest["suite_id"],
        "description": base_manifest["description"],
        "results_csv": base_manifest["results_csv"],
        "target_class_names": base_manifest["target_class_names"],
        "dataset_yaml": base_manifest["dataset_yaml"],
        "project": base_manifest["project"],
        "train_defaults": base_manifest["train_defaults"],
        "metadata_defaults": base_manifest["metadata_defaults"],
        "experiments": experiments,
    }
    return write_temp_manifest(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare, run, and summarize round-13 automation experiments.")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--weights", type=Path, help="Optional explicit N2 weights path.")
    parser.add_argument("--s2-weights", type=Path, help="Optional explicit round12 S2 weights path.")
    parser.add_argument("--negative-images-dir", type=Path, help="Optional standalone normal-surface negative image directory.")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=["E1", "E2", "S2a", "S2b", "L1", "HN1", "Final"],
        help="Only run the given round13 phases.",
    )
    args = parser.parse_args()

    py = sys.executable
    n2_weights = resolve_n2_weights(args.weights)
    s2_weights = resolve_s2_weights(args.s2_weights)

    selected = set(args.only or ["E1", "E2", "S2a", "S2b", "L1", "HN1", "Final"])
    if "L1" in selected:
        selected.update({"S2a", "S2b"})
    if "Final" in selected:
        selected.update({"S2a", "S2b", "L1", "HN1"})

    if not args.skip_prepare:
        stages: list[str] = []
        if {"S2a", "S2b", "L1", "Final"} & selected:
            stages.append("S2a")
            stages.append("S2b")
        if {"HN1", "Final"} & selected:
            stages.append("HN1_BASE")
            stages.append("HN1_NEG_ONLY")
        if stages:
            prepare_cmd = [
                py,
                str(ROOT / "scripts" / "prepare_aircraft_ai_round13.py"),
                "--weights",
                str(n2_weights),
                "--stage",
                *dict.fromkeys(stages),
            ]
            if args.negative_images_dir:
                prepare_cmd.extend(["--negative-images-dir", str(args.negative_images_dir.resolve())])
            rc = run(prepare_cmd)
            if rc != 0:
                return rc

    if "E1" in selected:
        rc = run(
            [
                py,
                str(ROOT / "scripts" / "search_sliced_inference_grid.py"),
                "--weights",
                str(n2_weights),
                "--dataset-yaml",
                str(SOURCE_YAML),
                "--split",
                "valid",
                "--slice-size",
                "512",
                "--overlap",
                "0.20",
                "--imgsz",
                "800",
                "--output-json",
                str(ROOT / "analysis" / "aircraft_ai_dataset_round13_e1_grid.json"),
            ]
        )
        if rc != 0:
            return rc

    if "E2" in selected:
        rc = run(
            [
                py,
                str(ROOT / "scripts" / "evaluate_round13_e2_suite.py"),
                "--n2-weights",
                str(n2_weights),
                "--s2-weights",
                str(s2_weights),
                "--dataset-yaml",
                str(SOURCE_YAML),
                "--split",
                "valid",
                "--slice-size",
                "512",
                "--overlap",
                "0.20",
                "--imgsz",
                "800",
                "--full-conf",
                "0.25",
                "--slice-conf",
                "0.20",
                "--final-conf",
                "0.25",
                "--merge-iou",
                "0.50",
                "--output-json",
                str(ROOT / "analysis" / "aircraft_ai_dataset_round13_e2_suite.json"),
            ]
        )
        if rc != 0:
            return rc

    base_only: list[str] = []
    if "S2a" in selected:
        base_only.append("R13-S2a")
    if "S2b" in selected:
        base_only.append("R13-S2b")
    if "HN1" in selected:
        base_only.append("R13-HN1")
    if base_only:
        rc = run(
            [
                py,
                str(ROOT / "run_experiment_suite.py"),
                "--manifest",
                str(ROUND13_MANIFEST),
                "--only",
                *base_only,
            ]
        )
        if rc != 0:
            return rc

    best_slice_row = select_best_slice_row()
    if "L1" in selected and best_slice_row:
        l1_manifest = build_l1_manifest(best_slice_row)
        rc = run([py, str(ROOT / "run_experiment_suite.py"), "--manifest", str(l1_manifest)])
        if rc != 0:
            return rc

    n2_anchor = find_success_row(ROUND4_CSV, "N2")
    if {"HN1", "Final"} & selected and best_slice_row and n2_anchor:
        if float(best_slice_row.get("map50_95", "") or 0.0) > float(n2_anchor.get("map50_95", "") or 0.0):
            hn_slice_manifest = build_hn_slice_manifest(best_slice_row)
            rc = run([py, str(ROOT / "run_experiment_suite.py"), "--manifest", str(hn_slice_manifest)])
            if rc != 0:
                return rc

    if "Final" in selected:
        best_row = select_best_round13_row(exclude_final=True)
        if best_row:
            final_manifest = build_final_manifest(best_row)
            rc = run([py, str(ROOT / "run_experiment_suite.py"), "--manifest", str(final_manifest)])
            if rc != 0:
                return rc

    return run(
        [
            py,
            str(ROOT / "scripts" / "summarize_aircraft_ai_round13.py"),
            "--round4-csv",
            str(ROUND4_CSV),
            "--round12-csv",
            str(ROUND12_CSV),
            "--round13-csv",
            str(ROUND13_CSV),
            "--e1-json",
            str(ROOT / "analysis" / "aircraft_ai_dataset_round13_e1_grid.json"),
            "--e2-json",
            str(ROOT / "analysis" / "aircraft_ai_dataset_round13_e2_suite.json"),
            "--output-md",
            str(ROOT / "analysis" / "aircraft_ai_dataset_round13_automation_summary.md"),
            "--output-csv",
            str(ROOT / "results" / "aircraft_ai_dataset_round13_summary.csv"),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
