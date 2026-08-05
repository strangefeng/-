from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROUND4_CSV = ROOT / "results" / "aircraft_ai_dataset_round4.csv"


def run(command: list[str]) -> int:
    return subprocess.call(command, cwd=ROOT)


def resolve_n2_weights(explicit: Path | None) -> Path:
    if explicit:
        weights = explicit.resolve()
        if not weights.exists():
            raise SystemExit(f"N2 weights not found: {weights}")
        return weights
    if not ROUND4_CSV.exists():
        raise SystemExit(f"Round-4 CSV not found: {ROUND4_CSV}")

    best_row: dict[str, str] | None = None
    with ROUND4_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("experiment_id") != "N2" or row.get("run_status") != "success":
                continue
            if best_row is None or float(row.get("map50_95", "0") or 0.0) > float(best_row.get("map50_95", "0") or 0.0):
                best_row = row
    if not best_row:
        raise SystemExit("Could not resolve N2 anchor weights from round-4 results.")
    weights = Path(best_row.get("weights_path", "")).resolve()
    if not weights.exists():
        raise SystemExit(f"Resolved N2 weights are missing: {weights}")
    return weights


def resolve_existing_run_dir(project_dir: Path, run_name: str) -> Path | None:
    exact = project_dir / run_name
    if exact.exists():
        return exact
    candidates = sorted(
        (
            path
            for path in project_dir.iterdir()
            if path.is_dir() and path.name.startswith(run_name)
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def build_resume_manifest(
    base_manifest_path: Path,
    experiment_ids: list[str],
) -> Path:
    manifest = json.loads(base_manifest_path.read_text(encoding="utf-8-sig"))
    project_dir = (ROOT / manifest.get("project", "runs/aircraft_ai_round12_automation")).resolve()
    if manifest.get("dataset_yaml"):
        manifest["dataset_yaml"] = str((ROOT / manifest["dataset_yaml"]).resolve())
    if manifest.get("results_csv"):
        manifest["results_csv"] = str((ROOT / manifest["results_csv"]).resolve())
    if manifest.get("project"):
        manifest["project"] = str(project_dir)

    updated = False
    for experiment in manifest.get("experiments", []):
        if experiment.get("dataset_yaml"):
            experiment["dataset_yaml"] = str((ROOT / experiment["dataset_yaml"]).resolve())
        if experiment.get("experiment_id") not in experiment_ids:
            continue
        run_dir = resolve_existing_run_dir(project_dir, experiment["name"])
        if not run_dir:
            continue
        last_pt = run_dir / "weights" / "last.pt"
        if not last_pt.exists():
            continue
        experiment["model"] = str(last_pt)
        overrides = dict(experiment.get("overrides", {}))
        overrides["resume"] = True
        experiment["overrides"] = overrides
        updated = True

    if not updated:
        return base_manifest_path

    tmp_root = ROOT / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="round12_resume_", dir=str(tmp_root)))
    tmp_manifest = tmp_dir / base_manifest_path.name
    tmp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return tmp_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare, run, and summarize round-12 automation experiments.")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip dataset preparation.")
    parser.add_argument("--skip-s1-eval", action="store_true", help="Skip sliced-inference-only evaluation.")
    parser.add_argument("--weights", type=Path, help="Optional explicit N2 weights path.")
    parser.add_argument("--only", nargs="*", help="Only run the given experiment ids in the suite.")
    parser.add_argument(
        "--resume-existing",
        nargs="*",
        default=[],
        help="Experiment ids that should resume from the latest existing run directory if available.",
    )
    parser.add_argument(
        "--prepare-stage",
        nargs="+",
        default=None,
        help="Optional subset of prepare stages to run instead of all.",
    )
    args = parser.parse_args()

    py = sys.executable
    n2_weights = resolve_n2_weights(args.weights)

    if not args.skip_prepare:
        prepare_cmd = [py, str(ROOT / "scripts" / "prepare_aircraft_ai_round12.py"), "--weights", str(n2_weights)]
        if args.prepare_stage:
            prepare_cmd.extend(["--stage", *args.prepare_stage])
        rc = run(prepare_cmd)
        if rc != 0:
            return rc

    s1_json = ROOT / "analysis" / "aircraft_ai_dataset_round12_s1_sliced_eval.json"
    if not args.skip_s1_eval:
        rc = run(
            [
                py,
                str(ROOT / "scripts" / "evaluate_sliced_inference.py"),
                "--weights",
                str(n2_weights),
                "--dataset-yaml",
                str(ROOT / "datasets" / "supplementary_3cls" / "aircraft_ai_dataset_3cls" / "data.yaml"),
                "--split",
                "valid",
                "--slice-size",
                "512",
                "--overlap",
                "0.20",
                "--imgsz",
                "800",
                "--output-json",
                str(s1_json),
            ]
        )
        if rc != 0:
            return rc

    manifest_path = ROOT / "experiments" / "aircraft_ai_dataset_round12_automation.json"
    if args.resume_existing:
        manifest_path = build_resume_manifest(manifest_path, list(args.resume_existing))

    suite_cmd = [
        py,
        str(ROOT / "run_experiment_suite.py"),
        "--manifest",
        str(manifest_path),
    ]
    if args.only:
        suite_cmd.extend(["--only", *args.only])
    rc = run(suite_cmd)
    if rc != 0:
        return rc

    return run(
        [
            py,
            str(ROOT / "scripts" / "summarize_aircraft_ai_round12.py"),
            "--round4-csv",
            str(ROOT / "results" / "aircraft_ai_dataset_round4.csv"),
            "--round12-csv",
            str(ROOT / "results" / "aircraft_ai_dataset_round12_automation.csv"),
            "--s1-json",
            str(s1_json),
            "--output",
            str(ROOT / "analysis" / "aircraft_ai_dataset_round12_automation_summary.md"),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
