from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Summarize stage-2 aircraft scratch experiments into Markdown and JSON.')
    parser.add_argument('--suite-csv', type=Path, default=Path('results/aircraft_skin_defects_new_mainline.csv'))
    parser.add_argument('--reference-id', default='B0')
    parser.add_argument('--experiment-ids', nargs='+', default=['Exp_A', 'Exp_B', 'Exp_C'])
    parser.add_argument('--output-dir', type=Path, default=Path('analysis/stage2_formal_summary'))
    return parser.parse_args()


def latest_success_row(frame: pd.DataFrame, experiment_id: str) -> pd.Series:
    subset = frame[(frame['run_status'] == 'success') & (frame['experiment_id'] == experiment_id)].copy()
    if subset.empty:
        raise KeyError(f'No successful rows found for experiment_id={experiment_id}')
    subset['epochs_num'] = pd.to_numeric(subset['epochs'], errors='coerce').fillna(-1)
    subset = subset.reset_index(names='row_order')
    subset = subset.sort_values(['epochs_num', 'row_order'], ascending=[False, False])
    return subset.iloc[0]


def metric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_curve_metrics(run_dir: Path) -> dict[str, Any]:
    results_csv = run_dir / 'results.csv'
    if not results_csv.exists():
        return {'results_csv': str(results_csv), 'overfit_flag': False, 'overfit_note': 'results.csv missing'}

    curve = pd.read_csv(results_csv)
    if 'metrics/mAP50-95(B)' not in curve.columns:
        return {'results_csv': str(results_csv), 'overfit_flag': False, 'overfit_note': 'mAP50-95 column missing'}

    curve['epoch'] = pd.to_numeric(curve['epoch'], errors='coerce')
    curve['metrics/mAP50-95(B)'] = pd.to_numeric(curve['metrics/mAP50-95(B)'], errors='coerce')
    curve = curve.dropna(subset=['epoch', 'metrics/mAP50-95(B)'])
    if curve.empty:
        return {'results_csv': str(results_csv), 'overfit_flag': False, 'overfit_note': 'No usable epoch rows'}

    best_idx = curve['metrics/mAP50-95(B)'].idxmax()
    best_row = curve.loc[best_idx]
    last_row = curve.iloc[-1]
    after_50 = curve[curve['epoch'] >= 50]
    overfit_flag = False
    overfit_note = 'No severe post-50 decline detected.'
    if not after_50.empty:
        peak_after_50 = after_50['metrics/mAP50-95(B)'].max()
        last_metric = float(last_row['metrics/mAP50-95(B)'])
        if peak_after_50 > 0 and last_metric < peak_after_50 * 0.9:
            overfit_flag = True
            overfit_note = (
                f"Post-50 validation mAP50-95 dropped from {peak_after_50:.4f} to {last_metric:.4f}; "
                'consider early stopping or stronger regularization.'
            )

    return {
        'results_csv': str(results_csv),
        'best_epoch': int(best_row['epoch']),
        'best_map50_95_curve': float(best_row['metrics/mAP50-95(B)']),
        'last_epoch': int(last_row['epoch']),
        'last_map50_95_curve': float(last_row['metrics/mAP50-95(B)']),
        'overfit_flag': overfit_flag,
        'overfit_note': overfit_note,
    }


def make_markdown_table(rows: list[dict[str, Any]]) -> str:
    header = '| Experiment | Precision | Recall | mAP50 | mAP50-95 | Scratch AP50-95 | Inference (ms) |'
    sep = '|---|---:|---:|---:|---:|---:|---:|'
    body = []
    for row in rows:
        body.append(
            '| {experiment_id} | {precision:.4f} | {recall:.4f} | {map50:.4f} | {map50_95:.4f} | {scratch_ap50_95:.4f} | {speed_inference_ms:.4f} |'.format(**row)
        )
    return '\n'.join([header, sep, *body])


def main() -> int:
    args = parse_args()
    suite_csv = args.suite_csv.resolve()
    frame = pd.read_csv(suite_csv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = latest_success_row(frame, args.reference_id)
    reference_metrics = {
        'map50_95': metric(reference['map50_95']) or 0.0,
        'scratch_ap50_95': metric(reference['scratch_ap50_95']) or 0.0,
    }

    summaries: list[dict[str, Any]] = []
    for experiment_id in args.experiment_ids:
        row = latest_success_row(frame, experiment_id)
        run_dir = Path(str(row['run_dir']))
        summary = {
            'experiment_id': experiment_id,
            'title': str(row['experiment_title']),
            'run_dir': str(run_dir),
            'weights_path': str(row['weights_path']),
            'precision': metric(row['precision']) or 0.0,
            'recall': metric(row['recall']) or 0.0,
            'map50': metric(row['map50']) or 0.0,
            'map50_95': metric(row['map50_95']) or 0.0,
            'scratch_ap50': metric(row['scratch_ap50']) or 0.0,
            'scratch_ap50_95': metric(row['scratch_ap50_95']) or 0.0,
            'speed_inference_ms': metric(row['speed_inference_ms']) or 0.0,
            'delta_vs_ref_map50_95': (metric(row['map50_95']) or 0.0) - reference_metrics['map50_95'],
            'delta_vs_ref_scratch_ap50_95': (metric(row['scratch_ap50_95']) or 0.0) - reference_metrics['scratch_ap50_95'],
        }
        summary.update(load_curve_metrics(run_dir))
        summaries.append(summary)

    summary_by_id = {row['experiment_id']: row for row in summaries}
    synergy = {}
    if {'Exp_A', 'Exp_B', 'Exp_C'}.issubset(summary_by_id):
        a = summary_by_id['Exp_A']
        b = summary_by_id['Exp_B']
        c = summary_by_id['Exp_C']
        scratch_super_additive = (
            a['delta_vs_ref_scratch_ap50_95'] > 0
            and b['delta_vs_ref_scratch_ap50_95'] > 0
            and c['delta_vs_ref_scratch_ap50_95'] > (a['delta_vs_ref_scratch_ap50_95'] + b['delta_vs_ref_scratch_ap50_95'])
        )
        map_super_additive = (
            a['delta_vs_ref_map50_95'] > 0
            and b['delta_vs_ref_map50_95'] > 0
            and c['delta_vs_ref_map50_95'] > (a['delta_vs_ref_map50_95'] + b['delta_vs_ref_map50_95'])
        )
        if a['delta_vs_ref_scratch_ap50_95'] <= 0 or b['delta_vs_ref_scratch_ap50_95'] <= 0:
            interpretation = 'At least one single-factor branch is negative versus the reference baseline, so a strict 1+1>2 claim is not scientifically supported.'
        elif scratch_super_additive:
            interpretation = 'Exp_C exceeded the sum of the individual scratch gains.'
        else:
            interpretation = 'Exp_C improved over part of the single-factor branches, but it did not reach a strict 1+1>2 effect on scratch AP50-95.'
        synergy = {
            'scratch_super_additive': scratch_super_additive,
            'map_super_additive': map_super_additive,
            'scratch_delta_ref': {
                'Exp_A': a['delta_vs_ref_scratch_ap50_95'],
                'Exp_B': b['delta_vs_ref_scratch_ap50_95'],
                'Exp_C': c['delta_vs_ref_scratch_ap50_95'],
            },
            'map_delta_ref': {
                'Exp_A': a['delta_vs_ref_map50_95'],
                'Exp_B': b['delta_vs_ref_map50_95'],
                'Exp_C': c['delta_vs_ref_map50_95'],
            },
            'interpretation': interpretation,
        }

    markdown = []
    markdown.append('# Stage-2 Experiment Summary')
    markdown.append('')
    markdown.append(f"Reference baseline: {args.reference_id} -> {reference['experiment_title']}")
    markdown.append('')
    markdown.append(make_markdown_table(summaries))
    markdown.append('')
    markdown.append('## Overfitting Check')
    markdown.append('')
    for row in summaries:
        markdown.append(
            f"- {row['experiment_id']}: best_epoch={row.get('best_epoch', 'NA')}, best_curve_mAP50-95={row.get('best_map50_95_curve', 0.0):.4f}, "
            f"last_epoch={row.get('last_epoch', 'NA')}, last_curve_mAP50-95={row.get('last_map50_95_curve', 0.0):.4f}, overfit_flag={row['overfit_flag']}. {row['overfit_note']}"
        )
    markdown.append('')
    markdown.append('## Synergy Check')
    markdown.append('')
    if synergy:
        markdown.append(f"- Scratch AP50-95 super-additive: {synergy['scratch_super_additive']}")
        markdown.append(f"- Overall mAP50-95 super-additive: {synergy['map_super_additive']}")
        markdown.append(f"- Interpretation: {synergy['interpretation']}")
        markdown.append(
            '- Scratch deltas vs reference: ' + ', '.join(f"{k}={v:.4f}" for k, v in synergy['scratch_delta_ref'].items())
        )
        markdown.append(
            '- mAP50-95 deltas vs reference: ' + ', '.join(f"{k}={v:.4f}" for k, v in synergy['map_delta_ref'].items())
        )
    else:
        markdown.append('- Missing one or more experiments among Exp_A / Exp_B / Exp_C; synergy check skipped.')

    payload = {
        'reference': {
            'experiment_id': args.reference_id,
            'title': str(reference['experiment_title']),
            'map50_95': reference_metrics['map50_95'],
            'scratch_ap50_95': reference_metrics['scratch_ap50_95'],
        },
        'experiments': summaries,
        'synergy': synergy,
    }

    (output_dir / 'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_dir / 'summary.md').write_text('\n'.join(markdown) + '\n', encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

