# Round16 AeroDefect-YOLO SCI Experiment Plan

## Objective

Use the N2 standard validation result as the fixed anchor:

- N2 mAP50-95: `0.352565`
- 1-point target: `0.362565`
- 2-point target: `0.372565`

The Round16 goal is to find a structurally defensible SCI candidate that improves the N2 standard validation result by at least `+0.010000` and preferably `+0.020000`.

## Main Hypothesis

Previous rounds showed that bigger models, generic attention, plain P2, slicing, Focaler-IoU, and simple rebalancing did not reliably beat N2. Therefore, the next candidate should not add generic complexity. It should preserve local pixels early and add an aircraft-defect-specific high-resolution branch.

## Experiment Matrix

| ID | Variant | Purpose | Promotion Gate |
| --- | --- | --- | --- |
| A0 | N2 anchor | Fixed standard baseline from Round4 | Reference only |
| A1 | SPD stem | Tests whether space-to-depth early downsampling preserves micro-defect cues | Beat N2 or improve missing-rivet without hurting total mAP heavily |
| A2 | Plain P2 control | Re-validates whether generic P2 alone is unstable | Mainly a control; not expected to win |
| A3 | SPD + defect-aware P2 | Main tiny-defect contribution with morphology gate | Beat A1/A2 and improve missing-rivet or scratch |
| A4 | SPD + defect-aware P2 + dynamic fusion | Full AeroDefect-YOLO candidate | Target model if it reaches +0.01 over N2 |
| A5 | Best A-model + TTA/iou sweep | Deployment-stage enhancement only | Report separately from standard validation |

## Implemented Components

- `SPDConv`: space-to-depth downsampling followed by convolution.
- `DefectMorphGate`: P2 morphology gate with round, horizontal, vertical, and diffuse branches.
- `WeightedConcat`: learnable scalar weighting before multi-scale neck concatenation.

## Files

- Manifest: `experiments/aircraft_ai_dataset_round16_aerodefect.json`
- Runner: `scripts/run_aircraft_ai_round16.py`
- Summarizer: `scripts/summarize_aircraft_ai_round16.py`
- Results CSV: `results/aircraft_ai_dataset_round16_aerodefect.csv`
- Summary: `analysis/aircraft_ai_dataset_round16_aerodefect_summary.md`
- Models:
  - `custom_yolo/models/yolo11n_spd.yaml`
  - `custom_yolo/models/yolo11n_p2.yaml`
  - `custom_yolo/models/yolo11n_spd_defect_p2.yaml`
  - `custom_yolo/models/yolo11n_spd_defect_p2_dyfuse.yaml`

## Continue-Until-Win Rule

After each completed experiment, run the summarizer and compare against N2.

If A4 does not reach `0.362565`, continue with the smallest targeted next branch instead of broad architecture churn:

1. A4b: keep A4, reduce `DefectMorphGate` reduction from `4` to `2`.
2. A4c: keep A4, apply SPDConv at the P3 downsample only instead of both stem layers if A1/A3 show overfitting.
3. A4d: combine the best A-model with the hard-negative branch if mAP is close but deployment false positives remain high.
4. A4e: retry clean-train labels only after QA2 manual correction is complete.

The paper candidate should be promoted only when standard validation, not TTA-only validation, exceeds the N2 anchor by at least one point.
