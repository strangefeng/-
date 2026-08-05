# aircraft_ai_dataset_round2 Experiment Summary

## Overview

- Total rows: `3`
- Successful runs: `3`
- Blocked runs: `0`

## Best Current Model

- Experiment: `R1` - Higher-resolution baseline at 800
- mAP50: `0.751459`
- mAP50-95: `0.351250`
- Precision: `0.778088`
- Recall: `0.706005`
- Run dir: `D:\桌面\毕设\数据集\yolo-baseline-repro\runs\aircraft_ai_round2\R1_yolo11n_i800_default`

## Result Table

| exp | title | map50 | map50-95 | precision | recall | missing-rivet | rust | scratch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | Higher-resolution baseline at 800 | 0.751459 | 0.351250 | 0.778088 | 0.706005 | 0.310734 | 0.396002 | 0.347015 |
| R2 | 800 resolution with conservative defect-friendly augmentation | 0.729644 | 0.337552 | 0.772692 | 0.688576 | 0.303653 | 0.377006 | 0.331997 |
| R3 | 960 resolution with conservative augmentation | 0.733794 | 0.329408 | 0.780804 | 0.681650 | 0.282806 | 0.379838 | 0.325579 |

## Recommendation

- Recommended baseline reference: `R1` should remain the baseline anchor if it is the official default run.
- Recommended best model candidate: `R1` because it currently has the highest `mAP50-95`.
- Use the gap between the official baseline and the best candidate as the main evidence in the ablation section.
