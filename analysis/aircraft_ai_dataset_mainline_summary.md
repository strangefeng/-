# aircraft_ai_dataset_mainline Experiment Summary

## Overview

- Total rows: `7`
- Successful runs: `7`
- Blocked runs: `0`

## Best Current Model

- Experiment: `B0` - Official YOLO11n baseline
- mAP50: `0.753719`
- mAP50-95: `0.350426`
- Precision: `0.792231`
- Recall: `0.701870`
- Run dir: `D:\桌面\毕设\数据集\yolo-baseline-repro\runs\aircraft_ai_mainline\B0_yolo11n_default3`

## Result Table

| exp | title | map50 | map50-95 | precision | recall | missing-rivet | rust | scratch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | Official YOLO11n baseline | 0.753719 | 0.350426 | 0.792231 | 0.701870 | 0.302157 | 0.402333 | 0.346787 |
| B1 | Weak-texture friendly baseline | 0.742231 | 0.345652 | 0.766403 | 0.727914 | 0.310213 | 0.378994 | 0.347750 |
| M3 | WTConv plus NWD loss | 0.707074 | 0.313103 | 0.732761 | 0.672427 | 0.265171 | 0.358487 | 0.315652 |
| M2 | WTConv backbone downsampling | 0.695591 | 0.304053 | 0.751032 | 0.653010 | 0.261491 | 0.356991 | 0.293678 |
| M5 | Final structure with WTConv, NWD and Rep neck | 0.678500 | 0.292802 | 0.707480 | 0.646542 | 0.253565 | 0.325614 | 0.299228 |
| M4 | WTConv plus NWD cooperative supervision | 0.672583 | 0.290965 | 0.715383 | 0.643328 | 0.249878 | 0.316022 | 0.306996 |
| M1 | Neck WT refinement on original dataset | 0.635574 | 0.281306 | 0.706397 | 0.578569 | 0.235247 | 0.324076 | 0.284596 |

## Recommendation

- Recommended baseline reference: `B0` should remain the baseline anchor if it is the official default run.
- Recommended best model candidate: `B0` because it currently has the highest `mAP50-95`.
- Use the gap between the official baseline and the best candidate as the main evidence in the ablation section.
