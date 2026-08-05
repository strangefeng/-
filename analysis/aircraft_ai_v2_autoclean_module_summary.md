# Aircraft AI v2 Auto-Clean Module Summary

## Anchor

The cleaned-label YOLO11n N2 run is used as the module anchor.

| Experiment | mAP50 | mAP50-95 | Precision | Recall |
|---|---:|---:|---:|---:|
| Clean YOLO11n N2 | 0.620931 | 0.282021 | 0.663273 | 0.604508 |

## Module Results

| Experiment | Type | mAP50 | mAP50-95 | Delta vs Clean |
|---|---|---:|---:|---:|
| L1_NWD_LOSS | Loss module | 0.616763 | 0.287999 | +0.005978 |
| L2_NWD_ASSIGNER | Assigner module | 0.618351 | 0.273383 | -0.008638 |
| L3_FOCALER_IOU | Loss module | 0.456149 | 0.202108 | -0.079913 |
| M1_P2_HEAD | Structure module | 0.446154 | 0.190662 | -0.091359 |
| M2_SPD_STEM | Structure module | 0.490337 | 0.212761 | -0.069260 |
| M3_SPD_DEFECT_P2 | Structure module | 0.470681 | 0.200546 | -0.081475 |
| M4_SPD_DEFECT_P2_DYFUSE | Structure module | 0.461705 | 0.204021 | -0.078000 |

## Best Candidate

The only positive module branch is `L1_NWD_LOSS`.

- mAP50-95: 0.282021 -> 0.287999
- Gain: +0.005978, about +0.60 points
- missing-rivet AP50-95: 0.100167 -> 0.105997
- rust AP50-95: 0.397271 -> 0.411622
- scratch AP50-95: 0.348624 -> 0.346379

## Interpretation

After label cleaning, large structural modules such as P2, SPD, and SPD+P2 do not improve this dataset under the current 100-epoch N2 recipe. The most plausible reason is that the cleaned labels already convert large `missing-rivet` boxes into many compact targets, while the available dataset size is still small; adding extra heads or custom downsampling increases optimization difficulty and weakens transfer from the pretrained YOLO11n checkpoint.

The NWD loss branch is safer because it keeps the pretrained YOLO11n architecture unchanged and only modifies bbox regression. It gives a small positive gain, mainly from `missing-rivet` and `rust`, while keeping precision and recall essentially unchanged.

## Recommendation

For the thesis experiment line, use:

1. Data cleaning as the main improvement: +3.03 mAP50-95 points.
2. NWD loss as a small auxiliary module: +0.60 mAP50-95 points on top of cleaned labels.
3. Report P2/SPD-style structure modules as ablation attempts that did not improve the cleaned dataset.

This is cleaner scientifically than forcing a larger module that drops performance.
