# aircraft_ai_dataset_r1_rect Summary

## Main Comparison

| exp | title | map50 | map50-95 | precision | recall | missing-rivet | rust | scratch | rect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | Official YOLO11n baseline | 0.753719 | 0.350426 | 0.792231 | 0.701870 | 0.302157 | 0.402333 | 0.346787 | false |
| R1 | Higher-resolution baseline at 800 | 0.751459 | 0.351250 | 0.778088 | 0.706005 | 0.310734 | 0.396002 | 0.347015 | false |
| R1_rect | R1 baseline with rect=True | 0.582716 | 0.251744 | 0.652743 | 0.546561 | 0.211802 | 0.301809 | 0.241620 | true |

## Key Deltas

- `R1_rect` vs `R1` on `mAP50-95`: `-0.099506`
- `R1_rect` vs `R1` on `missing-rivet AP50-95`: `-0.098932`
- `R1_rect` vs `B0` on `mAP50-95`: `-0.098682`

## Recommendation

- `rect=True` did not provide a clear gain over the current `R1`, so the non-rectangular setup should remain the preferred 800-resolution reference.
