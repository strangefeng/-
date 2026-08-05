# aircraft_ai_dataset_round3 Comparison Summary

## Included Models

- `B0`: round-1 official baseline anchor
- `R1`: round-2 high-resolution anchor
- `T1/T2`: round-3 newly trained experiments

## Main Comparison

| exp | title | map50 | map50-95 | precision | recall | missing-rivet | rust | scratch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | Official YOLO11n baseline | 0.753719 | 0.350426 | 0.792231 | 0.701870 | 0.302157 | 0.402333 | 0.346787 |
| R1 | Higher-resolution baseline at 800 | 0.751459 | 0.351250 | 0.778088 | 0.706005 | 0.310734 | 0.396002 | 0.347015 |
| T1 | R1 + P2 small-object detection head | 0.547129 | 0.236019 | 0.603093 | 0.539379 | 0.219471 | 0.276375 | 0.212213 |

## Round-3 Best vs Anchors

- Best round-3 candidate: `T1` - R1 + P2 small-object detection head
- Delta vs `R1` on `mAP50-95`: `-0.115231`
- Delta vs `R1` on `missing-rivet AP50-95`: `-0.091263`
- Delta vs `B0` on `mAP50-95`: `-0.114407`

## Recommendation

- Round 3 did not surpass the anchors clearly enough; the current evidence still favors `B0/R1` as the most stable references.
