# aircraft_ai_dataset_round4 Summary

## Main Comparison

| exp | title | map50 | map50-95 | precision | recall | missing-rivet | rust | scratch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | Official YOLO11n baseline | 0.753719 | 0.350426 | 0.792231 | 0.701870 | 0.302157 | 0.402333 | 0.346787 |
| R1 | Higher-resolution baseline at 800 | 0.751459 | 0.351250 | 0.778088 | 0.706005 | 0.310734 | 0.396002 | 0.347015 |
| N2 | R1 without mosaic and erasing | 0.744904 | 0.352565 | 0.782191 | 0.706429 | 0.306116 | 0.401327 | 0.350252 |
| N1 | R1 with weak defect-friendly augmentation | 0.739670 | 0.337809 | 0.783339 | 0.692644 | 0.299122 | 0.378733 | 0.335573 |
| N3 | R1 with weak augmentation and no mosaic | 0.740463 | 0.336018 | 0.779842 | 0.721114 | 0.281455 | 0.393747 | 0.332853 |
| N4 | N3 with a longer training schedule | 0.721463 | 0.327680 | 0.776702 | 0.692137 | 0.281065 | 0.373352 | 0.328622 |

## Best New Candidate

- Best round-4 experiment: `N2` - R1 without mosaic and erasing
- Delta vs `R1` on `mAP50-95`: `+0.001315`
- Delta vs `R1` on `missing-rivet AP50-95`: `-0.004618`
- Delta vs `B0` on `mAP50-95`: `+0.002139`

## Recommendation

- A training-strategy variant surpassed `R1`, so the next anchor should move from the raw high-resolution baseline to this new configuration.
