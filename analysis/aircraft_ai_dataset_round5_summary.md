# aircraft_ai_dataset_round5 Summary

## Main Comparison

| exp | title | map50 | map50-95 | precision | recall | missing-rivet | rust | scratch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | Official YOLO11n baseline | 0.753719 | 0.350426 | 0.792231 | 0.701870 | 0.302157 | 0.402333 | 0.346787 |
| R1 | Higher-resolution baseline at 800 | 0.751459 | 0.351250 | 0.778088 | 0.706005 | 0.310734 | 0.396002 | 0.347015 |
| N2 | R1 without mosaic and erasing | 0.744904 | 0.352565 | 0.782191 | 0.706429 | 0.306116 | 0.401327 | 0.350252 |
| F2 | Fit-within export with 800-resolution baseline recipe | 0.752022 | 0.349023 | 0.776811 | 0.726855 | 0.308472 | 0.394153 | 0.344443 |
| F4 | Fit-within export with a conservative 640 no-mosaic recipe | 0.743189 | 0.348478 | 0.778227 | 0.709695 | 0.288021 | 0.395914 | 0.361499 |
| F1 | Fit-within export with official 640 baseline recipe | 0.744588 | 0.347596 | 0.780635 | 0.690708 | 0.300100 | 0.391350 | 0.351337 |
| F3 | Fit-within export with the round-4 best N2 recipe | 0.742339 | 0.340085 | 0.795577 | 0.706421 | 0.295574 | 0.384854 | 0.339827 |

## Best New Candidate

- Best round-5 experiment: `F2` - Fit-within export with 800-resolution baseline recipe
- Delta vs `N2` on `mAP50-95`: `-0.003542`
- Delta vs `N2` on `missing-rivet AP50-95`: `+0.002356`
- Delta vs `R1` on `mAP50-95`: `-0.002227`
- Delta vs `B0` on `mAP50-95`: `-0.001403`

## Recommendation

- The fit-within export mainly improved the key small-object class while keeping global accuracy close, so it is worth keeping as a task-oriented data preprocessing choice.
