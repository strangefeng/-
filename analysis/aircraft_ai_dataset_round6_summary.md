# aircraft_ai_dataset_round6 Summary

## Main Comparison

| exp | title | map50 | map50-95 | precision | recall | missing-rivet | rust | scratch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N2 | R1 without mosaic and erasing | 0.744904 | 0.352565 | 0.782191 | 0.706429 | 0.306116 | 0.401327 | 0.350252 |
| F2 | Fit-within export with 800-resolution baseline recipe | 0.752022 | 0.349023 | 0.776811 | 0.726855 | 0.308472 | 0.394153 | 0.344443 |
| C1 | N2 recipe on manually cleaned focus-crop dataset | 0.728683 | 0.337972 | 0.747590 | 0.722142 | 0.289093 | 0.387899 | 0.336926 |
| S1 | N2 recipe on v1 focus-crop augmented dataset | 0.733502 | 0.335786 | 0.787041 | 0.713782 | 0.277317 | 0.382957 | 0.347084 |
| S3 | Focus-crop v1 with longer N2 schedule | 0.715093 | 0.335490 | 0.774253 | 0.668980 | 0.289412 | 0.386825 | 0.330234 |
| S2 | N2 recipe on fit-within focus-crop augmented dataset | 0.721565 | 0.330668 | 0.775807 | 0.691442 | 0.273346 | 0.379769 | 0.338890 |

## Best New Candidate

- Best round-6 experiment: `C1` - N2 recipe on manually cleaned focus-crop dataset
- Delta vs `N2` on `mAP50-95`: `-0.014593`
- Delta vs `N2` on `missing-rivet AP50-95`: `-0.017023`
- Delta vs `F2` on `mAP50-95`: `-0.011051`

## Recommendation

- The focus-crop branch did not beat N2 clearly enough yet, so manual cleaning should become the next priority before more training-only ablations.
