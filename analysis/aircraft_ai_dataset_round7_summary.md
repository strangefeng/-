# aircraft_ai_dataset_round7 Summary

## Main Comparison

| exp | title | map50 | map50-95 | precision | recall | missing-rivet | rust | scratch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N2 | R1 without mosaic and erasing | 0.744904 | 0.352565 | 0.782191 | 0.706429 | 0.306116 | 0.401327 | 0.350252 |
| C1 | N2 recipe on manually cleaned focus-crop dataset | 0.728683 | 0.337972 | 0.747590 | 0.722142 | 0.289093 | 0.387899 | 0.336926 |
| G2 | N2 recipe after expanded targeted cleanup | 0.746027 | 0.347892 | 0.791161 | 0.695136 | 0.298377 | 0.395736 | 0.349563 |
| G1 | N2 recipe on pilot manually cleaned v1 dataset | 0.735247 | 0.346616 | 0.741623 | 0.716148 | 0.293932 | 0.392676 | 0.353241 |

## Best New Candidate

- Best round-7 experiment: `G2` - N2 recipe after expanded targeted cleanup
- Delta vs `N2` on `mAP50-95`: `-0.004673`
- Delta vs `N2` on `missing-rivet AP50-95`: `-0.007739`
- Delta vs `C1` on `mAP50-95`: `+0.009920`

## Recommendation

- Round 7 did not beat N2 yet, but cleaning alone already outperformed the round-6 cleaned focus-crop branch, so the next investment should be a larger 50-150 image targeted cleanup.
