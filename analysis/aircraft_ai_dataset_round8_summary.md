# aircraft_ai_dataset_round8 Summary

## Main Comparison

| exp | title | map50 | map50-95 | precision | recall | missing-rivet | rust | scratch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N2 | R1 without mosaic and erasing | 0.744904 | 0.352565 | 0.782191 | 0.706429 | 0.306116 | 0.401327 | 0.350252 |
| G2 | N2 recipe after expanded targeted cleanup | 0.746027 | 0.347892 | 0.791161 | 0.695136 | 0.298377 | 0.395736 | 0.349563 |
| C1 | N2 recipe on manually cleaned focus-crop dataset | 0.728683 | 0.337972 | 0.747590 | 0.722142 | 0.289093 | 0.387899 | 0.336926 |
| H2 | N2 recipe after 90-sample targeted train-only cleanup | 0.730463 | 0.339661 | 0.798344 | 0.690860 | 0.295636 | 0.379195 | 0.344151 |
| H1 | N2 recipe after 72-sample targeted train-only cleanup | 0.719583 | 0.339191 | 0.767231 | 0.683020 | 0.302854 | 0.379935 | 0.334783 |

## Best New Candidate

- Best round-8 experiment: `H2` - N2 recipe after 90-sample targeted train-only cleanup
- Delta vs `G2` on `mAP50-95`: `-0.008231`
- Delta vs `G2` on `missing-rivet AP50-95`: `-0.002741`
- Delta vs `G2` on `scratch AP50-95`: `-0.005412`
- Delta vs `N2` on `mAP50-95`: `-0.012904`

## Recommendation

- Round 8 still failed to deliver a real gain over G2, so the public 3-class dataset is likely saturated and should not receive further cleanup-driven iterations.
