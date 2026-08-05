# aircraft_ai_dataset_round10_methodmatrix Summary

## Main Comparison

| exp | title | dataset_role | map50 | map50-95 | precision | recall | missing-rivet | rust | scratch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N2 | R1 without mosaic and erasing | round4_candidate | 0.744904 | 0.352565 | 0.782191 | 0.706429 | 0.306116 | 0.401327 | 0.350252 |
| F2 | Fit-within export with 800-resolution baseline recipe | fit_within_comparison | 0.752022 | 0.349023 | 0.776811 | 0.726855 | 0.308472 | 0.394153 | 0.344443 |
| G2 | N2 recipe after expanded targeted cleanup | manual_clean_v1_targeted | 0.746027 | 0.347892 | 0.791161 | 0.695136 | 0.298377 | 0.395736 | 0.349563 |
| A1 | Single-point P3-ECA attention on the public N2 anchor | public_3cls | 0.635600 | 0.281828 | 0.680952 | 0.599154 | 0.253353 | 0.322187 | 0.269943 |
| A3 | Single-point P3-ECA attention on the targeted-clean dataset branch | manual_clean_v1_targeted | 0.616645 | 0.276971 | 0.662372 | 0.585842 | 0.248972 | 0.310617 | 0.271324 |
| A2 | Single-point P3-ECA attention on the fit-within export | fit_within_v2 | 0.623761 | 0.274872 | 0.690591 | 0.571269 | 0.246141 | 0.310324 | 0.268151 |

## Best New Candidate

- Best round-10 experiment: `A1` - Single-point P3-ECA attention on the public N2 anchor
- Delta vs `N2` on `mAP50-95`: `-0.070737`
- Delta vs `N2` on `missing-rivet AP50-95`: `-0.052763`
- Delta vs `N2` on `scratch AP50-95`: `-0.080309`

## Recommendation

- The lightweight attention line did not beat the N2 anchor yet, so it should remain a documented ablation direction rather than the final model.
