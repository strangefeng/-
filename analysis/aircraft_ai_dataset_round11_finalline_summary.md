# aircraft_ai_dataset_round11_finalline Summary

## Main Comparison

| exp | title | dataset_role | map50 | map50-95 | precision | recall | missing-rivet | rust | scratch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N2 | R1 without mosaic and erasing | round4_candidate | 0.744904 | 0.352565 | 0.782191 | 0.706429 | 0.306116 | 0.401327 | 0.350252 |
| G2 | N2 recipe after expanded targeted cleanup | manual_clean_v1_targeted | 0.746027 | 0.347892 | 0.791161 | 0.695136 | 0.298377 | 0.395736 | 0.349563 |
| O3 | Targeted-clean branch with missing-rivet train-image oversampling x2 | manual_clean_v1_targeted_mrdup_x2 | 0.733527 | 0.341244 | 0.792567 | 0.697944 | 0.294499 | 0.388215 | 0.341017 |
| O1 | Public branch with missing-rivet train-image oversampling x2 | public_mrdup_x2 | 0.731076 | 0.337595 | 0.762102 | 0.710468 | 0.296003 | 0.388583 | 0.328200 |
| O2 | Public branch with missing-rivet train-image oversampling x3 | public_mrdup_x3 | 0.728278 | 0.334344 | 0.791509 | 0.689062 | 0.285781 | 0.375625 | 0.341626 |

## Best New Candidate

- Best round-11 experiment: `O3` - Targeted-clean branch with missing-rivet train-image oversampling x2
- Delta vs `N2` on `mAP50-95`: `-0.011321`
- Delta vs `N2` on `missing-rivet AP50-95`: `-0.011617`
- Delta vs `N2` on `rust AP50-95`: `-0.013112`
- Delta vs `N2` on `scratch AP50-95`: `-0.009235`

## Recommendation

- Missing-rivet oversampling did not beat the N2 anchor clearly enough, so N2 should remain the final mainline reference.
