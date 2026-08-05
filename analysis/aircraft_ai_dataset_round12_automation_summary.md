# aircraft_ai_dataset_round12_automation Summary

## Main Comparison

| exp | title | dataset_role | map50 | map50-95 | precision | recall | missing-rivet | rust | scratch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N2 | R1 without mosaic and erasing | round4_candidate | 0.744904 | 0.352565 | 0.782191 | 0.706429 | 0.306116 | 0.401327 | 0.350252 |
| S1 | N2 weights with sliced inference only | sliced_inference_only | 0.332006 | 0.119878 | 0.035441 | 0.694468 | 0.142417 | 0.108440 | 0.108777 |
| S2 | Sliced train tiles only with 512 edge and 0.20 overlap | slice_train | 0.759635 | 0.344840 | 0.773454 | 0.733010 | 0.299891 | 0.383286 | 0.351343 |
| S3 | Mixed original plus sliced training with 512 edge and 0.20 overlap | slice_mixed_train | 0.746223 | 0.342361 | 0.802500 | 0.709428 | 0.294288 | 0.388442 | 0.344354 |
| P2 | Missing-rivet plus scratch patch augmentation | patch_missing_rivet_scratch | 0.740337 | 0.342049 | 0.779114 | 0.730397 | 0.289881 | 0.395445 | 0.340821 |

## Best New Candidate

- Best round-12 candidate: `S2` - Sliced train tiles only with 512 edge and 0.20 overlap
- Delta vs `N2` on `mAP50-95`: `-0.007725`
- Delta vs `N2` on `missing-rivet AP50-95`: `-0.006225`
- Delta vs `N2` on `scratch AP50-95`: `+0.001091`

## Recommendation

- None of the round-12 data-information branches beat the N2 anchor clearly enough, so N2 should remain the final mainline reference.
