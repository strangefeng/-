# aircraft_ai_dataset_round13_automation Summary

## Main Comparison

| exp | title | dataset_role | map50 | map50-95 | precision | recall | missing-rivet | rust | scratch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N2 | R1 without mosaic and erasing | round4_candidate | 0.744904 | 0.352565 | 0.782191 | 0.706429 | 0.306116 | 0.401327 | 0.350252 |
| S2 | Sliced train tiles only with 512 edge and 0.20 overlap | slice_train | 0.759635 | 0.344840 | 0.773454 | 0.733010 | 0.299891 | 0.383286 | 0.351343 |
| R13-S2b | Slice512 plus Slice640 mixed training | slice512_640_mix_train | 0.741452 | 0.341361 | 0.791501 | 0.703119 | 0.297033 | 0.377667 | 0.349382 |
| R13-S2a | Slice640 training with 0.20 overlap | slice640_train | 0.725520 | 0.335114 | 0.770644 | 0.697565 | 0.285296 | 0.382908 | 0.337138 |

## E1 Best Run

- Best sliced-inference threshold search: conf=`0.05`, merge_iou=`0.5`
- Metrics: precision=`0.232810`, recall=`0.530213`, map50=`0.312862`, map50_95=`0.114130`

## E2 Comparison

| exp | model | mode | map50-95 | precision | recall |
| --- | --- | --- | --- | --- | --- |
| E2-1 | N2 | full | 0.310946 | 0.673732 | 0.757447 |
| E2-2 | N2 | slice | 0.102519 | 0.404959 | 0.417021 |
| E2-3 | N2 | fused | 0.270185 | 0.491488 | 0.761702 |
| E2-4 | S2 | full | 0.291685 | 0.701600 | 0.746383 |
| E2-5 | S2 | slice | 0.123530 | 0.485637 | 0.460426 |
| E2-6 | S2 | fused | 0.258578 | 0.549517 | 0.774468 |

## Best Round13 Candidate

- Best round-13 candidate: `R13-S2b` - Slice512 plus Slice640 mixed training
- Delta vs `N2` on `mAP50-95`: `-0.011204`
- Delta vs `S2` on `mAP50-95`: `-0.003479`
- Delta vs `N2` on `missing-rivet AP50-95`: `-0.009083`
- Delta vs `N2` on `rust AP50-95`: `-0.023660`
- Delta vs `N2` on `scratch AP50-95`: `-0.000870`

## Recommendation

- Round 13 did not beat either N2 or S2 clearly enough, so N2 remains the final mainline reference.
