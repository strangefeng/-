# aircraft_ai_dataset_round13_phase2 Summary

## Main Comparison

| exp | title | map50 | map50-95 | precision | recall | missing-rivet | rust | scratch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N2 | R1 without mosaic and erasing | 0.744904 | 0.352565 | 0.782191 | 0.706429 | 0.306116 | 0.401327 | 0.350252 |
| S2 | Sliced train tiles only with 512 edge and 0.20 overlap | 0.759635 | 0.344840 | 0.773454 | 0.733010 | 0.299891 | 0.383286 | 0.351343 |
| R13-L1_N2_FocalerIoU | N2-style full-image training with Focaler-IoU | 0.550361 | 0.254132 | 0.599359 | 0.527490 | 0.250119 | 0.284150 | 0.228126 |
| R13-L2_S2_FocalerIoU | S2 slice512 training with Focaler-IoU | 0.617696 | 0.275201 | 0.695342 | 0.594515 | 0.266297 | 0.313104 | 0.246201 |
| R13-HN1_N2_HardNegative50 | N2-style full-image training with 50 hard negatives | 0.762654 | 0.350875 | 0.785577 | 0.711224 | 0.312158 | 0.390841 | 0.349625 |

## C0 Consistency Check

- N2 official val: map50_95=`0.352222`, delta vs expected=`-0.000343`
- S2 official val: map50_95=`0.345007`, delta vs expected=`+0.000167`

## Hard-Negative FP Audit

- N2 on negative images: total_fp=`3179`, high_conf_fp=`59`, avg_fp_per_image=`63.5800`
- HN1 on negative images: total_fp=`1560`, high_conf_fp=`5`, avg_fp_per_image=`31.2000`

## Best Candidate

- Best phase2 candidate: `R13-HN1_N2_HardNegative50` - N2-style full-image training with 50 hard negatives
- Delta vs N2 on mAP50-95: `-0.001690`
- Delta vs S2 on mAP50-95: `+0.006035`
