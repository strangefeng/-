# aircraft_ai_dataset_round14 Summary

## Main Comparison

| exp | title | map50 | map50-95 | precision | recall | missing-rivet | rust | scratch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N2 | R1 without mosaic and erasing | 0.744904 | 0.352565 | 0.782191 | 0.706429 | 0.306116 | 0.401327 | 0.350252 |
| R13-HN1_N2_HardNegative50 | N2-style full-image training with 50 hard negatives | 0.762654 | 0.350875 | 0.785577 | 0.711224 | 0.312158 | 0.390841 | 0.349625 |
| R14-M1_YOLO11s_N2 | YOLO11s upper-bound validation under the N2 recipe | 0.734911 | 0.344610 | 0.796162 | 0.686438 | 0.297802 | 0.391330 | 0.344699 |

## C0 Consistency Check

- N2 official val map50-95: `0.352222` (delta vs historical N2 `-0.000343`)

## QA1 Audit

- Audit candidate count: `420`
- Train fix recommended count: `296`
- Valid record-only count: `124`
- Suggested clean_train branch: `yes`

## Deployment Negative Set

- `N2`: total_fp=`3179`, high_conf_fp=`59`, avg_fp_per_image=`63.5800`
- `R13-HN1_N2_HardNegative50`: total_fp=`1560`, high_conf_fp=`5`, avg_fp_per_image=`31.2000`
- `R14-M1_YOLO11s_N2`: total_fp=`2749`, high_conf_fp=`35`, avg_fp_per_image=`54.9800`

## Blocked / Skipped

- `R14-HN2_N2_HardNegative150`: Skipped because no curated train-only negative image directory was available.
- `R14-HN3_N2_HardNegative300`: Skipped because no curated train-only negative image directory was available.
- `R14-M2_YOLO11s_HN150`: Skipped because neither YOLO11s nor HN150 met the round14 trigger condition.
- `R14-QA2_N2_CleanTrain`: QA1 suggests a clean-train branch, but QA2 remains pending manual label correction and a reviewed label_fix_log.

## Best Round14 Candidate

- Best completed candidate: `R14-M1_YOLO11s_N2` - YOLO11s upper-bound validation under the N2 recipe
- Delta vs N2 on mAP50-95: `-0.007955`
- Delta vs HN1 on mAP50-95: `-0.006265`
