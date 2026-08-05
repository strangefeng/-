# aircraft_ai_dataset_round16_aerodefect Summary

- N2 standard validation mAP50-95: `0.352565`
- 1-point target: `0.362565`
- 2-point target: `0.372565`
- current target status: `not_reached`

## Main Matrix

| exp | title | mAP50 | mAP50-95 | delta vs N2 | precision | recall | missing-rivet | rust | scratch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | SPD detail-preserving stem under the N2 recipe | 0.624349 | 0.275601 | -0.076964 | 0.652442 | 0.605724 | 0.238004 | 0.317250 | 0.271550 |
| A2 | Plain P2 small-object head control under the N2 recipe | 0.556663 | 0.239904 | -0.112661 | 0.638549 | 0.511004 | 0.206843 | 0.273087 | 0.239782 |
| A3 | SPD stem plus defect-aware morphology P2 branch | 0.581045 | 0.265322 | -0.087243 | 0.597989 | 0.561007 | 0.232159 | 0.313169 | 0.250640 |
| A4 | Full AeroDefect-YOLO with dynamic multi-scale fusion | 0.569829 | 0.244916 | -0.107649 | 0.627144 | 0.536638 | 0.201089 | 0.281511 | 0.252150 |

## Recommendation

- Continue the matrix. If A4 still misses the +0.01 target, the next branch should tune the defect-aware P2 gate or add a clean-train/hard-negative combination rather than broad architecture churn.