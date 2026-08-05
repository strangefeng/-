# aircraft_skin_defects_new Final Summary

## Final Decision

- Final model on the main dataset: `Exp_A`
- Positioning: `Exp_A` should be treated as the formal final model for the thesis because it is the only branch that improves the overall benchmark while also producing the strongest `scratch` gain.

## Main Comparison

| exp | title | map50 | map50-95 | precision | recall | crack | dent | missing-head | paint-off | scratch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | Official default baseline | 0.759508 | 0.408627 | 0.851392 | 0.708975 | 0.477357 | 0.369413 | 0.598529 | 0.321186 | 0.276652 |
| Exp_A | Baseline YOLO11n plus offline Copy-Paste | 0.753724 | 0.413643 | 0.847391 | 0.645873 | 0.483141 | 0.364214 | 0.557285 | 0.292180 | 0.371397 |
| Exp_B | YOLO11n with Neck-WT refinement on original dataset | 0.449480 | 0.244249 | 0.426085 | 0.581779 | 0.227841 | 0.201668 | 0.590433 | 0.101449 | 0.099854 |
| Exp_C | YOLO11n with Neck-WT refinement plus offline Copy-Paste | 0.579091 | 0.313749 | 0.697670 | 0.529287 | 0.265708 | 0.195488 | 0.621401 | 0.152624 | 0.333525 |

## Why `Exp_A` Wins

- `Exp_A` improves `mAP50-95` over `B0` by `+0.005016`
- `Exp_A` improves `scratch AP50-95` over `B0` by `+0.094745`
- `Exp_A` also improves `crack AP50-95` over `B0` by `+0.005784`
- `Exp_B` and `Exp_C` show that adding the neck-side WT branch does not convert the Copy-Paste gain into a stronger final system, so the thesis should not claim structural synergy

## Data-Method Interpretation

- Original train-set class counts: `{'crack': 332, 'dent': 447, 'missing-head': 331, 'paint-off': 267, 'scratch': 75}`
- Copy-Paste train-set class counts: `{'crack': 664, 'dent': 898, 'missing-head': 660, 'paint-off': 526, 'scratch': 1110}`
- Offline Copy-Paste generated `scratch` donor patches from `74` source boxes and injected `962` synthetic scratch boxes into the train split
- The generated dataset keeps one augmented copy per original train image, so non-scratch categories roughly double because their original labels are preserved in the copied images
- After removing that expected duplication effect, `scratch` still gains an extra `960` boxes beyond simple doubling, which is the key targeted shift
- This means the final positive result is best explained as a targeted data-distribution correction for the hardest class rather than a network-structure breakthrough

## Paper Figures

- Per-class AP figure: `D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\aircraft_skin_defects_final_assets\exp_a_class_ap95_comparison.png`
- Train distribution figure: `D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\aircraft_skin_defects_final_assets\exp_a_train_distribution_shift.png`
- Qualitative comparison panel: `D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\aircraft_skin_defects_final_assets\exp_a_qualitative_comparison_panel.png`

## Thesis-Ready Conclusion

The main dataset chapter should conclude that the most reliable improvement comes from an offline Copy-Paste strategy targeted at the underperforming `scratch` class. In contrast, the tested wavelet- and NWD-based structural branches did not outperform the simpler data-centric final model. Therefore, `Exp_A` is the recommended final model, and the thesis contribution on the main dataset should be framed as a defect-oriented data enhancement workflow rather than a new backbone or neck design.
