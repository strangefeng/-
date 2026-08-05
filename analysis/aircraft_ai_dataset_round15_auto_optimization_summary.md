# Round15 Auto Optimization Summary

## Baseline

| Model / mode | Precision | Recall | mAP50 | mAP50-95 | Missing-rivet AP50-95 | Rust AP50-95 | Scratch AP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N2 standard val | 0.782191 | 0.706429 | 0.744904 | 0.352565 | 0.306116 | 0.401327 | 0.350252 |

## Completed Training Runs

| Run | Main idea | Precision | Recall | mAP50 | mAP50-95 | Delta vs N2 | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| R15-C1_CleanV3_N2 | targeted manual-clean v3 + N2 recipe | 0.735496 | 0.683358 | 0.705941 | 0.325426 | -0.027139 | Reject for main metric |
| R15-C3_MRdupX2_N2 | conservative missing-rivet rebalance | 0.760870 | 0.701425 | 0.709911 | 0.325512 | -0.027053 | Reject for main metric |
| R15-C4_N2FinetuneLowLR | continue N2 checkpoint with low LR | 0.791924 | 0.695795 | 0.741218 | 0.350605 | -0.001960 | Close, but below N2 |

`R15-C2_FitWithinV2_N2` did not complete because its dataset lacked `labels.cache` and the sandbox blocked Ultralytics cache generation. It should be retried outside the sandbox or after cache generation.

## Successful Gain

| Model / mode | Precision | Recall | mAP50 | mAP50-95 | Delta vs N2 | Missing-rivet AP50-95 | Rust AP50-95 | Scratch AP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N2 + TTA (`augment=True`) | 0.776161 | 0.707687 | 0.742319 | 0.355628 | +0.003063 | 0.314386 | 0.400205 | 0.352294 |
| C4 + TTA (`augment=True`) | 0.773754 | 0.698279 | 0.739970 | 0.354743 | +0.002178 | 0.322402 | 0.402686 | 0.339141 |
| N2 + TTA + `iou=0.650` | 0.781047 | 0.714087 | 0.751070 | 0.355834 | +0.003269 | 0.316252 | 0.400277 | 0.350973 |
| N2 + TTA + `iou=0.675` | 0.781335 | 0.711758 | 0.749230 | 0.356065 | +0.003500 | 0.316547 | 0.400049 | 0.351599 |
| N2 + TTA + `iou=0.685` | 0.781094 | 0.709158 | 0.747680 | 0.356407 | +0.003842 | 0.316509 | 0.400238 | 0.352474 |
| C4 + TTA + `iou=0.685` | 0.777789 | 0.703632 | 0.743012 | 0.355437 | +0.002872 | 0.322641 | 0.403403 | 0.340267 |

Best current round15 result: `N2 + TTA + iou=0.685`, mAP50-95 `0.356407`.

Additional probes that did not beat the best setting:

| Probe | mAP50-95 | Note |
| --- | ---: | --- |
| N2 + TTA, imgsz 960 | 0.353340 | Higher resolution helped scratch but hurt overall balance |
| N2 + TTA, imgsz 1024 | 0.348830 | Too large for this trained anchor |
| N2 + TTA + `iou=0.600` | 0.354679 | mAP50 improved, mAP50-95 below the peak |
| N2 + TTA + `iou=0.800` | 0.345751 | Too many duplicate boxes survive NMS |
| N2 + TTA + `iou=0.625` | 0.355155 | Below the local peak |
| N2 + TTA + `iou=0.690` | 0.355612 | Slightly below `0.685` |
| N2 + TTA + `iou=0.695` | 0.355457 | Slightly below `0.685` |

## Practical Recommendation

Use `N2` as the model weights and report two inference settings:

1. Standard inference: mAP50-95 `0.352565`.
2. TTA inference: mAP50-95 `0.355628`, a gain of `+0.003063`.
3. TTA with tuned NMS: mAP50-95 `0.356407`, a gain of `+0.003842`.

This is a real validation gain without changing the training data or architecture. The cost is slower inference: approximately `2.4 ms` standard inference versus about `9.1 ms` to `22.3 ms` with TTA depending on the validation path used.

## Reproduction Command

```powershell
$env:YOLO_CONFIG_DIR = "D:\桌面\毕设\数据集\yolo-baseline-repro\.yolo-config"
& 'D:\yolo11\.conda\Scripts\yolo.exe' detect val `
  model='.\yolo-baseline-repro\runs\aircraft_ai_round4\N2_yolo11n_i800_no_mosaic\weights\best.pt' `
  data='.\yolo-baseline-repro\datasets\supplementary_3cls\aircraft_ai_dataset_3cls\data.yaml' `
  imgsz=800 device=0 workers=0 augment=True iou=0.685 `
  project='.\yolo-baseline-repro\runs\aircraft_ai_round15_github_uav' `
  name='R15_TTA_N2' exist_ok=True
```

Precise metrics JSON:

- `analysis/aircraft_ai_dataset_round15_tta_n2.json`
- `analysis/aircraft_ai_dataset_round15_tta_c4.json`
- `analysis/aircraft_ai_dataset_round15_tta_n2_i800_iou0685.json`
- `analysis/aircraft_ai_dataset_round15_tta_c4_i800_iou0685.json`

## Next Optimization Ideas

The next experiments should avoid broad architecture churn and focus on small, testable changes:

1. Retry `R15-C2_FitWithinV2_N2` outside the sandbox or after prebuilding `labels.cache`.
2. Validate `N2 + TTA + iou=0.685` on the test split to check whether the validation gain transfers. This was attempted, but the test split has no `labels.cache` and the sandbox blocked Ultralytics cache generation.
3. Try prediction-time class-specific thresholds for deployment visuals only; keep official mAP reporting threshold-free.
4. If training continues, use N2 as pretrained weights but reduce augmentation further for scratch preservation instead of switching datasets.
