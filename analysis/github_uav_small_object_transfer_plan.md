# GitHub UAV Small-Object Transfer Plan

## Search Scope

I looked for GitHub cases around UAV/VisDrone/UAVDT object detection, especially YOLO-based small-object work. The useful recurring ideas are:

- higher effective resolution for small targets
- P2 or extra small-object detection head
- attention modules such as ECA/CBAM/coordinate attention
- sliced training or sliced inference
- hard-negative mining for texture-like false positives
- class rebalance for rare small targets

Representative GitHub cases from the VisDrone topic include:

- `fcakyon/small-object-detection-benchmark`: SAHI benchmark on VisDrone/xView.
- `xuanandsix/VisDrone-yolov8`: YOLOv8 models trained on VisDrone.
- `Mohd-Abdul-Rafay/small-object-detection-visdrone`: YOLOv8-L with EMA, P2 head, PIoU, and SAHI evaluation.
- `irfan112/YOLO-TLP`: tiny-object oriented YOLO work for objects smaller than 15 px.

Source index: https://github.com/topics/visdrone

## Local Evidence Check

Your experiment history already covers many of these ideas:

| UAV idea | Local status | Decision |
| --- | --- | --- |
| Higher resolution / no destructive augmentation | `N2`, YOLO11n, imgsz 800, no mosaic/erasing, is still the anchor: mAP50-95 `0.352565` | Keep as main recipe |
| Sliced inference | Round12/13 showed sliced-only and fused inference are below N2 | Do not use as the final inference trick |
| Sliced training | `S2` got close but still below N2 | Keep only as secondary, not the main line |
| P2 / attention / new modules | Existing custom models include P2/ECA/DWT variants and prior rounds did not beat N2 | Do not spend the next run here |
| Focaler-IoU / loss-only tweak | Round13 phase2 dropped strongly below N2 | Stop this branch unless a new stable implementation is added |
| Hard negatives | HN50 helped deployment false positives but did not beat N2 on standard validation | Use for deployment robustness, not paper main metric |
| Label cleaning | Round14 identified label quality as the main bottleneck | Highest-priority remaining line |

## Applied Round15 Candidates

I added `experiments/aircraft_ai_dataset_round15_github_uav.json` with three ready-to-run candidates:

1. `R15-C1_CleanV3_N2`
   - Uses `manual_clean_v3_targeted90`.
   - This is the best chance for a real mAP gain because it attacks the current bottleneck rather than adding more architecture.

2. `R15-C2_FitWithinV2_N2`
   - Uses the fit-within detail-preserving dataset.
   - This is the cleanest UAV small-object transfer: preserve small target pixels without changing the network.

3. `R15-C3_MRdupX2_N2`
   - Uses conservative missing-rivet rebalance.
   - This tests rare-class recovery without synthetic copy-paste noise.

## Run Command

From `D:/桌面/毕设/数据集/yolo-baseline-repro`:

```powershell
$env:YOLO_CONFIG_DIR = "$PWD\.yolo-config"
& 'D:\yolo11\.conda\python.exe' .\run_experiment_suite.py --manifest .\experiments\aircraft_ai_dataset_round15_github_uav.json
```

## Expected Outcome

The realistic target is to beat `N2` by a small but defensible amount:

- baseline to beat: mAP50-95 `0.352565`
- useful paper gain: `+0.003` or more
- strong gain for this dataset: `+0.008` or more

If none of the three candidates beats N2, the clean conclusion is still useful: your strongest result is not from open-source UAV tricks, but from a carefully validated N2-style recipe plus dataset-quality control.
