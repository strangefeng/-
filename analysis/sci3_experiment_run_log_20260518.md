# SCI-Q3 Experiment Run Log

Date: 2026-05-18

## Current evidence chain

The current publishable story should be framed as:

1. A cleaned three-class UAV aircraft skin-defect benchmark is used as the controlled evaluation bed.
2. YOLO11n is a stronger lightweight baseline than YOLOv8n on the clean split.
3. NWD loss improves overall localization quality with no model-size increase.
4. NWD assigner is not a universal winner, but it improves the tiny/small-object subset and missing-rivet behavior, which is useful as a targeted ablation.
5. Robustness and ONNX deployment experiments are now available to support the UAV scene and edge-deployment claims.

YOLO26 is still blocked by local Ultralytics/checkpoint compatibility and should not be the core claim until it can be trained reproducibly.

## Main clean-split results

Source files:

- `results/aircraft_ai_uav_paper_baselines.csv`
- `results/aircraft_ai_uav_paper_ablation.csv`

| Experiment | mAP50 | mAP50-95 | Precision | Recall | GPU inference ms |
|---|---:|---:|---:|---:|---:|
| B03_YOLOV8N | 0.487821 | 0.206883 | 0.494180 | 0.542641 | 2.798740 |
| B09_YOLO11N | 0.503023 | 0.211000 | 0.580612 | 0.527775 | 3.172427 |
| A00_YOLO11N_NWD_LOSS | 0.507261 | 0.220197 | 0.540633 | 0.539672 | 2.920256 |
| A00_YOLO11N_NWD_ASSIGNER | 0.492199 | 0.201720 | 0.569269 | 0.516199 | 3.297491 |

Interpretation:

- YOLO11n improves over YOLOv8n on mAP50 and mAP50-95.
- NWD loss is currently the best overall method: +0.009197 mAP50-95 over YOLO11n.
- NWD assigner should be written as a small-object/missing-rivet targeted ablation, not as the final overall method.

## Scale-group validation

Source file: `results/aircraft_ai_uav_paper_scale_groups.csv`

Scale split statistics:

- tiny: 565 boxes
- small: 321 boxes
- medium_large: 423 boxes

| Experiment | Group | mAP50 | mAP50-95 |
|---|---|---:|---:|
| B09_YOLO11N | medium_large | 0.375861 | 0.182120 |
| B09_YOLO11N | small | 0.155211 | 0.056264 |
| B09_YOLO11N | tiny | 0.080652 | 0.025147 |
| A00_YOLO11N_NWD_LOSS | medium_large | 0.374210 | 0.189008 |
| A00_YOLO11N_NWD_LOSS | small | 0.140911 | 0.053797 |
| A00_YOLO11N_NWD_LOSS | tiny | 0.087254 | 0.027525 |
| A00_YOLO11N_NWD_ASSIGNER | medium_large | 0.264319 | 0.125304 |
| A00_YOLO11N_NWD_ASSIGNER | small | 0.165309 | 0.060362 |
| A00_YOLO11N_NWD_ASSIGNER | tiny | 0.100894 | 0.035300 |

Interpretation:

- NWD loss gives a modest tiny-object gain while preserving medium/large performance.
- NWD assigner gives the clearest tiny/small gain but sacrifices medium/large objects.
- This supports a nuanced claim: NWD-based localization is valuable for micro-defect detection, but assignment design needs balancing.

## Robustness validation

Source file: `results/aircraft_ai_uav_paper_robustness.csv`

Benchmarks completed:

- clean
- defocus_blur_s1/s2/s3
- gaussian_noise_s1/s2/s3
- glare_s1/s2/s3
- jpeg_s1/s2/s3
- low_light_s1/s2/s3
- motion_blur_s1/s2/s3
- scale_s1/s2/s3

All four selected models have 22/22 robustness rows.

Important metric note:

- The main clean-split table and the robustness clean row are different evaluation subsets.
- The robustness matrix uses the 98-image test/corruption benchmark, so its clean mAP is lower and should be reported in a separate robustness table.

| Experiment | Robust clean mAP50-95 | Corruption avg mAP50-95 | Corruption rows |
|---|---:|---:|---:|
| B03_YOLOV8N | 0.164397 | 0.123799 | 21 |
| B09_YOLO11N | 0.175345 | 0.131423 | 21 |
| A00_YOLO11N_NWD_LOSS | 0.178375 | 0.132911 | 21 |
| A00_YOLO11N_NWD_ASSIGNER | 0.154297 | 0.116372 | 21 |

Interpretation:

- NWD loss has the best average robustness among the tested lightweight YOLO11n variants.
- Gaussian noise remains the hardest perturbation and may need a separate discussion or augmentation follow-up.

## Edge deployment

Source files:

- `results/aircraft_ai_uav_paper_exports.csv`
- `results/aircraft_ai_uav_paper_onnx_latency.csv`

ONNX export status:

| Experiment | Export status | ONNX size MB |
|---|---|---:|
| B03_YOLOV8N | success | 11.789020 |
| B09_YOLO11N | success | 10.204863 |
| A00_YOLO11N_NWD_LOSS | success | 10.204863 |
| A00_YOLO11N_NWD_ASSIGNER | success | 10.204863 |

ONNXRuntime CPU latency, dummy 1x3x800x800 input, 10 warmup, 50 iterations:

| Experiment | Mean ms | Median ms | P95 ms |
|---|---:|---:|---:|
| B03_YOLOV8N | 46.982464 | 45.831350 | 56.078900 |
| B09_YOLO11N | 43.214508 | 42.851850 | 47.681000 |
| A00_YOLO11N_NWD_LOSS | 48.023230 | 47.529400 | 59.776300 |
| A00_YOLO11N_NWD_ASSIGNER | 49.142934 | 49.926950 | 62.967400 |

Interpretation:

- YOLO11n ONNX is smaller and faster than YOLOv8n ONNX in this CPU benchmark.
- NWD variants retain the same ONNX size as YOLO11n because the architectural graph is unchanged; latency differences are small and should be treated as measurement variability unless repeated on a target edge device.

## Completed outputs

- `datasets/scale_groups/aircraft_ai_uav_clean_split_val/*.yaml`
- `results/aircraft_ai_uav_paper_scale_groups.csv`
- `results/aircraft_ai_uav_paper_robustness.csv`
- `results/aircraft_ai_uav_paper_exports.csv`
- `results/aircraft_ai_uav_paper_onnx_latency.csv`
- `runs/aircraft_ai_uav_paper_baselines/B03_yolov8n_uav_clean/weights/best.onnx`
- `runs/aircraft_ai_uav_paper_baselines/B09_yolo11n_uav_clean/weights/best.onnx`
- `runs/aircraft_ai_uav_paper_ablation/A00_yolo11n_nwd_loss_clean/weights/best.onnx`
- `runs/aircraft_ai_uav_paper_ablation/A00_yolo11n_nwd_assigner_clean/weights/best.onnx`

## Remaining recommended experiments

Priority 1:

- Repeat NWD-loss training with 2 additional random seeds if time permits.
- Add noise-aware augmentation or denoising augmentation and test whether gaussian_noise_s1/s2/s3 improves.

Priority 2:

- Fix YOLO26 checkpoint compatibility by matching the original Ultralytics/YOLO26 codebase, then rerun B11_YOLO26N.
- If YOLO26 remains blocked, remove it from the main claim and keep it as a limitation/future-work note.

Priority 3:

- Run ONNX latency on a real edge target or at least repeat CPU benchmark 3 times to report mean plus standard deviation across runs.
