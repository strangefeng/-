# Aircraft AI Dataset 3-Class Baseline Conclusion

## Dataset

- Dataset: `Aircraft AI Dataset.v1i.yolov11`
- Task: 3-class object detection
- Classes: `missing-rivet`, `rust`, `scratch`
- Source note: exported from Roboflow
- Export note: `Auto-orientation` and `Resize to 640x640 (Stretch)` were applied
- Augmentation note: none

Reference files:

- `D:\桌面\毕设\数据集\Aircraft AI Dataset.v1i.yolov11\README.roboflow.txt`
- `D:\桌面\毕设\数据集\Aircraft AI Dataset.v1i.yolov11\data.yaml`
- `D:\桌面\毕设\数据集\yolo-baseline-repro\datasets\supplementary_3cls\aircraft_ai_dataset_3cls\data.yaml`

## Run

- Model: `yolo11n`
- Image size: `640`
- Epochs: `100`
- Batch size: `16`
- Export: skipped (`none`)
- Run directory: `D:\桌面\毕设\数据集\yolo-baseline-repro\runs\supplementary_baseline_3cls\bl_aircraft_ai_3cls_y11n_e100_i640_b16`
- Best weights: `D:\桌面\毕设\数据集\yolo-baseline-repro\runs\supplementary_baseline_3cls\bl_aircraft_ai_3cls_y11n_e100_i640_b16\weights\best.pt`

## Final Metrics

Validation results from `best.pt`:

- Overall precision: `0.748`
- Overall recall: `0.720`
- mAP50: `0.745`
- mAP50-95: `0.343`

Per class:

| Class | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| missing-rivet | 0.668 | 0.658 | 0.668 | 0.297 |
| rust | 0.811 | 0.784 | 0.814 | 0.389 |
| scratch | 0.766 | 0.718 | 0.751 | 0.342 |

## Judgment

This dataset **can be used as a new baseline dataset** for a focused 3-class aircraft skin defect detection setting.

Reasons:

1. The training run completed normally and converged stably.
2. The final metrics are usable as a baseline, especially for `rust` and `scratch`.
3. Class behavior is reasonably balanced; no class collapsed to near-zero performance.
4. The data source is easier to explain than some earlier noisy datasets because class definitions are clear and the split is complete.

## Caveats

This dataset is **not a drop-in replacement** for the previous 5-class mainline baseline.

Reasons:

1. It is a 3-class dataset, while the previous mainline experiments used 5 classes.
2. The Roboflow export is not fully raw because it includes `640x640 stretch resize`.
3. Stretch resize may distort shape cues, so this baseline is best described as a Roboflow-export baseline rather than a raw-source baseline.

## Recommendation

Recommended positioning in the thesis:

- Use this dataset as a **supplementary 3-class baseline** or a **cleaner external baseline**.
- Do not silently replace the previous 5-class mainline with it.
- If you want to upgrade it from supplementary baseline to core baseline, first try to obtain or rebuild a less-processed export without stretch distortion.

## One-Sentence Conclusion

`Aircraft AI Dataset.v1i.yolov11` is good enough to serve as a clean and defensible **3-class baseline**, but it should be presented as a supplementary or parallel baseline rather than a direct substitute for the existing 5-class mainline dataset.
