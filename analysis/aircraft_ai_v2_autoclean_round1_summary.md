# Aircraft AI v2 Auto-Clean Round 1 Summary

## Dataset Cleaning

- Source dataset: `Aircraft AI Dataset.v2i.yolov11`
- Cleaned dataset: `Aircraft_AI_relabel_dataset_auto_clean`
- Final re-split dataset: `Aircraft_AI_relabel_dataset_auto_clean_split`
- Cleaning policy: conservative automatic repair of suspicious `missing-rivet` large boxes only.
- Change log: `dataset_relabel_tool/outputs/review_lists/auto_clean_change_log.csv`

The cleaning pass split large `missing-rivet` boxes into compact dark single-hole candidates when possible, and removed large `missing-rivet` boxes when no compact dark component was found.

## Label Statistics After Cleaning

- `missing-rivet`: 3429 boxes, mean area 0.001737
- `rust`: 2039 boxes, mean area 0.021035
- `scratch`: 2068 boxes, mean area 0.018537
- YOLO validation: 0 format errors, 2 empty labels after cleaning

## Fair Experiment Design

Both models were evaluated on the same cleaned validation labels.

- Baseline: original v2 train labels + cleaned validation labels
- Cleaned branch: auto-cleaned v2 train labels + cleaned validation labels
- Model: YOLO11n
- Image size: 800
- Epochs: 100
- Augmentation profile: N2 style, mosaic/erasing/mixup disabled

## Results

| Experiment | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Raw train + clean eval | 0.606774 | 0.503940 | 0.537558 | 0.251717 |
| Clean train + clean eval | 0.663273 | 0.604508 | 0.620931 | 0.282021 |

Absolute gains:

- Precision: +0.056499
- Recall: +0.100568
- mAP50: +0.083373
- mAP50-95: +0.030304

Per-class mAP50-95:

| Class | Raw Train | Clean Train | Delta |
|---|---:|---:|---:|
| missing-rivet | 0.012432 | 0.100167 | +0.087735 |
| rust | 0.391104 | 0.397271 | +0.006167 |
| scratch | 0.351615 | 0.348624 | -0.002991 |

## Conclusion

The conservative relabeling branch improves mAP50-95 by about 3.03 points on the same cleaned validation split, exceeding the target of roughly 2 points. The main gain comes from `missing-rivet`, which is consistent with the cleaning policy that corrected large multi-target or seam-like `missing-rivet` boxes.
