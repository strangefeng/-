# Class Mapping Report

This report audits the current class systems of the four available datasets and records the normalization decision for the unified 5-class baseline.

Unified target classes:

| Target ID | Target Name |
| --- | --- |
| 0 | crack |
| 1 | dent |
| 2 | missing-head |
| 3 | paint-off |
| 4 | scratch |

## Main 5-Class Datasets

### 1. Aircraft Surface Defect Detection.v1i.yolov5pytorch

- Current `nc`: `5`
- Current classes:
  - `0 crack`
  - `1 dent`
  - `2 missing-head`
  - `3 paint-off`
  - `4 scratch`
- Label IDs observed in annotation files: `0, 1, 2, 3, 4`
- Annotation instance counts by class ID:
  - `0 crack`: `7995`
  - `1 dent`: `7135`
  - `2 missing-head`: `370`
  - `3 paint-off`: `294`
  - `4 scratch`: `83`
- Consistency with target 5 classes: `exact match`
- Extra classes: `none`
- Missing classes: `none`
- Normalization action: no semantic remapping required

### 2. aircraft-skin-defects-new-dataset.v1i.yolov5pytorch

- Current `nc`: `5`
- Current classes:
  - `0 crack`
  - `1 dent`
  - `2 missing-head`
  - `3 paint-off`
  - `4 scratch`
- Label IDs observed in annotation files: `0, 1, 2, 3, 4`
- Annotation instance counts by class ID:
  - `0 crack`: `398`
  - `1 dent`: `517`
  - `2 missing-head`: `370`
  - `3 paint-off`: `294`
  - `4 scratch`: `83`
- Consistency with target 5 classes: `exact match`
- Extra classes: `none`
- Missing classes: `none`
- Normalization action: no semantic remapping required

### 3. aircraft_skin_defects.v1i.yolov5pytorch

- Current `nc`: `5`
- Current classes:
  - `0 crack`
  - `1 dent`
  - `2 missing-head`
  - `3 paint-peel-off`
  - `4 scratch`
- Label IDs observed in annotation files: `0, 1, 2, 3, 4`
- Annotation instance counts by class ID:
  - `0 crack`: `431`
  - `1 dent`: `2156`
  - `2 missing-head`: `2834`
  - `3 paint-peel-off`: `2106`
  - `4 scratch`: `922`
- Consistency with target 5 classes: `semantic mismatch in class name only`
- Mapping suggestion: `paint-peel-off -> paint-off`
- Adopted normalization: keep class ID `3`, update only the standardized descriptor name
- Extra classes: `none`
- Missing classes:
  - strict name comparison: `paint-off`
  - after semantic mapping: `none`

## Supplementary 2-Class Dataset

### 4. YOLO.v1i.yolov11

- Current `nc`: `2`
- Current classes:
  - `0 crack`
  - `1 dent`
- Label IDs observed in annotation files: `0, 1`
- Annotation instance counts by class ID:
  - `0 crack`: `8141`
  - `1 dent`: `3477`
- Consistency with target 5 classes: `subset only`
- Semantic mapping:
  - `0 crack -> 0 crack`
  - `1 dent -> 1 dent`
- Extra classes: `none`
- Missing classes:
  - `2 missing-head`
  - `3 paint-off`
  - `4 scratch`
- Main experiment inclusion: `excluded`
- Role: retained as a supplementary 2-class dataset only

## Standardization Decision

- Main 5-class baseline datasets:
  - `Aircraft Surface Defect Detection.v1i.yolov5pytorch`
  - `aircraft-skin-defects-new-dataset.v1i.yolov5pytorch`
  - `aircraft_skin_defects.v1i.yolov5pytorch`
- Supplementary dataset kept outside the main baseline:
  - `YOLO.v1i.yolov11`

## Non-Destructive Guarantee

- The original image folders were not copied.
- The original labels were not modified.
- The original dataset YAML files were not overwritten.
- Normalization is implemented only through standardized descriptor directories and new dataset YAML files.
