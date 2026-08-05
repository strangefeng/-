# YOLO Baseline Repro

This repository standardizes three custom aircraft defect datasets into one shared 5-class label space for baseline Ultralytics YOLO experiments.

Target classes:

1. `crack`
2. `dent`
3. `missing-head`
4. `paint-off`
5. `scratch`

`YOLO.v1i.yolov11` is preserved as a separate 2-class supplementary dataset and is not included in the main 5-class baseline runs.

## Layout

- `datasets/standardized_5cls/`: standardized dataset descriptors for the 3 main datasets
- `datasets/supplementary_2cls/`: preserved descriptor for the 2-class supplementary dataset
- `datasets/main_5cls_datasets.json`: batch manifest for the main 5-class experiments
- `class_mapping_report.md`: class audit and normalization report
- `train_one.py`: train one dataset, export ONNX, and append results to CSV
- `train_all.py`: train all 3 standardized 5-class datasets
- `val_all.py`: validate all 3 standardized 5-class datasets
- `summarize_results.py`: summarize the append-only experiment CSV
- `results/experiments.csv`: experiment log generated automatically

## Environment

Use the dedicated YOLO environment and keep Ultralytics config inside the repo:

```powershell
$env:YOLO_CONFIG_DIR = "$PWD\.yolo-config"
& 'D:\yolo11\.conda\python.exe' -m pip install -r requirements.txt
```

Optional Git setup if `git` is available on your machine:

```bash
git init
```

## One-Click Launcher

Use the repo launcher when you do not want to remember commands:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_yolo.ps1
```

Direct actions are also supported:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_yolo.ps1 -Action verify
powershell -ExecutionPolicy Bypass -File .\start_yolo.ps1 -Action smoke
powershell -ExecutionPolicy Bypass -File .\start_yolo.ps1 -Action status
```

## Standardized Dataset YAMLs

- [datasets/standardized_5cls/aircraft_surface_defect_detection/data.yaml](D:/桌面/毕设/数据集/yolo-baseline-repro/datasets/standardized_5cls/aircraft_surface_defect_detection/data.yaml)
- [datasets/standardized_5cls/aircraft_skin_defects_new/data.yaml](D:/桌面/毕设/数据集/yolo-baseline-repro/datasets/standardized_5cls/aircraft_skin_defects_new/data.yaml)
- [datasets/standardized_5cls/aircraft_skin_defects/data.yaml](D:/桌面/毕设/数据集/yolo-baseline-repro/datasets/standardized_5cls/aircraft_skin_defects/data.yaml)

## Smoke Test

The smoke test uses the smallest complete 5-class dataset:

- dataset: `aircraft_skin_defects.v1i.yolov5pytorch`
- model: `yolo11n.pt`
- epochs: `3`
- imgsz: `640`
- batch: `16`

Run it with:

```powershell
$env:YOLO_CONFIG_DIR = "$PWD\.yolo-config"
& 'D:\yolo11\.conda\python.exe' .\train_one.py ^
  --dataset-yaml datasets/standardized_5cls/aircraft_skin_defects/data.yaml ^
  --model yolo11n.pt ^
  --imgsz 640 ^
  --epochs 3 ^
  --batch 16 ^
  --device auto ^
  --project runs/smoke ^
  --name smoke_aircraft_skin_defects
```

## Main 5-Class Baseline

Train all three main datasets:

```powershell
$env:YOLO_CONFIG_DIR = "$PWD\.yolo-config"
& 'D:\yolo11\.conda\python.exe' .\train_all.py ^
  --model yolo11n.pt ^
  --imgsz 640 ^
  --epochs 100 ^
  --batch 16 ^
  --device auto ^
  --project runs/baseline ^
  --name baseline5cls
```

Validate all three main datasets using the latest successful training record for each dataset:

```powershell
$env:YOLO_CONFIG_DIR = "$PWD\.yolo-config"
& 'D:\yolo11\.conda\python.exe' .\val_all.py ^
  --imgsz 640 ^
  --epochs 100 ^
  --batch 16 ^
  --device auto ^
  --project runs/baseline ^
  --name baseline5cls
```

Summarize experiment CSV records:

```powershell
$env:YOLO_CONFIG_DIR = "$PWD\.yolo-config"
& 'D:\yolo11\.conda\python.exe' .\summarize_results.py --project runs/baseline --name baseline5cls
```

## Notes

- The standardized YAMLs point to the original dataset folders without copying images or labels.
- The original labels are never rewritten by this repository.
- `aircraft_skin_defects.v1i.yolov5pytorch` is normalized by changing only the standardized YAML class name from `paint-peel-off` to `paint-off`.


