# Smoke Test Command

Run this from the repository root after activating the clean environment.

```powershell
$env:YOLO_CONFIG_DIR = "$PWD\.yolo-config"
$DEVICE = & 'D:\yolo11\.conda\python.exe' -c "import torch; print('0' if torch.cuda.is_available() else 'cpu')"
& 'D:\yolo11\.conda\python.exe' .\train_one.py --dataset-yaml datasets/standardized_5cls/aircraft_skin_defects/data.yaml --model ..\yolo11n.pt --imgsz 640 --epochs 3 --batch 16 --device $DEVICE --project runs/smoke_clean --name smoke_aircraft_skin_defects_clean
```

This uses the smallest complete standardized 5-class dataset and keeps smoke-test outputs separate from future baseline runs.

