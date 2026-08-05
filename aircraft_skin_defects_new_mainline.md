# aircraft_skin_defects_new Mainline

This note turns `aircraft_skin_defects_new` into the primary mainline dataset for the thesis workflow:

1. Build a clean official baseline (`B0`).
2. Build a task-adapted baseline with weak-texture-preserving augmentation (`B1`).
3. Add feature or supervision modules one by one only after the previous stage is stable.
4. Do deployment only after the final model exists.

## Files

- Manifest: `experiments/aircraft_skin_defects_new_mainline.json`
- Runner: `run_experiment_suite.py`
- Result CSV: `results/aircraft_skin_defects_new_mainline.csv`
- Local patch package: `custom_yolo/`

## Environment

Use the clean YOLO environment instead of the broken base Python:

```powershell
$env:YOLO_CONFIG_DIR='D:\桌面\毕设\数据集\yolo-baseline-repro\.yolo-config'
& 'D:\yolo11\.conda\python.exe' 'D:\桌面\毕设\数据集\yolo-baseline-repro\run_experiment_suite.py' --list
```

## First batch to run

List the suite:

```powershell
$env:YOLO_CONFIG_DIR='D:\桌面\毕设\数据集\yolo-baseline-repro\.yolo-config'
& 'D:\yolo11\.conda\python.exe' 'D:\桌面\毕设\数据集\yolo-baseline-repro\run_experiment_suite.py' --list
```

Run the two baselines first:

```powershell
$env:YOLO_CONFIG_DIR='D:\桌面\毕设\数据集\yolo-baseline-repro\.yolo-config'
& 'D:\yolo11\.conda\python.exe' 'D:\桌面\毕设\数据集\yolo-baseline-repro\run_experiment_suite.py' --only B0 B1
```

## Weak-texture-preserving augmentation used in `B1`

- Reduce rotation, translation and scaling strength.
- Keep horizontal flip but disable vertical flip.
- Keep only mild HSV jitter.
- Lower Mosaic to `0.2`.
- Keep MixUp off.
- Keep erasing off.

## Local custom modules

- `DWTDown`: implemented in `custom_yolo/modules.py` and injected into the YOLO parser by `custom_yolo/patches.py`.
- `NWD Loss/Assigner`: implemented as context-aware patches in `custom_yolo/patches.py`.
- `Rep Neck`: implemented through the local `yolo11n_dwt_repneck.yaml` neck definition that uses `RepC3` blocks.

## Current implementation status

Ready now:

- `B0`: official default baseline
- `B1`: task-adapted baseline
- `M1`: DWTDown feature stage
- `M2`: DWT + NWD loss stage
- `M3`: DWT + NWD cooperative supervision stage
- `M4`: DWT + NWD + Rep neck final structure

Still blocked by dependency on final weights:

- `M5`: deployment export and benchmark
