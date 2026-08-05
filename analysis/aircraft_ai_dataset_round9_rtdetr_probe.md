# aircraft_ai_dataset_round9_rtdetr Probe

## What was checked

- Goal: test whether `RT-DETR-L` is a stronger comparison model than the current public-dataset YOLO anchor.
- Dataset: public 3-class Aircraft AI dataset.
- Training recipe: aligned to the current `N2` direction as much as practical.
  - `imgsz=800`
  - `rect=false`
  - `mosaic=0`
  - `erasing=0`
  - `auto_augment=none`
- Model: `rtdetr-l.pt`
- Hardware: `CUDA:0`, RTX 4060 Laptop GPU.

## Engineering note

- The original experiment framework instantiated every model through `YOLO(...)`.
- `RT-DETR` needs to be instantiated through `RTDETR(...)`.
- The compatibility layer was patched in:
  - [pipeline_utils.py](D:/桌面/毕设/数据集/yolo-baseline-repro/pipeline_utils.py)
  - [run_experiment_suite.py](D:/桌面/毕设/数据集/yolo-baseline-repro/run_experiment_suite.py)

## Probe status

- Formal 100-epoch run did not finish inside the single command time budget.
- The run itself was valid and reached epoch `87`.
- We then validated the saved `best.pt` checkpoint directly.

Artifacts:
- Training run: [D1_rtdetr_l_i800_no_mosaic2](D:/桌面/毕设/数据集/yolo-baseline-repro/runs/aircraft_ai_round9_rtdetr/D1_rtdetr_l_i800_no_mosaic2)
- Validation run: [D1_rtdetr_l_i800_no_mosaic2_val2](D:/桌面/毕设/数据集/yolo-baseline-repro/runs/aircraft_ai_round9_rtdetr/D1_rtdetr_l_i800_no_mosaic2_val2)

## Validation result from saved best checkpoint

| model | precision | recall | mAP50 | mAP50-95 | missing-rivet | rust | scratch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RT-DETR-L @800 | 0.7514 | 0.7079 | 0.7252 | 0.3244 | 0.2802 | 0.3701 | 0.3229 |

## Comparison against the current YOLO anchor

Current best public-dataset YOLO anchor:
- `N2`: `mAP50=0.7449`, `mAP50-95=0.3526`
- `missing-rivet AP50-95=0.3061`
- `rust AP50-95=0.4013`
- `scratch AP50-95=0.3503`

RT-DETR-L probe deltas vs `N2`:
- `mAP50`: `-0.0197`
- `mAP50-95`: `-0.0282`
- `missing-rivet AP50-95`: `-0.0259`
- `rust AP50-95`: `-0.0313`
- `scratch AP50-95`: `-0.0274`

## Deployment view

- RT-DETR-L validation speed: about `28.4 ms` inference per image.
- Current YOLO11n public anchor is much lighter and faster.
- RT-DETR-L also required far more training time on the same GPU.

## Takeaway

- `RT-DETR-L` can run in the current framework after the model-loader compatibility fix.
- On this dataset and recipe, it did **not** outperform the current `YOLO11n N2` anchor.
- It is slower and heavier, while also producing worse validation metrics.
- Recommendation: keep `RT-DETR` as a checked comparison model, not as the new mainline direction.
