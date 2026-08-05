# Aircraft AI Dataset round12 自动化实验方案

## 1. 目标

`round12` 不再继续做结构堆叠，而是只验证三条真正可能改变样本信息量的路线：

1. 切片训练 / 切片推理
2. 缺陷级 patch copy-paste 增强
3. hard negative mining

所有训练统一固定在 `N2` 配方上：

- `YOLO11n`
- `imgsz=800`
- `mosaic=0`
- `erasing=0`
- `auto_augment=none`

## 2. 自动化组成

### 数据准备

- `scripts/build_sliced_yolo_dataset.py`
- `scripts/build_defect_patch_dataset.py`
- `scripts/build_hard_negative_dataset.py`
- `scripts/prepare_aircraft_ai_round12.py`

### 推理评估

- `scripts/evaluate_sliced_inference.py`

### 训练与汇总

- `experiments/aircraft_ai_dataset_round12_automation.json`
- `scripts/run_aircraft_ai_round12.py`
- `scripts/summarize_aircraft_ai_round12.py`

## 3. 实验编号

| 编号 | 方向 | 说明 |
| --- | --- | --- |
| `S1` | sliced inference only | 直接用 `N2 best.pt` 做切片推理评估，不重训 |
| `S2` | sliced train | 只用切片 train tiles 训练 |
| `S3` | mixed sliced train | 原图 + 切片混合训练 |
| `P1` | patch aug | 只对 `missing-rivet` 做缺陷级贴片增强 |
| `P2` | patch aug | 对 `missing-rivet + scratch` 做贴片增强 |
| `P3` | patch aug | 在 `P2` 基础上加入轻微光照/模糊/噪声扰动 |
| `HN1` | hard negative | 采 50 个 false-positive crop 作为空标注负样本 |
| `HN2` | hard negative | 采 100 个 false-positive crop 作为空标注负样本 |

## 4. 默认参数

### 切片

- `slice_size=512`
- `overlap=0.20`
- `max_empty_tiles_per_image=1`

### 贴片增强

- `copies_per_image=1`
- `pastes_per_copy=2`
- `min_scale=0.90`
- `max_scale=1.15`
- `max_iou=0.05`

### hard negative

- `conf=0.25`
- `max_negatives=50/100`
- `context_scale=3.0`
- `min_crop_edge=160`
- `max_crop_edge=320`

## 5. 一键运行方式

完整自动化：

```powershell
python .\scripts\run_aircraft_ai_round12.py
```

只准备数据：

```powershell
python .\scripts\prepare_aircraft_ai_round12.py
```

只跑部分训练实验：

```powershell
python .\scripts\run_aircraft_ai_round12.py --only S2 S3 P2
```

跳过准备，直接重跑训练：

```powershell
python .\scripts\run_aircraft_ai_round12.py --skip-prepare
```

## 6. 输出位置

### 数据集

- `datasets/generated_3cls/aircraft_ai_dataset_3cls_slice512_o20`
- `datasets/generated_3cls/aircraft_ai_dataset_3cls_mixedslice512_o20`
- `datasets/generated_3cls/aircraft_ai_dataset_3cls_patch_mr`
- `datasets/generated_3cls/aircraft_ai_dataset_3cls_patch_mr_scratch`
- `datasets/generated_3cls/aircraft_ai_dataset_3cls_patch_mr_scratch_photo`
- `datasets/generated_3cls/aircraft_ai_dataset_3cls_hardneg50`
- `datasets/generated_3cls/aircraft_ai_dataset_3cls_hardneg100`

### 结果

- 训练结果表：`results/aircraft_ai_dataset_round12_automation.csv`
- 切片推理评估：`analysis/aircraft_ai_dataset_round12_s1_sliced_eval.json`
- 汇总结论：`analysis/aircraft_ai_dataset_round12_automation_summary.md`

## 7. 结果解释建议

如果 `S1/S2/S3` 有明显收益，可以把论文主线推进到：

> 提高缺陷在输入图中的有效像素占比，比继续增加结构复杂度更有效。

如果 `P2/P3` 涨点更明显，可以推进到：

> 缺陷级样本多样性扩展优于整图重复过采样。

如果 `HN1/HN2` 的 `Precision` 和可视化误检明显改善，即使 `mAP50-95` 只小幅变化，也依然值得写进部署章节。
