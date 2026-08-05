# YOLO11 三数据集 Baseline 逐个执行清单

## 0. 全局执行原则

在整个 baseline 阶段，统一遵守以下规则：

- [ ] 不改动代码
- [ ] 不升级/降级依赖
- [ ] 不切换到 GPU
- [ ] 不修改类别定义
- [ ] 不混用 smoke test 输出目录和 baseline 输出目录
- [ ] 所有训练都固定使用 `workers=0`
- [ ] 所有训练都固定使用 `device=cpu`
- [ ] 所有训练都固定使用 `imgsz=640`
- [ ] 所有训练都固定使用 `batch=16`
- [ ] 所有训练都固定使用 `epochs=100`

## 1. 统一预检查

### 1.1 环境预检查

- [ ] PowerShell 已打开
- [ ] 当前环境与 smoke test 成功环境一致
- [ ] `D:\yolo11\.conda\Scripts\yolo.exe version` 可正常返回版本号
- [ ] `D:\yolo11\yolo11n.pt` 文件存在
- [ ] baseline 主输出根目录存在或可创建：

```text
D:\runs\detect\yolo-baseline-repro\runs\baseline_5cls
```

### 1.2 配置目录固定

先执行：

```powershell
$env:YOLO_CONFIG_DIR='D:\桌面\毕设\数据集\yolo-baseline-repro\.yolo-config'
```

然后检查：

- [ ] `YOLO_CONFIG_DIR` 已设置
- [ ] 不再使用默认用户目录配置

### 1.3 数据集登记确认

- [ ] 已确认三主数据集来自 `main_5cls_datasets.json`
- [ ] 已确认三个 `data.yaml` 路径真实存在
- [ ] 已确认三个数据集均为统一 5 类定义：
- [ ] `crack`
- [ ] `dent`
- [ ] `missing-head`
- [ ] `paint-off`
- [ ] `scratch`

## 2. aircraft_skin_defects 的完整执行清单

### 2.1 训练前检查

- [ ] `data.yaml` 路径存在：

```text
D:\桌面\毕设\数据集\yolo-baseline-repro\datasets\standardized_5cls\aircraft_skin_defects\data.yaml
```

- [ ] 输出目录命名确认：

```text
bl_aircraft_skin_y11n_e100_i640_b16_cpu
```

- [ ] 本次训练命令已确认不再修改

### 2.2 执行训练

执行：

```powershell
& 'D:\yolo11\.conda\Scripts\yolo.exe' detect train `
  model='D:\yolo11\yolo11n.pt' `
  data='D:\桌面\毕设\数据集\yolo-baseline-repro\datasets\standardized_5cls\aircraft_skin_defects\data.yaml' `
  epochs=100 `
  imgsz=640 `
  batch=16 `
  workers=0 `
  device=cpu `
  project='D:\runs\detect\yolo-baseline-repro\runs\baseline_5cls' `
  name='bl_aircraft_skin_y11n_e100_i640_b16_cpu'
```

### 2.3 训练中观察

- [ ] 已进入训练阶段
- [ ] 日志出现 `Starting training for ... epochs`
- [ ] 未在首个 batch 读图阶段报错
- [ ] epoch 正常推进
- [ ] 无明显 `nan` / 全 0 指标异常

### 2.4 训练后文件核查

执行：

```powershell
$run='D:\runs\detect\yolo-baseline-repro\runs\baseline_5cls\bl_aircraft_skin_y11n_e100_i640_b16_cpu'
Test-Path "$run\results.csv"
Test-Path "$run\weights\best.pt"
Test-Path "$run\weights\last.pt"
Get-ChildItem -Force $run
```

核查：

- [ ] 输出目录已生成
- [ ] `results.csv` 已生成
- [ ] `weights\best.pt` 已生成
- [ ] `weights\last.pt` 已生成
- [ ] `args.yaml` 已生成
- [ ] 结果图文件已生成（若存在）

### 2.5 指标记录

- [ ] 已记录 `precision`
- [ ] 已记录 `recall`
- [ ] 已记录 `mAP50`
- [ ] 已记录 `mAP50-95`
- [ ] 已记录各类别结果
- [ ] 已保存最后 30 行关键日志

### 2.6 本数据集结论

- [ ] 训练成功
- [ ] 训练失败
- [ ] 训练部分完成但需补跑

## 3. aircraft_skin_defects_new 的完整执行清单

### 3.1 训练前检查

- [ ] `data.yaml` 路径存在：

```text
D:\桌面\毕设\数据集\yolo-baseline-repro\datasets\standardized_5cls\aircraft_skin_defects_new\data.yaml
```

- [ ] 输出目录命名确认：

```text
bl_aircraft_skin_new_y11n_e100_i640_b16_cpu
```

- [ ] 确认沿用与上一个数据集相同环境和参数

### 3.2 执行训练

执行：

```powershell
& 'D:\yolo11\.conda\Scripts\yolo.exe' detect train `
  model='D:\yolo11\yolo11n.pt' `
  data='D:\桌面\毕设\数据集\yolo-baseline-repro\datasets\standardized_5cls\aircraft_skin_defects_new\data.yaml' `
  epochs=100 `
  imgsz=640 `
  batch=16 `
  workers=0 `
  device=cpu `
  project='D:\runs\detect\yolo-baseline-repro\runs\baseline_5cls' `
  name='bl_aircraft_skin_new_y11n_e100_i640_b16_cpu'
```

### 3.3 训练中观察

- [ ] 已进入训练阶段
- [ ] 日志出现 `Starting training for ... epochs`
- [ ] 未在首个 batch 读图阶段报错
- [ ] epoch 正常推进
- [ ] 无明显 `nan` / 全 0 指标异常

### 3.4 训练后文件核查

执行：

```powershell
$run='D:\runs\detect\yolo-baseline-repro\runs\baseline_5cls\bl_aircraft_skin_new_y11n_e100_i640_b16_cpu'
Test-Path "$run\results.csv"
Test-Path "$run\weights\best.pt"
Test-Path "$run\weights\last.pt"
Get-ChildItem -Force $run
```

核查：

- [ ] 输出目录已生成
- [ ] `results.csv` 已生成
- [ ] `weights\best.pt` 已生成
- [ ] `weights\last.pt` 已生成
- [ ] `args.yaml` 已生成
- [ ] 结果图文件已生成（若存在）

### 3.5 指标记录

- [ ] 已记录 `precision`
- [ ] 已记录 `recall`
- [ ] 已记录 `mAP50`
- [ ] 已记录 `mAP50-95`
- [ ] 已记录各类别结果
- [ ] 已保存最后 30 行关键日志

### 3.6 本数据集结论

- [ ] 训练成功
- [ ] 训练失败
- [ ] 训练部分完成但需补跑

## 4. aircraft_surface_defect_detection 的完整执行清单

### 4.1 训练前检查

- [ ] `data.yaml` 路径存在：

```text
D:\桌面\毕设\数据集\yolo-baseline-repro\datasets\standardized_5cls\aircraft_surface_defect_detection\data.yaml
```

- [ ] 输出目录命名确认：

```text
bl_aircraft_surface_y11n_e100_i640_b16_cpu
```

- [ ] 确认沿用与前两个数据集相同环境和参数

### 4.2 执行训练

执行：

```powershell
& 'D:\yolo11\.conda\Scripts\yolo.exe' detect train `
  model='D:\yolo11\yolo11n.pt' `
  data='D:\桌面\毕设\数据集\yolo-baseline-repro\datasets\standardized_5cls\aircraft_surface_defect_detection\data.yaml' `
  epochs=100 `
  imgsz=640 `
  batch=16 `
  workers=0 `
  device=cpu `
  project='D:\runs\detect\yolo-baseline-repro\runs\baseline_5cls' `
  name='bl_aircraft_surface_y11n_e100_i640_b16_cpu'
```

### 4.3 训练中观察

- [ ] 已进入训练阶段
- [ ] 日志出现 `Starting training for ... epochs`
- [ ] 未在首个 batch 读图阶段报错
- [ ] epoch 正常推进
- [ ] 无明显 `nan` / 全 0 指标异常

### 4.4 训练后文件核查

执行：

```powershell
$run='D:\runs\detect\yolo-baseline-repro\runs\baseline_5cls\bl_aircraft_surface_y11n_e100_i640_b16_cpu'
Test-Path "$run\results.csv"
Test-Path "$run\weights\best.pt"
Test-Path "$run\weights\last.pt"
Get-ChildItem -Force $run
```

核查：

- [ ] 输出目录已生成
- [ ] `results.csv` 已生成
- [ ] `weights\best.pt` 已生成
- [ ] `weights\last.pt` 已生成
- [ ] `args.yaml` 已生成
- [ ] 结果图文件已生成（若存在）

### 4.5 指标记录

- [ ] 已记录 `precision`
- [ ] 已记录 `recall`
- [ ] 已记录 `mAP50`
- [ ] 已记录 `mAP50-95`
- [ ] 已记录各类别结果
- [ ] 已保存最后 30 行关键日志

### 4.6 本数据集结论

- [ ] 训练成功
- [ ] 训练失败
- [ ] 训练部分完成但需补跑

## 5. 三数据集全部完成后的总核查

### 5.1 结果文件完整性

- [ ] 数据集 1 输出完整
- [ ] 数据集 2 输出完整
- [ ] 数据集 3 输出完整

### 5.2 指标整理

- [ ] 三个数据集的最终指标已汇总到总表
- [ ] 三个数据集的 `results.csv` 已单独备份
- [ ] 三个数据集的 `best.pt` 路径已登记
- [ ] 三个数据集的 `last.pt` 路径已登记

### 5.3 可复现性检查

- [ ] 三个实验的命令均已保存
- [ ] 三个实验的 `data.yaml` 路径均已保存
- [ ] `YOLO_CONFIG_DIR` 设置已记录
- [ ] 当前环境信息已记录
- [ ] 未发生中途改依赖/改参数/改代码

### 5.4 最终结论

- [ ] 三数据集 baseline 全部成功完成
- [ ] 有部分数据集需要补跑
- [ ] 进入下一阶段：指标对比与结果分析

## 6. 建议实际执行顺序

按这个顺序跑：

- [ ] 先跑 `aircraft_skin_defects`
- [ ] 再跑 `aircraft_skin_defects_new`
- [ ] 最后跑 `aircraft_surface_defect_detection`

原因是前两个和你 smoke test 轨迹更接近，能先把正式 baseline 节奏跑顺。

## 7. 每跑完一个数据集后固定要做的动作

每次都重复，不要省略：

- [ ] 看是否完成全部 epoch
- [ ] 查 `results.csv`
- [ ] 查 `best.pt`
- [ ] 查 `last.pt`
- [ ] 记录 4 个核心指标
- [ ] 保存最后 30 行日志
- [ ] 填实验记录模板
- [ ] 再开始下一个数据集
