# SCI 三区目标实验矩阵

日期：2026-05-17

## 总体原则

所有进入主文的性能表必须满足：

- 使用同一 cleaned re-split 数据：`datasets/generated_3cls/aircraft_ai_uav_clean_split.yaml`
- 训练设置一致：`imgsz=800`、`epochs=100`、`seed=42`、N2 增强策略（mosaic/erasing/mixup 关闭）
- 指标统一：Precision、Recall、mAP50、mAP50-95、per-class AP50/AP50-95、Params、FLOPs、inference time
- 不把旧 Roboflow split、旧 cleaned anchor、不同 yaml 的结果混入同一张主表

## Stage 1：公平主表最小闭环

目标：先确认论文主方法相对主流轻量检测器是否有可写空间。

优先运行：

1. B09_YOLO11N
2. B11_YOLO26N
3. A01_YOLO26N
4. A02_YOLO26N_NWD_LOSS
5. A03_YOLO26N_NWD_ASSIGNER

已有可复用但需标记为同一 suite 结果：

- B01_YOLOV5N
- B02_YOLOV5S
- B03_YOLOV8N
- B04_YOLOV8S
- B05_YOLOV9T

补充条件允许后再跑：

- B06_YOLOV9S
- B07_YOLOV10N
- B08_YOLOV10S
- B10_YOLO11S
- B12_YOLO26S
- B13_RTDETR_L

成功标准：

- 至少形成 `YOLOv5n/YOLOv8n/YOLOv9t/YOLO11n/YOLO26n/YOLO26n+NWD` 的同 split 对比表。
- 如果 YOLO26n+NWD 未超过 YOLO11n，则论文题目不再强绑定 RTD-YOLO26。

## Stage 2：标签治理实验

目标：把数据清洗从“预处理”提升为正式贡献。

主对比：

1. Raw train + clean eval
2. Clean train + clean eval
3. Clean train + NWD loss

需要补充：

- 随机抽样 100-200 张清洗前后样本，人工统计错误框、过大 missing-rivet 框、误删率。
- 输出清洗前后 bbox area 分布和 class count 分布。

成功标准：

- 能证明 missing-rivet AP 的提升主要来自标签治理，而不是评估集变化或随机波动。

## Stage 3：小目标分组评测

目标：支撑“小缺陷检测”主张。

分组：

- tiny: area < 0.25%
- small: 0.25% <= area < 1%
- medium/large: area >= 1%

模型：

- YOLO11n
- YOLO26n
- YOLO26n + NWD loss
- 最终方法

输出：

- AP_tiny、AP_small、AP_medium_large
- AR_tiny、AR_small
- missing-rivet subgroup AP

## Stage 4：UAV 退化鲁棒性实验

目标：支撑 UAV 巡检真实成像退化。

退化集已存在：

- defocus_blur_s1-s3
- gaussian_noise_s1-s3
- glare_s1-s3
- jpeg_s1-s3
- low_light_s1-s3
- motion_blur_s1-s3
- scale_s1-s3

模型：

- YOLOv8n
- YOLO11n
- YOLO26n
- YOLO26n + NWD loss

指标：

- clean mAP50-95
- mean corruption mAP50-95
- relative degradation rate
- per-corruption degradation

## Stage 5：边缘部署实验

目标：支撑轻量可部署。

模型：

- YOLOv5n
- YOLOv8n
- YOLO11n
- YOLO26n
- YOLO26n + NWD loss

导出：

- ONNX 必做
- TensorRT FP16 有设备时补做

指标：

- model size
- latency
- FPS
- memory if available
- export success

## Stage 6：外部 UAV 验证

目标：形成真实场景泛化证据。

前提：金堂自采数据完成标注和记录。

设置：

1. public/cleaned train -> Jintang UAV test
2. public/cleaned train + 10% Jintang fine-tune -> remaining Jintang test

指标：

- Recall
- false negative rate
- mAP50
- per-class AP
- qualitative failure cases

## 当前运行顺序

先运行 Stage 1：

1. B09_YOLO11N
2. B11_YOLO26N
3. A01_YOLO26N
4. A02_YOLO26N_NWD_LOSS
5. A03_YOLO26N_NWD_ASSIGNER

然后根据结果决定论文命名是否继续使用 RTD-YOLO26。

