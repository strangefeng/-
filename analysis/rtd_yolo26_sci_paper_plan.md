# RTD-YOLO26 小论文算法、鲁棒性与边缘部署实验设计

## 1. 小论文定位

建议中文题目：

面向无人机巡检的飞机蒙皮小缺陷鲁棒轻量检测方法

建议英文题目：

RTD-YOLO26: A Robust Texture-aware Lightweight Detector for UAV-based Aircraft Skin Defect Detection

论文目标不是简单验证 YOLO26，而是提出一个面向飞机蒙皮小缺陷、弱纹理、强反光和边缘部署约束的鲁棒轻量检测方法。YOLO26n 作为实时检测底座，创新点放在纹理增强、小目标定位、真实无人机外部验证和边缘端部署。

这篇小论文服务大论文第 4 章和第 6 章：小论文负责算法与部署，大论文进一步承接到数字孪生平台，实现检测结果结构定位、复核派工和历史追踪。

## 2. 面向 SCI 三区的论文故事

近几年 YOLO 类 SCI 论文通常不是只报一个模型精度，而是形成以下证据链：

1. 任务困难性：小目标、弱纹理、复杂光照、强反光、无人机视角退化。
2. 方法针对性：不是泛泛加注意力，而是针对缺陷纹理、尺度丢失、定位不稳定设计模块。
3. 横向对比充分：YOLOv5、YOLOv7、YOLOv8、YOLOv9、YOLOv10、YOLOv11、YOLO26、RT-DETR 等。
4. 消融实验完整：每个模块单独有效，组合后最好。
5. 复杂场景评估：小目标、低照度、模糊、压缩、反光、外部数据集。
6. 工程部署验证：ONNX/TensorRT/边缘设备 FPS、延迟、模型大小。
7. 可视化解释：检测结果、热力图、特征图、失败案例。

你的论文应避免写成“YOLO26 + 某模块”，而应写成：

公开数据集训练和方法创新 -> 退化鲁棒性评测 -> 金堂真实无人机外部验证 -> 边缘设备部署 -> 数字孪生平台接入。

## 3. 核心创新点

### 3.1 RTFE 纹理频域增强模块

Robust Texture-Frequency Enhancement，面向 scratch、rust、missing-rivet 等弱纹理小缺陷。建议使用轻量高频分支提取局部边缘、细长划痕和腐蚀纹理，再与主干语义特征门控融合。

可写成：

- 高频分支：Haar/DWT 或 depthwise high-pass convolution。
- 形态分支：横向、纵向、局部圆形和扩散纹理。
- 门控融合：根据缺陷纹理响应动态增强浅层特征。

### 3.2 NWD 小目标定位优化

你的已有实验里 NWD loss / NWD-aware assigner 对 missing-rivet 更有帮助，因此建议保留为第二个关键点。它比 Focaler-IoU 更适合解释小框定位：极小缺陷框之间轻微偏移会导致 IoU 大幅波动，NWD 能缓解小目标定位不稳定。

### 3.3 UAV-Robust 退化评测协议

构造无人机场景退化测试集：

- motion blur
- defocus blur
- Gaussian noise
- low-light
- overexposure / glare
- JPEG compression
- scale down-up
- oblique perspective

使用 clean mAP、corrupt mAP、relative degradation rate 评价鲁棒性。

### 3.4 EdgeTwin 部署与平台接入

将模型导出为 ONNX/TensorRT，部署到边缘设备。推理结果以 JSON 输出到数字孪生平台，包括 defect_id、class、confidence、bbox、image_id、region_id、risk_level、timestamp。

## 4. 算法方案

### 4.1 主模型

RTD-YOLO26n:

Input -> YOLO26n shallow backbone -> RTFE -> neck feature fusion -> YOLO26 detect head -> NWD-aware localization -> detection output

推荐优先实现：

- YOLO26n + RTFE
- YOLO26n + NWD
- YOLO26n + RTFE + NWD

### 4.2 可选增强

这些作为增强实验，不建议全部作为主创新：

- P2 small-object head
- YOLO26n-P2
- RTFE placed at P2/P3/P4
- lightweight attention comparison: ECA、CBAM、SimAM、Triplet Attention
- C2f/C3k2 replacement comparison if工程允许

## 5. 数据集设计

### 5.1 公开/整理数据集

使用当前清洗后的 Aircraft AI 三类数据集。

类别：

- missing-rivet
- rust
- scratch

用途：

- 模型训练
- 常规测试
- 消融实验
- 小目标专项评测

### 5.2 金堂无人机自采外部验证集

在沃飞长空金堂实习期间采集真实无人机巡检图像。建议不参与主训练，作为 external validation。这样论文叙事更强：方法在公开/整理数据上训练，在真实无人机场景中验证泛化。

必须记录：

- 无人机型号
- 相机型号
- 采集高度或距离
- 拍摄角度
- 光照条件
- 飞机结构区域
- 缺陷类别
- 图像分辨率
- 是否参与训练、验证或仅测试

### 5.3 鲁棒性测试集

从 clean test 派生多种退化版本。每种退化建议设置 3 个强度等级：

- Level 1: mild
- Level 2: medium
- Level 3: severe

最终可形成：

Clean + 8 corruption types x 3 severity levels。

如果时间不够，优先做 6 类：

- motion blur
- defocus blur
- low-light
- glare / overexposure
- JPEG compression
- scale down-up

## 6. 实验设计总览

### 6.1 实验公平性设置

所有模型尽量统一：

- dataset split
- image size: 800 或 640，建议主实验 800，部署实验可补 640
- epochs: 100 或 150
- batch size: 按显存调整，但记录清楚
- optimizer
- seed
- augmentation profile
- pretrained weights
- evaluation confidence and IoU settings

如果不同模型官方训练策略差异较大，论文中要说明“all models are trained under the same dataset and comparable training settings”。

### 6.2 横向模型对比实验

原计划只放 YOLO11/YOLO26 太少。SCI 写法建议分四组。

必做模型：

- YOLOv5n
- YOLOv5s
- YOLOv7-tiny 或 YOLOv7
- YOLOv8n
- YOLOv8s
- YOLOv9t 或 YOLOv9s
- YOLOv10n
- YOLOv10s
- YOLO11n
- YOLO11s
- YOLO26n
- YOLO26s
- RT-DETR-r18 或 RT-DETR-l
- RTD-YOLO26n

增强版可加：

- YOLOv6n/s
- YOLOX-s
- YOLOv12n，如果代码和训练环境稳定
- YOLOv13n，如果有可靠实现且不影响主线
- Faster R-CNN 或 SSD，作为传统检测器对照

建议表格列：

- Model
- Input size
- Params
- FLOPs
- Precision
- Recall
- mAP50
- mAP50-95
- missing-rivet AP50
- rust AP50
- scratch AP50
- FPS on GPU or CPU

表名：

Table 2. Comparison with mainstream object detection models on the aircraft skin defect dataset.

### 6.3 同领域方法对比

如果能复现或近似复现，最好加“飞机蒙皮/表面缺陷改进 YOLO”对比。不能复现时，可以只在 Related Work 中定性对比，不强行放表。

可选对标类型：

- 改进 YOLOv8 aircraft skin defect methods
- YOLOv11-based steel/surface defect methods
- UAV small object methods such as LRDS-YOLO, TOE-YOLO, DDF-YOLO
- RT-DETR/YOLOv9 aircraft defect comparison papers

注意：如果没有开源代码或数据集不同，不要直接拿论文数值硬比。可以写“not directly comparable due to different datasets”。

### 6.4 消融实验

目的：证明每个设计不是装饰。

基础消融：

- YOLO26n
- YOLO26n + RTFE
- YOLO26n + NWD loss
- YOLO26n + NWD assigner
- YOLO26n + RTFE + NWD loss
- YOLO26n + RTFE + NWD assigner

增强消融：

- YOLO26n + P2
- YOLO26n + P2 + RTFE
- YOLO26n + RTFE at P2
- YOLO26n + RTFE at P3
- YOLO26n + RTFE at P2/P3

表名：

Table 3. Ablation study of RTFE and NWD-aware localization.

### 6.5 模块替换对比

目的：证明 RTFE 不是随便换一个注意力就能得到同样结果。

实验组：

- YOLO26n + SE
- YOLO26n + CBAM
- YOLO26n + ECA
- YOLO26n + SimAM
- YOLO26n + Triplet Attention
- YOLO26n + RTFE

指标：

- mAP50
- mAP50-95
- missing-rivet AP
- scratch AP
- Params increase
- FPS drop

表名：

Table 4. Comparison of different attention or texture enhancement modules.

### 6.6 小目标专项评测

目的：贴合你的数据集中小框比例高的问题。

按归一化框面积分组：

- tiny: area < 0.25%
- small: 0.25% <= area < 1%
- medium/large: area >= 1%

指标：

- AP_tiny
- AP_small
- AP_medium_large
- AR_tiny
- missing-rivet AP

表名：

Table 5. Detection performance under different object scales.

### 6.7 鲁棒性评测

目的：支撑三区论文创新，不只是刷 clean mAP。

模型建议：

- YOLOv8n
- YOLO11n
- YOLO26n
- RTD-YOLO26n

退化条件：

- motion blur
- defocus blur
- Gaussian noise
- low-light
- overexposure / glare
- JPEG compression
- scale degradation
- perspective disturbance

指标：

- mAP_clean
- mAP_corrupt
- mean corruption mAP
- relative degradation rate

公式：

Degradation Rate = (mAP_clean - mAP_corrupt) / mAP_clean

表名：

Table 6. Robustness evaluation under UAV imaging degradations.

图：

- robustness radar chart
- mAP degradation bar chart

### 6.8 外部真实无人机验证

目的：把沃飞长空金堂实习数据变成论文亮点。

推荐设置：

1. Train on public/cleaned dataset, test on JinTang UAV dataset。
2. Train on public/cleaned dataset + 10% JinTang fine-tuning subset, test on remaining JinTang dataset。
3. Compare YOLO26n and RTD-YOLO26n under the same external validation split。

指标：

- mAP50
- Recall
- false negative rate
- per-class AP
- qualitative examples

表名：

Table 7. External validation on real UAV inspection images from Jintang.

### 6.9 边缘部署实验

目的：服务小论文部署创新和大论文数字孪生平台。

部署链路：

PyTorch -> ONNX -> TensorRT FP16 -> TensorRT INT8 optional

平台：

- PC GPU
- CPU-only industrial computer if available
- Jetson Orin Nano 8GB preferred
- Jetson Nano optional

模型：

- YOLOv5n
- YOLOv8n
- YOLO11n
- YOLO26n
- RTD-YOLO26n

指标：

- model size
- latency
- FPS
- memory
- power or temperature if measurable
- export success or compatibility

表名：

Table 8. Edge deployment performance of lightweight models.

### 6.10 数字孪生平台接入实验

目的：把小论文和大论文打通。

实验内容：

- 边缘端输出 detection JSON。
- 数字孪生平台读取 JSON。
- 将缺陷绑定到结构区域。
- 生成复核优先级和维修队列。

指标不一定是 mAP，可以用系统指标：

- average inference-to-display latency
- JSON transmission latency
- defect record generation success rate
- region mapping success rate
- case study screenshots

表名：

Table 9. Digital twin integration performance and case summary.

## 7. 论文图片清单

### Figure 1. Overall research framework

无人机采集 -> 数据治理 -> RTD-YOLO26 检测 -> 鲁棒性评测 -> 边缘部署 -> 数字孪生平台。

### Figure 2. Dataset examples and defect characteristics

展示 missing-rivet、rust、scratch，每类至少 4 张。建议包含：

- clean close view
- small distant view
- glare view
- blur/low-light view

### Figure 3. Label and scale distribution

必须做。包括：

- class distribution bar chart
- bbox area histogram
- tiny/small/medium ratio pie chart or bar chart

### Figure 4. RTD-YOLO26 architecture

核心方法图。画出 YOLO26n 主干、RTFE 插入位置、NWD-aware localization 和输出。

### Figure 5. RTFE module

展示：

- main feature branch
- high-frequency texture branch
- morphology branch
- gate fusion
- output feature

### Figure 6. NWD localization motivation

画小目标框在 IoU 下对偏移敏感，而 NWD 更平滑。这个图能解释为什么 missing-rivet 需要 NWD。

### Figure 7. UAV degradation protocol

同一张 clean image 派生 motion blur、low-light、glare、JPEG、scale degradation 等。

### Figure 8. Qualitative comparison

YOLOv8n、YOLO11n、YOLO26n、RTD-YOLO26n 四列对比。场景包括：

- tiny missing-rivet
- weak scratch
- rust under uneven illumination
- glare false positive
- blurred UAV image

### Figure 9. Heatmap or feature visualization

可用 Grad-CAM、feature map 或 attention response。证明 RTFE 更关注缺陷纹理而不是反光背景。

### Figure 10. Robustness radar or bar chart

展示不同模型在多种退化条件下 mAP 变化。

### Figure 11. Speed-accuracy trade-off

x 轴 latency 或 FPS，y 轴 mAP50，点大小表示参数量。审稿人很喜欢这种图，因为能一眼看到轻量模型价值。

### Figure 12. Edge deployment pipeline

训练端 PC -> model export -> ONNX -> TensorRT -> edge inference -> JSON output -> digital twin platform。

### Figure 13. Edge experiment platform photo

实物图。建议拍：

- 无人机或采集设备
- Jetson/工控机
- 显示屏或平台界面
- 被检飞机或蒙皮样件

### Figure 14. Digital twin platform screenshot

展示缺陷队列、结构区域定位、置信度、复核状态和维修派工。

### Figure 15. Failure cases

展示失败案例和原因分析：

- extreme glare
- serious blur
- very tiny missing-rivet
- ambiguous rust/scratch boundary

## 8. 论文表格清单

### Table 1. Dataset statistics

内容：

- dataset name
- train/val/test images
- external UAV images
- missing-rivet/rust/scratch instances
- tiny/small ratio

### Table 2. Mainstream detector comparison

必须包含 YOLOv5、YOLOv7、YOLOv8、YOLOv9、YOLOv10、YOLO11、YOLO26、RT-DETR、RTD-YOLO26。

### Table 3. Ablation study

RTFE、NWD loss、NWD assigner、组合模型。

### Table 4. Module replacement comparison

SE、CBAM、ECA、SimAM、Triplet Attention、RTFE。

### Table 5. Small object evaluation

tiny、small、medium/large 分组 AP。

### Table 6. Robustness evaluation

不同退化条件下 mAP 和 degradation rate。

### Table 7. External UAV validation

公开数据训练，金堂自采数据测试。

### Table 8. Edge deployment results

PyTorch、ONNX、TensorRT FP16、TensorRT INT8 的 FPS、latency、模型大小、内存。

### Table 9. Digital twin integration case

缺陷编号、类别、置信度、结构区域、风险等级、复核状态、平台响应时间。

## 9. 小论文结构

### 1. Introduction

写作重点：

- 飞机蒙皮缺陷巡检的重要性。
- 无人机场景导致小目标、弱纹理、反光和模糊。
- 现有 YOLO 改进多关注 clean accuracy，缺少真实无人机场景、鲁棒性和边缘部署验证。
- 本文贡献：RTFE、NWD、UAV-Robust benchmark、edge deployment、digital-twin-ready output。

### 2. Related Work

建议分三节：

- Aircraft skin defect detection。
- UAV small object detection and YOLO-based detectors。
- Lightweight deployment and digital-twin-oriented inspection systems。

### 3. Method

建议分四节：

- YOLO26n baseline。
- RTFE module。
- NWD-aware localization。
- Overall RTD-YOLO26 and inference output format。

### 4. Experiments

建议分七节：

- Dataset and implementation details。
- Comparison with mainstream detectors。
- Ablation study。
- Small object evaluation。
- Robustness under UAV degradations。
- External validation on real UAV images。
- Edge deployment and digital twin integration。

### 5. Discussion

讨论：

- 为什么 RTFE 对 scratch/rust 有效。
- 为什么 NWD 对 missing-rivet 有效。
- 速度-精度权衡。
- 外部验证和失败案例。
- 数字孪生平台接入意义。

### 6. Conclusion

总结：

- 精度提升。
- 小目标提升。
- 鲁棒性提升。
- 边缘端实时性。
- 可支撑数字孪生运维闭环。

## 10. 实验优先级

如果时间有限，按这个顺序做：

1. YOLOv5n、YOLOv8n、YOLO11n、YOLO26n、RTD-YOLO26n 主对比。
2. RTFE 和 NWD 消融。
3. 小目标分组评测。
4. 6 类退化鲁棒性评测。
5. 金堂自采外部验证。
6. ONNX/TensorRT 边缘部署。
7. 模块替换对比和 Grad-CAM 可视化。
8. 数字孪生平台接入截图和响应时间。

## 11. 需要补充的材料

1. 自采数据采集记录表

每批次记录无人机型号、相机型号、距离、高度、角度、光照、区域、缺陷类别。

2. 边缘设备实物照片

包括 Jetson/工控机、相机或无人机、运行界面。

3. 平台接入截图

同一缺陷从检测图像到结构区域、复核队列、维修派工的流程截图。

4. 失败案例

强反光误检、极小铆钉漏检、严重模糊漏检。SCI 论文里有失败案例会更可信。

5. 统计图

类别分布、目标尺度分布、退化强度分布、速度-精度权衡图。
