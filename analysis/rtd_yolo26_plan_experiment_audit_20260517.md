# RTD-YOLO26 小论文计划与实验审视备忘录

日期：2026-05-17

## 1. 总体判断

当前计划的方向是对的，但需要收缩论文主张。按 Nature/DeepScientist 的审稿标准，论文现在最稳的主线不是“提出一个复杂新结构 RTD-YOLO26 并全面优于所有模型”，而是：

> 通过保守标签治理显著改善飞机蒙皮小缺陷检测的可学习性，并在此基础上用轻量定位损失、无人机场景退化测试和边缘部署验证，形成面向真实巡检的鲁棒轻量检测流程。

也就是说，最强证据已经在“数据治理 + 小目标定位 + 鲁棒评测协议”上；“RTFE/YOLO26 主创新”目前还缺少可支撑的完成结果，不能在摘要和贡献里提前写满。

## 2. 已有证据盘点

### 2.1 标签清洗是当前最强贡献

YOLO11n 在同一 cleaned validation 上：

| 对比 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Raw train + clean eval | 0.606774 | 0.503940 | 0.537558 | 0.251717 |
| Clean train + clean eval | 0.663273 | 0.604508 | 0.620931 | 0.282021 |

关键提升：

- mAP50: +0.083373
- mAP50-95: +0.030304
- missing-rivet AP50-95: 0.012432 -> 0.100167

这是论文里最有说服力的证据：清洗策略和 missing-rivet 小框重标注之间有明确因果关系。

### 2.2 NWD loss 可作为辅助改进，但不要夸大

在 cleaned YOLO11n N2 anchor 上：

| 分支 | mAP50 | mAP50-95 | missing-rivet AP50-95 | rust AP50-95 | scratch AP50-95 |
|---|---:|---:|---:|---:|---:|
| Clean YOLO11n | 0.620931 | 0.282021 | 0.100167 | 0.397271 | 0.348624 |
| NWD loss | 0.616763 | 0.287999 | 0.105997 | 0.411622 | 0.346379 |
| NWD assigner | 0.618351 | 0.273383 | 0.106475 | 0.386980 | 0.326695 |
| Focaler-IoU | 0.456149 | 0.202108 | 0.085062 | 0.296009 | 0.225252 |

允许主张：

- NWD loss 在不改变模型结构的情况下带来小幅 mAP50-95 增益。
- 对 missing-rivet 和 rust 更友好。

不能主张：

- NWD 全面显著优于 baseline。
- NWD assigner 是有效模块；它只提升 missing-rivet，但整体 mAP50-95 下降。

### 2.3 P2/SPD/DyFuse 结构分支目前应作为 negative ablation

P2、SPD、SPD+P2、DyFuse 这批结构分支在 cleaned dataset 上均显著下降。它们不适合继续被包装为主创新。更好的写法是：

- 放到消融或附录中说明“并非所有小目标结构增强都适合该 cleaned small-defect setting”。
- 用它们反衬 NWD loss 的低风险性：不破坏预训练结构，收益虽小但稳定。

### 2.4 最新 baseline suite 还不能作为最终主表

`aircraft_ai_uav_paper_baselines.csv` 当前只汇总到 YOLOv5n/s、YOLOv8n/s、YOLOv9t。B06 YOLOv9s 有 run 目录，但总表没有成功汇总，且本地结果只看到 40 epoch 记录。

已有 baseline 的 mAP50-95 约为 0.196-0.207，明显低于之前 YOLO11n cleaned anchor 的 0.282。这说明两批实验的 dataset yaml、训练批次、模型族或运行状态需要重新核对，不能直接混到一张“最终对比表”里。

## 3. 对当前小论文计划的主要修改

### 3.0 文献锚点

建议把论文定位建立在三个外部锚点上：

- Aircraft skin defect detection 领域已有综述和 YOLO/RT-DETR 对比，说明任务存在明确研究背景。
- NWD 的原始动机是 tiny object 的框偏移敏感性，这与 missing-rivet 小框问题匹配。
- ImageNet-C/常见退化鲁棒性评测提供了“非对抗、真实成像退化”的 benchmark 思路，可借鉴为 UAV corruption protocol。

### 3.1 题目建议降一点风险

原题：

RTD-YOLO26: A Robust Texture-aware Lightweight Detector for UAV-based Aircraft Skin Defect Detection

更稳版本：

Label-Guided Robust Lightweight Detection for UAV-based Aircraft Skin Defects

如果 YOLO26 + RTFE 最终跑出稳定提升，再改回：

RTD-YOLO26: A Robust Lightweight Detector for UAV-based Aircraft Skin Defects with Label-Guided Tiny-Defect Localization

### 3.2 贡献重排

建议贡献顺序改成：

1. A conservative label-governance pipeline for tiny aircraft skin defects, especially missing-rivet relabeling.
2. A lightweight localization refinement using NWD loss for tiny-defect bounding boxes.
3. A UAV-oriented corruption benchmark with blur, low-light, glare, JPEG, noise and scale degradation.
4. A practical edge/digital-twin-ready inference output protocol.

RTFE 暂时放为 optional method extension，不放第一贡献，除非后续实验超过 YOLO11n+NWD 或 YOLO26n baseline。

### 3.3 论文类型从“新网络结构论文”改成“任务驱动方法与验证论文”

按 Nature 风格，读者最关心的是：

- 为什么 aircraft skin defects 在 UAV 场景下难？
- 为什么 label noise 对 missing-rivet 特别致命？
- 你的方法如何让检测结果更可信、更可部署？

而不是先看一个复杂模块图。因此 Introduction 应该从巡检场景和数据噪声切入，再引出模型。

## 4. 下一批实验优先级

### P0：先修复实验矩阵一致性

必须先确认：

- `aircraft_ai_v2_cleaneval_cleantrain.yaml` 与 `aircraft_ai_uav_clean_split.yaml` 是否指向同一 split。
- YOLO11n cleaned anchor 与 new baseline suite 是否使用同样 eval split、imgsz、augment、epoch、seed。
- B06 之后的 YOLOv9s、YOLOv10n/s、YOLO11n/s、YOLO26n/s、RT-DETR 是否完成并汇总。

输出：

- 一张 `paper_experiment_matrix.csv`，每行记录 dataset yaml、run id、weights、epoch、mAP50、mAP50-95、per-class AP、是否可进主文。

### P1：重新跑一个最小闭环主表

主表不要一开始贪多。建议先做：

- YOLOv5n
- YOLOv8n
- YOLOv9t
- YOLO11n
- YOLO26n
- YOLO11n + NWD loss
- YOLO26n + NWD loss

如果 YOLO26n 不明显优于 YOLO11n，则论文不要强行以 YOLO26 命名；改成 cleaned-label + NWD 的 task method。

### P2：做小目标分组评测

这是当前最缺但最能增强说服力的分析。按 bbox area 分 tiny/small/medium-large，分别报 AP/Recall。理由：

- 你的核心证据集中在 missing-rivet。
- missing-rivet 是极小目标。
- 审稿人会问整体 mAP 提升是否只是 rust/scratch 带来的。

### P3：跑鲁棒性评测

你已经生成了 6 类退化 x 3 强度的数据结构：

- defocus blur
- gaussian noise
- glare
- JPEG
- low light
- motion blur
- scale

建议先只评 3 个模型：

- YOLOv8n
- YOLO11n
- YOLO11n + NWD loss 或 YOLO26n + NWD loss

指标：

- clean mAP50-95
- mean corruption mAP50-95
- relative degradation rate
- per-corruption degradation

### P4：外部验证和边缘部署作为加分项，不要压过主线

金堂外部数据和 Jetson/ONNX/TensorRT 很亮眼，但如果主实验矩阵还没闭合，它们会分散论文。建议作为最后两张表：

- External validation: 证明泛化。
- Edge deployment: 证明可用。

数字孪生平台接入建议放 case study，不要作为算法论文的主性能证据。

## 5. Figure/Table 精简建议

原计划 15 个 Figure、9 个 Table 对小论文太重。建议压缩为：

### 主文 Figures

1. Dataset and label-governance overview: 原始错标、清洗后、类别/尺度分布。
2. Method overview: cleaned-label training + NWD localization + robust/deployment evaluation.
3. Main results: speed-accuracy 或 per-class AP comparison。
4. UAV corruption robustness: degradation bars/radar。
5. Qualitative/failure cases: tiny missing-rivet、glare、blur、rust/scratch ambiguity。

### 主文 Tables

1. Dataset statistics and cleaning changes。
2. Main detector comparison。
3. Ablation: clean labels, NWD loss, NWD assigner, failed structure branches。
4. Robustness evaluation。
5. Edge/external validation summary。

其余放 appendix 或 supplementary。

## 6. 不能写或需要降级的主张

目前不能写：

- RTFE 已证明有效。
- RTD-YOLO26 全面优于 YOLO11n/YOLO26n。
- P2/SPD 结构增强适合该数据集。
- NWD assigner 提升整体性能。
- 已完成真实无人机外部泛化验证。
- 已完成边缘部署闭环。

可以写：

- Conservative relabeling substantially improves tiny missing-rivet detection.
- NWD loss provides a small but consistent localization-oriented gain on cleaned labels.
- Structure-heavy small-object modifications can hurt this small cleaned dataset.
- A corruption-based evaluation protocol is prepared and should be used to assess UAV robustness.

## 7. 一句话新版论文故事

飞机蒙皮无人机巡检中的小缺陷检测瓶颈不只是模型结构，而是小目标错标、弱纹理退化和部署约束共同造成的可信检测问题；本文通过保守标签治理、轻量小框定位优化和 UAV 退化评测，构建一个更可复现、更可部署的飞机蒙皮缺陷检测流程。
