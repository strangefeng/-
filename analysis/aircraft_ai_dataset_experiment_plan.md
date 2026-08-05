# Aircraft AI Dataset 论文实验方案表

## 研究目标

面向无人机视角下的飞机蒙皮缺陷检测，围绕 `missing-rivet`、`rust`、`scratch` 三类目标，验证在公开数据集 `Aircraft AI Dataset` 上进行轻量结构改进与监督策略改进后，是否能够在保持部署可行性的前提下提升检测性能。

## 实验设置

- 数据集: `Aircraft AI Dataset.v1i.yolov11`
- 类别: `missing-rivet`, `rust`, `scratch`
- 划分: `688 train / 197 val / 98 test`
- 基线模型: `YOLO11n`
- 统一训练预算: `imgsz=640`, `epochs=100`, `batch=16`, `optimizer=auto`, `seed=0`
- 统一比较指标: `Precision`, `Recall`, `mAP50`, `mAP50-95`, 各类 `AP50`, 各类 `AP50-95`

## 论文实验方案表

| 组别 | 类型 | 模型/配置 | 核心改动 | 指标 | 预期作用 |
| --- | --- | --- | --- | --- | --- |
| B0 | Baseline | YOLO11n default | 官方默认训练配置 | P, R, mAP50, mAP50-95, 各类 AP | 作为主基线，建立公开数据集上的可复现实验起点 |
| B1 | Baseline+ | YOLO11n texture-preserving | 减弱强增强，保留更稳定的弱纹理信息 | 同上 | 验证对细小裂纹、锈蚀边界、弱纹理目标更友好的训练策略是否有效 |
| M1 | Ablation | YOLO11n + Neck WT Refine | 在 neck 融合阶段引入小波细节增强 | 同上 | 强化多尺度融合中的细节恢复，改善复杂背景下的小目标与弱纹理特征 |
| M2 | Ablation | YOLO11n + WTConv | 关键下采样处采用小波辅助下采样 | 同上 | 减少下采样过程中细节丢失，提升小目标和细长目标保真度 |
| M3 | Ablation | M2 + NWD Loss | 将边框回归加入 NWD 相似度监督 | 同上 | 缓解 IoU 对小目标回归过于敏感的问题，提升定位稳定性 |
| M4 | Ablation | M2 + NWD Loss + NWD Assigner | 在样本分配阶段引入 NWD 协同监督 | 同上 | 提升难样本和微小目标的正负样本匹配质量 |
| M5 | Final | M4 + Rep Neck | 在最终结构上加入更强 neck 表达 | 同上 | 形成论文最终模型，验证结构增强与监督增强的叠加效果 |

## 消融逻辑

1. `B0 -> B1`
   先验证训练策略本身是否对该类弱纹理数据更友好。

2. `B1 -> M1 / M2`
   分别验证 neck 细节增强与 backbone 下采样增强的独立收益。

3. `M2 -> M3 -> M4`
   在结构改进成立的基础上，逐步验证 `NWD loss` 与 `NWD assigner` 的增益。

4. `M4 -> M5`
   验证更强 neck 表达是否能带来最终性能提升，形成论文主模型。

## 论文中建议重点汇报的结果

- 总体指标:
  - `mAP50`
  - `mAP50-95`
  - `Precision`
  - `Recall`
- 分类别指标:
  - `missing-rivet AP50/AP50-95`
  - `rust AP50/AP50-95`
  - `scratch AP50/AP50-95`
- 复杂度指标:
  - 参数量
  - FLOPs
  - 推理速度
- 可视化结果:
  - `missing-rivet` 小目标密集区域
  - `rust` 纹理不稳定区域
  - `scratch` 细长弱纹理区域

## 预期论文结论方向

- 公开飞机蒙皮缺陷数据集上的 `YOLO11n` 可以作为稳定基线；
- 保守增强策略更适合弱纹理、细粒度工业缺陷；
- 小波结构增强与 NWD 类监督机制对小目标和细长目标更敏感；
- 最终模型有望在 `mAP50-95` 和难类指标上优于纯官方基线。
