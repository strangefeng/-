# 预先固定的实验规则

日期：2026-09-11。研究目标：采集质量 → 缺陷检测 → 检测可靠性。准确率优先，不沿用 YOLO11n 轻量化实验。

## 来源与证据

- https://universe.roboflow.com/strangefeng/aircraft-ai-dataset-3dl7k/dataset/2 ：本地导出 YAML 声明 CC BY 4.0、3 类；应继续追溯原始贡献者与逐图授权。
- https://universe.roboflow.com/strangefeng/yolo-tb1l6-yyu4i/dataset/1 ：本地导出 YAML 声明 CC BY 4.0、2 类，含 polygon 标注。
- https://docs.ultralytics.com/models/yolo26 ：2026-09-11 核实 YOLO26 与官方 P2。P2 只有架构，不存在官方 scale-specific P2 pretrained checkpoints。
- https://pytorch.org/get-started/previous-versions/ ：torch 2.10.0 + torchvision 0.25.0 的 CUDA 12.6 安装组合。

本机固定 ultralytics 8.4.22。其源码参数 `end2end=False` 才是 one-to-many 路径；`nms` 在本版本主要是导出选项，不能照搬较新文档来切检测头。checkpoint 来源、实际版本与 SHA 在每次结果里记录；不得未经核实写成 Objects365+COCO 预训练。

science 技能的专用 bash_exec/artifact.science 工具未在本会话提供，因此使用本机 shell 执行，并将证据存于 reports，不宣称已写入专用证据图。

## 数据划分

全部原始数据只读。多边形外接框用于检测，语义仍受原始标注质量限制；原图名、完全像素哈希和 pHash 构建连通分组。分组 SHA + seed 42 确定 70/15/15 目标比例。实际比例由整组归属决定。测试集仅允许数据结构审计，在最终模型和阈值确定前不用于性能选择。

未知飞机和拍摄批次是当前公开数据的局限。不能据此宣称按飞机无泄漏，也不能保证 pHash 捕获所有裁剪与视角变化。对于已知来源相同的后续数据必须补充 group 元数据再建立新版本。

自建真实数据与公开数据不同域；禁止自动将未检测到目标的照片标为无缺陷。模型候选框只能是预标注，须人工/专家复核。两架飞机不足以支持同时独立的 train/val/test 三分组，可以做预先固定的一架适配、一架留出或双向 leave-one-aircraft-out；采集轮次仍须分组。

## 筛选与对照

E01 60 epochs 是统一预算的尺度筛选，不是充分收敛证明。E02 依序筛分辨率而非穷举尺度×分辨率。学习率、SGD、增强、seed、microbatch=1、nbs=64 固定。B0 使用 120 epochs 与 seeds 42/3407/2026。P2 首轮 seed 42，后续若参与最终论文比较应补齐同样种子。

P2 从常规模型迁移兼容权重；它与完整预训练 B0 的初始化覆盖不同，报告迁移比例，必要时加入同初始化训练对照，不能把结果全归因于架构。

优先 mAP50:95、Recall、每类 Recall。Ultralytics 常规 Recall 往往对应内部选定工作点；失效分析固定 confidence=0.25，两者需明确区分。官方 COCO 小目标 AP 需要正确 areaRng/maxDets/全量预测评估，不能用小目标样本上的普通 AP 代替。

## 改进与统计

M1（小目标细节融合）与 M2（局部对比度/光照稳健特征）目前都是假设，尚无实现与提升证据。P2、增强和调高分辨率不能冒充新方法。改进若无效也保存记录，不通过改测试集或挑 seed 获得正结果。

在训练/验证上开发可靠性模型，应使用交叉拟合或独立 calibration 数据；不能把同一源图的退化版本分别放进可靠性训练与测试。GT 局部尺寸与质量仅用于回顾分析，不能作为部署时已知变量。IMU/VIO 本身不提供表面距离和法向，仍需表面几何、标定和统一尺度。

退化版本继承源图组。距离/角度若来自仿真参数必须标明 synthetic，不填入真实位姿列。最终对 baseline 与 selected model 做同测试组配对 bootstrap；以源图/机体或采集批次为重采样单位，避免将高度相关帧当独立样本。
