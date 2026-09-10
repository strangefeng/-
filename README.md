# 飞机蒙皮缺陷：YOLO26 与采集可靠性实验

本项目按用户提供的新版论文方向建立。当前是数据准备与基线实验阶段，尚无“改进模型提升”的结论。

## 当前数据

- 公开源：Aircraft AI Dataset v2（缺失铆钉、锈蚀、划伤）与 YOLO v1（裂纹、凹陷）。原 YAML 声明 CC BY 4.0；见 `docs/protocol.md` 的来源说明。
- 4,577 张源图，去除 34 张完全重复且标注一致的图，保留 4,543 张。多边形以外接矩形转为检测框；不修改原始文件。
- 固定划分：训练 2,953、验证 766、测试 824。原图名归组并合并 pHash 距离 ≤4 的近似组，共 1,102 组。原划分有 204 个关联组跨集合。
- 该划分是“原图/近似图分组”，不是经过确认的“飞机/飞行批次分组”。衍生混合数据集暂不重复导入；外部 scratch 来源待核实。
- 自采集：221 个 Live Photo 和 7 张照片提取为 228 张真实静态图，全部尚未标注。来源、质量特征、缩略图、COCO 标注任务在 `data/custom_unannotated_v1/`。
- 用户说明有滑翔机与退役运输机，IMU/VIO 以后补充。目录混有其他素材，具体机体归属待核实。空标注任务不能当作负样本或测试真值。
- 已目视筛出 216 张待标注候选，另 12 张为非目标候选。人工分组记录仍需核对。
- `data/degradation_val_v1/` 含 100 个验证源图组 ×27 条件＝2,700 张模拟退化图和对应检测框。仅用于验证阶段开发；不代表真实采集位姿。

## 运行

Windows PowerShell，工作目录为本仓库。GPU 环境放在 `.venv`，不替换原 Python 的 CPU PyTorch。

```powershell
.\.venv\Scripts\python.exe scripts/train_campaign.py --smoke
.\.venv\Scripts\python.exe scripts/train_campaign.py
python scripts/validate_data.py
python tests/test_matching.py
```

正式队列：E01 s/m/l/x @800（60 epochs）→ E02 胜出尺度 @640/800/1024/1280（60 epochs）→ 冻结尺度/分辨率 → B0 三种子（120 epochs）→ P2 对照（120 epochs）。所有筛选只使用验证集。

8 GB 显存默认 microbatch=1，nbs=64；可能需数天。若 OOM，程序记录失败并停止，不自动降分辨率破坏对照。所有参数、日志、结果与 checkpoint hash 保留；失败后检查 `reports/campaign_status.json`。

基线完成后运行：

```powershell
.\.venv\Scripts\python.exe scripts/failure_analysis.py --weights <best.pt的路径> --imgsz <所选分辨率>
```

`failure_analysis.py` 报告固定置信度 0.25、IoU 0.5 的逐 GT 命中、漏检、FP、尺寸和图像质量代理指标。尺寸 Recall 不等于 COCO AP_small。

已实测训练期间默认验证路径不会应用 `end2end=False`；`scripts/otm_trainer.py` 在保留双头训练损失的同时，从 one-to-many 输出计算验证指标，确保 checkpoint 选优与最终推理一致。其数值与直接 one-to-many 推理的等价性测试，以及 1 epoch 训练/验证 smoke test 均已通过。Smoke 指标只验证执行，不构成论文性能证据。

## 仍需完成

1. 正式基线、P2、失效分析后，才定义 M1；有反光失效证据后才考虑 M2。
2. 与 YOLO11、RT-DETR、Faster R-CNN 公平对比；消融与三种子重复。
3. COCO AP_small、分组 paired bootstrap 置信区间；最终模型确定前不评估 test。
4. 自采集专家标注复核、飞机/批次身份确认；补齐同步真实 IMU/VIO、内外参、时间戳与表面几何。
5. 位姿可靠性和 EuRoC 验证。合成退化只能声称模拟实验，不能冒充真实距离、角度或 IMU 测量。

`reports/` 是证据，未运行的实验不得填写指标。当前框架不承诺任何预设提升幅度。原始图像、权重与本地路径清单不纳入 Git。
