# Aircraft AI Dataset 第六轮实验设计

## 1. 这一轮要解决什么问题

前五轮已经把方向筛得很清楚了：

- 复杂结构改造没有形成稳定收益。
- `N2` 仍然是当前最强锚点。
- `Fit within` 只对 `missing-rivet` 有轻微帮助，但整体没有超过 `N2`。

所以第六轮不再问“模型结构怎么变”，而是直接问：

**怎样通过更贴近任务本身的数据侧方法，把当前 `N2` 真的往上推一点。**

这比继续做结构花活更有希望得到实际效果。

## 2. 第六轮核心思路

第六轮分成两条线，但优先级不同：

### 主线：Focus Crop

针对 `missing-rivet` 这类小目标，给训练集增加“以小目标为中心的局部裁剪样本”，让模型在训练时更频繁地看到放大的缺陷区域。

这条线的好处是：

- 不改模型结构，风险低
- 更符合你前面暴露出来的真实瓶颈：目标太小、细节容易被破坏
- 只增强训练集，验证集和测试集不动，比较公平

### 备用线：Manual Clean

保留一个“人工清洗后重训”的分支，但默认不直接运行。先把 review pack 自动生成出来，等你人工处理完再把这个分支放开。

## 3. 第六轮实验矩阵

| 编号 | 数据集 | 配方 | 作用 |
| --- | --- | --- | --- |
| `S1` | `v1 + focus crop` | `N2` 配方 | 验证小目标焦点裁剪能否直接推动现有最优锚点 |
| `S2` | `v2 fit-within + focus crop` | `N2` 配方 | 验证保比例导出和 focus crop 是否有协同 |
| `S3` | `v1 + focus crop` | `N2` 配方 + 更长训练 | 判断 richer train split 是否需要更长收敛 |
| `C1` | 手工清洗数据占位 | `N2` 配方 | 预留给人工清洗后的重训分支，默认阻塞 |

对应锚点：

- `N2`：当前主锚点
- `F2`：第五轮最优 fit-within 对照

## 4. 为什么 focus crop 比继续改结构更值得做

这一点其实已经被前面结果暗示出来了：

- `P2` 头失败，不代表“小目标问题不存在”
- 它只说明“直接加结构”不是当前这套数据上最有效的解法
- 第五轮里 `missing-rivet` 的轻微改善说明，小目标仍是最值得攻的点

换句话说：

**问题大概率不在 backbone 不够花，而在训练时小目标可见性不够。**

第六轮就是专门围绕这个问题下手。

## 5. 自动化流程设计

我把第六轮设计成了三段自动化：

1. 生成数据
   用 `build_focus_crop_dataset.py` 从原始 3 类数据集中生成带 focus crop 的训练集变体。

2. 生成 review pack
   用现有 `review_tools` 自动导出一份可疑样本复核包，供后续人工清洗分支使用。

3. 跑训练并汇总
   用 `run_aircraft_ai_round6.py` 一次性完成 prepare、train、summary。

## 6. 这轮的成功判据

优先顺序仍然是：

1. `mAP50-95`
2. `missing-rivet AP50-95`
3. 其余类别是否明显塌陷

更具体地说：

- 如果 `S1` 超过 `N2`，说明 focus crop 这条主线成立
- 如果 `S2` 超过 `S1`，说明 fit-within 只有在配合小目标裁剪时才真正有价值
- 如果 `S3` 超过 `S1`，说明 richer train split 需要更长收敛时间
- 如果三者都没有明显收益，第七轮就不该继续做训练侧小改动，而应转向 manual clean

## 7. 已经落到仓库里的文件

- 方案文档：`analysis/aircraft_ai_dataset_round6_plan.md`
- 实验清单：`experiments/aircraft_ai_dataset_round6.json`
- 数据构建脚本：`scripts/build_focus_crop_dataset.py`
- 准备脚本：`scripts/prepare_aircraft_ai_round6.py`
- 运行脚本：`scripts/run_aircraft_ai_round6.py`
- 汇总脚本：`scripts/summarize_aircraft_ai_round6.py`

## 8. 推荐执行顺序

如果你准备正式开第六轮，我建议顺序是：

1. 先 dry-run 检查 manifest 和数据准备逻辑
2. 正式运行 `S1 S2 S3`
3. 看 summary
4. 再决定是否启动 `C1` 的人工清洗分支

一句话总结：

**第六轮已经从“继续调模型”切换成“围绕 `N2` 做小目标可见性增强和清洗预备”，这是目前最有希望带来实际效果的路线。**
