# 四模型准确率结果汇总（2026-11-17）

本表汇总四个模型版本在 MedMCQA 和 MedExpQA 上的最终盲测结果。README 和准确率对比图使用本表的 Normalized Accuracy。

## 1. 评估范围与统计口径

- **MedMCQA 最终盲测集**为官方 Validation 的 4,183 条。
- **MedExpQA 最终盲测集**为完整 618 条。上游 Train / Validation / Test 分别为 430 / 63 / 125 条，三个标签记录样本来源。
- 最终发布轮次的模型和评估设置在盲测前确定。模型筛选使用从 MedMCQA 官方 Train 划出的 9,139 条内部验证样本，两套盲测数据均未用于该轮的模型筛选。
- 主展示指标为 **Normalized Accuracy**，分母为全部样本数；无效答案按错误计入。
- 结果表同时列出 Strict Accuracy、正确数和无效答案数。Qwen3.5-4B 两项 SFT 的 Normalized 正确数由表中准确率和样本总数换算。

## 2. 最终准确率

### MedMCQA 官方 Validation（4,183 条最终盲测）

| 模型                  | 类型   | Strict Accuracy | Normalized Accuracy | Strict 正确数 | Normalized 正确数 | 无效答案 |
| ------------------- | ---- | --------------: | ------------------: | ---------: | -------------: | ---: |
| Qwen2.5-3B-Instruct | Base |          36.91% |          **47.98%** |      1,544 |          2,007 |   27 |
| Qwen2.5-3B-Instruct | SFT  |          49.13% |          **50.44%** |      2,055 |          2,110 |    3 |
| Qwen3.5-4B          | Base |          62.13% |          **62.13%** |      2,599 |          2,599 |    1 |
| Qwen3.5-4B          | SFT  |          64.02% |          **64.02%** |      2,678 |          2,678 |    0 |

### MedExpQA 全集（618 条最终盲测）

| 模型                  | 类型   | Strict Accuracy | Normalized Accuracy | Normalized 正确数 | 无效答案 |
| ------------------- | ---- | --------------: | ------------------: | -------------: | ---: |
| Qwen2.5-3B-Instruct | Base |          47.57% |          **47.57%** |            294 |   49 |
| Qwen2.5-3B-Instruct | SFT  |          53.40% |          **53.40%** |            330 |    3 |
| Qwen3.5-4B          | Base |          77.35% |          **77.35%** |            478 |    2 |
| Qwen3.5-4B          | SFT  |          78.96% |          **78.96%** |            488 |    1 |

## 3. SFT 相对 Base 的变化

变化量按表中展示的 SFT 和 Base 的 Normalized Accuracy 相减，单位为百分点（pp）。

| 模型 | MedMCQA | MedExpQA |
|---|---:|---:|
| Qwen2.5-3B-Instruct | +2.46 pp | +5.83 pp |
| Qwen3.5-4B | +1.89 pp | +1.61 pp |

Qwen2.5-3B-Instruct 在 MedExpQA 上的无效答案由 49 条降至 3 条。

## 4. Qwen2.5-3B Base 的 MedMCQA 口径

Qwen2.5-3B Base 在 MedMCQA 上的 Normalized Accuracy 为 **47.98%**，即 4,183 道题中答对 2,007 道。

## 5. 评估设置摘要

- Greedy decoding；
- `max_new_tokens=32`；
- `use_cache=True`；
- BF16；
- 不进行生成截断；
- Qwen3.5 关闭 thinking；
- 同一数据集的题目 Prompt 保持一致，各模型使用各自原生 Chat Template；
- Normalized Accuracy 以全部样本为分母，无效答案计为错误。

同一盲测集的 Base / SFT 使用相同的题目 Prompt 和生成设置。MedMCQA 最终盲测使用 Answer-only，SFT 训练使用 Answer + Explanation。

## 6. 图表

### 最终盲测准确率

- [MedMCQA 准确率对比](../figures/accuracy/medmcqa_accuracy_comparison.png)
- [MedExpQA 准确率对比](../figures/accuracy/medexpqa_accuracy_comparison.png)

### 训练损失

- **Qwen2.5-3B-Instruct** 的前两个 500 Step 区间平均 Loss 从约 2.23 降至 1.94，中后期 100 Step 移动平均逐渐稳定在 1.85 附近。[逐 Step 曲线](../figures/training_loss/qwen25_3b_loss_compact.png) · [500 Step 区间趋势](../figures/training_loss/qwen25_3b_loss_500step.png)
- **Qwen3.5-4B** 的前两个 500 Step 区间平均 Loss 从约 2.01 降至 1.81，后期区间平均 Loss 在 1.70 附近波动。[逐 Step 曲线](../figures/training_loss/qwen35_4b_loss_compact.png) · [500 Step 区间趋势](../figures/training_loss/qwen35_4b_loss_500step.png)

## 7. 结果下载

[准确率汇总 CSV](final_accuracy_results_20261117.csv)记录本表的评估明细，也是绘图脚本的输入。
