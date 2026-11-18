# 最终 SFT 实验协议

状态：2026-11-09 确认，2026-11-10 补充评估口径约束。本文是本项目最终实验的数据划分、模型和评估协议。项目代码、README、训练配置与评估入口均应以本文为准。

## 1. 最终模型

最终参与 SFT 训练和对比的模型只有两个：

| 模型 | Hugging Face ID | 固定 Revision |
|---|---|---|
| Qwen2.5-3B-Instruct | `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |

Qwen3.5-0.8B、Qwen3-1.7B-Base、Qwen2.5-3B-Base 等早期探索模型不属于最终 SFT 模型。

## 2. MedMCQA 官方 Train 的初始划分

原始来源为 MedMCQA 官方 Train，共 **182,822 条**。初始训练范围按官方 Train 扣除内部验证集计算。

采用 **按规范化题干分组的随机划分**，避免相同题干被拆到训练集和内部验证集两侧。

| 参数 | 固定值 |
|---|---|
| 分组键 | 规范化题干 |
| 规范化方式 | 去首尾空白、连续空白合并为一个空格、casefold |
| 内部验证目标比例 | 5% |
| 随机种子 | `20261020` |
| 随机数生成器 | `random.Random(20261020)` |
| 分组顺序 | 对题干分组键执行固定种子的 shuffle |
| 选取方式 | 按打乱后的分组顺序逐组加入，达到目标数量后停止；一个题干组整体进入同一侧 |

最终划分结果：

| 数据用途 | 数量 | 文件 |
|---|---:|---|
| 初始训练范围 | **173,683** | 按官方 Train 扣除内部验证集计算 |
| 内部验证集 | **9,139** | `data/processed/validation/internal_medmcqa/dev_eval.jsonl` |
| 原始官方 Train | **182,822** | `data/raw/medmcqa/train.parquet` |

`182,822 − 9,139 = 173,683`。清洗时使用的冻结 ID 清单和 SHA256 见[数据清单](../../data_release_manifest.json)的 `artifacts.train_scope`。

**9,139 条内部验证集是在后续训练数据清洗、长度筛选之前，从官方 Train 中预先划出的固定集合。它不是从 137,928 条最终训练文件中再次切分出来的。** 旧路径 `data/processed/dev_eval.jsonl` 是同一份数据的历史副本，不是另一份验证集。

## 3. 从初始训练范围到最终训练文件

初始训练范围共 **173,683 条**。经过数据质量检查、答案与重复问题处理，以及两个模型各自的长度准入检查，最终得到 **137,928 条共同训练样本**。

| 数据文件 | 数量 | 用途 |
|---|---:|---|
| `data/processed/train/common/medmcqa_train_common_137928.jsonl` | 137,928 | 两模型共同训练候选 |
| `data/processed/train/model_specific/qwen25_3b/medmcqa_train.jsonl` | 137,928 | Qwen2.5-3B-Instruct 训练文件 |
| `data/processed/train/model_specific/qwen35_4b/medmcqa_train.jsonl` | 137,928 | Qwen3.5-4B 训练文件 |

两份模型专属文件的样本 ID、顺序和训练文本一致；Token 长度元数据按各自 Tokenizer / Processor 计算。两份文件均通过全量真实 Tokenizer / Processor 长度预检，超过 512 Tokens 的样本为 0，未启用静默截断。当前训练运行时还会验证两个模型专属文件的 source_id 全部属于 SHA256 锁定的训练 ID 清单，并与内部验证及两个独立测试集保持 ID 隔离。

按初始训练范围统计，共有 **35,755 条**未进入最终训练集（`173,683 − 137,928`）。

## 4. 内部验证集

内部验证集固定为 MedMCQA 官方 Train 预先划分出的 **9,139 条**：

- 路径：`data/processed/validation/internal_medmcqa/dev_eval.jsonl`
- 比例目标：5%
- 随机种子：20261020
- 划分方式：按规范化题干分组随机划分
- 用途：训练过程监控、内部验证与模型选择
- 禁止：进入训练 Dataset、参与梯度更新，或与两个独立测试集混称

## 5. 两个独立测试集

### 测试集一：MedMCQA 官方 Validation

- 数量：**4,183 条**
- 文件：`data/evaluation/medmcqa_official_validation/validation.jsonl`
- 用途：最终独立测试
- 不参与 SFT 训练，也不属于内部验证集。

### 测试集二：MedExpQA 全集

项目评估集共 **618 条**，由三个官方 split 组成：

| 官方 split | 数量 |
|---|---:|
| Train | 430 |
| Validation | 63 |
| Test | 125 |
| **合计** | **618** |

评估时完整纳入项目保留的 430 条 Train、63 条 Validation 和 125 条 Test 记录；Train 中的超长隔离记录与质量排除记录仍属于评估集，不能套用训练准入规则再次删除。

MedExpQA 只用于独立测试，绝不加入最终 SFT 训练数据。评估结果需保留并报告原始 split 来源分组。

## 6. 训练方法

两个最终模型采用相同的核心 SFT 配方：

- 监督目标：Answer + Explanation；
- 方法：BF16 基座权重 + LoRA；
- 最大序列长度：512，禁止静默截断；
- Micro Batch：2；
- Gradient Accumulation：4；
- Effective Batch：8；
- Learning Rate：5e-5；
- Epoch：1；
- Optimizer：AdamW；
- Scheduler：Linear；
- Warmup：518 steps；
- Seed：42；
- LoRA：r=8、alpha=16、dropout=0、target modules 为 `q_proj` 和 `v_proj`。

模型固定 revision、模型专属数据 SHA256 和完整 Tokenizer / Processor 预检报告，以各自配置文件及预检报告为准。

## 7. Base / SFT 对比必须遵守的同口径要求

如果报告 Base 与 SFT 的准确率变化，同一模型家族的 Base 和 SFT 必须使用完全一致的评估输入协议：

- 相同题目、选项顺序和选项文本；
- 相同的最终 Answer + Explanation 用户指令，即训练数据中的指令模板；
- 同一模型家族使用相同的 Tokenizer / Chat Template；
- 相同的生成参数、最大生成长度、答案解析器和无效输出计分规则；
- Base 与 SFT 使用同一测试集，并分别报告 MedMCQA 官方 Validation 和 MedExpQA 全集结果。

### 7.1 2026-11-05 Qwen2.5-3B-Instruct Base 历史结果的处理

历史结果保留在：

- 历史探索评估代码与说明：保留在本地历史归档中，不属于当前 GitHub 源码发布内容。
- 原始结果：本地历史产物 `outputs/qwen25_3b_instruct_baseline_validation_20261105/`，不随 Git 发布。
- 按最终 Prompt 准备、尚未执行的 Base runner：`src/medical_sft/evaluation/qwen25_instruct_baseline_matched_20261110.py`

该次历史评估使用 Answer-only 的官方 Validation Prompt，并额外追加“只输出 A/B/C/D”的限制句；最终 SFT 训练指令则要求 Answer + Explanation。两者不是同一 Prompt 协议。此外，该官方 Validation 曾用于早期模型筛选。

因此，历史严格准确率 44.9438%、规范化准确率 47.3344% 只描述当时的探索性评估，**不得用于计算或宣称最终 SFT 相对 Base 的提升幅度**。只删除历史 Prompt 后缀也不足以修复，因为原始基础 Prompt 本身仍然是 Answer-only。

本项目不为修正这组历史数据而重新启动 GPU 推理。`src/medical_sft/evaluation/qwen25_instruct_baseline_matched_20261110.py` 已准备为使用最终 Answer + Explanation 指令的 Base 评估入口，并要求设置 `ALLOW_APPROVED_GPU_EVALUATION=1` 才能启动；目前只完成 CPU 单元测试及 4,183 条官方 Validation Prompt 重建预检，已逐字符确认最终指令与训练数据一致；未执行推理，也没有生成新基线结果。后续 SFT 评估必须复用该脚本的 Prompt 构造、greedy generation、`max_new_tokens=32`、`use_cache=True`、Chat Template 和解析器，才可以进行同口径比较。该 Validation 已用于早期模型筛选，结果仍需注明这一限制。未获明确批准前不得启动评估，也不得补造或推算结果。

## 8. 不得混淆的数量与角色

- **182,822**：MedMCQA 官方 Train 原始总量。
- **173,683**：官方 Train 扣除内部验证集后的初始训练范围。
- **9,139**：从官方 Train 预先划出的内部验证集。
- **137,928**：经过后续训练准入与模型长度适配后，两个模型实际使用的训练样本数。
- **4,183**：MedMCQA 官方 Validation，独立测试集一。
- **618**：MedExpQA 全集，独立测试集二。

任何 README、配置、训练脚本、评估脚本和实验报告都不得把 MedExpQA 写成训练来源，不得把官方 Validation 写成内部验证集，也不得把 173,683 写成最终训练文件的实际样本数。
