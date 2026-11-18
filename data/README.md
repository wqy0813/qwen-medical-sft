# 数据目录说明

本目录按最终实验中的数据角色组织。数据划分和数量见[根目录数据清单](../data_release_manifest.json)。上游数据源见 [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) 和 [MedExpQA](https://huggingface.co/datasets/HiTZ/MedExpQA)。

**GitHub 发布范围：** 为避免上传约 3.5 GB 的本地数据正文，原始数据、训练 JSONL、内部验证集和大部分评估 JSONL 均不纳入 Git。公开仓库只保留目录说明、Manifest 和空目录标记。克隆仓库后需从官方数据源或受控的数据备份恢复文件到下列路径，并按 Manifest 核对 SHA256；没有恢复数据文件时，完整数据验收、训练预检和评估测试无法运行。

**测试分组：** 安装开发依赖后，可先运行 `python scripts/run_tests.py public`，执行 106 项不依赖冻结数据正文的测试。恢复本文件所列冻结数据后，再运行 `python scripts/run_tests.py data` 执行 11 项数据集成测试；`python scripts/run_tests.py all` 运行完整 117 项。测试不需要恢复模型权重或 `outputs/`。

## 一、MedMCQA 官方 Train：初始划分

原始官方 Train 共 **182,822** 条。初始训练范围按官方 Train 扣除内部验证集计算。

采用按规范化题干分组的随机划分：

- 题干规范化：去首尾空白、连续空白合并、casefold。
- 内部验证目标比例：5%。
- 固定随机种子：`20261020`。
- 随机方式：`random.Random(20261020)`。
- 先打乱规范化题干分组，再按完整题干组分配，避免相同题干跨训练/验证两侧。

| 数据角色 | 数量 | 路径 |
|---|---:|---|
| 官方 Train 原始数据 | 182,822 | `raw/medmcqa/train.parquet` |
| 初始训练范围 | 173,683 | 按官方 Train 扣除内部验证集计算 |
| 内部验证集 | 9,139 | `processed/validation/internal_medmcqa/dev_eval.jsonl` |

`182,822 − 9,139 = 173,683`。清洗时使用的冻结 ID 清单和 SHA256 见[数据清单](../data_release_manifest.json)的 `artifacts.train_scope`。

**内部验证集 9,139 条是在后续训练数据清洗和模型长度适配之前，从官方 Train 预先划出的固定集合。它不是从最终 137,928 条训练文件中重新切分出来的。** 根目录旧路径 `data/processed/dev_eval.jsonl` 是同一数据的历史副本。

## 二、最终 SFT 训练数据

初始训练范围为 **173,683 条**。后续按项目冻结的数据质量规则、答案/重复问题处理和模型长度准入处理，最终得到 **137,928 条共同训练样本**。

| 文件 | 数量 | 用途 |
|---|---:|---|
| `processed/train/common/medmcqa_train_common_137928.jsonl` | 137,928 | 共同训练候选 |
| `processed/train/model_specific/qwen25_3b/medmcqa_train.jsonl` | 137,928 | Qwen2.5-3B-Instruct 训练文件 |
| `processed/train/model_specific/qwen35_4b/medmcqa_train.jsonl` | 137,928 | Qwen3.5-4B 训练文件 |

两个模型专属文件的样本 ID、顺序和训练文本一致；完整 Token 长度由各自 Tokenizer / Processor 计算。两份文件均已通过真实 Tokenizer / Processor 全量预检，长度上限 512，超长样本 0，静默截断 0。最大完整长度分别为 Qwen2.5-3B-Instruct **512**、Qwen3.5-4B **506**。

按初始训练范围统计，共有 **35,755 条**未进入最终训练集（`173,683 − 137,928`）。

## 三、内部验证集

- 路径：`processed/validation/internal_medmcqa/dev_eval.jsonl`
- 数量：**9,139**。
- 来源：MedMCQA 官方 Train。
- 划分：规范化题干分组随机划分，5% 目标比例，seed=20261020。
- 用途：内部验证、训练监控和模型选择。
- 不参与梯度更新，也不属于最终测试集。

## 四、两个独立测试集

### 测试集一：MedMCQA 官方 Validation

- 数量：**4,183**。
- 原始数据：`raw/medmcqa/validation.parquet`。
- 评估副本：`evaluation/medmcqa_official_validation/validation.jsonl`。
- 用途：最终独立测试，不是内部验证集。

### 测试集二：MedExpQA 全集

MedExpQA 在本项目中只用于评估，不参与 SFT 训练。评估集共 **618** 条：

| 官方 split | 数量 | 组成 |
|---|---:|---|
| Train | 430 | 378 条长度合格候选 + 31 条超长隔离 + 21 条质量排除 |
| Validation | 63 | 官方 Validation 保留记录 |
| Test | 125 | 官方 Test 保留记录 |
| **合计** | **618** | 全部纳入评估 |

评估时必须纳入 MedExpQA Train split 的全部 430 条，包括不满足训练长度或训练准入条件的记录；不能套用训练准入规则再次删除。四条跨 split 重复 Train 记录已按评估集构建规则隔离，Validation/Test 记录保持保留。

数据文件位于 `evaluation/medexpqa_full/`，组成与 SHA256 见该目录的 `manifest.json`。原始官方 Train、Validation、Test 文件位于 `raw/medexpqa/`。

## 五、目录结构

```text
data/
├── raw/
│   ├── medmcqa/
│   │   ├── train.parquet
│   │   └── validation.parquet
│   └── medexpqa/
├── processed/
│   ├── train/
│   │   ├── medmcqa_original_train_scope_ids.jsonl
│   │   ├── common/
│   │   └── model_specific/
│   │       ├── qwen25_3b/
│   │       └── qwen35_4b/
│   └── validation/
│       └── internal_medmcqa/
│           └── dev_eval.jsonl
└── evaluation/
    ├── medmcqa_official_validation/
    │   └── validation.jsonl
    └── medexpqa_full/
        ├── manifest.json
        ├── medexpqa_train_candidates_512.jsonl
        ├── medexpqa_over_512_isolated.jsonl
        ├── medexpqa_quality_excluded.jsonl
        ├── validation_preserved.jsonl
        └── test_preserved.jsonl
```

## 六、数据使用边界

1. 最终训练模型只有 Qwen2.5-3B-Instruct 和 Qwen3.5-4B。
2. 最终 SFT 训练只读取各自的 MedMCQA 模型专属训练文件。
3. 内部验证集 9,139 条不参与梯度更新。
4. MedMCQA 官方 Validation 4,183 条和 MedExpQA 全集 618 条是两个独立测试集。
5. MedExpQA 的任何 split 均不得加入最终 SFT 训练数据。
6. 评估集不得套用训练集 A/B/C/D 训练准入规则再次删题。
7. 不重新生成或重新划分现有 Train/内部验证文件；代码整理只核对路径、数量和哈希，不改动数据内容。
