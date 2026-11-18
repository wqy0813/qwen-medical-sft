# MedExpQA Full 独立评估集

本目录用于最终模型评估，不用于 SFT 训练。

## Split 组成

| 官方 Split | 本项目采用的数据 | 数量 |
|---|---|---:|
| Train | 512-token 候选 378 条 + 超长隔离 31 条 + 质量排除 21 条 | 430 |
| Validation | 官方 Validation 原样保留 | 63 |
| Test | 官方 Test 原样保留 | 125 |
| **总计** | | **618** |

Train 的 4 条跨 split 重复记录已隔离，不纳入这 618 条；Train 内部的 430 条必须全部参与评估，包括不满足训练长度或质量准入条件的记录。评估集不能只取 378 条训练候选。

## 当前文件组成

| 文件 | 数量 | 官方来源 split | 用途 |
|---|---:|---|---|
| `medexpqa_train_candidates_512.jsonl` | 378 | Train | 评估 |
| `medexpqa_over_512_isolated.jsonl` | 31 | Train | 评估，不能因超长而删除 |
| `medexpqa_quality_excluded.jsonl` | 21 | Train | 评估，不能因训练准入排除而删除 |
| `validation_preserved.jsonl` | 63 | Validation | 评估 |
| `test_preserved.jsonl` | 125 | Test | 评估 |

前三个文件合起来是完整的项目评估 Train split。评估程序统一标记 `source_split=train`。文件名、数量和 SHA256 以本目录 `manifest.json` 为准。

**GitHub 发布说明：** 为避免上传数据集正文，以上 JSONL 文件被 `.gitignore` 排除；公开仓库只保留本说明和 `manifest.json`。克隆仓库后必须按 Manifest 恢复数据文件，才能运行完整评估和依赖真实数据的测试。

## 数据边界

- 不用于训练。
- 不按训练集 A/B/C/D 规则删除评估题目。
- 不因超过 512 Tokens 而从评估集删除题目。
- 保留四选项/五选项题目和原始选项顺序。
- 报告必须分别给出 Train、Validation、Test 来源分组结果，同时给出全体 618 条总体结果。
- 程序结构检查不代表医学答案或解释已经通过专家审核。

上游 revision、snapshot、各 split 数量及当前评估文件 SHA256 以本目录 `manifest.json` 为准。
