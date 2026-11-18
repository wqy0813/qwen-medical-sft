# 项目文件索引

最终数据、训练参数及评估口径以 [最终实验协议](docs/experiment/FINAL_EXPERIMENT_PROTOCOL.md) 为准；项目入口见 [README](README.md)。

## 目录

- `src/medical_sft/data/`、`training/`、`evaluation/`：正式 Python 包，项目路径由 `src/medical_sft/paths.py` 管理。
- `pyproject.toml`：使用 `python -m pip install -e .` 安装项目。
- `docs/experiment/`：冻结的最终实验协议。
- `docs/data/`：MedMCQA 清洗规则及第三方来源与许可。
- `docs/engineering/`：工程主线导航。
- 本地原始数据、训练与内部验证数据、独立评估数据分别位于 `data/raw/`、`data/processed/`、`data/evaluation/`。GitHub 源码仅包含必要的目录说明、数据来源、清单与目录标记。数据正文保存在本地。
- `configs/medmcqa_only_training/` 保存两份正式训练配置和训练入口读取的长度预检参考文件。
- `tests/`：正式链路测试；`tests/test_medmcqa_final.py` 是清洗规则测试。
- `reports/results/` 和 `reports/figures/` 分别保存公开发布的正式实验结果与最终图表。[准确率汇总](reports/results/final_accuracy_results_20261117.md)和[对应 CSV](reports/results/final_accuracy_results_20261117.csv)记录四模型的 Strict Accuracy、Normalized Accuracy、正确数和无效答案数。
- `scripts/` 保存准确率和训练 Loss 绘图入口，以及 `run_tests.py`（公开测试、冻结数据集成测试和完整测试套件入口）。
- `delete/` 是本地历史归档，不随 GitHub 源码发布。历史实验产物和旧模型快照分别归档在 `delete/outputs_legacy/` 和 `delete/models_legacy/`。
- `outputs/`、`artifacts/` 和 `models/` 保存本地实验资产，不随 GitHub 源码发布。4B 模型位于项目外的 `../model_assets/`。

```text
reports/
├── results/
└── figures/
    ├── accuracy/
    └── training_loss/
```

## 正式代码

| 文件 | 用途 |
|---|---|
| `src/medical_sft/data/medmcqa_answers.py`、`src/medical_sft/data/medmcqa_final_rules.py`、`src/medical_sft/data/medmcqa_training_contract.py` | 答案、清洗规则及训练消息格式 |
| `src/medical_sft/data/build_medmcqa_final.py`、`src/medical_sft/data/verify_medmcqa_final.py` | 清洗构建及核验 |
| `src/medical_sft/data/build_model_specific_medmcqa.py`、`src/medical_sft/data/verify_model_specific_medmcqa.py`、`src/medical_sft/data/preflight_model_specific_medmcqa.py` | 模型专属数据构建、内容核验与长度预检 |
| `src/medical_sft/training/current_medmcqa_training.py` | 两模型共享训练运行时 |
| `src/medical_sft/training/train_qwen25_3b_medmcqa.py`、`src/medical_sft/training/train_qwen35_4b_medmcqa.py` | 两个最终训练入口 |
| `src/medical_sft/evaluation/medmcqa_evaluation.py`、`src/medical_sft/evaluation/medexpqa_evaluation.py` | 两个测试集的读取、评估和评分辅助 |
| `src/medical_sft/evaluation/qwen25_instruct_baseline_matched_20261110.py` | 尚未执行推理的 Base 匹配 Prompt 准备脚本；依赖本地模型和历史输出，独立克隆后不能直接运行 |
| `scripts/plot_accuracy_comparison.py` | 从 `reports/results/final_accuracy_results_20261117.csv` 生成 MedMCQA 与 MedExpQA 准确率对比图 |
| `scripts/plot_training_loss.py` | 读取正式训练日志并绘制 Loss 图 |
| `scripts/run_tests.py` | 分组运行 106 项公开测试、11 项冻结数据集成测试或完整 117 项测试 |

最终训练数据来自 MedMCQA 官方 Train；内部验证 9,139 条，模型专属训练文件各 137,928 条。MedMCQA 官方 Validation 4,183 条和 MedExpQA 618 条仅作独立测试。不得重新划分、重建覆盖已有数据，也不得擅自启动 GPU 训练或推理。

Qwen2.5-3B-Instruct 的正式配置使用本地快照 `models/Qwen2.5-3B-Instruct-aa8e72537993/`。项目外的同 revision 副本 `../model_assets/Qwen2.5-3B-Instruct/aa8e72537993ba99e69dfaafa59ed015b17504d1/` 仅含配置和 Tokenizer 文件。数据清单 `data_release_manifest.json` 位于项目根目录。
