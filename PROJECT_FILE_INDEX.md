# 项目文件索引

项目入口见 [README](README.md)。数据划分与评估范围见[数据目录说明](data/README.md)，训练参数见[训练配置](configs/medmcqa_only_training/)。

## 目录

- `src/medical_sft/data/`、`training/`、`evaluation/`：正式 Python 包，项目路径由 `src/medical_sft/paths.py` 管理。
- `pyproject.toml`：使用 `python -m pip install -e .` 安装项目。
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
| `src/medical_sft/evaluation/qwen25_instruct_baseline_matched_20261110.py` | 历史探索的 Base 匹配 Prompt 对照脚本，使用 Answer + Explanation，依赖本地模型和历史输出，尚未执行推理 |
| `scripts/plot_accuracy_comparison.py` | 从 `reports/results/final_accuracy_results_20261117.csv` 生成 MedMCQA 与 MedExpQA 准确率对比图 |
| `scripts/plot_training_loss.py` | 读取正式训练日志并绘制 Loss 图 |
| `scripts/run_tests.py` | 分组运行 106 项公开测试、11 项冻结数据集成测试或完整 117 项测试 |

最终训练数据来自 MedMCQA 官方 Train。内部验证 9,139 条，模型专属训练文件各 137,928 条。MedMCQA 官方 Validation 4,183 条和 MedExpQA 全集 618 条均用于最终盲测，两个数据集均未参与该轮模型筛选。

Qwen2.5-3B-Instruct 的正式配置使用本地快照 `models/Qwen2.5-3B-Instruct-aa8e72537993/`。项目外的同 revision 副本 `../model_assets/Qwen2.5-3B-Instruct/aa8e72537993ba99e69dfaafa59ed015b17504d1/` 仅含配置和 Tokenizer 文件。数据清单 `data_release_manifest.json` 位于项目根目录。

## 运行代码测试

修改数据清洗、答案解析或命令行入口后，可运行 `public` 测试组检查这些功能。它包含 106 项测试，使用 CPU 和测试样例。以下命令在仓库根目录运行，使用 Python 3.12 和 Windows PowerShell。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install numpy transformers pyarrow
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts/run_tests.py public
```

在 Linux 和 macOS 上，虚拟环境的解释器路径为 `.venv/bin/python`。按平台选择 CPU 版 PyTorch 的安装命令。[PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)
