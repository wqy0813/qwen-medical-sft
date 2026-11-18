# 最终实验工程导航

最终实验的模型、数据划分、训练参数和评估口径以 [实验协议](../experiment/FINAL_EXPERIMENT_PROTOCOL.md) 为准；[数据清洗规则](../data/MEDMCQA_FINAL_RULES.md) 描述 MedMCQA 清洗方法。项目入口和完整文件导航分别见 [README](../../README.md) 和 [文件索引](../../PROJECT_FILE_INDEX.md)。

当前主线依次为：MedMCQA 数据清洗、共同训练数据和模型专属文件、两个模型的 BF16 LoRA 训练、两个独立测试集的评估。MedExpQA 只用于评估。正式配置位于 `configs/medmcqa_only_training/`；现有冻结数据、模型及运行结果应原位保留。

MedMCQA 清洗入口为 `src/medical_sft/data/build_medmcqa_final.py` 和 `src/medical_sft/data/verify_medmcqa_final.py`；模型专属数据入口为 `src/medical_sft/data/build_model_specific_medmcqa.py`、`src/medical_sft/data/verify_model_specific_medmcqa.py` 和 `src/medical_sft/data/preflight_model_specific_medmcqa.py`。训练入口为 `src/medical_sft/training/train_qwen25_3b_medmcqa.py` 和 `src/medical_sft/training/train_qwen35_4b_medmcqa.py`，共用 `src/medical_sft/training/current_medmcqa_training.py`。评估功能位于 `src/medical_sft/evaluation/medmcqa_evaluation.py` 和 `src/medical_sft/evaluation/medexpqa_evaluation.py`。现阶段尚无经过最终协议验收的统一 Base/SFT 推理入口。

现有数据已经构建，无需为了整理再次清洗、构建或训练。完整的数据来源和许可说明见 [第三方声明](../data/THIRD_PARTY_NOTICES.md)。

在项目根目录安装开发依赖并运行 `python -m pip install -e .`。测试分为不依赖冻结数据的公开组（106 项）和需要恢复冻结数据的集成组（11 项）；完整套件共 117 项。测试入口：

```bash
python scripts/run_tests.py public  # 106 项，无需恢复数据正文
python scripts/run_tests.py data    # 11 项，需要恢复冻结数据
python scripts/run_tests.py all     # 完整 117 项
```

常用只读检查：

```bash
python -m medical_sft.data.build_medmcqa_final --mode preflight
python -m medical_sft.data.verify_model_specific_medmcqa
python -m medical_sft.evaluation.medmcqa_evaluation --split validation
python -m medical_sft.evaluation.medexpqa_evaluation --help
python -m medical_sft.training.train_qwen25_3b_medmcqa --help
python -m medical_sft.training.train_qwen35_4b_medmcqa --help
python scripts/run_tests.py all
```

需要重新构建时，`python -m medical_sft.data.build_model_specific_medmcqa --help` 和 `python -m medical_sft.data.preflight_model_specific_medmcqa --help` 列出入口；现有数据和报告不可覆盖。模型长度预检默认写入 `outputs/preflight/model_specific/`，训练入口的 `--preflight` 写入 `outputs/preflight/medmcqa_training/`。训练入口读取的长度参考文件随训练配置保存在 `configs/medmcqa_only_training/`。完整 GPU 运行须另获批准。
