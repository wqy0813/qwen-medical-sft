# 复算 Qwen2.5 的 MedExpQA 成绩

这份文件记录同一组 618 道题上 Base 和 SFT 的原始回答。克隆仓库后，在仓库根目录用 Python 运行评分脚本即可复算成绩和逐题变化。[下载逐题预测](predictions.jsonl)

```bash
python scripts/rescore_qwen25_medexpqa.py --changes qwen25-medexpqa-changes.csv
```

| 模型 | Normalized Accuracy | 无效输出 |
| --- | ---: | ---: |
| Base | 294 / 618（47.57%） | 49 |
| SFT | 330 / 618（53.40%） | 3 |

72 道从未答对变为答对，36 道从答对变为未答对。分母为全部 618 道题，无法解析的输出计错。在导出的 CSV 中，空白作答表示原始输出无法解析。[评分规则](../../src/medical_sft/evaluation/medexpqa_evaluation.py)

## 预测文件

每行一个 JSON 对象。题目编号 `id` 和上游划分 `source_split` 可用于关联原数据。文件提供标准答案 `gold`、可用选项字母 `allowed_options`，以及两次生成的原始回答 `base_output` 和 `sft_output`。这里的上游划分记录题目来源，618 道题均用于本项目的盲测。[题目划分与数据清单](../../data/evaluation/medexpqa_full/manifest.json)

- **基座**使用 Qwen2.5-3B-Instruct，revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`。[基座模型](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/tree/aa8e72537993ba99e69dfaafa59ed015b17504d1)
- **SFT**使用仓库里的最后一步 LoRA。适配器权重的 SHA256 为 `864a6c534abd2e6dff168a67c24520aafb587b74d30fe08e197dad10cf37b40e`。[适配器与许可](../qwen25-3b-final-lora/)
- **生成设置**在题目和选项末尾加 `Return exactly one line in the format: Answer: X`。两次均用基座自带的聊天模板，使用 BF16、贪心解码、`max_new_tokens=32` 和 `use_cache=True`。
- **文件校验**按 LF 换行计算，`predictions.jsonl` 的 SHA256 为 `6532fe8678d2a52e37802bfc64d81e832fa336aa6bb4dd6345434ad33a6f83d5`。

## 来源

题目编号、划分和答案来自 Alonso、Oronoz 和 Agerri 发布的 MedExpQA 英文数据。答案数字转换成对应的选项字母，项目生成的 Base 和 SFT 回答按题目编号配对加入。上游数据采用 CC BY 4.0。[数据快照](https://huggingface.co/datasets/HiTZ/MedExpQA/tree/78b57699763e00e38acd550f28de24d8a65f1643) · [论文](https://doi.org/10.1016/j.artmed.2024.102938) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
