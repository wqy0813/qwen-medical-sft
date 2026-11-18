# MedMCQA 最终清洗规则与执行规格

状态：**2026-10-31 已冻结并应用于全量数据，独立验收通过**。本文件与 `medmcqa_final_rules.py` 是 MedMCQA 内容清洗的唯一最终规则规格和实现；最终流程不导入历史筛选器，也不读取历史筛选版本输出。
规则版本：`medmcqa-final-20261031`。训练长度基线：512 Tokens。最终模型与完整实验划分见[最终实验协议](../experiment/FINAL_EXPERIMENT_PROTOCOL.md)。

## 0. 执行边界

- 不增加医学、语义或其他筛选规则，不恢复历史规则，不调用模型逐条审题或改写。
- 输入范围、路径、数量和 SHA256 由根目录 `data_release_manifest.json` 锁定。
- 原始数据和既有 train/dev/保留评估文件只读；所有新结果必须写入独立新目录，不覆盖已验收交付。
- 不读取其他候选集、历史筛选结果或外部隔离清单来改变本轮输入范围。
- 所有修改必须有原文、新文、字段、ID、规则、位置和说明；不修改原始 gold。
- 弱提示不改变质量类别，不确定时保留证据；不能把低置信度提示升级为异常。
- 冻结 ID 清单中的记录已完成正式清洗，并通过独立全量验证。
- 本文件冻结的是数据清洗规则；最终 SFT 模型、137,928 条模型专属训练文件、9,139 条内部验证集和两个测试集的角色见[最终实验协议](../experiment/FINAL_EXPERIMENT_PROTOCOL.md)。
- A/B 不代表医学事实经过核验，也不等于正式训练发布许可。

## 1. 输入范围、字段与来源

以根目录 `data_release_manifest.json` 定位输入，不从目录扫描或历史报告推断数据入口。

| 用途 | manifest 逻辑键与文件 | 数量 |
|---|---|---:|
| 本轮 ID 范围和输入顺序 | `train_scope`：`data/processed/train/medmcqa_original_train_scope_ids.jsonl` | 见 `artifacts.train_scope.rows` |
| 原始题干、选项、答案、解释 | `raw.train`：`data/raw/medmcqa/train.parquet` | 182,822 |
| 唯一活跃评估入口 | `dev_eval`：`data/processed/validation/internal_medmcqa/dev_eval.jsonl` | 9,139 |
| 历史退役的消息别名（非活跃输入） | `dev_sft`：已退役的历史消息别名，当前不在活跃 manifest 中 | 9,139，不重复计数 |

`train_scope` 仅保留原始训练子集的 ID 和顺序，不含训练 prompt、答案或解释。按 ID 从原始 Parquet 读取：
`id`、`question`、`opa/opb/opc/opd`、`cop`、`exp`、`choice_type`、`subject_name`、`topic_name`。
保留完整原始记录、源文件哈希、源行号和本轮输入行号。

来源：`openlifescienceai/medmcqa`，本地登记版本 `download-20261022`；未固定上游提交号，使用文件 SHA256 锁定准确输入。
初始训练范围按官方 Train 扣除内部验证集计算，共 **173,683 条**。清洗运行以冻结的 `train_scope` ID 清单为输入。
初始 Train/内部验证划分使用规范化题干分组随机划分，目标验证比例 5%，固定随机种子 `20261020`，随机生成器为 `random.Random(20261020)`；同一规范化题干组整体分配，不拆到训练与验证两侧。规范化方式为去首尾空白、合并连续空白、casefold。当前最终清洗入口不重新生成该划分，而是直接读取并校验已冻结的 `train_scope` ID 顺序清单、`dev_eval` 文件及 `data_release_manifest.json`。划分成员与顺序以这些冻结数据文件和 Manifest 为准；最终清洗不导入或读取历史筛选版本。
旧 Answer-only 训练文件 `train_sft.jsonl` 已退役删除；仅保留无训练文本的 `train_scope` ID 清单，保证原始 train/dev 划分成员和顺序可追溯。`dev_sft` 是 `dev_eval` 的历史消息格式别名，已从活跃 manifest 移除；历史副本单独归档，不作为当前数据输入。
官方 validation 保留文件不作为训练输入，也不新增本文件以外的隔离规则。

输入数量、哈希、ID 唯一性、原始映射或 dev 别名不符时立即停止；不得静默丢行、猜测或继续。
不读取其他筛选结果、候选包或旁路隔离清单来改变本轮输入范围。
`choice_type=multi` **不作为排除条件**。

## 2. 类别、优先级与修复提交

| 类别 | 定义 |
|---|---|
| A | 未发现需要修复或隔离的明确问题，保留原文 |
| B | 只有明确、局部、允许的安全修复；修复并复检后保留 |
| C | 有明确问题或有结构证据的不确定性，不能安全自动处理，隔离 |
| D | 不具备文本型 Answer + Explanation 训练基本条件，排除 |

唯一质量分类优先级：**D > C > B > A**。每条记录可保存多个原因，但只能计入一个质量类别。
弱提示不参与优先级。C/D 始终保存原始记录；候选修复不覆盖原文。
局部分类只提交通过复检的 B 修复。数据集级冲突之后可能将 B 升为 C；已提交的修复仍全部保留日志和原文。

## 3. 严格答案解析

统一入口：`medmcqa_answers.parse_answer(value=None)`，返回 `(status, letter)`。

| 原始值 | status | letter / 质量影响 |
|---|---|---|
| 整数 0、1、2、3 | valid | A、B、C、D；不因此降级 |
| 字符串 A～D，允许大小写及首尾空白 | valid | 统一大写；不因此降级 |
| None、字段缺失、空字符串、纯空白 | missing | None；D |
| AB、C123、整数 4、True、浮点数、字符串数字等 | invalid | None；C |

必须先排除 bool，不接受字符串 `"0"`。禁止子串成员判断和整数强制转换。
旧数据准备脚本及其 `answer_index()` 适配器已退役；当前统一使用 `medmcqa_answers.parse_answer()`，新分类器必须保留 status，不能只看 None。

## 4. D 类条件

任一明确 D 条件优先于所有 C 和 B 条件。

### 4.1 图片或明确图片依赖

题干、任何选项、解释和记录资源中存在实际图片即 D，包括：
结构化图片 payload、URL、路径、资源 ID、附件；HTML `<img>`；Markdown 图片；明确图片占位；明确要求观察图片、影像、图表、示意图才能作答的题目。
图片缺失但题目明确要求识别其中所示结构，也归 D。先检查图片，再处理 HTML，不能去掉图片后保留题目。
普通 `image processing`、`figure`、`picture` 单词不能判 D。模糊依赖只生成弱提示和原文位置证据。
原因：`contains_image_or_image_dependency`。

### 4.2 基本内容缺失

- 题干缺失或空白：`missing_question`。
- gold missing：`missing_gold_answer`。
- 全部选项为空或为明确无内容标签：`all_options_missing_or_placeholder`。
- 解释缺失或空白：`missing_explanation`。

选项占位只作保守完整匹配，不按长度、词数或语义判断，不建立合法选项白名单。
单字、数字、判断类、组合类、`Both A and B`、`All of the above`、`None of the above` 都不能仅凭表达形式判异常。
解释占位词表不得套用到选项。

### 4.3 明确占位解释

首尾去空白、忽略大小写后的**完整字符串**匹配：

```python
PLACEHOLDERS = {
    ".", "-", "--", "none", "null", "n/a", "na", "no explanation",
    "not available", "explanation not available here", "no explanation provided",
    "no explanation available", "not provided",
}
```

命中归 D：`placeholder_explanation`。不使用子串匹配。

### 4.4 Answer Only

只识别明确答案声明、答案重复表述和有结构证据的引用区块，检查剩余正文。
以下没有正文的情况归 D：仅字母；`Ans. C`；`Ans. C) Option text`；`C is correct`；只重复正确选项文字；答案加明确参考文献。
有实际正文，不要求 because/therefore，不设 160 字符阈值，也不评判正文是否充分或医学正确。
原因：`answer_only_no_explanation_body`。

若 gold=C，解释只有 `Ans. A) Urogenital sinus`，即使字母可修复，仍因无正文最终 D。
若相同答案行后有：

```text
The urogenital sinus develops into several structures
of the urinary system.
```

则满足安全修复条件时归 B，只改答案字母。

## 5. C 类条件与答案声明

无 D 时以下条件归 C：
invalid gold；部分选项缺失；保守规范化后选项完全重复；明确内容乱码或严重损坏；复杂 HTML 无法无损处理；有结构证据但无法分离的引用和正文；无法安全处理的明确答案冲突。

### 5.1 声明范围与匹配

只识别解释开头或独立答案声明行中的明确声明：
`Ans. C`、`Answer: C`、`Correct answer: C`、`The correct answer is C`。
`C is correct` 的独立答案重复行也可识别。普通正文的 A/B/C/D 和正常选项列表不是声明。

| 情况 | 处理 |
|---|---|
| 声明字母与 gold 一致，无论是否重复选项文字 | 不触发答案冲突 |
| 字母不一致，答案文字唯一完全对应 gold 选项 | B，只改解释字母 |
| 字母不一致，文字对应其他选项 | C：`explicit_answer_content_conflict` |
| 字母不一致，无文字或无法唯一匹配 | C：`answer_text_mapping_uncertain` |
| 多个互相矛盾的明确声明 | C：`contradictory_answer_declarations` |

安全修复还必须是**解释开头**的明确声明，且选项唯一完全匹配。
独立行中的非开头错误声明有可匹配文字也不自动修复，保留证据并归 C。
不进行同义词、语义、医学知识判断；不删除标点、数字、单位或选项文字来凑匹配。
保守实现只在答案行具有明确文字边界时提取文字；答案文字与正文在同一行且无法可靠分界时不猜测。

### 5.2 选项、乱码和损坏

部分选项缺失：`missing_option`；规范化精确重复：`duplicate_option_exact`。
仅语义相似不触发。NFKC、首尾去空白、合并连续空白、casefold 后比较，保留数字、标点、术语。
明确 U+FFFD、非法 Unicode 损坏影响内容：`text_corruption_unrepairable`。
低置信度 mojibake 只提示；不能自动恢复医学术语。严重损坏没有可靠结构证据时不能凭正则升 C。
重复单词和大小写边界疑似粘连只提示，不删除、不插空格；正常 had had、very very 不因此降级。

### 5.3 引用和正文

识别独立 `References:`、`Reference:`、`Bibliography:`、`Sources:`、`Ref:`、`Refs:` 标题或区块，结合行边界、编号、版本页码等结构证据。
不能看到 Ref 就删除后面全部内容。引用后的正文必须保留。
纯答案加可明确识别引用归 D；有正文不触发；边界不可靠归 C：`reference_body_boundary_uncertain`。
单凭含糊书名不能猜成引用。识别器没有证据时保留原文，不调用模型判断。

## 6. B 类允许修复与日志

只允许：

1. 第 5.1 节的开头答案字母安全修复：`answer_letter_typo_safe_repair`。
2. 纯格式 HTML 和实体：`html_format_safe_cleanup`。
3. 连续完整完全相同句/段的后续副本删除：`consecutive_exact_duplicate_sentence_removed` / `consecutive_exact_duplicate_paragraph_removed`。
4. 无意义首尾空白、换行统一、明确多余空白：`whitespace_safe_cleanup`。

### 6.1 HTML

可以清理 `<b>Vitamin C</b>` 等纯格式标签和 `&nbsp;`、`&amp;`、`&lt;`、`&gt;` 等实体。
必须保留可见文字和段落边界；保守实现的纯格式标签为无属性的 b/strong/i/em/u/p/br/div/span。
标签属性可能承载信息、嵌套实体解码边界不清时不剥除，按无法安全处理的 HTML 结构归 C。
`<sub>`、`<sup>`、公式、MathML、表格、链接或其他可能承载信息的复杂结构不得简单剥标签，无法无损处理归 C：`html_structure_unsafe`。
不是 HTML 的比较符号不得误当标签删除。不能处理图片以尝试保留样本。

### 6.2 连续重复完整句/段

必须连续相邻、保守规范化完全相同、边界明确、非标题/引用/选项/公式/代码，且删除不改变含义或结构。
只删后出现副本，保留第一份和其原有文本。缩写、小数、句子边界不明时不自动删。
重复单词、短语、不连续重复、近似重复不自动删，不改变分类。
句子安全性不足可只提示；段落也必须有明确完整正文边界，不能把两行相同标题当段落删。

### 6.3 日志与复检

每次已提交修改记录：ID、本轮行号、字段、原文、新文、规则、说明、变动位置和片段；字母修复还记录前后字母和匹配选项。
复检确保没有出现新的 D/C 条件。再次清理已修复内容应不再产生修改。
NFKC/casefold 主要用于比较键，不因此统一改写所有训练原文；不改句子、数字、单位、药名、术语、标点含义或 gold，不补写内容。

## 7. 禁用规则与统计

最终流程及其传递依赖不得计算或输出以下筛选标签：医学风险关键词、剂量关键词、指代不清、题解词面重合度、理由关键词要求、多选项列表独立 Review。
不得使用 Answer Only 的 160 字符阈值。
少于 50 字符和超过 1000 字符仅计数，不是质量门槛；不截短解释。
词面重合度彻底不计算、不打标签。旧脚本只归档，不导入其分类逻辑。

## 8. 实际训练序列与长度

模型：`Qwen/Qwen3.5-0.8B`；本地已验证快照：`2fc06364715b967f1860aea9cf38778875588b17`。
使用真实 AutoProcessor 的 Qwen3.5 tokenizer 和完整 chat template，不加载模型权重。
冻结训练消息：一个 user（题干、A～D 选项、格式要求）加一个 assistant：

```text
Answer: X
Explanation: <原文或允许修复后的解释>
```

user 格式要求：

```text
Answer the question and provide a concise explanation in English. Use exactly this format:
Answer: X
Explanation: <brief explanation>
```

统一编码入口：`medmcqa_training_contract.encode_training_sequence`。
完整序列 add_generation_prompt=False；prompt add_generation_prompt=True；tokenize=True、return_dict=True、return_tensors=pt；不覆盖模板默认特殊 Token 行为，不额外重复加 BOS/EOS。
完整模板生成的 input_ids 数量为长度，包含模板当前产生的 think 等特殊内容。
必须校验 prompt 是完整序列前缀，且有监督 Token。

B 修复后计算长度；A/B ≤512 为 within_limit，>512 为 over_limit；未检查为 not_checked。
超长不改变质量类别，另行隔离。训练 collator 超长直接报错，禁止切片截断。
报告最终 A/B 检查数、≤512 和 >512 数量比例、超长分布及 A/B 分布。数据集级冲突前已计数的样本保留测量结果并另报，不混入最终 A/B 分母。
本轮不调整为 1024、不启动 GPU 训练。后续是否改变长度另结合显存 smoke test 决策。

## 9. 原始与修复后指纹、重复与 dev 交叉

同时保存原始键和修复后键的指纹：题干键、题干+有序选项键、完整记录键。
所有比较字段分别 NFKC、首尾去空白、合并连续空白、casefold；保留数字、标点、术语和 A/B/C/D 顺序。
完整记录字段为题干、四个选项、严格解析后的标准答案、解释正文；不含 ID 和元数据。
比较使用字段数组，禁止无边界拼接导致碰撞。解析失败保留原始答案值与类型。

最终重复判断以**修复后内容**为准：

- 完整记录完全重复：保留输入顺序第一条，其余 exact_duplicate_copy，保留副本原文及代表 ID，不静默丢行。
- 同题同选项同答案、解释不同：全部保留，报告 same_prompt_multiple_explanations。
- 同题同选项、答案不同：相关记录标记 duplicate_answer_conflict；无 D 时归 C，不自动选答案。
- 仅选项顺序不同：不自动判完全重复；仅题干相同只报告，不作为新增隔离依据。
- 同 ID 重复必须报告；输入 ID 无法唯一映射时作为输入契约错误停止，不能覆盖字典项而丢行。

与唯一活跃入口 dev_eval 比较题干+有序选项键：

- 匹配均隔离训练侧；不删 dev，不计两份 dev。
- gold 不同：额外标记 dev_answer_conflict，无 D 时归 C。
- 以修复后的题干和有序选项键与 dev 的实际内容键比较；原始键只作追溯，不增加原始键单独命中的隔离规则。
- 不加载历史 4,455 条清单作过滤，不用语义相似度，不新增仅题干命中的隔离。

## 10. 独立状态与所有阻断原因

quality_class：A/B/C/D。
length_status：within_limit / over_limit / not_checked。
duplicate_status：unique / exact_duplicate_copy / same_prompt_multiple_explanations / duplicate_answer_conflict / dev_eval_overlap。
可同时命中多个重复维度，另存 duplicate_flags；展示状态按 dev overlap、答案冲突、完全重复副本、多解释、unique 顺序选择，不改变质量规则。

training_block_reasons 保存**所有适用原因**：quality_D、quality_C、dev_eval_overlap、dev_answer_conflict、exact_duplicate_copy、duplicate_answer_conflict、over_512 等。
其中也包含导致 C/D 的每个具体内容问题；另存 quality_block_reasons，避免只留下质量类别而丢失具体阻断原因。修复规则和弱提示不作为阻断原因。
training_status 的展示优先级固定为：

1. excluded_quality_D
2. isolated_quality_C
3. isolated_dev_overlap
4. isolated_duplicate
5. isolated_over_512
6. eligible_candidate

只有 A/B 且长度通过、无重复副本/答案冲突/dev交叉才能 eligible_candidate。
质量拒绝与各隔离原因分开统计；提供每个原因数量和去重记录并集，不相加冒充记录总数。

## 11. 输出与运行保护

所有正式输出进入全新独立目录；输出目录已存在则拒绝覆盖。至少包括：

```text
medmcqa_A.jsonl
medmcqa_B_cleaned.jsonl
medmcqa_C_review.jsonl
medmcqa_D_excluded.jsonl
medmcqa_AB_quality_candidates.jsonl
medmcqa_train_candidates_512.jsonl
medmcqa_over_512.jsonl
medmcqa_train_dev_overlap_isolated.jsonl
medmcqa_duplicate_groups.jsonl
medmcqa_repair_log.jsonl
medmcqa_cleaning_report.json
README.md
```

类别文件包含原始记录、有效内容、来源、指纹、四个独立状态和全部原因；训练候选包含统一训练 messages 和追溯信息。
AB_quality_candidates 是内容质量候选侧视图，不等于通过长度和交叉检查。
重复组报告包括 ID、输入顺序、gold、解释差异及代表记录。各隔离文件可能重叠，报告必须说明。
报告固定输入、规则、源码、依赖版本、tokenizer/template 文件哈希、参数、样本范围和输出哈希。
小规模结果明确 scope=development_sample、training_release_ready=false，不得伪装全量结果。
默认命令只预检。sample 模式最多 1,000 条且明确记录所选 ID；full 必须显式附加确认参数。正式全量清洗已经完成，本次文档维护不重新运行清洗。
活跃 manifest 和历史验收 manifest 保持原样，不用旧 contract 的历史 quarantine；独立加载器仅读取 manifest 的原始输入角色和 dev 别名。
2026-10-31 的全量清洗交付包已退役，现存于本地 `delete/data_derived_legacy/medmcqa_final_20261031/`。`delete/reports_legacy/medmcqa_final_full_20261031/final_delivery_manifest.json` 记录了 13 个历史交付文件的清单、大小和 SHA-256。当前正式训练数据及内部验证集位于 `data/processed/`，独立评估数据位于 `data/evaluation/`。

## 12. 验收

必须覆盖：

- 输入路径/哈希/数量、ID、原始映射、dev 别名、182,822 去向守恒；漂移失败停止。
- 严格解析所有 valid/missing/invalid 正反例，包括 None、缺失字段、字符串数字、bool。
- D>C>B>A；字母可修复但无正文 D、有正文 B；gold 永不修改；多个矛盾声明 C。
- 图片在题干、任一选项、解释、结构化资源及明确缺失依赖均 D；普通 image processing 不命中；模糊只提示。
- Answer Only、选项复述、答案重复、引用前后正文保留、无法分界 C；无理由词正常。
- 安全 HTML/实体 B；上下标/公式/表格 C；比较符号与实际文字不被误删。
- 连续完整同句/同段 B；非连续/近似/重复词/短语/缩写/小数/标题/代码不自动删除。
- 禁用规则没有进入新流程依赖；医学词、剂量、multi、字母列表、短长解释不单独降级。
- 原始与修复后指纹；完整重复保留输入第一条；同答案不同解释全保留；答案冲突 C；选项顺序不变。
- dev 匹配训练侧隔离，不同 gold 加冲突 C；D 优先；多阻断原因完整，主要状态顺序正确。
- 真实 tokenizer 的 511/512/513 边界、完整模板特殊 Token、prompt 和 padding mask、完整解释监督、超长报错。
- 修复幂等；相同输入两次生成的数据文件和确定性报告一致；所有输入哈希不变。

单元、反例和小规模检查及获准正式全量执行的开发阶段门槛已经完成。历史独立全量验收记录 `delete/reports_legacy/medmcqa_final_full_20261031/independent_verification.json` 为通过，覆盖冻结清单中的全部记录、708 次修复事件及真实 tokenizer 长度复核。历史完整复现记录与当前数据交付完整性分别报告；历史记录通过不等于当前环境已重新完成全量复现。

## 13. 技术边界

不验证医学事实、临床安全、指南时效或专家水平；不使用模型逐条审核，不改写医学内容。
自然语言图片依赖、模糊引用、严重损坏无法靠有限程序穷尽识别。仅按本文件明确证据执行；无法识别不意味着医学正确，低置信度线索不得成为新过滤器。
本阶段目标是可解释、可复现、可追溯的有限流程，不追求扩展规则数量。

## 14. 正式完成标准

只有获准全量运行后，冻结清单中的全部记录有去向、A/B/C/D 守恒、修复有日志、图片证据按规则排除、长度真实无截断、重复/答案冲突/dev交叉处理完整、禁用规则未参与、输入哈希不变、全量验收通过，才可宣布最终清洗完成。
上述正式清洗完成条件已经满足：冻结清单中的全部记录已完成 A/B/C/D 分类，A 144,396 条、B 342 条、C 3,284 条、D 25,626 条；B 类 342 条已完成允许的安全修复，共 708 次事件，gold 未修改。144,738 条 A/B 记录完成 512-token 长度筛选后，形成 **138,663 条历史中间候选**，另有 6,075 条超长隔离。该 138,663 条是清洗阶段的历史统计，不是当前最终训练文件数量。后续最终训练准入和模型专属数据发布形成的共同训练文件为 **137,928 条/模型**，具体路径、SHA256 和模型长度预检见[最终实验协议](../experiment/FINAL_EXPERIMENT_PROTOCOL.md)及 `configs/medmcqa_only_training/`。正式交付 manifest 的 13 个历史清洗文件已核对通过，历史独立全量验收记录为通过。

当前规则文档是持续维护的唯一有效版本；历史验收 manifest 中记录的代码及文档 SHA256 对应历史版本，不再要求与当前工作版本一致。历史验收记录及其原始数据文件不作修改；重要数据文件继续按 SHA256 检查完整性。
