# 多智能体可靠生成实验执行手册

> AI 每次先阅读 [AGENTS.md](AGENTS.md)，研究原理见 [plan.md](plan.md)。本文件负责指导 AI 按顺序写代码、记录进度和验收结果。

## 1. 怎么使用

以后每次只让 AI 完成当前步骤：

1. 查看“总进度”中的当前任务；
2. 复制对应步骤下面的“给 AI 的提示词”；
3. AI 写代码、运行测试并生成产物；
4. 验收通过后，AI 把 `[ ]` 改成 `[x]`，更新总进度和完成记录；
5. 再进入下一步。

状态说明：

- `[ ]` 未完成；
- `[~]` 正在做；
- `[x]` 已完成并验证；
- `[!]` 被外部条件阻塞。

共同要求：

- 每次只做指定步骤，不提前实现后续内容；
- 原始数据只读，不覆盖；
- API Key 通过运行时环境读取；按用户于 2026-10-09 的明确选择，可由本人填写被 Git 忽略的 `config/api.local.json`，程序只在 live 模式将其读入内存中的调用环境，不将密钥写入公开配置、日志或结果；
- 随机过程固定种子；
- 模型调用支持缓存、重试和断点续跑；
- 至少提供测试或可重复验证命令；
- 只有代码、测试和产物都通过后才能标记完成；
- 每次完成后更新本文件末尾的“完成记录”。

## 2. 已确定的实验方案

| 项目 | 决定 |
| --- | --- |
| 主数据集 | FActScore 人工标注数据 |
| 初始规模 | 最多 120 条回答 |
| 数据划分 | 按人物或主题分组，训练/验证/测试为 60%/20%/20% |
| Verifier | 3 个独立 API 判断者：DeepSeek、GPT、Claude；用户于 2026-10-09 增加 Claude，第三个判断者改用 Claude |
| 个人风险 | 根据训练集历史对错统计，不直接相信模型自报置信度 |
| 风险模型 | 逻辑回归，CPU 训练 |
| 风险阈值 | 只在验证集选择，测试集不得调参 |
| 证据处理 | 支持则保留、反驳则局部修改、证据不足则标记不确定 |
| 对照组 | M0 不处理、M1 多数投票、M2 全部验证、M3 风险触发 |
| 人工抽查 | 约 50 条 |

以后再决定：

- API 后端可追溯的实际版本，以及正式批量前的配置冻结；
- 第 4 步使用哪一种固定外部知识源；
- 人工抽查由一人还是两人完成。

## 3. 总进度

| 顺序 | 阶段 | 状态 | 核心产物 |
| --- | --- | --- | --- |
| 0 | 研究方案和执行手册 | `[x]` | `plan.md`、`experiment.md` |
| 1 | 数据准备 | `[x]` | 固定的 train/dev/test |
| 1A | 本地小模型可运行性检查 | 历史任务：安装检查已完成，正式提示词验收未进行；Qwen 已退出当前三模型组合 | 历史安装试跑见 [small_model_setup.md](small_model_setup.md) |
| 2 | 三个 Verifier | `[~]` DeepSeek、GPT、Claude 使用统一 verifier-v3，各 10 条格式有效（30/30），缓存复跑 0 新请求；正式配置冻结和全量判断尚未完成 | [可发布试跑汇总](data/reports/verifier_pilot_report.json)、[运行器](src/run_verifiers.py)；尚无全量三份判断或基线 |
| 3 | 风险模型 | `[ ]` | 个人风险、最终风险、固定阈值 |
| 4 | 证据验证和局部修改 | `[ ]` | 证据、修改及复核记录 |
| 5 | M0—M3 对比实验 | `[ ]` | 主结果表和逐 Claim 结果 |
| 6 | 人工抽查 | `[ ]` | 人工评审表和最终结论 |

当前下一任务：**冻结 DeepSeek、GPT、Claude 的模型 ID、提示词 verifier-v3、推理参数和数据划分，保存不含密钥的配置快照；随后按用户正式批量指令收集 3743 条 Claim 的三份判断，共 11229 个逻辑判断。同版本已有的 30 条结果应命中缓存，失败保留日志并可续跑。当前三模型不依赖 Qwen 运行电脑。只发送 topic/Claim，不发送金标签或其他模型意见；不得根据测试标签修改任何设置。完成三份正式判断后再生成基线，随后进入第 3 步可靠性统计和风险模型。费用按用户决定暂不估算。** 目前只完成小样本，未设置全量放行标志、未启动全量。模型 ID、Base URL 和密钥仍集中在 `config/api.local.json`。

### 第 1A 步：检查本地小模型能否运行（历史记录）

用户新增 Claude 后，当前 V3 改为 Claude API；本节只保留此前 Qwen 的安装和检查记录，不再阻塞当前三模型实验。

本步骤只确认本地推理是否可行，不运行全量 Claim，也不改变数据集。实际在 D 盘部署 Windows Ollama 便携 CLI 0.34.4 和 `hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M`。用户已接受约 23.6 小时 CPU 全量估时，该模型作为 V3 候选；全量运行可等到有合适电脑时进行。安装试跑实际使用 CPU、上下文 4096、温度 0、输出上限 256、JSON Schema 及 `/no_think`。经只读检查旧试跑脚本，安装检查提示词含额外格式要求，与本文件 4.1 节的正式提示词不同；既有 10 条结果不能视为正式 `verifier-v1` 验收。第 2 步需用统一的 10 条训练 Claim 重新检查，并记录 seed 42 等正式候选参数。部署、硬件、历史试跑和存储记录见 [small_model_setup.md](small_model_setup.md)。

### 待办

- [x] 获取本机 RAM、CPU、显卡/显存和可用磁盘空间；
- [x] 根据硬件初步选择 Qwen3 4B Q4、CPU 推理；
- [x] 将 Ollama 便携 CLI 和模型放在 D:\research\ollama，模型、配置、日志、临时文件与试跑结果均指向 D 盘；
- [x] 对 10 条临时 Claim 试跑：10/10 JSON 有效，10/10 标签匹配；平均 22.7 秒/条，标签为 supported 6、unsupported 4；
- [x] 用户接受全量 3743 条按 CPU 实测估算约 23.6 小时；确认 `hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M` 和 Ollama 0.34.4 为 V3 候选，不将安装冒烟结果当作正式提示词验收；
- [~] train/dev/test 全量调用暂缓，等有合适电脑时运行；当前 10 条试跑只验证可运行性和格式，不能代表准确率；
- [x] 更新本节状态、独立安装说明和完成记录。

### 验收条件

- 有可复核的硬件信息和明确的软件/模型版本；
- 10 条试跑均有结构合法的结果或明确失败记录；
- 有运行耗时和资源占用记录，可判断全量运行是否现实；
- 只有格式和运行稳定性通过、人工抽查输出无明显异常后，才纳入三个 Verifier；
- 尚未获得本机配置或尚未试跑时，不得标记完成。

### 给 AI 的提示词

```text
请阅读 AGENTS.md、plan.md、experiment.md 和 small_model_setup.md。用户已接受 Qwen3-4B-GGUF Q4_K_M 在 CPU 上全量运行约 23.6 小时，该模型为 V3 候选；运行电脑未定，不重复询问是否接受耗时。
按“当前下一任务”继续第 2 步，复用已有代码。安装检查提示词与正式 verifier-v1 不同；Qwen 运行电脑确定后需要在正式提示词下重新完成同一批 10 条验收。不要把历史安装结果当作正式结果，也不要开始全量或第 3 步。
```

---

## 第 1 步：数据准备

### 目的和原因

把 FActScore 理解成“带标准答案的练习册”。后续需要用人工真假标签统计 Verifier 是否可靠，并检查风险模型预测得准不准。

### 待办

- [x] 从官方来源取得人工标注数据，不下载完整 Wikipedia；
- [x] 记录来源 URL、下载日期、文件大小和 SHA-256；
- [x] 检查真实文件结构、字段和标签含义；
- [x] 建立最小 Python 项目、配置和测试结构；
- [x] 将原始数据转换成统一 Claim 格式；
- [x] 处理空值、重复项、未知标签和重复 ID；
- [x] 选择最多 120 条回答；
- [x] 按人物或主题分组切分 60%/20%/20%；
- [x] 检查三个集合不存在主题重叠；
- [x] 生成数据统计报告。

### 已办

- [x] 确定使用 FActScore；
- [x] 确定按主题分组，避免数据泄漏；
- [x] 确定样本规模和划分比例。

### 统一数据格式

```json
{
  "answer_id": "唯一回答编号",
  "claim_id": "唯一Claim编号",
  "topic": "人物或主题",
  "source_model": "原回答模型，如数据提供",
  "claim_text": "原子事实",
  "gold_label": "supported或unsupported",
  "gold_label_binary": 0,
  "source_file": "原文件名"
}
```

### 应产生的文件

```text
data/raw/factscore/          官方原始数据
data/processed/claims.jsonl 统一后的 Claim
data/splits/train.jsonl
data/splits/dev.jsonl
data/splits/test.jsonl
data/reports/data_report.md
```

### 验收条件

- 所有 JSONL 都可以逐行读取；
- Claim ID 唯一，标签合法；
- 同一主题不会跨集合出现；
- 每个集合都包含正确和错误 Claim；
- 相同配置重复运行得到相同划分；
- 报告包含回答数、Claim 数、标签比例和排除数量。

### 给 AI 的提示词

```text
请阅读 plan.md 和 experiment.md，只执行“第 1 步：数据准备”。

从 FActScore 官方来源取得人工标注数据，不下载完整 Wikipedia。
先记录来源、下载日期、文件大小和 SHA-256，再检查真实文件结构。
建立最小 Python 3.12 项目和测试，将原始数据转换成 experiment.md 规定的
统一 Claim JSONL。原始文件必须只读，未知标签不能静默猜测。

选择最多 120 条回答，固定随机种子，按 topic 分组切分为 60% 训练、
20% 验证、20% 测试。自动检查主题零重叠、ID 唯一、标签合法、正负样本
存在和结果可复现，并生成数据报告。

不要实现 Verifier 或后续步骤。完成测试和真实数据验证后，更新 experiment.md
中的复选框、总进度和完成记录，并给出下一步提示词。如官方数据必须人工下载，
不要换用非官方数据，记录准确阻塞原因和人工操作方法。
```

---

## 第 2 步：运行三个 Verifier

### 目的和原因

让三个独立判断者分别判断每条 Claim。一个模型可能犯错，三个模型还能提供一致或分歧信息。单模型和多数投票基线直接复用这批结果，不重复调用。

### 待办

- [x] 已接入 DeepSeek、GPT、Claude 三个 API；均在同批训练 Claim 上获得有效输出，实际返回字段与请求别名一致，但中转别名不能独立证明真实后端版本；
- [x] 当前提示词版本 `verifier-v3`：强化中文短理由和直接 JSON 输出；三模型同版本试跑通过，严格 50 字符校验不变，旧版结果原样保留；正式批量前仍需保存配置快照；
- [x] 实现 OpenAI 兼容接口与 Ollama 的统一调用及命令行入口；
- [x] 实现环境变量及用户自填的本地 API 配置读取、失败重试、顺序限速、成功结果缓存、原子检查点和断点续跑；
- [x] 严格校验标签、置信度、理由、必需字段及重复 JSON 键；Ollama 使用 JSON Schema；
- [x] 离线单元测试 27 项通过，含 Claude 私有配置读取/不泄露密钥、新旧版本输出隔离、缓存、重试与全量门禁；三 API 的 PowerShell 包装器离线测试通过；
- [x] 两套 API 完成同一批 10 条训练 Claim 的真实试跑：GPT 10/10 格式有效，DeepSeek 9/10；DeepSeek 断点续跑跳过 9 条已成功结果，失败 Claim 又尝试 3 次仍为理由超长，保持未通过；
- [x] verifier-v3 三个 API 均为 10/10 格式有效；复跑 30 条全部命中缓存、0 新请求。已检查失败率、标签分布、实际返回模型、用量和耗时，并扫描产物不含三把密钥；
- [x] 用户新增 Claude 为 V3，当前组合不再等待 Qwen 运行电脑；Qwen 正式提示词验收未做，历史结果不作 Claude 结果；
- [x] 现有数据准备、模型调用、命令行、测试和 PowerShell 脚本已补中文注释；整理无密钥试跑汇总和新电脑配置说明，发布不包含本地密钥及原始数据；
- [ ] 保存三模型、提示词、推理参数与数据划分的正式配置快照，核对本地覆盖值一致后再记录全量放行；
- [ ] 配置冻结后运行 train/dev/test；
- [ ] 复用输出生成 V1 单模型和多数投票基线；
- [ ] 计算 Accuracy、F1、AUPRC 和混淆矩阵。

### 已办

- [x] 确定三个 Verifier 独立判断；
- [x] 确定原始置信度只记录，不直接作为个人风险；
- [x] 写出首版运行提示词。

### 当前候选配置与用户条件

| ID | 模型与调用方式 | 状态 |
| --- | --- | --- |
| V1 | DeepSeek 4.1 Flash；当前 API ID `deepseek-flash` | verifier-v3：10/10 格式有效；supported 6、uncertain 4；6 条明确判断与训练人工标签一致，4 条弃判不算判断正确；平均 0.969 秒/条 |
| V2 | `gpt-6.1-sol`；用户中转站 | verifier-v3：10/10 格式有效；supported 10；与训练人工标签一致 8 条，2 条错误 Claim 被判支持；平均 3.519 秒/条 |
| V3 | `claude-sonnet-4-6`；用户中转站的 OpenAI 兼容接口 | verifier-v3：10/10 格式有效；supported 6、unsupported 1、uncertain 3；7 条明确判断中 6 条与训练人工标签一致，3 条弃判不算判断正确；平均 3.306 秒/条 |

用户于 2026-10-09 指定“费用暂时不管”，本轮未估算费用。此次 Claude 接入及格式排查共 55 次请求尝试：verifier-v2 阶段 25 次（含 Claude 代码块诊断），最终 verifier-v3 30 次；缓存复跑 0 次。最终同版 30 条返回用量为输入 52567 token、输出 1070 token，仅为服务商响应统计，未独立核对账单。以前 verifier-v1 的 26 次请求另存历史记录。

离线采样只使用训练集，种子为 42，三个 Verifier 共享同一批 10 条；mock 结果保存在独立目录并明确标记，不得用于准确率、基线或风险模型。API 只发送当前 topic/Claim，不发送金标准标签、原回答或其他 Verifier 意见。正式结果用配置/实际提示词/端点/全部请求参数摘要作缓存键；配置变更不会复用旧判断。外部 API 使用温度 0，不宣称服务输出逐字节可复现。

可执行命令和结果位置见 [README.md](README.md)。当前训练试跑的 10 条中有 4 条三模型存在分歧；人工标签为 supported 8、unsupported 2。样本只作格式和初步质量检查，不用于宣称总体准确率或校准风险，也不能为了判中这 10 条而改事实判断策略。默认命令是离线计划，正式配置冻结并记录放行前程序拒绝全量模式。

### 输出格式

```json
{
  "claim_id": "...",
  "verifier_id": "v1",
  "model": "实际模型名和版本",
  "prompt_version": "verifier-v3",
  "label": "supported",
  "raw_confidence": 0.82,
  "reason": "简短理由",
  "created_at": "ISO时间"
}
```

### 验收条件

- 每条 Claim 对三个 Verifier 都有结果或明确失败记录；
- 标签只包含 `supported/unsupported/uncertain`；
- 重新运行不会产生重复付费请求；
- API Key 不出现在代码、日志和结果中；
- 单模型和投票基线来自同一批输出。

### 给 AI 的提示词

```text
请阅读 plan.md、experiment.md 和第 1 步产物，只执行第 2 步。
模型已经由用户确定：V1 DeepSeek 4.1 Flash（API ID deepseek-flash）、
V2 gpt-6.1-sol、V3 claude-sonnet-4-6（两个中转 API）。当前正式候选提示词为 verifier-v3。
复用 src/verifiers.py、src/run_verifiers.py、config/verifiers.example.json 和现有测试；
不要重写已通过的接口。三个模型同版各 10 条已完成，先读取
data/reports/verifier_pilot_report.json；本机另有 outputs/verifiers/live/verifier-v3/ 下的三个结果文件；
不得展示 config/api.local.json 全文、密钥或把它们写到日志/结果。
保存不含密钥的正式配置快照并核对本地模型覆盖值。报告实际模型、
报告格式、失败率、标签分布、实际返回模型、尝试次数、token 用量和耗时；费用按用户决定暂不估算。
当前不使用 Qwen，不再要求用户确定其运行电脑。
接口拒绝候选参数时报告具体失败，不静默修改配置；不得据测试集调整任何设置。
配置冻结并按用户正式批量指令放行后，运行 train/dev/test；同版已成功的小样本复用缓存。

复用同一输出生成 V1 单模型和多数投票基线，计算 Accuracy、F1、AUPRC 和
混淆矩阵。不得根据测试标签修改提示词。验收后更新 experiment.md，
不要实现第 3 步。
```

---

## 第 3 步：计算 Claim 错误风险

### 目的和原因

解决“三个模型意见不同时相信谁”的问题。先从训练集估计每个 Verifier 过去做出某种判断时对应的实际错误率，再综合票数、可靠性和分歧得到最终错误概率。

### 个人风险公式

```text
r_i(v) = (Verifier i 输出 v 且 Claim 实际错误的数量 + 1)
         / (Verifier i 输出 v 的总数量 + 2)
```

训练集必须用 5 折 out-of-fold 计算，防止一条样本使用自己的答案构造特征。验证集和测试集只能使用训练集统计结果。

### 待办

- [ ] 实现带平滑的个人风险表；
- [ ] 实现训练集 5 折 out-of-fold 计算；
- [ ] 生成票数、平均风险、风险差、分歧、历史准确率加权和共同犯错特征；
- [ ] 为所有公式编写可手算测试；
- [ ] 使用训练集拟合逻辑回归；
- [ ] 在验证集检查概率和选择阈值；
- [ ] 优先满足错误召回率至少 80%，再最小化验证数量；
- [ ] 达不到 80% 时选择验证集 F1 最高阈值；
- [ ] 阈值冻结后只评估一次测试集；
- [ ] 与 V1 和多数投票比较。

### 已办

- [x] 确定个人风险公式；
- [x] 确定 5 折防泄漏方式；
- [x] 确定逻辑回归和阈值选择规则。

### 验收条件

- 测试能手算复核个人风险；
- 训练行不能参与自身个人风险统计；
- 测试标签不参与特征、训练、校准或阈值选择；
- 输出概率在 0 到 1；
- 保存模型、特征顺序、风险表、阈值和指标；
- 相同输入和种子得到相同结果。

### 给 AI 的提示词

```text
请阅读 plan.md、experiment.md 以及第 1、2 步产物，只执行第 3 步。
按照文档公式实现每个 Verifier 的个人风险。训练集必须使用 5 折 out-of-fold；
验证集和测试集只能使用完整训练集统计。实现票数、个人风险、分歧、历史准确率
加权和共同犯错特征，并为公式和数据泄漏编写可手算测试。

使用 LogisticRegression(class_weight="balanced", random_state=固定值) 训练。
只在验证集检查校准和选择阈值；规则冻结后才评估一次测试集。
报告 V1、简单投票和风险模型的 Accuracy、Precision、Recall、F1、AUPRC、
AUROC、混淆矩阵及触发验证比例。保存模型和全部配置，验收后更新 experiment.md。
不要开始检索证据。
```

---

## 第 4 步：验证并修改高风险 Claim

### 目的和原因

高风险只表示“可疑”，不能证明它错了。因此必须查外部证据，确认后才保留、修改或标记不确定。

### 固定处理规则

| 证据结果 | 操作 |
| --- | --- |
| `support` | 保留原 Claim |
| `refute` | 根据证据局部修改 |
| `insufficient` | 标记不确定，不强行修改 |

### 待办

- [ ] 确定固定知识源、版本和许可；
- [ ] 固定查询生成和 top-k 规则；
- [ ] 实现检索、缓存和来源记录；
- [ ] 实现 Evidence Judge；
- [ ] 保存查询、标题、片段、URL/文档 ID 和排名；
- [ ] 只有 `refute` 才调用修改器；
- [ ] 修改器只能修改当前 Claim；
- [ ] 修改后的 Claim 再次经过 Evidence Judge；
- [ ] 复核失败时回退为不确定；
- [ ] 为无结果、格式错误和调用失败编写测试。

### 已办

- [x] 确定三种证据结论；
- [x] 确定只修改高风险 Claim；
- [x] 确定局部修改和修改后复核规则；
- [x] 写出首版 Judge 和修改提示词。

### 验收条件

- 每个处理决定能追溯到具体证据；
- 无证据时不会生成肯定修改；
- 非高风险 Claim 不产生检索；
- 修改结果只有复核通过才接受；
- 检索或 API 失败不会被当成 `support`。

### 给 AI 的提示词

```text
请阅读 plan.md、experiment.md 和前三步产物，只执行第 4 步。
先根据当前硬件、网络和预算提出一个最小、固定、可追溯的知识源方案；
需要我决定时先问我，优先避免下载超大知识库。

实现查询生成、top-k 检索、缓存和来源记录。Evidence Judge 只能依据给定片段
输出 support、refute 或 insufficient。只有 refute 才调用局部修改器；
修改器不得重写整篇回答或引入证据外事实。修改后再次验证，失败则回退为不确定。

使用 mock 测试覆盖无结果、证据不足、格式错误、调用失败和复核失败。
先试跑 10 条并人工检查，再更新 experiment.md。不要运行 M0—M3 正式对比。
```

---

## 第 5 步：运行 M0—M3 对比实验

### 目的和原因

证明风险触发是否比简单投票更合理，以及是否能用更少验证次数接近“全部验证”的效果。

### 四组方法

| 方法 | 含义 |
| --- | --- |
| M0 | 原 Claim，不处理 |
| M1 | 多数票认为错误时触发验证和修改 |
| M2 | 所有 Claim 全部验证和处理 |
| M3 | 风险达到冻结阈值时触发验证和修改 |

M1、M2、M3 除触发方式外，检索、Judge、修改器和复核规则必须相同。

### 待办

- [ ] 冻结数据、模型、提示词、阈值、知识源和处理规则；
- [ ] 确认测试集没有参与任何调整；
- [ ] 实现统一 M0—M3 运行器；
- [ ] 记录每条 Claim 的触发和处理路径；
- [ ] 统计错误发现率、修正率、误修改率和最终错误率；
- [ ] 统计检索、Judge、修改次数、时间和费用；
- [ ] 检查 M0 验证数为 0、M2 覆盖率为 100%；
- [ ] 输出逐 Claim 结果和 Markdown 主结果表；
- [ ] 生成第 6 步人工抽查候选文件。

### 已办

- [x] 确定 M0—M3 定义；
- [x] 确定效果和成本指标；
- [x] 明确修改后的新 Claim 不能直接沿用原标签。

### 验收条件

- 四组使用完全相同的测试 Claim；
- 唯一方法差异是触发策略；
- M3 使用冻结阈值；
- 结果表可以从逐 Claim 记录重新生成；
- 失败和证据不足单独计数；
- 不根据测试结果回头修改方法。

### 给 AI 的提示词

```text
请阅读 experiment.md 和全部已完成产物，只执行第 5 步。
先生成并冻结数据、模型、提示词、阈值、知识源和处理规则清单，确认测试集
没有参与任何调参。实现统一 M0—M3 运行器，除触发方式外保持所有模块一致。

复用缓存节省费用，但分别记录逻辑调用数、实际新增调用、耗时和估算费用。
输出逐 Claim 处理路径、机器可读指标和 Markdown 主结果表。自动检查 M0 零验证、
M2 全覆盖、M3 使用冻结阈值，并生成第 6 步抽查候选文件。
不得根据测试结果修改方法。验收后更新 experiment.md。
```

---

## 第 6 步：人工抽查

### 目的和原因

修改后产生的是新 Claim，原数据标签不再是它的标准答案；自动 Judge 也可能犯错，因此需要人工检查一小部分。

### 待办

- [ ] 固定随机种子，分层抽取约 50 条；
- [ ] 覆盖成功修改、复核失败、证据支持、证据不足、误触发和漏检；
- [ ] 生成不显示风险分数和系统预期的盲评表；
- [ ] 人工填写证据相关性、充分性、修改正确性和是否引入新错误；
- [ ] 未知情况填写 `unclear`，不能猜测；
- [ ] 原始评审表只读保存；
- [ ] 汇总最终正确率、正确修正率、误修改率和失败类型；
- [ ] 有两位评审者时计算一致率和 Cohen's kappa；
- [ ] 生成成功与失败案例摘要。

### 已办

- [x] 确定约 50 条人工抽查；
- [x] 确定主要人工判断字段；
- [x] 明确自动 Judge 不能冒充人工金标准。

### 验收条件

- 抽样规则固定且可重复；
- 盲评表不显示风险分数或系统预期；
- 所有必填字段通过校验；
- 汇总结果能追溯到每条评审记录；
- `unclear` 单独报告；
- 最终同时报告成功和失败案例。

### 给 AI 的提示词

```text
请阅读 experiment.md 和第 5 步产物，只执行第 6 步。
按照固定类别和随机种子生成约 50 条盲评样本，隐藏风险分数和系统预期，
创建评审表、字段说明、合法取值和抽样清单。不要替我填写人工判断。

等我填写完成后，先只读备份原始评审表，再校验和汇总；不得擅自修改或推测
人工答案。分别报告最终正确率、正确修正率、误修改率、不确定数量和失败类型；
如有两位评审者，再计算一致率和 Cohen's kappa。生成最终 MVP 总结并更新状态。
```

---

## 4. 实验运行时提示词

以下提示词由代码调用模型。正式运行前先小样本试跑并确定版本，不能根据测试集结果修改。

### 4.1 Verifier

```text
System:
你是独立的事实核查初审员。请判断 Claim 是否符合事实。
你看不到其他核查员的结论。信息不足、内容含糊或无法可靠确认时选择 uncertain，
不要猜测。supported 表示核心事实正确；unsupported 表示至少一个核心事实错误；
uncertain 表示无法可靠确定。直接输出一个JSON对象，不要Markdown代码块、
代码围栏、JSON标签或任何额外文字。回复第一个字符必须是{，最后一个字符必须是}。
reason 只写一句简短中文依据，尽量控制在30个字符以内，最多50个字符
（包含标点、数字和英文字母）。避免复述长人名、标题或完整Claim；不要添加不必要的细节。

User:
Topic: {{topic}}
Claim: {{claim_text}}

输出：
{
  "label": "supported | unsupported | uncertain",
  "confidence": 0到1之间的数字,
  "reason": "一句中文短理由，尽量不超过30字符，最多50字符"
}
```

### 4.2 Evidence Judge

```text
System:
你只能依据用户提供的证据判断 Claim，不能使用记忆或其他资料。
support 表示证据支持全部核心事实；refute 表示证据明确反驳至少一个核心事实；
insufficient 表示证据无关、含糊、冲突或不充分。证据没有提到不等于反驳。
只输出 JSON。

User:
Topic: {{topic}}
Claim: {{claim_text}}
Evidence passages: {{evidence_passages_with_ids}}

输出：
{
  "label": "support | refute | insufficient",
  "evidence_ids": ["使用的片段ID"],
  "reason": "不超过80字的证据说明"
}
```

### 4.3 局部修改器

```text
System:
只有证据明确反驳原 Claim 时才进行最小修改。只修改当前 Claim，不重写整篇回答；
修改后的每个事实必须有证据支持；不得引入证据中没有的新人物、地点、日期、数字
或因果关系。证据不足时返回 cannot_revise。保持原 Claim 的语言，只输出 JSON。

User:
Topic: {{topic}}
Original Claim: {{claim_text}}
Refuting evidence: {{evidence_passages_with_ids}}

输出：
{
  "action": "revised | cannot_revise",
  "revised_claim": "修改后的单条Claim；不能修改时为null",
  "evidence_ids": ["使用的片段ID"],
  "change_summary": "不超过50字的修改说明"
}
```

## 5. 完成记录

| 日期 | 完成内容 | 验证命令或结果 | 下一任务 |
| --- | --- | --- | --- |
| 2026-10-08 | 完成研究方案和单文件实验执行手册 | 已检查章节、待办和提示词 | 第 1 步数据准备 |
| 2026-10-08 | 完成第 1 步：从 FActScore 官方 `data.zip` 整理人工原子事实；生成统一 Claim、固定 topic 分组切分和数据报告；未下载 Wikipedia | `python -m unittest discover -s tests -v`：5 项通过；`python -m src.prepare_data --check-reproducibility`：通过。120 条回答按 72/24/24 切分，共 3743 条选中 Claim，topic 零重叠 | 第 2 步运行三个 Verifier；先由用户确认可用 API、预算和模型 |
| 2026-10-08 | 初始化 Git 仓库并发布代码、配置、测试、研究文档、来源清单和数据报告；原始及生成数据由 `.gitignore` 排除 | `python -m unittest discover -s tests -v`：5 项通过；密钥特征扫描无命中；成功推送 `main` 到 `git@github.com:weirdo-lyx/-mulagent.git` | 第 2 步运行三个 Verifier；先由用户确认可用 API、预算和模型 |
| 2026-10-09 | 用户接受 Qwen3 CPU 全量约 23.6 小时；冻结 `hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M`（Ollama 0.34.4）为 V3 候选，参数为 CPU、上下文 4096、温度 0、JSON Schema、`/no_think`、提示词 `verifier-v1`。用户可能改在其他电脑运行，因此 train/dev/test 全量调用暂缓；10 条结果仅为冒烟检查，不作准确率结论 | 没有运行新的模型调用或测试；配置与 10 条既有试跑结果见 [small_model_setup.md](small_model_setup.md)。全量尚未运行 | 第 2 步准备：确定 V1/V2 的模型、API 和预算；实现统一运行器并进行 mock 与小样本试跑；所有正式评估前让三个 Verifier 覆盖相同 Claim |
| 2026-10-09 | 第 2 步离线准备：记录用户的 DeepSeek 4.1 Flash、GPT 中转站 `gpt-6.1-sol`（更正连字符）、中转地址及费用暂不估算决定；新增 `src/verifiers.py`、`src/run_verifiers.py`、`config/verifiers.example.json`、`tests/test_verifiers.py`、`tests/test_api_pilot.ps1`、`scripts/run_api_pilot.ps1`，更新 README、本手册及 plan 的已确认耗时状态。实现统一接口、严格输出、限速/重试、缓存、原子检查点、失败日志、固定种子采样与 mock/live 分离。只读核对旧 Qwen 脚本，确认安装提示词不是正式提示词，需要重新试跑 | `python -m unittest discover -s tests -v`：22 项通过；`python -m src.run_verifiers --report outputs/verifiers/plan.json`：通过；`--mode mock` 首次各 10 条成功（30/30），再次各 10 条命中缓存、0 新尝试，报告在 `outputs/verifiers/mock_first.json`、`mock_resume.json`。PS 语法检查通过；直接执行 PS 测试被本机策略拦截，改 `powershell -NoProfile -ExecutionPolicy Bypass -File .\tests\test_api_pilot.ps1` 后成功/失败两种流程与密钥恢复检查通过，仅作用新进程。当前两套 Key 均未配置；真实 API/新的 Qwen 调用均为 0 | 在本机隐藏输入两把密钥，执行两个 API 各 10 条真实试跑；复核结果后再冻结参数。Qwen 等运行电脑确定后完成正式 10 条；全量和第 3 步尚未开始 |
| 2026-10-09 | 按用户“写到一个 config，我自己加完你再跑”的选择，新建空密钥的 `config/api.example.json` 与用户自填的 `config/api.local.json`，预填两个 Base URL；将本地文件加入 `.gitignore`。修改 `src/run_verifiers.py` 仅在 live 模式读取本地配置并在内存传给调用接口；空值回退到环境变量，错误不回显私密内容。新增配置读取与保密测试，更新 README 与本手册的下一任务；没有改动原始数据或请求参数 | `python -m unittest discover -s tests -v`：26 项通过；`powershell -NoProfile -ExecutionPolicy Bypass -File .\tests\test_api_pilot.ps1`：离线成功/失败及密钥恢复检查通过；`git check-ignore config/api.local.json`：确认忽略。新增测试覆盖本地配置读取、环境变量回退、解析错误/报告不暴露密钥，以及 plan/mock 不读私密文件；真实模型请求为 0 | 用户填写本地配置的两个 API_KEY 并通知“加好了”，由 AI 直接跑 DeepSeek/GPT 各 10 条并检查结果；Qwen 电脑待定，不执行全量 |
| 2026-10-09 | Added DEEPSEEK_MODEL and GPT_RELAY_MODEL support in the ignored local API config; live runner applies configured model IDs and reports them. Preserved existing local settings, added regression coverage, and sanitized the public example so both key fields are blank. No model requests were made. | python -m unittest discover -s tests -v: 26 passed; powershell API pilot wrapper: passed with 0 external requests; git diff --check: passed; git check-ignore config/api.local.json: passed. | User confirms model IDs, Base URLs, and API keys are correct in local config; then run DeepSeek/GPT 10-claim pilots. If credential-like values previously shown in the example were real, revoke and replace those keys first. |


| 2026-10-09 | 按用户要求只保留一个 API 配置文件：删除 config/api.example.json，保留被 Git 忽略的 config/api.local.json；README 改为只说明本地配置文件。 | 搜索确认 README 和当前操作说明不再引用已删除模板；未运行测试或真实模型调用。 | 用户在 config/api.local.json 填妥两套 API Key 并确认模型 ID、Base URL 后通知，随后运行两个 API 各 10 条试跑。 |
| 2026-10-09 | 用户通知两套 API 已配置；按既有授权，使用种子 42、正式 verifier-v1 和同一批 10 条训练 Claim 完成 DeepSeek/GPT 真实试跑。GPT 10/10 格式有效；DeepSeek 9/10，有 1 条理由超过 50 字符。保留失败日志，DeepSeek 断点续跑仅重试该 Claim，仍失败，未将其标记通过。输出及检查报告位于 outputs/verifiers/live；修复完成记录中的转义换行显示 | `python -m unittest discover -s tests -v`：26 项通过；`python -m src.run_verifiers --mode live --verifier v1`：9 成功/1 失败、13 尝试；`--verifier v2`：10 成功/0 失败、10 尝试；DeepSeek 续跑：9 缓存、1 失败、3 尝试。共 26 次尝试；输出密钥扫描通过；同批 9 条共同有效结果中 4 条分歧。GPT 判全部支持，其中 2 条与训练人工标签不符；不据此宣称总体准确率 | 解决 DeepSeek 理由长度格式问题，记录任何提示词新版本并按相同版本复测两套 API；Qwen 等运行电脑确定后验收同批 10 条。暂不全量或进入第 3 步 |
| 2026-10-09 | 用户在同一个本地配置中新增 Claude，确认当前组合为 DeepSeek/GPT/Claude。将 V3 从 Qwen 接为 Claude 的 OpenAI 兼容中转接口，新增字段映射；强化简短中文理由得到 verifier-v2，保留严格 50 字符校验。Claude 1 条预检及诊断确认返回 Markdown 代码块，未伪造通过；进一步明确纯 JSON 得到 verifier-v3，三模型同版各 10 条完成。按版本隔离新输出、保留全部旧结果；更新 src/verifiers.py、src/run_verifiers.py、config/verifiers.example.json、scripts/run_api_pilot.ps1、tests/test_verifiers.py、tests/test_api_pilot.ps1、README.md、plan.md 和本手册；未改动本地密钥及原始数据 | `python -m unittest discover -s tests -v`：27 项通过；PowerShell 包装器离线测试通过；verifier-v3 三模型 30/30 格式有效，完整复跑缓存 30/30、0 新请求。整个接入和诊断阶段共 55 次尝试（v2 25、v3 30）；新报告与三份结果在 outputs/verifiers/live/verifier-v3。校验训练 Claim、版本、缓存摘要、标签及密钥扫描通过；4/10 Claim 有分歧；小样本不作总体可靠性结论。全量尚未运行，未设置全量放行标志 | 固定三个 API 的配置、verifier-v3 提示词和数据划分，保存无密钥快照；按用户正式批量指令收集 3743 条 Claim 的三份判断并生成基线；之后进入第 3 步风险模型 |
| 2026-10-09 | 按用户要求补齐现有代码的中文模块说明、函数说明和关键流程注释，覆盖 src/prepare_data.py、src/verifiers.py、src/run_verifiers.py、测试与 PowerShell 脚本；保持算法和提示词不变。PowerShell 使用带 BOM 的 UTF-8，使 Windows PowerShell 5.1 正确读取中文注释。新增可发布的 data/reports/verifier_pilot_report.json 汇总，README 补代码入口及新电脑单文件 API 配置方式，并整理当前阶段成果准备上传既有 GitHub 仓库 | 27 项单元测试通过；PowerShell 成功/失败密钥恢复测试通过；mock 初跑及缓存复跑通过、0 外部请求；原有 30 条 live 缓存键保持一致；数据准备 AST 比较确认只改注释；候选发布文件密钥扫描通过，本地私密配置被 Git 忽略。远端 main 核对为 3e2cc1d，待本轮提交推送验证 | 提交并验证 GitHub 推送；之后继续冻结三模型配置和正式批量判断 |
