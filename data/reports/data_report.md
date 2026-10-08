# FActScore 数据准备报告

## 来源与完整性

- 官方仓库：https://github.com/shmsw25/FActScore
- 官方下载目录：https://drive.google.com/drive/folders/1kFey69z8hGXScln01mVxrOhrqgM62X7I?usp=sharing
- 下载日期：2026-10-08
- 原始压缩包：`data/raw/factscore/data.zip`
- 文件大小：2197467 bytes
- SHA-256：`404482c9a7d588d581a2f7a78d609fe6bfec4ddc0f729c8560a61b1743dbffab`
- 未下载 Wikipedia 数据库。

## 字段与标签

官方每行是一条模型回答，主要字段为 `input`、`output`、`topic`、`cat` 和 `annotations`。
本实验读取 `annotations[].human-atomic-facts[]`：`S` 映射为 `supported/0`，
`NS` 映射为 `unsupported/1`；`IR` 表示无关事实，按方案排除并计数。

## 原始数据统计

- 回答数：549
- 人工原子事实数：16040
- 原标签分布：{"IR": 1515, "NS": 4419, "S": 10106}
- 转换后可用 Claim 数：14502
- 各来源回答数：{"ChatGPT": 183, "InstructGPT": 183, "PerplexityAI": 183}
- 各来源可用 Claim 数：{"ChatGPT": 4879, "InstructGPT": 4063, "PerplexityAI": 5560}

## 排除与清洗

- `answers_with_empty_output`：1
- `answers_with_null_annotations`：44
- `conflicting_duplicate_claims`：4
- `label_IR_irrelevant`：1507
- `same_label_duplicate_claims`：27
- `sentences_with_null_human_facts`：232

同一回答内文本规范化后完全相同且标签相同的 Claim 只保留第一次；
标签冲突的重复 Claim 整组排除。未知标签不会被猜测或映射。

## 抽样与切分

- 随机种子：42
- 最大回答数：120
- 选中 topic 数：40
- 实际回答数：120
- 实际 Claim 数：3743
- 切分单位：topic；同一 topic 的三个来源回答始终在同一集合。

| 集合 | Topic | 回答 | Claim | supported | unsupported | unsupported 比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 24 | 72 | 2306 | 1689 | 617 | 0.2676 |
| dev | 8 | 24 | 705 | 485 | 220 | 0.3121 |
| test | 8 | 24 | 732 | 529 | 203 | 0.2773 |

## 验收检查

- [x] JSONL 可逐行解析；
- [x] Claim ID 全局唯一；
- [x] 标签仅为 `supported` / `unsupported` 且二值映射一致；
- [x] train/dev/test 的 topic 两两零重叠；
- [x] 每个集合均含正、负两类 Claim；
- [x] 固定配置重复运行得到逐字节一致的四个 JSONL 文件。
