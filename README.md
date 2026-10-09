# 多 Verifier 事实风险实验

第 1 步 FActScore 数据准备已通过验收；第 2 步的调用接口和离线验证已完成。
三个 API 的同版小样本已通过格式检查（各 10 条）；正式配置冻结和全量结果尚未完成，
进度以 `experiment.md` 为准。可随仓库查看的汇总见
[三模型试跑报告](data/reports/verifier_pilot_report.json)，详细输出保留在本机。

代码入口：`src/prepare_data.py` 准备与划分数据，`src/verifiers.py` 负责请求、校验、
重试和缓存，`src/run_verifiers.py` 负责选择运行模式、组织样本和生成报告。
关键函数和实验约束已用中文注释说明。

```powershell
python -m unittest discover -s tests -v
powershell -NoProfile -ExecutionPolicy Bypass -File .\tests\test_api_pilot.ps1
python -m src.prepare_data --check-reproducibility
```

Git 仓库不包含下载的 FActScore 原始数据或生成的 JSONL。官方来源、文件大小和
SHA-256 记录在 `data/raw/factscore/source_manifest.json`；取得官方 `data.zip`
并解压到清单对应目录后，即可运行上述命令重建数据产物。

## 第 2 步：先检查，再运行

候选配置见 `config/verifiers.example.json`：V1 为 DeepSeek 4.1 Flash（API ID
`deepseek-flash`），V2 为用户中转站的 `gpt-6.1-sol`，V3 为用户新加的
`claude-sonnet-4-6`。三个模型使用 `verifier-v3`，API 兼容性和实际返回模型需通过小样本确认。

默认命令只生成计划，不读取密钥、不访问模型服务：

```powershell
python -m src.run_verifiers --report outputs/verifiers/plan.json
python -m src.run_verifiers --mode mock --report outputs/verifiers/mock_summary.json
```

### API 配置和 10 条试跑

所有 API 设置放在本机的 `config/api.local.json`：模型 ID、Base URL 和 API Key
都在这一个文件里。三个模型 ID 字段为 `DEEPSEEK_MODEL`、`GPT_RELAY_MODEL`、
`CLAUDE_RELAY_MODEL`；各自对应的 Base URL 和 API Key 也放在同一个文件中。
只在本地填写 API Key，不要发到聊天里。该文件已被 Git 忽略。

在新电脑上需要自行创建这一个配置文件，并填写三套密钥：

```json
{
  "DEEPSEEK_MODEL": "deepseek-flash",
  "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
  "DEEPSEEK_API_KEY": "",
  "GPT_RELAY_MODEL": "gpt-6.1-sol",
  "GPT_RELAY_BASE_URL": "https://api.sbbbbbbbbb.xyz/v1",
  "GPT_RELAY_API_KEY": "",
  "CLAUDE_RELAY_MODEL": "claude-sonnet-4-6",
  "CLAUDE_RELAY_BASE_URL": "https://api.sbbbbbbbbb.xyz/v1",
  "CLAUDE_RELAY_API_KEY": ""
}
```

填好配置并通知后，AI 使用种子 42、同一批训练样本试跑三个模型各 10 条。
每条最多尝试 3 次。旧版输出保留，新版结果按提示词版本分目录保存。

```powershell
python -m src.run_verifiers --mode live --report outputs/verifiers/live/verifier-v3/summary.json
```

当前 V3 已改为 Claude。Qwen 的历史安装和试跑记录见 `small_model_setup.md`。

当前真实结果和错误保存在 `outputs/verifiers/live/verifier-v3/`。运行器记录标签、实际返回模型、
重试次数、已知 token 用量和耗时；密钥不会写入报告或错误日志。小样本尚未验收时，
禁止全量调用。不得用 mock 结果判断准确率。

GitHub 保存代码、文档和无密钥试跑汇总；本地配置、原始数据和逐条运行输出
不随 Git 推送。小样本格式合格不代表事实判断全部正确。
