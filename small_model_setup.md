# 本地小模型：D 盘安装与运行评估

> 本文记录实验第 1A 步的实际部署。所有安装文件、模型、配置、临时文件和试跑结果均放在 `D:\research\ollama`；不需要把模型或 Ollama 安装到 C 盘。

## 本机与结论

- 系统：Windows 版本信息：23H2，build 22631
- CPU：AMD Ryzen 7 5700G
- 内存：16 GB
- GPU：AMD Radeon 集成显卡；报告专用显存 0.5 GB
- D 盘部署空间：下载前约有 167.5 GB 可用
- 推理设备：CPU。此次未配置或依赖 GPU 加速
- 运行器：Ollama Windows 便携 CLI，版本 0.34.4
- 模型：`hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M`，Qwen 官方 GGUF 仓库，Q4_K_M 量化
- 10 条临时 Claim：10/10 JSON 结构有效，10/10 标签与预设答案一致；平均 22.7 秒/条，范围 15.83–29.45 秒
- 按 3743 条线性估算，单独运行本模型约需 23.6 小时。此估算不含失败重试、电脑同时运行其他任务造成的减速；10 条临时样例只是接口与粗略运行检查，不代表模型准确率评估
- 结论：技术上可以运行，但全量 CPU 时间较长。是否将其纳入正式三个 Verifier，待用户决定能否接受约 23.6 小时

## 存储估算

实际部署数据：

| 项目 | 大小 |
| --- | ---: |
| Ollama 官方便携 ZIP 下载包 | 约 1.46 GB |
| 解压后的运行文件 | 约 1.90 GB |
| Qwen3 4B Q4_K_M 模型缓存 | 约 2.33 GB |
| D:\research\ollama 实际保留总量 | 约 5.9 GB |
| 首次下载失败时的峰值临时分片 | 约 2.33 GB，已清理 |

建议 D 盘至少预留 8 GB，以容纳下载包、解压文件、模型和更新/临时文件。现在模型和运行器已安装，不需要再下载即可试用。试跑 JSONL、日志、脚本也都在 D 盘。

## 安装记录与位置

使用 Ollama 官方发布的 Windows AMD64 便携 CLI ZIP，没有运行图形安装器。ZIP 位于：

```text
D:\research\ollama\downloads\ollama-windows-amd64-v0.34.4.zip
```

SHA-256：

```text
535193F38F3344E5B08F5D1C171C31CE11AA17F0124FF69AE26D8EC7FE06FA62
```

运行文件和所有运行数据位置：

```text
D:\research\ollama\ollama.exe
D:\research\ollama\models\
D:\research\ollama\profile\
D:\research\ollama\temp\
D:\research\ollama\logs\
D:\research\ollama\pilot\
```

`start_ollama_d.ps1` 会将 Ollama 模型目录、用户配置目录和临时目录设为 D 盘路径，并将服务绑定到本机地址 `127.0.0.1:11434`。它启动本地服务时使用隐藏窗口，不改变系统执行策略，也不新增全局 PATH。

## 启动与使用

在 PowerShell 中运行：

```powershell
& 'D:\research\ollama\start_ollama_d.ps1'
```

看到版本号和 `OLLAMA_MODELS=D:\research\ollama\models` 后，模型服务已在本机启动。查询已安装模型：

```powershell
& 'D:\research\ollama\ollama.exe' list
```

进入交互式对话：

```powershell
& 'D:\research\ollama\ollama.exe' run 'hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M'
```

退出对话输入 `/bye`。安装完成后，运行模型无需联网；初次下载模型需要联网。

## 本次下载经过

原计划从 Ollama 模型库获取 Qwen3 4B Instruct Q4_K_M。该地址的模型分片连接失败（TLS 握手失败），没有完成下载。随后改用 Qwen 团队维护的官方 Hugging Face GGUF 仓库：

```powershell
& 'D:\research\ollama\ollama.exe' pull 'hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M'
```

下载和摘要校验成功。现在实际安装的是 Qwen3-4B-GGUF 仓库中的 Q4_K_M 标签，不是原计划的 `4b-instruct-2507` 精确构建，因此后续正式使用前应冻结并记录此实际模型标识。

## 10 条小样本记录

输入是专为安装检查准备的 10 条临时基础常识 Claim，不取自项目 train/dev/test 数据，也不用于调整提示词、阈值或模型。使用 CPU、上下文长度 4096、温度 0、JSON Schema 约束和 `/no_think`。

结果文件：

```text
D:\research\ollama\pilot\claims.json
D:\research\ollama\pilot\pilot_results.jsonl
D:\research\ollama\pilot\run_pilot.ps1
```

结果：10 条均成功返回、10 条均通过格式检查、10 条均与预设标签一致。标签分布为 supported 6 条、unsupported 4 条。平均 22.7 秒/条，样本中位数 22.03 秒，最长 29.45 秒。该结果只验证模型能响应以及结构化输出可用；样例规模太小且内容简单，不能据此声称模型准确率为 100%。

按 3743 条估算：

```text
3743 × 22.7 秒 ÷ 3600 ≈ 23.6 小时
```

本次试跑后模型已从内存卸载；服务器进程可由上面的启动脚本重新启动。试跑时没有采集推理过程中的峰值内存，因此不要把模型成功加载解释为已验证所有并发场景的内存余量。

## 验证与待决定事项

本次已验证：

- Ollama CLI 版本为 0.34.4；
- 模型成功下载、校验并由本机 CPU 推理；
- 正式试跑 10/10 结构合法、10/10 标签匹配；
- 下载包、程序、模型、配置、日志、临时目录和试跑数据均位于 `D:\research\ollama`；
- Ollama 服务仅监听本机回环地址。

待决定：3743 条预计需要约 23.6 小时 CPU 时间。用户接受后，才把该确切模型和参数冻结为正式 V3 Verifier；若不接受，应在进入正式三模型步骤前改评估更小的候选或其他独立 Verifier，并单独验证，不能直接假定替代模型合格。

## 官方资料

- [Ollama Windows 安装、便携 CLI 与模型存储文档](https://github.com/ollama/ollama/blob/main/docs/windows.mdx)
- [Ollama 官方 Windows 下载页](https://ollama.com/download/windows)
- [Qwen 团队的 Qwen3-4B-GGUF 模型仓库](https://huggingface.co/Qwen/Qwen3-4B-GGUF)
- [Qwen 团队的 Qwen3-4B 模型卡](https://huggingface.co/Qwen/Qwen3-4B)
