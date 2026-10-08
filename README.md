# 多 Verifier 事实风险实验

当前仅完成 `experiment.md` 的第 1 步：FActScore 人工标注数据准备。

```powershell
python -m unittest discover -s tests -v
python -m src.prepare_data --check-reproducibility
```

原始文件位于 `data/raw/factscore/`，处理脚本只读取该目录。统一 Claim 和按 topic
隔离的 train/dev/test 划分分别写入 `data/processed/` 与 `data/splits/`。

Git 仓库不包含下载的 FActScore 原始数据或生成的 JSONL。官方来源、文件大小和
SHA-256 记录在 `data/raw/factscore/source_manifest.json`；取得官方 `data.zip`
并解压到清单对应目录后，即可运行上述命令重建数据产物。
