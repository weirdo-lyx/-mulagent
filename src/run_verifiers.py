"""第 2 步命令行入口：计划、离线模拟、真实试跑与正式调用。

默认 plan 只生成调用计划；mock 验证数据流和缓存；只有 live 会访问模型。
小样本只从训练集抽取，正式全量需要单独标记验收，防止意外大批量调用。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.verifiers import (
    PROMPT_VERSION,
    VerifierConfig,
    VerifierError,
    _read_jsonl,
    build_messages,
    build_payload,
    resolve_base_url,
    run_claims,
)


def load_configs(path: Path) -> tuple[dict[str, Any], list[VerifierConfig]]:
    """读取无密钥的公共实验参数，将全局默认值合并到每个判断者的配置。"""
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("verifiers"), list):
        raise ValueError("configuration requires a verifiers list")
    if any(not isinstance(value, dict) for value in document["verifiers"]):
        raise ValueError("each Verifier configuration must be a JSON object")
    defaults = {key: document[key] for key in (
        "prompt_version", "max_attempts", "timeout_seconds", "min_interval_seconds", "seed",
    ) if key in document}
    configs = [VerifierConfig.from_mapping({**defaults, **value}) for value in document["verifiers"]]
    ids = [value.verifier_id for value in configs]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("configuration must contain unique Verifier IDs")
    if any(value.prompt_version != PROMPT_VERSION for value in configs):
        raise ValueError(f"this runner implements only {PROMPT_VERSION}")
    return document, configs


def load_api_environment(
    path: Path,
    configs: list[VerifierConfig],
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """将本地私有配置合入环境副本；非空本地值优先，空值回退到环境变量。

    不修改进程环境，不打印私有文件；只允许已配置判断者使用的字段。
    本函数仅在 live 模式调用，plan/mock 不依赖密钥文件。
    """
    environment = dict(os.environ if environ is None else environ)
    if not path.exists():
        return environment
    try:
        settings = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        # JSON 错误只报告行列，不输出可能含密钥的源文本。
        raise ValueError(
            f"API configuration is invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from None
    if not isinstance(settings, dict):
        raise ValueError("API configuration must be a JSON object")
    allowed = {config.base_url_env for config in configs}
    allowed.update(config.api_key_env for config in configs if config.api_key_env)
    allowed.update(config.model_env for config in configs if config.model_env)
    if set(settings) - allowed:
        raise ValueError("API configuration contains unsupported field names")
    for name, value in settings.items():
        if not isinstance(value, str) or any(ord(character) < 32 for character in value):
            raise ValueError("API configuration values must be single-line strings")
        if value.strip():
            environment[name] = value.strip()
    return environment


def select_claims(root: Path, limit: int, seed: int, full_run: bool = False) -> list[dict[str, Any]]:
    """按固定种子选择共同试跑集；全量模式读取三个已经固定的划分。"""
    if limit < 1:
        raise ValueError("limit must be positive")
    claims = []
    splits = ("train", "dev", "test") if full_run else ("train",)
    for split in splits:
        for row in _read_jsonl(root / "data" / "splits" / f"{split}.jsonl"):
            # 白名单构造模型输入，主动去掉人工标签、原回答和其他判断者的输出。
            claims.append({
                "claim_id": row["claim_id"], "topic": row["topic"],
                "claim_text": row["claim_text"], "_split": split,
            })
    if len({row["claim_id"] for row in claims}) != len(claims):
        raise ValueError("split inputs contain duplicate Claim IDs")
    claims.sort(key=lambda row: row["claim_id"])
    # 先排序再随机抽样，文件行顺序变化不会改变相同种子的样本。
    if full_run:
        return claims
    sample = random.Random(seed).sample(claims, min(limit, len(claims)))
    return sorted(sample, key=lambda row: row["claim_id"])


def mock_post(url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: int) -> dict:
    """固定返回 uncertain 的离线样例，只测格式和流程，不能用于准确率统计。"""
    content = json.dumps({
        "label": "uncertain", "confidence": 0.0, "reason": "离线 mock，不是事实判断。",
    }, ensure_ascii=False)
    if url.endswith("/api/chat"):
        return {"model": payload["model"], "message": {"content": content}}
    return {"model": payload["model"], "choices": [{"message": {"content": content}}]}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析模式、判断者和试跑大小；默认模式始终是无需 API 的 plan。"""
    parser = argparse.ArgumentParser(description="Plan, mock, or run independent Verifiers")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("config/verifiers.example.json"))
    parser.add_argument("--api-config", type=Path, default=Path("config/api.local.json"),
                        help="private connection settings, loaded only in live mode")
    parser.add_argument("--mode", choices=("plan", "mock", "live"), default="plan")
    parser.add_argument("--verifier", default="all", help="all or one configured ID")
    parser.add_argument("--limit", type=int, default=10, help="fixed-seed training pilot size")
    parser.add_argument("--full-run", action="store_true", help="all splits; live needs accepted pilots")
    parser.add_argument("--report", type=Path, help="optional generated JSON report destination")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """组织输入、验证当前模式、运行所选判断者并输出不含密钥的报告。"""
    args = parse_args(argv)
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    document, all_configs = load_configs(config_path)
    configs = [config for config in all_configs if args.verifier in {"all", config.verifier_id}]
    if not configs:
        raise ValueError("requested Verifier ID was not found")
    if args.mode == "live" and args.limit > 10 and not args.full_run:
        raise ValueError("live pilots are limited to 10 training Claims")
    if args.mode == "live" and args.full_run:
        # 试跑通过后须明确记录放行；仅提供 --full-run 不足以启动全量请求。
        if not all(value.get("pilot_accepted") is True for value in document["verifiers"]):
            raise ValueError("full execution requires accepted real pilots for every Verifier")
    seed = int(document.get("seed", 42))
    claims = select_claims(root, args.limit, seed, args.full_run)
    report: dict[str, Any] = {
        "mode": args.mode, "seed": seed, "claim_count": len(claims),
        "claim_ids": [row["claim_id"] for row in claims],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "prompt_version": PROMPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "external_api_requests": 0 if args.mode != "live" else None,
        "verifiers": [],
    }
    # 先检查所有选中判断者的地址和密钥，避免只因后一个缺配置而半途停止。
    if args.mode == "live":
        api_path = args.api_config if args.api_config.is_absolute() else root / args.api_config
        live_environment = load_api_environment(api_path, all_configs)
        configs = [
            # API 本地配置里的模型 ID 是实际请求值，也应出现在报告和缓存键中。
            replace(config, model=live_environment[config.model_env].strip())
            if config.model_env and live_environment.get(config.model_env, "").strip()
            else config
            for config in configs
        ]
        for config in configs:
            resolve_base_url(config, live_environment)
            if config.api_key_env and not live_environment.get(config.api_key_env, "").strip():
                raise VerifierError(f"missing API key environment variable: {config.api_key_env}")
    for config in configs:
        payload = build_payload(claims[0], config) if claims else None
        details: dict[str, Any] = {
            "verifier_id": config.verifier_id, "model": config.model,
            "provider": config.provider, "claim_count": len(claims),
            "max_attempts": config.max_attempts,
            "max_output_tokens": config.max_output_tokens,
            "prompt_sha256": hashlib.sha256(
                json.dumps([build_messages(row, config.no_think) for row in claims],
                           ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "request_parameters": {key: value for key, value in (payload or {}).items()
                                   if key != "messages"},
        }
        if args.mode != "plan":
            output = (root / "outputs" / "verifiers" / args.mode / config.prompt_version
                      / f"{config.verifier_id}.jsonl")
            errors = output.with_name(f"{config.verifier_id}_errors.jsonl")
            if args.mode == "mock":
                # mock 使用假地址和假密钥，并显式替换传输层，绝不回退到真实 HTTP。
                env = {config.base_url_env: "https://mock.invalid"}
                if config.api_key_env:
                    env[config.api_key_env] = "MOCK-NOT-A-CREDENTIAL"
                settings = replace(config, min_interval_seconds=0)
                stats = run_claims(claims, settings, output, errors, environ=env,
                                   post_json=mock_post, execution_mode="mock")
            else:
                stats = run_claims(claims, config, output, errors,
                                   environ=live_environment, execution_mode="live")
            details.update({"stats": stats, "output": str(output.relative_to(root))})
        report["verifiers"].append(details)
    if args.mode == "live":
        report["external_api_requests"] = sum(
            row["stats"]["transport_attempts"] for row in report["verifiers"]
            if row["provider"] == "openai_compatible"
        )
    if args.report:
        destination = args.report if args.report.is_absolute() else root / args.report
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if any(row.get("stats", {}).get("failed", 0) for row in report["verifiers"]) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, VerifierError, OSError) as exc:
        raise SystemExit(f"Verifier run stopped: {exc}") from exc
