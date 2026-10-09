"""第 2 步的模型调用基础层：请求、输出校验、重试、缓存和失败记录。

这里仅收集各 Verifier 的独立判断，不读取人工标签，也不计算错误风险。
API Key 只通过调用环境进入请求头；写盘内容不包含调用环境或请求头。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlsplit


VALID_LABELS = {"supported", "unsupported", "uncertain"}
PROMPT_VERSION = "verifier-v3"
SYSTEM_PROMPT = (
    "你是独立的事实核查初审员。请判断 Claim 是否符合事实。"
    "你看不到其他核查员的结论。信息不足、内容含糊或无法可靠确认时选择 uncertain，"
    "不要猜测。supported 表示核心事实正确；unsupported 表示至少一个核心事实错误；"
    "uncertain 表示无法可靠确定。直接输出一个JSON对象，不要Markdown代码块、"
    "代码围栏、JSON标签或任何额外文字。回复第一个字符必须是{，最后一个字符必须是}。"
    "reason 只写一句简短中文依据，"
    "尽量控制在30个字符以内，最多50个字符（包含标点、数字和英文字母）。"
    "避免复述长人名、标题或完整Claim；不要添加不必要的细节。"
)
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": sorted(VALID_LABELS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "minLength": 1, "maxLength": 50},
    },
    "required": ["label", "confidence", "reason"],
    "additionalProperties": False,
}


class VerifierError(RuntimeError):
    """一次 Claim 调用失败；附带尝试次数和已知用量供错误日志统计。"""
    def __init__(self, message: str, *, attempts: int = 0, usage: dict[str, int] | None = None):
        super().__init__(message)
        self.attempts = attempts
        self.usage = usage


class RetryableVerifierError(VerifierError):
    """可在当前尝试上限内重试的网络、限流或输出格式错误。"""
    pass


@dataclass(frozen=True)
class VerifierConfig:
    """单个判断者的非私密配置；密钥字段保存变量名，不保存密钥值。"""
    verifier_id: str
    provider: str
    model: str
    base_url_env: str
    base_url_default: str = ""
    api_key_env: str | None = None
    model_env: str | None = None
    temperature: float = 0.0
    context_length: int = 4096
    max_attempts: int = 3
    timeout_seconds: int = 120
    no_think: bool = False
    prompt_version: str = PROMPT_VERSION
    max_output_tokens: int = 256
    output_token_field: str = "max_tokens"
    min_interval_seconds: float = 0.0
    seed: int = 42
    request_options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VerifierConfig":
        """合并配置后校验类型、参数范围与允许的请求选项，避免静默换参数。"""
        required = ("verifier_id", "provider", "model", "base_url_env")
        missing = [key for key in required if not value.get(key)]
        if missing:
            raise ValueError(f"missing verifier configuration fields: {', '.join(missing)}")
        provider = str(value["provider"])
        if provider not in {"openai_compatible", "ollama"}:
            raise ValueError(f"unsupported provider: {provider}")
        attempts = int(value.get("max_attempts", 3))
        if attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if int(value.get("timeout_seconds", 120)) < 1 or int(value.get("max_output_tokens", 256)) < 1:
            raise ValueError("timeout_seconds and max_output_tokens must be positive")
        token_field = str(value.get("output_token_field", "max_tokens"))
        if token_field not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("unsupported output token field")
        interval = float(value.get("min_interval_seconds", 0))
        if not math.isfinite(interval) or interval < 0 or interval > 60:
            raise ValueError("min_interval_seconds must be between 0 and 60")
        options = value.get("request_options", {})
        if not isinstance(options, dict) or set(options) - {"thinking", "reasoning_effort"}:
            raise ValueError("request_options supports only thinking and reasoning_effort")
        verifier_id = str(value["verifier_id"])
        if not verifier_id.isidentifier():
            raise ValueError("verifier_id must be a simple identifier")
        return cls(
            verifier_id=verifier_id,
            provider=provider,
            model=str(value["model"]),
            base_url_env=str(value["base_url_env"]),
            base_url_default=str(value.get("base_url_default", "")),
            api_key_env=value.get("api_key_env"),
            model_env=value.get("model_env"),
            temperature=float(value.get("temperature", 0)),
            context_length=int(value.get("context_length", 4096)),
            max_attempts=attempts,
            timeout_seconds=int(value.get("timeout_seconds", 120)),
            no_think=bool(value.get("no_think", False)),
            prompt_version=str(value.get("prompt_version", PROMPT_VERSION)),
            max_output_tokens=int(value.get("max_output_tokens", 256)),
            output_token_field=token_field,
            min_interval_seconds=interval,
            seed=int(value.get("seed", 42)),
            request_options=dict(options),
        )


def build_messages(claim: Mapping[str, Any], no_think: bool = False) -> list[dict[str, str]]:
    """只向模型发送主题和当前 Claim，保持三个判断者互相独立。"""
    system = SYSTEM_PROMPT
    if no_think:
        system = "/no_think\n" + system
    user = (
        f"Topic: {claim['topic']}\nClaim: {claim['claim_text']}\n\n输出：\n"
        '{\n  "label": "supported | unsupported | uncertain",\n'
        '  "confidence": 0到1之间的数字,\n'
        '  "reason": "一句中文短理由，尽量不超过30字符，最多50字符"\n}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_payload(claim: Mapping[str, Any], config: VerifierConfig) -> dict[str, Any]:
    """将同一组任务消息适配为 OpenAI 兼容或 Ollama 的请求格式。"""
    messages = build_messages(claim, config.no_think)
    if config.provider == "ollama":
        return {
            "model": config.model, "messages": messages, "stream": False,
            "format": OUTPUT_SCHEMA,
            "options": {
                "temperature": config.temperature, "num_ctx": config.context_length,
                "num_predict": config.max_output_tokens, "seed": config.seed, "num_gpu": 0,
            },
        }
    return {
        "model": config.model, "messages": messages, "temperature": config.temperature,
        "response_format": {"type": "json_object"},
        config.output_token_field: config.max_output_tokens,
        **config.request_options,
    }


def resolve_base_url(config: VerifierConfig, env: Mapping[str, str]) -> str:
    """优先使用运行环境的地址，并拒绝含凭据、查询参数的 URL。"""
    base = env.get(config.base_url_env, config.base_url_default).strip().rstrip("/")
    if not base:
        raise VerifierError(f"missing endpoint environment variable: {config.base_url_env}")
    parts = urlsplit(base)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise VerifierError("endpoint must be an HTTP(S) base URL")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise VerifierError("endpoint must not contain credentials, query parameters, or a fragment")
    return base


def cache_key(
    claim: Mapping[str, Any], config: VerifierConfig, base_url: str, execution_mode: str = "live",
) -> str:
    """以实际输入、提示词、模型、端点和请求参数生成成功结果的缓存键。

    改模型或提示词会产生新键；mock 和 live 也互不复用。密钥不参与摘要，
    因而仅更换同一服务的密钥不会重复请求已经保存的判断。
    """
    material = {
        "claim_id": claim["claim_id"],
        "claim_text": claim["claim_text"],
        "topic": claim["topic"],
        "verifier_id": config.verifier_id,
        "provider": config.provider,
        "base_url": base_url.rstrip("/"),
        "model": config.model,
        "prompt_version": config.prompt_version,
        "temperature": config.temperature,
        "context_length": config.context_length,
        "no_think": config.no_think,
        "execution_mode": execution_mode,
        "payload": build_payload(claim, config),
    }
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _extract_content(response: Mapping[str, Any], provider: str) -> tuple[str, dict[str, int] | None]:
    """统一提取回答文本和 token 用量；没有完整用量时返回 None。"""
    if provider == "ollama":
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        usage = None
        if all(isinstance(response.get(k), int) and not isinstance(response[k], bool)
               and response[k] >= 0 for k in ("prompt_eval_count", "eval_count")):
            usage = {
                "input_tokens": response["prompt_eval_count"],
                "output_tokens": response["eval_count"],
            }
    else:
        choices = response.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        raw_usage = response.get("usage")
        usage = None
        if isinstance(raw_usage, dict):
            if all(isinstance(raw_usage.get(k), int) and not isinstance(raw_usage[k], bool)
                   and raw_usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens")):
                usage = {
                    "input_tokens": raw_usage["prompt_tokens"],
                    "output_tokens": raw_usage["completion_tokens"],
                }
    if not isinstance(content, str):
        raise RetryableVerifierError("provider response did not contain text content")
    return content, usage


def parse_verifier_output(content: str) -> dict[str, Any]:
    """严格校验模型的 JSON，不截断理由或修补非法回答来伪造格式通过。

    置信度保存在 raw_confidence 中，仅作原始记录，后续不能直接当作错误风险。
    """
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        # 默认 JSON 解析会保留重复键的最后一个值；实验要求直接拒绝这种歧义。
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise RetryableVerifierError("model output contains duplicate JSON keys")
            value[key] = item
        return value

    try:
        value = json.loads(content, object_pairs_hook=unique_object)
    except json.JSONDecodeError as exc:
        raise RetryableVerifierError("model output was not valid JSON") from exc
    if not isinstance(value, dict):
        raise RetryableVerifierError("model output must be a JSON object")
    if set(value) != {"label", "confidence", "reason"}:
        raise RetryableVerifierError("model output must contain exactly label, confidence, and reason")
    if not isinstance(value.get("label"), str) or value["label"] not in VALID_LABELS:
        raise RetryableVerifierError("model output has an invalid label")
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise RetryableVerifierError("confidence must be a number")
    if not math.isfinite(float(confidence)) or not 0 <= confidence <= 1:
        raise RetryableVerifierError("confidence must be between 0 and 1")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 50:
        raise RetryableVerifierError("reason must be a non-empty string of at most 50 characters")
    return {"label": value["label"], "raw_confidence": float(confidence), "reason": reason.strip()}


def _post_json(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout: int,
) -> Mapping[str, Any]:
    """执行一次 HTTP 请求；仅暴露状态码或错误类型，不记录请求头和错误正文。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        # 限流及服务端暂时故障可重试；鉴权或参数错误应直接交给调用者处理。
        if exc.code == 429 or exc.code >= 500:
            raise RetryableVerifierError(f"provider returned HTTP {exc.code}") from exc
        raise VerifierError(f"provider returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, ConnectionError) as exc:
        raise RetryableVerifierError(type(exc).__name__) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetryableVerifierError("provider returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise RetryableVerifierError("provider returned a non-object response")
    return parsed


def call_provider(
    claim: Mapping[str, Any],
    config: VerifierConfig,
    *,
    environ: Mapping[str, str] | None = None,
    post_json: Callable[..., Mapping[str, Any]] = _post_json,
    rate_limiter: "RateLimiter | None" = None,
) -> tuple[dict[str, Any], dict[str, int] | None]:
    """调用一个判断者并执行有限重试，返回合法判断及所有已知尝试的用量。

    post_json 可替换成离线传输函数，使测试不用访问真实服务。
    格式失败也可能已经计费，因此成功前的已知用量必须一并累计。
    """
    env = os.environ if environ is None else environ
    base_url = resolve_base_url(config, env)
    headers = {"Content-Type": "application/json"}
    if config.api_key_env:
        api_key = env.get(config.api_key_env, "").strip()
        if not api_key:
            raise VerifierError(f"missing API key environment variable: {config.api_key_env}")
        headers["Authorization"] = f"Bearer {api_key}"

    url = base_url + ("/api/chat" if config.provider == "ollama" else "/chat/completions")
    payload = build_payload(claim, config)
    limiter = rate_limiter or RateLimiter(config.min_interval_seconds)

    last_error: Exception | None = None
    total_usage = {"input_tokens": 0, "output_tokens": 0}
    usage_complete = True
    known_usage_attempts = 0
    for attempt in range(config.max_attempts):
        limiter.wait()
        try:
            response = post_json(url, headers, payload, config.timeout_seconds)
            content, usage = _extract_content(response, config.provider)
            if usage is not None:
                known_usage_attempts += 1
                for name in total_usage:
                    total_usage[name] += usage[name]
            else:
                usage_complete = False
            parsed = parse_verifier_output(content)
            parsed.update({
                "attempts": attempt + 1,
                "usage_complete": usage_complete,
                "response_model": response.get("model"),
            })
            return parsed, total_usage if known_usage_attempts else None
        except RetryableVerifierError as exc:
            last_error = exc
        except VerifierError as exc:
            raise VerifierError(
                str(exc), attempts=attempt + 1,
                usage=total_usage if known_usage_attempts else None,
            ) from exc
        except Exception as exc:
            raise VerifierError(
                "unexpected response or transport failure", attempts=attempt + 1,
                usage=total_usage if known_usage_attempts else None,
            ) from exc
        if known_usage_attempts < attempt + 1:
            # 超时等失败没有返回用量时，统计只能代表已知部分，不能冒充完整账单。
            usage_complete = False
        if attempt + 1 < config.max_attempts:
            time.sleep(min(2**attempt, 8))
    raise VerifierError(
        f"request failed after {config.max_attempts} attempts: {last_error}",
        attempts=config.max_attempts, usage=total_usage if known_usage_attempts else None,
    ) from last_error


class RateLimiter:
    """同一判断者的全部 Claim 与重试共用限速器，约束请求开始时间的间隔。"""

    def __init__(self, interval: float):
        self.interval = interval
        self.last_start: float | None = None

    def wait(self) -> None:
        # 单调时钟不受系统日期调整影响，适合测量请求之间的等待时间。
        if self.last_start is not None:
            remaining = self.interval - (time.monotonic() - self.last_start)
            if remaining > 0:
                time.sleep(remaining)
        self.last_start = time.monotonic()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """按行读取 JSON 对象，损坏时报告位置并停止，避免静默丢弃实验记录。"""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            yield value


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    """追加并刷盘保存失败事件；历史失败保留，便于追踪重跑发生了什么。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """先写同目录临时文件并刷盘，再替换成功结果文件，保护上一次完整检查点。"""
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _safe_error(exc: Exception, config: VerifierConfig, env: Mapping[str, str]) -> str:
    """只保存受控错误摘要，并移除当前判断者的密钥，限制日志长度。"""
    message = str(exc) if isinstance(exc, VerifierError) else "unexpected response or transport failure"
    if config.api_key_env:
        secret = env.get(config.api_key_env, "").strip()
        if secret:
            message = message.replace(secret, "[redacted]")
    return message[:300]


def load_cached_keys(path: Path) -> set[str]:
    """读取已有成功结果的键；失败事件不会被视为成功缓存。"""
    if not path.exists():
        return set()
    return {str(row["cache_key"]) for row in _read_jsonl(path)}


def run_claims(
    claims: Iterable[Mapping[str, Any]],
    config: VerifierConfig,
    output_path: Path,
    error_path: Path,
    *,
    environ: Mapping[str, str] | None = None,
    post_json: Callable[..., Mapping[str, Any]] = _post_json,
    limit: int | None = None,
    execution_mode: str = "live",
) -> dict[str, int]:
    """顺序运行 Claim，跳过成功缓存，记录失败，每成功一条就保存检查点。

    启动前先校验全部输入和旧缓存。模型失败可以记录后继续；写盘失败必须
    终止，否则继续调用会产生无法持久保存的结果和重复付费风险。
    """
    env = os.environ if environ is None else environ
    base_url = resolve_base_url(config, env)
    if execution_mode not in {"live", "mock"}:
        raise ValueError("execution_mode must be live or mock")
    if config.api_key_env and not env.get(config.api_key_env, "").strip():
        raise VerifierError(f"missing API key environment variable: {config.api_key_env}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    claims = list(claims)
    # 在任何模型请求前检查重复 ID，避免同一输入在一轮中被意外调用两次。
    seen: set[str] = set()
    for claim in claims:
        if any(not isinstance(claim.get(name), str) or not claim[name].strip()
               for name in ("claim_id", "topic", "claim_text")):
            raise ValueError("every Claim requires non-empty claim_id, topic, and claim_text")
        if claim["claim_id"] in seen:
            raise ValueError("input contains duplicate claim_id")
        seen.add(claim["claim_id"])
    saved_rows = list(_read_jsonl(output_path)) if output_path.exists() else []
    # 已存输出也必须通过同样校验，不能把损坏记录或 mock 记录当作真实成功。
    for row in saved_rows:
        if not isinstance(row.get("cache_key"), str) or len(row["cache_key"]) != 64:
            raise ValueError("cache contains an invalid key")
        parse_verifier_output(json.dumps({
            "label": row.get("label"), "confidence": row.get("raw_confidence"),
            "reason": row.get("reason"),
        }))
    cached = {str(row["cache_key"]) for row in saved_rows}
    if len(cached) != len(saved_rows):
        raise ValueError("cache contains duplicate keys")
    if any(row.get("execution_mode") != execution_mode for row in saved_rows):
        raise ValueError("mock and live records must use separate output files")
    limiter = RateLimiter(config.min_interval_seconds)
    stats = {"completed": 0, "cached": 0, "failed": 0, "transport_attempts": 0}
    for index, claim in enumerate(claims):
        if limit is not None and index >= limit:
            break
        key = cache_key(claim, config, base_url, execution_mode)
        if key in cached:
            stats["cached"] += 1
            continue
        started = time.monotonic()
        try:
            parsed, usage = call_provider(
                claim, config, environ=env, post_json=post_json, rate_limiter=limiter,
            )
        except Exception as exc:
            stats["transport_attempts"] += getattr(exc, "attempts", 0) or 0
            _append_jsonl(
                error_path,
                {
                    "claim_id": claim.get("claim_id"),
                    "verifier_id": config.verifier_id,
                    "model": config.model,
                    "error_type": type(exc).__name__,
                    "error": _safe_error(exc, config, env),
                    "attempts": getattr(exc, "attempts", None),
                    "usage": getattr(exc, "usage", None),
                    "execution_mode": execution_mode,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "cache_key": key,
                },
            )
            stats["failed"] += 1
            continue
        row = {
            "claim_id": claim["claim_id"],
            "split": claim.get("_split"),
            "verifier_id": config.verifier_id,
            "model": config.model,
            "provider": config.provider,
            "prompt_version": config.prompt_version,
            "execution_mode": execution_mode,
            **parsed,
            "usage": usage,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cache_key": key,
        }
        # 先成功落盘，再把键加入内存缓存；磁盘错误向外抛出，停止后续请求。
        _write_jsonl_atomic(output_path, [*saved_rows, row])
        saved_rows.append(row)
        cached.add(key)
        stats["completed"] += 1
        stats["transport_attempts"] += parsed["attempts"]
    return stats
