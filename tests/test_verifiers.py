"""Verifier 离线回归测试：用临时数据和假传输检查校验、保密、缓存及失败恢复。

测试中的密钥均为无效样例；CLI 的 live 模式也会替换传输层或直接拦截网络。
"""
from __future__ import annotations

import json
import io
from contextlib import redirect_stdout
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.verifiers import (
    PROMPT_VERSION,
    RetryableVerifierError,
    OUTPUT_SCHEMA,
    RateLimiter,
    VerifierConfig,
    VerifierError,
    call_provider,
    parse_verifier_output,
    run_claims,
)
from src.run_verifiers import load_api_environment, load_configs, main, select_claims

ROOT = Path(__file__).resolve().parents[1]


CLAIM = {
    "claim_id": "c-1",
    "topic": "Example topic",
    "claim_text": "A concise factual claim.",
    "gold_label": "supported",
}
ENV = {
    "TEST_BASE_URL": "https://example.invalid/v1",
    "TEST_API_KEY": "test-secret",
}
GOOD_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": json.dumps(
                    {"label": "supported", "confidence": 0.7, "reason": "符合已知事实。"},
                    ensure_ascii=False,
                )
            }
        }
    ],
    "usage": {"prompt_tokens": 23, "completion_tokens": 12},
}


def config(**overrides: object) -> VerifierConfig:
    """提供最小非付费配置，测试按需覆盖模型、重试上限或接口类型。"""
    values: dict[str, object] = {
        "verifier_id": "v1",
        "provider": "openai_compatible",
        "model": "test-model",
        "base_url_env": "TEST_BASE_URL",
        "api_key_env": "TEST_API_KEY",
        "max_attempts": 2,
    }
    values.update(overrides)
    return VerifierConfig.from_mapping(values)


class VerifierTests(unittest.TestCase):
    """验证底层模型调用行为，特别是失败时不泄露密钥、不损坏成功缓存。"""
    def test_output_validation_rejects_bad_label_and_confidence(self) -> None:
        with self.assertRaisesRegex(VerifierError, "invalid label"):
            parse_verifier_output('{"label":"maybe","confidence":0.5,"reason":"x"}')
        with self.assertRaisesRegex(VerifierError, "between 0 and 1"):
            parse_verifier_output('{"label":"supported","confidence":1.1,"reason":"x"}')

    def test_openai_compatible_request_uses_fixed_prompt_and_json_mode(self) -> None:
        captured: dict[str, object] = {}

        def fake_post(url: str, headers: object, payload: object, timeout: int) -> object:
            captured.update({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
            return GOOD_RESPONSE

        parsed, usage = call_provider(CLAIM, config(), environ=ENV, post_json=fake_post)
        payload = captured["payload"]
        self.assertEqual(captured["url"], "https://example.invalid/v1/chat/completions")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["temperature"], 0)
        prompt_text = json.dumps(payload["messages"], ensure_ascii=False)
        self.assertNotIn("gold_label", prompt_text)
        self.assertEqual(parsed["label"], "supported")
        self.assertEqual(usage, {"input_tokens": 23, "output_tokens": 12})

    def test_ollama_request_uses_frozen_local_settings(self) -> None:
        captured: dict[str, object] = {}

        def fake_post(url: str, headers: object, payload: object, timeout: int) -> object:
            captured.update({"url": url, "payload": payload})
            return {
                "message": {
                    "content": '{"label":"uncertain","confidence":0.4,"reason":"信息不足。"}'
                },
                "prompt_eval_count": 20,
                "eval_count": 9,
            }

        local = config(
            verifier_id="v3",
            provider="ollama",
            model="qwen-local",
            base_url_env="OLLAMA_BASE_URL",
            api_key_env=None,
            context_length=4096,
            no_think=True,
        )
        parsed, usage = call_provider(
            CLAIM,
            local,
            environ={"OLLAMA_BASE_URL": "http://127.0.0.1:11434"},
            post_json=fake_post,
        )
        self.assertEqual(captured["url"], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(captured["payload"]["options"], {
            "temperature": 0, "num_ctx": 4096, "num_predict": 256, "seed": 42, "num_gpu": 0,
        })
        self.assertEqual(captured["payload"]["format"], OUTPUT_SCHEMA)
        self.assertTrue(captured["payload"]["messages"][0]["content"].startswith("/no_think"))
        self.assertEqual(parsed["label"], "uncertain")
        self.assertEqual(usage, {"input_tokens": 20, "output_tokens": 9})

    def test_resume_skips_cached_claim_without_another_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            errors = Path(directory) / "errors.jsonl"
            calls = 0

            def fake_post(url: str, headers: object, payload: object, timeout: int) -> object:
                nonlocal calls
                calls += 1
                return GOOD_RESPONSE

            first = run_claims([CLAIM], config(), output, errors, environ=ENV, post_json=fake_post)
            second = run_claims([CLAIM], config(), output, errors, environ=ENV, post_json=fake_post)
            self.assertEqual(first, {"completed": 1, "cached": 0, "failed": 0, "transport_attempts": 1})
            self.assertEqual(second, {"completed": 0, "cached": 1, "failed": 0, "transport_attempts": 0})
            self.assertEqual(calls, 1)
            result = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertNotIn("gold_label", result)
            self.assertNotIn("test-secret", output.read_text(encoding="utf-8"))

    def test_transient_failure_retries_and_does_not_log_credentials(self) -> None:
        attempts = 0

        def fake_post(url: str, headers: object, payload: object, timeout: int) -> object:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RetryableVerifierError("temporary 503")
            return GOOD_RESPONSE

        with patch("src.verifiers.time.sleep"):
            parsed, _ = call_provider(CLAIM, config(), environ=ENV, post_json=fake_post)
        self.assertEqual(attempts, 2)
        self.assertEqual(parsed["label"], "supported")
        self.assertEqual(parsed["attempts"], 2)
        self.assertFalse(parsed["usage_complete"])

    def test_fatal_http_error_is_not_retried(self) -> None:
        from unittest.mock import Mock

        transport = Mock(side_effect=VerifierError("provider returned HTTP 401"))
        with self.assertRaises(VerifierError) as caught:
            call_provider(CLAIM, config(), environ=ENV, post_json=transport)
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(caught.exception.attempts, 1)

    def test_failed_claim_is_logged_without_key_and_can_resume(self) -> None:
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as directory:
            output, errors = Path(directory) / "out.jsonl", Path(directory) / "err.jsonl"
            transport = Mock(side_effect=RetryableVerifierError("temporary test-secret failure"))
            with patch("src.verifiers.time.sleep"):
                stats = run_claims([CLAIM], config(), output, errors, environ=ENV, post_json=transport)
            self.assertEqual(stats["failed"], 1)
            self.assertEqual(stats["transport_attempts"], 2)
            self.assertFalse(output.exists())
            self.assertNotIn("test-secret", errors.read_text(encoding="utf-8"))
            stats = run_claims([CLAIM], config(), output, errors, environ=ENV,
                               post_json=lambda *args: GOOD_RESPONSE)
            self.assertEqual(stats["completed"], 1)

    def test_model_change_invalidates_cache(self) -> None:
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as directory:
            output, errors = Path(directory) / "out.jsonl", Path(directory) / "err.jsonl"
            transport = Mock(return_value=GOOD_RESPONSE)
            run_claims([CLAIM], config(), output, errors, environ=ENV, post_json=transport)
            changed = run_claims([CLAIM], config(model="another-model"), output, errors,
                                 environ=ENV, post_json=transport)
            self.assertEqual(changed["completed"], 1)
            self.assertEqual(transport.call_count, 2)

    def test_mock_cache_cannot_be_used_as_real_judgment(self) -> None:
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as directory:
            output, errors = Path(directory) / "out.jsonl", Path(directory) / "err.jsonl"
            transport = Mock(return_value=GOOD_RESPONSE)
            run_claims([CLAIM], config(), output, errors, environ=ENV, post_json=transport,
                       execution_mode="mock")
            with self.assertRaisesRegex(ValueError, "separate output files"):
                run_claims([CLAIM], config(), output, errors, environ=ENV, post_json=transport)
            self.assertEqual(transport.call_count, 1)

    def test_duplicate_input_stops_before_any_requests(self) -> None:
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as directory:
            transport = Mock(return_value=GOOD_RESPONSE)
            with self.assertRaisesRegex(ValueError, "duplicate claim_id"):
                run_claims([CLAIM, CLAIM], config(), Path(directory) / "out.jsonl",
                           Path(directory) / "err.jsonl", environ=ENV, post_json=transport)
            transport.assert_not_called()

    def test_failed_checkpoint_write_keeps_previous_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output, errors = Path(directory) / "out.jsonl", Path(directory) / "err.jsonl"
            run_claims([CLAIM], config(), output, errors, environ=ENV,
                       post_json=lambda *args: GOOD_RESPONSE)
            previous = output.read_bytes()
            second = {**CLAIM, "claim_id": "c-2"}
            with patch("src.verifiers.os.replace", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    run_claims([second], config(), output, errors, environ=ENV,
                               post_json=lambda *args: GOOD_RESPONSE)
            self.assertEqual(output.read_bytes(), previous)

    def test_duplicate_json_keys_and_boolean_confidence_are_rejected(self) -> None:
        for content in (
            '{"label":"supported","label":"unsupported","confidence":0.3,"reason":"x"}',
            '{"label":"supported","confidence":true,"reason":"x"}',
            '{"label":[],"confidence":0.3,"reason":"x"}',
        ):
            with self.assertRaises(VerifierError):
                parse_verifier_output(content)

    def test_rate_limiter_spaces_requests(self) -> None:
        limiter = RateLimiter(1.0)
        with patch("src.verifiers.time.monotonic", side_effect=[10.0, 10.4, 11.0]):
            with patch("src.verifiers.time.sleep") as sleep:
                limiter.wait()
                limiter.wait()
        self.assertAlmostEqual(sleep.call_args.args[0], 0.6)


class VerifierCommandTests(unittest.TestCase):
    """验证命令行的数据隔离、私有配置、同版输出目录和全量门禁。"""
    def make_training_fixture(self, root: Path) -> None:
        path = root / "data" / "splits" / "train.jsonl"
        path.parent.mkdir(parents=True)
        rows = [{"claim_id": f"c-{index:02d}", "topic": f"T {index}",
                 "claim_text": f"Fact {index}", "gold_label": "unsupported"}
                for index in range(20)]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_pilot_selection_is_fixed_seed_and_contains_no_gold_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_training_fixture(root)
            first = select_claims(root, 10, 42)
            self.assertEqual(first, select_claims(root, 10, 42))
            self.assertEqual(len(first), 10)
            self.assertTrue(all("gold_label" not in row for row in first))

    def test_default_plan_and_mock_use_no_network_and_mock_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_training_fixture(root)
            # Offline modes must not even parse a private API file.
            (root / "config").mkdir()
            (root / "config" / "api.local.json").write_text("invalid private file", encoding="utf-8")
            legacy = root / "outputs" / "verifiers" / "mock" / "v1.jsonl"
            legacy.parent.mkdir(parents=True)
            legacy.write_text("legacy snapshot must remain untouched", encoding="utf-8")
            common = ["--root", str(root), "--config", str(ROOT / "config/verifiers.example.json")]
            with patch("src.verifiers.urlopen", side_effect=AssertionError("network forbidden")):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(common + ["--report", "plan.json"]), 0)
                    self.assertEqual(main(common + ["--mode", "mock", "--report", "first.json"]), 0)
                    self.assertEqual(main(common + ["--mode", "mock", "--report", "second.json"]), 0)
            plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
            first = json.loads((root / "first.json").read_text(encoding="utf-8"))
            second = json.loads((root / "second.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["external_api_requests"], 0)
            self.assertEqual(first["external_api_requests"], 0)
            self.assertTrue(all(row["stats"]["completed"] == 10 for row in first["verifiers"]))
            self.assertTrue(all(row["stats"]["cached"] == 10 for row in second["verifiers"]))
            self.assertTrue(all(row["stats"]["transport_attempts"] == 0 for row in second["verifiers"]))
            self.assertEqual(first["claim_ids"], second["claim_ids"])
            self.assertEqual(legacy.read_text(encoding="utf-8"),
                             "legacy snapshot must remain untouched")
            self.assertTrue(all(Path(row["output"]).parent.name == PROMPT_VERSION
                                for row in first["verifiers"]))

    def test_example_uses_correct_model_id_and_merges_global_defaults(self) -> None:
        _, configs = load_configs(ROOT / "config/verifiers.example.json")
        by_id = {config.verifier_id: config for config in configs}
        self.assertEqual(by_id["v2"].model, "gpt-6.1-sol")
        self.assertEqual(by_id["v2"].base_url_default, "https://api.sbbbbbbbbb.xyz/v1")
        self.assertEqual(by_id["v1"].min_interval_seconds, 1)
        self.assertEqual(by_id["v1"].request_options, {"thinking": {"type": "disabled"}})
        self.assertEqual(by_id["v3"].provider, "openai_compatible")
        self.assertEqual(by_id["v3"].model_env, "CLAUDE_RELAY_MODEL")

    def test_live_full_run_is_blocked_until_real_pilots_are_accepted(self) -> None:
        with patch("src.verifiers.urlopen", side_effect=AssertionError("network forbidden")):
            with self.assertRaisesRegex(ValueError, "accepted real pilots"):
                main(["--root", str(ROOT), "--mode", "live", "--full-run"])

    def test_local_api_file_overrides_nonempty_values_and_preserves_environment_fallback(self) -> None:
        _, configs = load_configs(ROOT / "config/verifiers.example.json")
        original = {"DEEPSEEK_API_KEY": "EXISTING-OFFLINE-FIXTURE"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "api.local.json"
            path.write_text(json.dumps({
                "DEEPSEEK_API_KEY": "", "GPT_RELAY_API_KEY": " LOCAL-OFFLINE-FIXTURE ",
                "GPT_RELAY_BASE_URL": "https://example.invalid/v1",
                "GPT_RELAY_MODEL": "fixture-model-name",
                "CLAUDE_RELAY_API_KEY": "CLAUDE-OFFLINE-FIXTURE",
                "CLAUDE_RELAY_MODEL": "claude-fixture-model",
                "CLAUDE_RELAY_BASE_URL": "https://claude.invalid/v1",
            }), encoding="utf-8-sig")
            loaded = load_api_environment(path, configs, original)
            self.assertEqual(loaded["DEEPSEEK_API_KEY"], original["DEEPSEEK_API_KEY"])
            self.assertEqual(loaded["GPT_RELAY_API_KEY"], "LOCAL-OFFLINE-FIXTURE")
            self.assertEqual(loaded["GPT_RELAY_MODEL"], "fixture-model-name")
            self.assertEqual(loaded["CLAUDE_RELAY_MODEL"], "claude-fixture-model")
            self.assertEqual(loaded["CLAUDE_RELAY_API_KEY"], "CLAUDE-OFFLINE-FIXTURE")
            self.assertNotIn("GPT_RELAY_API_KEY", original)

    def test_invalid_private_json_error_does_not_include_key_text(self) -> None:
        _, configs = load_configs(ROOT / "config/verifiers.example.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "api.local.json"
            path.write_text('{"GPT_RELAY_API_KEY":"PRIVATE-OFFLINE-FIXTURE', encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                load_api_environment(path, configs, {})
            self.assertNotIn("PRIVATE-OFFLINE-FIXTURE", str(caught.exception))

    def test_invalid_private_config_values_are_rejected_without_echoing_them(self) -> None:
        _, configs = load_configs(ROOT / "config/verifiers.example.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "api.local.json"
            for value in ({"GPT_RELAY_API_KEY": ["PRIVATE-OFFLINE-FIXTURE"]},
                          {"PRIVATE-OFFLINE-FIXTURE": "unknown field"}):
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError) as caught:
                    load_api_environment(path, configs, {})
                self.assertNotIn("PRIVATE-OFFLINE-FIXTURE", str(caught.exception))

    def test_live_cli_passes_private_keys_in_memory_without_writing_them_to_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_training_fixture(root)
            (root / "config").mkdir()
            (root / "config" / "api.local.json").write_text(json.dumps({
                "GPT_RELAY_BASE_URL": "https://example.invalid/v1",
                "GPT_RELAY_API_KEY": "PRIVATE-OFFLINE-FIXTURE",
                "GPT_RELAY_MODEL": "configured-offline-model",
            }), encoding="utf-8")
            output = io.StringIO()
            with patch("src.run_verifiers.run_claims", return_value={
                "completed": 10, "cached": 0, "failed": 0, "transport_attempts": 10,
            }) as runner:
                with patch("src.verifiers.urlopen", side_effect=AssertionError("network forbidden")):
                    with redirect_stdout(output):
                        status = main([
                            "--root", str(root), "--config", str(ROOT / "config/verifiers.example.json"),
                            "--mode", "live", "--verifier", "v2", "--report", "report.json",
                        ])
            self.assertEqual(status, 0)
            self.assertEqual(runner.call_args.kwargs["environ"]["GPT_RELAY_API_KEY"],
                             "PRIVATE-OFFLINE-FIXTURE")
            self.assertEqual(runner.call_args.args[1].model, "configured-offline-model")
            self.assertEqual(json.loads(output.getvalue())["verifiers"][0]["model"],
                             "configured-offline-model")
            self.assertNotIn("PRIVATE-OFFLINE-FIXTURE", output.getvalue())
            self.assertNotIn("PRIVATE-OFFLINE-FIXTURE", (root / "report.json").read_text(encoding="utf-8"))

    def test_claude_live_cli_uses_private_model_endpoint_and_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_training_fixture(root)
            (root / "config").mkdir()
            private = {
                "CLAUDE_RELAY_BASE_URL": "https://claude.invalid/v1",
                "CLAUDE_RELAY_MODEL": "claude-offline-model",
                "CLAUDE_RELAY_API_KEY": "PRIVATE-CLAUDE-OFFLINE-FIXTURE",
            }
            (root / "config" / "api.local.json").write_text(json.dumps(private), encoding="utf-8")
            output = io.StringIO()
            with patch("src.run_verifiers.run_claims", return_value={
                "completed": 10, "cached": 0, "failed": 0, "transport_attempts": 10,
            }) as runner, patch("src.verifiers.urlopen", side_effect=AssertionError("network forbidden")):
                with redirect_stdout(output):
                    status = main([
                        "--root", str(root), "--config", str(ROOT / "config/verifiers.example.json"),
                        "--mode", "live", "--verifier", "v3", "--report", "report.json",
                    ])
            self.assertEqual(status, 0)
            self.assertEqual(runner.call_args.args[1].model, private["CLAUDE_RELAY_MODEL"])
            self.assertEqual(runner.call_args.kwargs["environ"]["CLAUDE_RELAY_BASE_URL"],
                             private["CLAUDE_RELAY_BASE_URL"])
            self.assertEqual(runner.call_args.kwargs["environ"]["CLAUDE_RELAY_API_KEY"],
                             private["CLAUDE_RELAY_API_KEY"])
            self.assertNotIn(private["CLAUDE_RELAY_API_KEY"], output.getvalue())
            self.assertNotIn(private["CLAUDE_RELAY_API_KEY"],
                             (root / "report.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
