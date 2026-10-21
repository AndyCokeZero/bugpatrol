from __future__ import annotations

import io
import json
import os
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from bugpatrol.agents import (
    DEEPSEEK_ANTHROPIC_BASE_URL,
    DEEPSEEK_DEFAULT_MODEL,
    AgentInvocation,
    build_triage_agent_invocation,
    detect_sandbox_denial,
    is_bare_gateway_rejection,
    is_retryable_agent_failure,
    is_transient_agent_failure,
    load_agent_json,
    parse_claude_token_usage,
    probe_model_available,
)
from bugpatrol.config import load_project_config, parse_project_config


def _config_with_provider(provider: str, *, model: str = ""):
    config = load_project_config(Path("projects/example.toml"))
    return parse_project_config(
        {
            "github_repo": config.github_repo,
            "lark": {
                "chat_id": config.lark.chat_id,
                "app_id": config.lark.app_id,
                "app_secret_env": config.lark.app_secret_env,
                "bot_open_id": config.lark.bot_open_id,
            },
            "triage_agent": {
                "provider": provider,
                "model": model,
                "runner_labels": list(config.triage_agent.runner_labels),
            },
            "prd": {"root_wiki_node": config.prd.root_wiki_node},
            "intake": {"language": config.intake.language},
            "issue_field_names": dict(config.issue_field_names),
        }
    )


class LoadAgentJsonTest(unittest.TestCase):
    def test_parses_well_formed_json(self) -> None:
        self.assertEqual(load_agent_json('{"a": 1, "b": "x"}'), {"a": 1, "b": "x"})

    def test_repairs_unescaped_inner_quotes(self) -> None:
        # Real failure mode from fived #4129: the agent quoted a code comment
        # verbatim inside a string value without escaping it.
        raw = (
            '{\n'
            '  "summary": "ok",\n'
            '  "blame_suggestion": "注释也注明 "a lone Leave button hugs its label"，'
            '设计意图是自适应。",\n'
            '  "assignee": ""\n'
            '}'
        )
        with self.assertRaises(json.JSONDecodeError):
            json.loads(raw)
        result = load_agent_json(raw)
        self.assertEqual(
            result["blame_suggestion"],
            '注释也注明 "a lone Leave button hugs its label"，设计意图是自适应。',
        )
        self.assertEqual(result["assignee"], "")

    def test_repairs_multiple_values_with_inner_quotes(self) -> None:
        raw = (
            '{"a": "say "hi" now", "b": "and "bye" too"}'
        )
        result = load_agent_json(raw)
        self.assertEqual(result["a"], 'say "hi" now')
        self.assertEqual(result["b"], 'and "bye" too')

    def test_preserves_already_escaped_quotes(self) -> None:
        raw = '{"a": "he said \\"hi\\"", "b": 2}'
        self.assertEqual(load_agent_json(raw), {"a": 'he said "hi"', "b": 2})

    def test_reraises_original_error_when_unrepairable(self) -> None:
        raw = '{"a": 1 "b": 2}'  # missing comma, no inner-quote issue
        with self.assertRaises(json.JSONDecodeError):
            load_agent_json(raw)


class AgentsTest(unittest.TestCase):
    def test_builds_subscription_friendly_codex_exec_command(self) -> None:
        config = load_project_config(Path("projects/example.toml"))
        invocation = build_triage_agent_invocation(
            config,
            issue_number=123,
            prompt_path=Path("prompts/triage.zh.md"),
            schema_path=Path("triage.schema.json"),
            output_path=Path("triage-output.json"),
            context_path=Path("triage-context.md"),
        )

        self.assertEqual(invocation.provider, "codex")
        self.assertEqual(invocation.command[:2], ["codex", "exec"])
        self.assertIn("--output-schema", invocation.command)
        self.assertNotIn("OPENAI_API_KEY", " ".join(invocation.command))
        self.assertIn("Issue: #123", invocation.command[-1])
        self.assertIn("triage-context.md", invocation.command[-1])

    def test_supports_claude_provider_boundary(self) -> None:
        config = load_project_config(Path("projects/example.toml"))
        claude_config = parse_project_config(
            {
                "github_repo": config.github_repo,
                "lark": {
                    "chat_id": config.lark.chat_id,
                    "app_id": config.lark.app_id,
                    "app_secret_env": config.lark.app_secret_env,
                    "bot_open_id": config.lark.bot_open_id,
                },
                "triage_agent": {
                    "provider": "claude",
                    "runner_labels": list(config.triage_agent.runner_labels),
                },
                "prd": {"root_wiki_node": config.prd.root_wiki_node},
                "intake": {"language": config.intake.language},
                "issue_field_names": dict(config.issue_field_names),
            }
        )

        invocation = build_triage_agent_invocation(
            claude_config,
            issue_number=123,
            prompt_path=Path("prompts/triage.zh.md"),
            schema_path=Path("triage.schema.json"),
            output_path=Path("triage-output.json"),
            context_path=None,
            workspace_dirs=(Path("/runner/bugpatrol/prompts"), Path("/runner/out")),
        )

        self.assertEqual(invocation.provider, "claude")
        self.assertEqual(invocation.command[0], "claude")
        # Non-interactive `claude -p` cannot answer approval prompts, so the
        # runner bypasses them wholesale (Bash dedup/git + output write).
        self.assertIn("--dangerously-skip-permissions", invocation.command)
        self.assertNotIn("--permission-mode", invocation.command)
        # Each workspace dir outside the checkout is re-admitted via --add-dir.
        self.assertEqual(
            [
                invocation.command[i + 1]
                for i, token in enumerate(invocation.command)
                if token == "--add-dir"
            ],
            ["/runner/bugpatrol/prompts", "/runner/out"],
        )


    def test_deepseek_provider_uses_claude_cli_with_endpoint_env(self) -> None:
        config = _config_with_provider("deepseek")

        with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}):
            invocation = build_triage_agent_invocation(
                config,
                issue_number=7,
                prompt_path=Path("prompts/triage.zh.md"),
                schema_path=Path("triage.schema.json"),
                output_path=Path("triage-output.json"),
                context_path=None,
            )

        self.assertEqual(invocation.provider, "deepseek")
        self.assertEqual(invocation.command[0], "claude")
        self.assertIn("--dangerously-skip-permissions", invocation.command)
        self.assertEqual(
            invocation.command[invocation.command.index("--model") + 1],
            DEEPSEEK_DEFAULT_MODEL,
        )
        self.assertEqual(invocation.env["ANTHROPIC_BASE_URL"], DEEPSEEK_ANTHROPIC_BASE_URL)
        self.assertEqual(invocation.env["ANTHROPIC_API_KEY"], "sk-test")
        self.assertEqual(invocation.env["ANTHROPIC_AUTH_TOKEN"], "sk-test")
        # The key must never leak into argv (visible in ps / run logs).
        self.assertNotIn("sk-test", " ".join(invocation.command))

    def test_deepseek_provider_honors_configured_model(self) -> None:
        config = _config_with_provider("deepseek", model="deepseek-v4-pro[1m]")

        with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}):
            invocation = build_triage_agent_invocation(
                config,
                issue_number=7,
                prompt_path=Path("prompts/triage.zh.md"),
                schema_path=Path("triage.schema.json"),
                output_path=Path("triage-output.json"),
                context_path=None,
            )

        self.assertEqual(
            invocation.command[invocation.command.index("--model") + 1],
            "deepseek-v4-pro[1m]",
        )

    def test_deepseek_provider_requires_api_key(self) -> None:
        config = _config_with_provider("deepseek")

        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                build_triage_agent_invocation(
                    config,
                    issue_number=7,
                    prompt_path=Path("prompts/triage.zh.md"),
                    schema_path=Path("triage.schema.json"),
                    output_path=Path("triage-output.json"),
                    context_path=None,
                )

    def test_non_deepseek_providers_have_empty_env(self) -> None:
        config = load_project_config(Path("projects/example.toml"))
        invocation = build_triage_agent_invocation(
            config,
            issue_number=1,
            prompt_path=Path("prompts/triage.zh.md"),
            schema_path=Path("triage.schema.json"),
            output_path=Path("triage-output.json"),
            context_path=None,
        )

        self.assertEqual(invocation.env, {})


class ParseClaudeTokenUsageTest(unittest.TestCase):
    def test_splits_real_input_from_cache(self) -> None:
        # Shape emitted by the deepseek `result` event in stream-json output.
        stdout = json.dumps(
            {
                "type": "result",
                "usage": {
                    "input_tokens": 36705,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 604160,
                    "output_tokens": 11525,
                },
            }
        )
        self.assertEqual(parse_claude_token_usage(stdout), (36705, 604160, 11525))

    def test_returns_zeros_when_unparseable(self) -> None:
        self.assertEqual(parse_claude_token_usage(""), (0, 0, 0))
        self.assertEqual(parse_claude_token_usage("not json"), (0, 0, 0))


class DetectSandboxDenialTest(unittest.TestCase):
    def _tool_result_event(self, *, text: str, is_error: bool) -> str:
        return json.dumps(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "is_error": is_error, "content": text}
                    ],
                },
            }
        )

    def test_flags_bash_command_approval_denial(self) -> None:
        stdout = "\n".join(
            [
                self._tool_result_event(text="This command requires approval", is_error=True),
                json.dumps({"type": "result", "usage": {}}),
            ]
        )
        self.assertEqual(detect_sandbox_denial(stdout), "This command requires approval")

    def test_flags_out_of_workspace_read_denial(self) -> None:
        stdout = self._tool_result_event(
            text="Error: this tool may only search files in the allowed working directories",
            is_error=True,
        )
        self.assertIn("allowed working directories", detect_sandbox_denial(stdout) or "")

    def test_ignores_non_error_and_unrelated_errors(self) -> None:
        stdout = "\n".join(
            [
                # Ordinary error output must not trip the guard.
                self._tool_result_event(text="No such file or directory", is_error=True),
                # The phrase quoted in a *successful* result (e.g. issue text) is ignored.
                self._tool_result_event(text="the PR requires approval", is_error=False),
                json.dumps({"type": "result", "usage": {}}),
            ]
        )
        self.assertIsNone(detect_sandbox_denial(stdout))

    def test_returns_none_for_empty(self) -> None:
        self.assertIsNone(detect_sandbox_denial(""))


def _result_event(message: str) -> str:
    return json.dumps(
        {"type": "result", "subtype": "success", "is_error": True, "result": message}
    )


class IsTransientAgentFailureTest(unittest.TestCase):
    """#6246: a dropped stream killed a 7-minute run that a retry would finish."""

    def test_flags_a_stream_cut_mid_response(self) -> None:
        stdout = _result_event(
            "API Error: Connection closed mid-response. The response above may be incomplete."
        )
        self.assertTrue(is_transient_agent_failure(stdout))

    def test_flags_provider_5xx_and_socket_failures(self) -> None:
        for message in (
            "API Error: 503 Service is too busy. We advise users to temporarily switch to another model.",
            "API Error: 502 Bad Gateway",
            "API Error: terminated",
            "API Error: The operation timed out.",
            "API Error: socket hang up",
        ):
            with self.subTest(message=message):
                self.assertTrue(is_transient_agent_failure(_result_event(message)))

    def test_keeps_credential_and_balance_errors_fatal(self) -> None:
        # Retrying these just triples the cost of a failure a human must fix:
        # the account state is identical on every attempt.
        # The bare 400 is here because the *log alone* cannot settle it — no
        # diagnostic in the body. `is_retryable_agent_failure` adds a model
        # probe for exactly that shape; see IsRetryableAgentFailureTest.
        for message in (
            "Failed to authenticate. API Error: 401 Insufficient balance. Manage your billing here: https://opencode.ai",
            'API Error: 402 {"error":{"message":"Insufficient Balance","type":"unknown_error"}}',
            'API Error: 400 {"model":"deepseek-v4-flash"}',
            "Failed to authenticate. API Error: 401 {\"type\":\"error\",\"error\":{\"type\":\"authentication_error\"}}",
        ):
            with self.subTest(message=message):
                self.assertFalse(is_transient_agent_failure(_result_event(message)))

    def test_ignores_transient_looking_text_outside_the_error_event(self) -> None:
        # An issue body quoting the string must not turn a real failure into a
        # retry loop: only the agent's own fatal error event is classified.
        stdout = "\n".join(
            [
                json.dumps({"type": "assistant", "message": {"content": "the log said Connection closed mid-response"}}),
                _result_event("API Error: 402 Insufficient Balance"),
            ]
        )
        self.assertFalse(is_transient_agent_failure(stdout))

    def test_is_false_without_any_error_event(self) -> None:
        self.assertFalse(is_transient_agent_failure(""))
        self.assertFalse(is_transient_agent_failure(None))
        self.assertFalse(is_transient_agent_failure(json.dumps({"type": "result", "is_error": False})))


def _deepseek_invocation(model: str = DEEPSEEK_DEFAULT_MODEL) -> AgentInvocation:
    return AgentInvocation(
        provider="deepseek",
        command=[],
        env={
            "ANTHROPIC_BASE_URL": DEEPSEEK_ANTHROPIC_BASE_URL,
            "ANTHROPIC_API_KEY": "test-key",
            "ANTHROPIC_AUTH_TOKEN": "test-key",
        },
        model=model,
    )


class BareGatewayRejectionTest(unittest.TestCase):
    """The gateway's `{"model": ...}` 400 carries no diagnosis at all."""

    def test_matches_a_body_with_only_the_model(self) -> None:
        self.assertTrue(
            is_bare_gateway_rejection('API Error: 400 {"model":"deepseek-v4.1-flash"}')
        )

    def test_does_not_match_a_structured_error_body(self) -> None:
        # The gateway *does* diagnose real refusals; those must stay fatal and
        # must not be mistaken for the undiagnosable shape.
        for message in (
            'API Error: 400 {"type":"error","error":{"type":"api_error","message":"Upstream request failed: Model is unavailable."}}',
            'API Error: 400 {"type":"error","error":{"type":"MissingSessionID","message":"Request is missing x-opencode-session"}}',
            'API Error: 402 {"error":{"message":"Insufficient Balance","type":"unknown_error"}}',
            "Failed to authenticate. API Error: 401 Insufficient balance.",
        ):
            with self.subTest(message=message):
                self.assertFalse(is_bare_gateway_rejection(message))

    def test_does_not_match_other_statuses(self) -> None:
        self.assertFalse(is_bare_gateway_rejection('API Error: 503 {"model":"x"}'))


class IsRetryableAgentFailureTest(unittest.TestCase):
    """A bare 400 is retried only when the model still answers a probe."""

    def test_retries_a_bare_400_when_the_model_still_answers(self) -> None:
        with mock.patch(
            "bugpatrol.agents.probe_model_available", return_value=True
        ) as probe:
            self.assertTrue(
                is_retryable_agent_failure(
                    _result_event('API Error: 400 {"model":"deepseek-v4.1-flash"}'),
                    _deepseek_invocation(),
                )
            )
        probe.assert_called_once_with(_deepseek_invocation())

    def test_keeps_a_bare_400_fatal_when_the_model_is_gone(self) -> None:
        with mock.patch("bugpatrol.agents.probe_model_available", return_value=False):
            self.assertFalse(
                is_retryable_agent_failure(
                    _result_event('API Error: 400 {"model":"deepseek-v4-flash"}'),
                    _deepseek_invocation("deepseek-v4-flash"),
                )
            )

    def test_network_failures_skip_the_probe(self) -> None:
        # Already a known-transient shape: no need to spend a request.
        with mock.patch("bugpatrol.agents.probe_model_available") as probe:
            for message in (
                "API Error: 503 Service is too busy.",
                "API Error: socket hang up",
            ):
                with self.subTest(message=message):
                    self.assertTrue(
                        is_retryable_agent_failure(
                            _result_event(message), _deepseek_invocation()
                        )
                    )
        probe.assert_not_called()

    def test_credential_errors_never_probe(self) -> None:
        with mock.patch("bugpatrol.agents.probe_model_available") as probe:
            for message in (
                'API Error: 402 {"error":{"message":"Insufficient Balance","type":"unknown_error"}}',
                "Failed to authenticate. API Error: 401 Insufficient balance.",
            ):
                with self.subTest(message=message):
                    self.assertFalse(
                        is_retryable_agent_failure(
                            _result_event(message), _deepseek_invocation()
                        )
                    )
        probe.assert_not_called()


class ProbeModelAvailableTest(unittest.TestCase):
    def _response(self, status: int = 200) -> mock.MagicMock:
        response = mock.MagicMock()
        response.status = status
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        return response

    def test_sends_one_token_to_the_agent_endpoint(self) -> None:
        with mock.patch(
            "urllib.request.urlopen", return_value=self._response()
        ) as urlopen:
            self.assertTrue(probe_model_available(_deepseek_invocation()))
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, DEEPSEEK_ANTHROPIC_BASE_URL + "/v1/messages")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], DEEPSEEK_DEFAULT_MODEL)
        self.assertEqual(payload["max_tokens"], 1)
        headers = {key.lower(): value for key, value in request.header_items()}
        self.assertEqual(headers["x-api-key"], "test-key")
        # The OpenCode gateway cannot route a request without a session id, so
        # omitting it would make every probe fail and nothing ever retry.
        self.assertIn("x-opencode-session", headers)

    def test_http_error_is_not_available(self) -> None:
        error = urllib.error.HTTPError(
            "https://example.invalid", 400, "bad request", {}, io.BytesIO(b"{}")
        )
        with mock.patch("urllib.request.urlopen", side_effect=error):
            self.assertFalse(probe_model_available(_deepseek_invocation()))

    def test_unreachable_endpoint_is_not_available(self) -> None:
        with mock.patch(
            "urllib.request.urlopen", side_effect=urllib.error.URLError("no route")
        ):
            self.assertFalse(probe_model_available(_deepseek_invocation()))

    def test_missing_credentials_stay_fatal_without_a_request(self) -> None:
        invocation = AgentInvocation(
            provider="deepseek",
            command=[],
            env={"ANTHROPIC_BASE_URL": DEEPSEEK_ANTHROPIC_BASE_URL},
            model=DEEPSEEK_DEFAULT_MODEL,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("urllib.request.urlopen") as urlopen:
                self.assertFalse(probe_model_available(invocation))
        urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
