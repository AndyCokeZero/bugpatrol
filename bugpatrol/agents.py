"""Triage agent invocation helpers for trusted self-hosted runners."""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from bugpatrol.config import ProjectConfig

# DeepSeek via the OpenCode Go subscription gateway (Anthropic-compatible
# endpoint). Host root WITHOUT /v1: the claude CLI appends /v1/messages itself.
DEEPSEEK_ANTHROPIC_BASE_URL = "https://opencode.ai/zen/go"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4.1-flash"

# Fatal agent errors a fresh attempt usually clears: an in-flight response
# dropped mid-stream, a provider 5xx, a reset socket. These are stochastic —
# observed fleet-wide (`Connection closed mid-response` on minici32g/16g/
# carbonci, `503 Service is too busy`, `The operation timed out`) at roughly
# 1% of runs, and never on the same run twice.
_TRANSIENT_AGENT_FAILURE_MARKERS = (
    "Connection closed mid-response",
    "API Error: terminated",
    "The operation timed out",
    "overloaded_error",
    "socket hang up",
    "ECONNRESET",
    "ETIMEDOUT",
    "fetch failed",
)
# `API Error: 503 Service is too busy...`; the code is the whole signal, and
# spelling the messages out here would miss the next 5xx shape.
_TRANSIENT_AGENT_FAILURE_RE = re.compile(r"API Error: 5\d\d")

# A gateway rejection whose body carries no diagnostic at all — just the echoed
# model name, e.g. `API Error: 400 {"model":"deepseek-v4.1-flash"}`. Real
# rejections (a retired model, a disabled model, a bad credential, a missing
# session) come back as a structured `{"type":"error","error":{...}}` body, so
# this shape says nothing about *why* the request was refused. It kills ~4% of
# triage runs (chronic since 2026-09-11) and every observed instance cleared on
# a re-run of the identical context, which is why it is probed, not trusted.
_BARE_GATEWAY_REJECTION_RE = re.compile(r"^API Error: 400\s+(\{.*\})\s*$", re.DOTALL)

# The claude CLI's endpoint when the provider config does not override it.
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_API_VERSION = "2023-06-01"
MODEL_PROBE_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class AgentInvocation:
    provider: str
    command: list[str]
    # Extra environment for the agent process (e.g. third-party endpoint
    # overrides). Merged over os.environ at execution time; never printed.
    env: dict[str, str] = field(default_factory=dict)
    # Model name actually used, for run-completion reporting.
    model: str = ""


def parse_claude_token_usage(stdout: str) -> tuple[int, int, int]:
    """Best-effort (input, cached_input, output) token counts from `claude -p`.

    ``input`` is the freshly-processed prompt tokens; ``cached_input`` is the
    cache-creation + cache-read tokens (cheap, reported separately so the real
    cost is legible). Returns (0, 0, 0) when the stdout can't be parsed (e.g.
    codex, or an older CLI).
    """
    data: object = None
    text = stdout.strip()
    if text:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            for line in reversed(text.splitlines()):
                candidate = line.strip()
                if candidate.startswith("{") and candidate.endswith("}"):
                    try:
                        data = json.loads(candidate)
                        break
                    except json.JSONDecodeError:
                        continue
    if not isinstance(data, dict):
        return (0, 0, 0)
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return (0, 0, 0)

    def _int(key: str) -> int:
        value = usage.get(key)
        return value if isinstance(value, int) else 0

    input_tokens = _int("input_tokens")
    cached_input = _int("cache_creation_input_tokens") + _int("cache_read_input_tokens")
    return (input_tokens, cached_input, _int("output_tokens"))


def load_agent_json(text: str) -> object:
    """Parse an agent's JSON output, tolerating unescaped inner double-quotes.

    Models occasionally quote source verbatim inside a string value without
    escaping it (e.g. ``"...代码注释也注明 "a lone Leave button hugs its label"..."``),
    which breaks strict ``json.loads`` with ``Expecting ',' delimiter``. This
    is deterministic for a given issue, so retrying the agent never recovers.

    Strict parsing is tried first. Only on failure do we attempt a single,
    bounded repair pass that escapes stray inner quotes, and we accept the
    result *only if it re-parses* — otherwise the original decode error is
    re-raised so the failure still surfaces honestly.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as original:
        repaired = _escape_unescaped_inner_quotes(text)
        if repaired is None or repaired == text:
            raise
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            raise original from None


def _escape_unescaped_inner_quotes(text: str) -> str | None:
    """Escape ``"`` characters that appear inside JSON string values.

    A quote inside a string is treated as the terminator only when the next
    non-whitespace character is a structural JSON token (``,`` ``}`` ``]``
    ``:``) or end-of-input; otherwise it is a stray inner quote and gets
    escaped. This recovers pretty-printed agent output where a value contains
    an unescaped quote. Returns ``None`` when nothing was changed.
    """
    out: list[str] = []
    in_string = False
    changed = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue
        if ch == "\\":
            out.append(ch)
            if i + 1 < n:
                out.append(text[i + 1])
                i += 2
            else:
                i += 1
            continue
        if ch == '"':
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j >= n or text[j] in ",}]:":
                out.append(ch)
                in_string = False
            else:
                out.append('\\"')
                changed = True
            i += 1
            continue
        out.append(ch)
        i += 1
    if not changed:
        return None
    return "".join(out)


# Stable substrings Claude Code emits in a tool_result when it refuses a Bash
# command or a file read that falls outside the sandbox. A denied run silently
# yields a degraded triage (no CODEOWNERS/repo/git/dup-search) while still
# exiting 0, so the runner treats any of these as a hard failure — this catches
# regressions the invocation-flag unit test can't (CLI renames a flag, a wrong
# --add-dir path, a missing cwd), independent of whether the model self-reports.
_SANDBOX_DENIAL_MARKERS = (
    "requires approval",
    "allowed working directories",
)


def _tool_result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(parts)
    return ""


def detect_sandbox_denial(stdout: str) -> str | None:
    """Return the first sandbox/permission-denial message in a stream-json turn
    log, or None. Only ``tool_result`` blocks flagged ``is_error`` are inspected,
    so ordinary error output (and issue text that merely quotes these phrases)
    never trips the guard.
    """
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = event.get("message") if isinstance(event, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            if not block.get("is_error"):
                continue
            text = _tool_result_text(block.get("content"))
            if any(marker in text for marker in _SANDBOX_DENIAL_MARKERS):
                return text.strip()
    return None


def agent_error_from_stdout(stdout: str | None) -> str:
    """Return the agent's own fatal error message from ``claude -p`` stdout.

    Some fatal failures (provider API errors like 402 "Insufficient Balance",
    401 auth, 5xx, or a model refusal) are reported by the CLI on **stdout** as
    a ``result`` event with ``is_error: true`` while ``stderr`` stays empty.
    Without surfacing this the failure comment only says "exited with code 1",
    hiding the actionable cause. Returns the last such ``result`` string, or ""
    when none is found.
    """
    if not stdout:
        return ""
    for raw in reversed(stdout.splitlines()):
        line = raw.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "result" and event.get("is_error"):
            result = event.get("result")
            if isinstance(result, str) and result.strip():
                return result.strip()
    return ""


def is_transient_agent_failure(stdout: str | None) -> bool:
    """True when the agent died of an API failure a fresh attempt would clear.

    Only the agent's own fatal error message is classified — never the
    transcript around it — so an issue body that happens to quote one of these
    strings cannot turn a real failure into a retry loop.

    Deliberately an allowlist. Credential and account errors (401 auth, 402
    "Insufficient Balance") are identical on every attempt: retrying them
    triples the cost of a failure that a human has to fix anyway, so anything
    unrecognized stays fatal.

    A bare gateway 400 is the one rejection this cannot decide from the log
    alone, so callers that can afford a probe should use
    `is_retryable_agent_failure` instead — it keeps this allowlist and adds a
    one-token model probe for that shape.
    """
    error = agent_error_from_stdout(stdout)
    if not error:
        return False
    if any(marker in error for marker in _TRANSIENT_AGENT_FAILURE_MARKERS):
        return True
    return _TRANSIENT_AGENT_FAILURE_RE.search(error) is not None


def is_bare_gateway_rejection(error: str) -> bool:
    """True for a 400 whose body is only the echoed model name.

    That body carries no error type and no message, so it identifies neither a
    permanent refusal nor a transient one. A structured body
    (`{"type":"error","error":{...}}`) is a real diagnosis and is not matched.
    """
    match = _BARE_GATEWAY_REJECTION_RE.match(error.strip())
    if match is None:
        return False
    try:
        body = json.loads(match.group(1))
    except json.JSONDecodeError:
        return False
    return isinstance(body, dict) and "model" in body and "error" not in body


def probe_model_available(
    invocation: AgentInvocation,
    *,
    timeout_seconds: float = MODEL_PROBE_TIMEOUT_SECONDS,
) -> bool:
    """Send one minimal request to the agent's own endpoint/model.

    True only when it answers 200. A retired or disabled model answers with a
    structured 4xx on every attempt, so this separates that permanent case from
    a gateway hiccup without retrying the expensive agent run to find out.

    Endpoint and credentials come from the environment the agent process
    actually ran with (`{**os.environ, **invocation.env}`), so the probe
    authenticates exactly as the agent did.
    """
    env = {**os.environ, **invocation.env}
    base_url = env.get("ANTHROPIC_BASE_URL") or DEFAULT_ANTHROPIC_BASE_URL
    api_key = env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN")
    model = invocation.model
    if not api_key or not model:
        # Both are required to run the agent at all, so reaching here means the
        # invocation is inconsistent. Stay fatal (the pre-probe behaviour)
        # rather than retrying blind, but say so instead of failing silently.
        print(
            "warning: cannot probe the model endpoint "
            f"(api_key={'set' if api_key else 'missing'}, model={model or 'missing'}); "
            "treating the agent failure as fatal",
            file=sys.stderr,
        )
        return False
    headers = {
        "content-type": "application/json",
        "anthropic-version": ANTHROPIC_API_VERSION,
        "x-api-key": api_key,
        "authorization": f"Bearer {api_key}",
        "user-agent": "bugpatrol-model-probe/1.0",
    }
    # The OpenCode gateway cannot route a request without a session id.
    if "opencode.ai" in base_url:
        headers["x-opencode-session"] = str(uuid.uuid4())
    body = json.dumps(
        {
            "model": model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }
    ).encode()
    url = base_url.rstrip("/") + "/v1/messages"
    request = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.status == 200
    except urllib.error.HTTPError as error:
        print(
            f"warning: model probe for {model} answered HTTP {error.code}; "
            "treating the agent failure as fatal",
            file=sys.stderr,
        )
        return False
    except OSError as error:
        # Unreachable/timed-out probe: the run already failed, and retrying a
        # network that just refused a one-token request is not worth budget.
        print(
            f"warning: model probe for {model} could not reach {url}: {error}; "
            "treating the agent failure as fatal",
            file=sys.stderr,
        )
        return False


def is_retryable_agent_failure(stdout: str | None, invocation: AgentInvocation) -> bool:
    """True when a fresh attempt is likely to clear the agent's failure.

    `is_transient_agent_failure` decides from the error text alone. This adds
    the one shape that text cannot settle: a bare gateway 400 is retried only
    when the same model still answers a one-token probe. A retired model stays
    fatal; the ~4% of runs killed by a gateway hiccup get the retry they need.
    """
    if is_transient_agent_failure(stdout):
        return True
    error = agent_error_from_stdout(stdout)
    if not error or not is_bare_gateway_rejection(error):
        return False
    return probe_model_available(invocation)


def build_triage_agent_invocation(
    config: ProjectConfig,
    *,
    issue_number: int,
    prompt_path: Path,
    schema_path: Path,
    output_path: Path,
    context_path: Path | None = None,
    workspace_dirs: Sequence[Path] = (),
) -> AgentInvocation:
    provider = config.triage_agent.provider
    env: dict[str, str] = {}
    if provider == "codex":
        command = _build_codex_command(
            config,
            issue_number=issue_number,
            prompt_path=prompt_path,
            schema_path=schema_path,
            output_path=output_path,
            context_path=context_path,
        )
    elif provider == "claude":
        command = _build_claude_command(
            config,
            issue_number=issue_number,
            prompt_path=prompt_path,
            schema_path=schema_path,
            output_path=output_path,
            context_path=context_path,
            workspace_dirs=workspace_dirs,
        )
    elif provider == "deepseek":
        # DeepSeek's Anthropic-compatible endpoint driven through the claude
        # CLI (codex+DeepSeek is broken since codex 0.128 dropped wire_api=chat).
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError("deepseek provider requires DEEPSEEK_API_KEY in the environment")
        command = _build_claude_command(
            config,
            issue_number=issue_number,
            prompt_path=prompt_path,
            schema_path=schema_path,
            output_path=output_path,
            context_path=context_path,
            workspace_dirs=workspace_dirs,
            default_model=DEEPSEEK_DEFAULT_MODEL,
        )
        env = {
            "ANTHROPIC_BASE_URL": DEEPSEEK_ANTHROPIC_BASE_URL,
            "ANTHROPIC_API_KEY": api_key,
            "ANTHROPIC_AUTH_TOKEN": api_key,
        }
    else:
        raise ValueError(f"unsupported triage agent provider: {provider}")
    model = config.triage_agent.model
    if provider == "deepseek" and not model:
        model = DEEPSEEK_DEFAULT_MODEL
    return AgentInvocation(provider=provider, command=command, env=env, model=model)


def _common_prompt(
    config: ProjectConfig,
    *,
    issue_number: int,
    prompt_path: Path,
    context_path: Path | None,
) -> str:
    lines = [
        f"Use prompt file: {prompt_path}",
        f"Project: {config.project}",
        f"Repository: {config.github_repo}",
        f"Issue: #{issue_number}",
    ]
    if context_path is not None:
        lines.append(f"Use triage context file: {context_path}")
    lines.append("Return only JSON matching the provided schema.")
    return "\n".join(lines)


def _build_codex_command(
    config: ProjectConfig,
    *,
    issue_number: int,
    prompt_path: Path,
    schema_path: Path,
    output_path: Path,
    context_path: Path | None,
) -> list[str]:
    command = [
        "codex",
        "exec",
        "--output-schema",
        str(schema_path),
        "-o",
        str(output_path),
    ]
    if config.triage_agent.model:
        command.extend(["--model", config.triage_agent.model])
    if config.triage_agent.effort:
        command.extend(["--effort", config.triage_agent.effort])
    command.append(
        _common_prompt(
            config,
            issue_number=issue_number,
            prompt_path=prompt_path,
            context_path=context_path,
        )
    )
    return command


def _build_claude_command(
    config: ProjectConfig,
    *,
    issue_number: int,
    prompt_path: Path,
    schema_path: Path,
    output_path: Path,
    context_path: Path | None,
    workspace_dirs: Sequence[Path] = (),
    default_model: str = "",
) -> list[str]:
    prompt = "\n".join(
        [
            _common_prompt(
                config,
                issue_number=issue_number,
                prompt_path=prompt_path,
                context_path=context_path,
            ),
            f"Schema file: {schema_path}",
            f"Write final JSON to: {output_path}",
        ]
    )
    return _claude_task_command(prompt, workspace_dirs=workspace_dirs, model=config.triage_agent.model or default_model)


def _claude_task_command(
    prompt: str,
    *,
    workspace_dirs: Sequence[Path] = (),
    model: str = "",
) -> list[str]:
    # The agent runs with cwd set to the repo checkout (main or a branch
    # worktree), so CODEOWNERS, source, and git history/blame of the right
    # branch are naturally in scope. `--dangerously-skip-permissions` is
    # required because non-interactive `claude -p` cannot answer approval
    # prompts: without it every Bash call (gh dedup search, git log) and the
    # output-JSON write is auto-denied. Runs happen on trusted runners.
    # `--add-dir` re-admits the runner-side workspace (prompt/context/schema/
    # output) that lives outside the checkout. stream-json (+ --verbose, which
    # it requires under -p) emits the full turn-by-turn conversation so the
    # runner can persist a turn log and parse token usage from the result event.
    command = [
        "claude",
        "-p",
        prompt,
        "--dangerously-skip-permissions",
        "--output-format",
        "stream-json",
        "--verbose",
    ]
    for workspace in workspace_dirs:
        command.extend(["--add-dir", str(workspace)])
    if model:
        command.extend(["--model", model])
    return command


def build_fix_agent_invocation(
    config: ProjectConfig,
    *,
    issue_number: int,
    prompt_path: Path,
    schema_path: Path,
    output_path: Path,
    context_path: Path,
    workspace_dirs: Sequence[Path] = (),
) -> AgentInvocation:
    """Build the agent command for an auto-fix run.

    Unlike triage, the fix agent edits the working tree (cwd = the fix branch
    worktree) and then writes a JSON summary. Only the claude/deepseek path is
    supported: fix relies on the Claude CLI's in-place file editing.
    """
    provider = config.fix_agent.provider
    if provider not in ("claude", "deepseek"):
        raise ValueError(
            f"fix runner requires a claude or deepseek provider, got: {provider}"
        )
    env: dict[str, str] = {}
    default_model = ""
    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError("deepseek provider requires DEEPSEEK_API_KEY in the environment")
        env = {
            "ANTHROPIC_BASE_URL": DEEPSEEK_ANTHROPIC_BASE_URL,
            "ANTHROPIC_API_KEY": api_key,
            "ANTHROPIC_AUTH_TOKEN": api_key,
        }
        default_model = DEEPSEEK_DEFAULT_MODEL
    prompt = "\n".join(
        [
            f"Use fix prompt file: {prompt_path}",
            f"Project: {config.project}",
            f"Repository: {config.github_repo}",
            f"Issue: #{issue_number}",
            f"Use fix context file: {context_path}",
            "Edit the code in the current working directory to fix the bug, "
            "adding or adjusting tests as needed.",
            f"Schema file: {schema_path}",
            f"After editing, write final JSON summary to: {output_path}",
        ]
    )
    model = config.fix_agent.model or default_model
    command = _claude_task_command(prompt, workspace_dirs=workspace_dirs, model=model)
    return AgentInvocation(provider=provider, command=command, env=env, model=model)
