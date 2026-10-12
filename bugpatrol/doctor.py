"""Environment checks for a bugpatrol project."""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import zlib
from dataclasses import dataclass
from pathlib import Path

from bugpatrol.config import ProjectConfig
from bugpatrol.fields import NATIVE_ISSUE_TYPES, default_field_specs
from bugpatrol.github import GitHubCliIssuesClient
from bugpatrol.github_fields import GitHubIssueFieldsClient
from bugpatrol.lark import DownloadedLarkResource, LarkOpenApiMessengerClient
from bugpatrol.resources import (
    DESCRIPTION_UNAVAILABLE_PREFIX,
    CommandResourceDescriber,
    LarkResourceRef,
    _format_command_part,
)

# The media probe is one real call to the vision endpoint. Cap it well below the
# production timeout (a 32x32 probe answers in seconds) so a hanging endpoint
# fails the doctor instead of stalling it for the project's full 300s budget.
_MEDIA_PROBE_TIMEOUT_SECONDS = 60
_MEDIA_PROBE_PIXELS = 32


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str


def run_doctor(
    *,
    config: ProjectConfig,
    github: GitHubCliIssuesClient,
    issue_fields: GitHubIssueFieldsClient,
    lark: LarkOpenApiMessengerClient | None = None,
) -> tuple[DoctorCheck, ...]:
    checks: list[DoctorCheck] = []
    checks.append(_check("config", lambda: _check_config(config)))
    checks.append(_check("github_repo", lambda: _check_repo(config, github)))
    checks.append(_check("issue_types", lambda: _check_issue_types(config, github)))
    checks.append(_check("issue_fields", lambda: _check_issue_fields(config, issue_fields)))
    checks.append(_check("asset_repo", lambda: _check_asset_repo(config, github)))
    checks.append(_check("media_vision_command", lambda: _check_media_command(config)))
    checks.append(_check("ffmpeg", lambda: _check_ffmpeg(config)))
    checks.append(_check("triage_agent", lambda: _check_triage_agent(config)))
    checks.append(_check("triage_agent_auth", lambda: _check_triage_agent_auth(config)))
    if lark is not None:
        checks.append(_check("lark_history", lambda: _check_lark(config, lark)))
    return tuple(checks)


def _check(name: str, fn: object) -> DoctorCheck:
    try:
        detail = fn()  # type: ignore[operator]
    except Exception as exc:  # noqa: BLE001
        return DoctorCheck(name=name, ok=False, detail=str(exc))
    return DoctorCheck(name=name, ok=True, detail=detail)


def _check_config(config: ProjectConfig) -> str:
    config.validate_against(default_field_specs())
    return config.github_repo


def _check_repo(config: ProjectConfig, github: GitHubCliIssuesClient) -> str:
    repo = github.get_repository(repo=config.github_repo)
    return f"{repo.get('full_name')} private={repo.get('private')}"


def _check_issue_types(config: ProjectConfig, github: GitHubCliIssuesClient) -> str:
    names = github.list_issue_types(repo=config.github_repo)
    missing = sorted(set(NATIVE_ISSUE_TYPES) - set(names))
    if missing:
        raise ValueError(f"missing issue types: {missing}")
    return ", ".join(names)


def _check_issue_fields(config: ProjectConfig, issue_fields: GitHubIssueFieldsClient) -> str:
    owner = config.github_repo.split("/", 1)[0]
    live = issue_fields.list_org_fields(org=owner)
    live_options = {name: field.options for name, field in live.items()}
    config.validate_github_field_options(live_options, default_field_specs())
    return f"{len(live)} org fields"


def _check_asset_repo(config: ProjectConfig, github: GitHubCliIssuesClient) -> str:
    if not config.assets.github_repo:
        return "not configured"
    repo = github.get_repository(repo=config.assets.github_repo)
    return f"{repo.get('full_name')} private={repo.get('private')}"


def _check_media_command(config: ProjectConfig) -> str:
    if not config.media.description_command:
        return "not configured"
    probe_ref = LarkResourceRef(message_id="doctor-probe", kind="image", resource_key="doctor-probe")
    probe_resource = DownloadedLarkResource(
        content=_probe_image_bytes(),
        content_type="image/png",
        filename="doctor-probe.png",
    )
    # Resolve {python} (and friends) first: the command usually *starts* with a
    # placeholder, and `which "{python}"` would report a missing binary for a
    # config whose interpreter is plainly there.
    executable = _format_command_part(
        config.media.description_command[0],
        path=Path(probe_resource.filename),
        ref=probe_ref,
        resource=probe_resource,
    )
    if shutil.which(executable) is None:
        raise ValueError(f"media command not found: {executable}")
    # `which` only proves the binary is there. A media command whose model name
    # was retired upstream, or whose API key is missing, passes that check and
    # then fails on every real attachment — quietly: the describer returns a
    # placeholder string instead of raising, so triage carries on and the only
    # trace is a line in agent-turns.jsonl. That is how the whole fleet lost its
    # media descriptions when the default model was retired (2026-10-12). Probe
    # the command for real, once, on an image generated here.
    describer = CommandResourceDescriber(
        command=config.media.description_command,
        timeout_seconds=min(config.media.description_timeout_seconds, _MEDIA_PROBE_TIMEOUT_SECONDS),
    )
    description = describer.describe(ref=probe_ref, resource=probe_resource).strip()
    if not description:
        raise ValueError("media command returned an empty description")
    if description.startswith(DESCRIPTION_UNAVAILABLE_PREFIX):
        raise ValueError(description)
    return f"{executable} (live probe ok)"


def _probe_image_bytes(size: int = _MEDIA_PROBE_PIXELS) -> bytes:
    """A minimal solid-red PNG. The command only has to prove it accepts the
    request end to end; what the model answers about it does not matter."""
    raw = (b"\x00" + b"\xff\x00\x00" * size) * size

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def _check_ffmpeg(config: ProjectConfig) -> str:
    if config.media.max_video_bytes <= 0:
        return "not required"
    if shutil.which("ffmpeg") is None:
        raise ValueError("ffmpeg not found")
    return "ffmpeg"


def _check_triage_agent(config: ProjectConfig) -> str:
    if config.triage_agent.provider not in {"codex", "claude", "deepseek"}:
        raise ValueError(f"unsupported provider: {config.triage_agent.provider}")
    if not config.triage_agent.runner_labels:
        raise ValueError("missing runner labels")
    return config.triage_agent.provider


def _check_triage_agent_auth(config: ProjectConfig) -> str:
    if config.triage_agent.provider == "deepseek":
        # deepseek rides the claude CLI with API-key auth; no login session needed.
        if shutil.which("claude") is None:
            raise ValueError("claude not found")
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise ValueError("deepseek provider requires DEEPSEEK_API_KEY in the environment")
        return "claude CLI + DEEPSEEK_API_KEY"
    command = _triage_agent_auth_command(config)
    executable = command[0]
    if shutil.which(executable) is None:
        raise ValueError(f"{executable} not found")
    completed = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or f"exit {completed.returncode}").strip()
        raise ValueError(f"{config.triage_agent.provider} auth check failed: {detail[:300]}")
    detail = (completed.stdout or completed.stderr).strip().splitlines()
    return detail[0] if detail else config.triage_agent.provider


def _triage_agent_auth_command(config: ProjectConfig) -> tuple[str, ...]:
    if config.triage_agent.provider == "codex":
        return ("codex", "login", "status")
    if config.triage_agent.provider == "claude":
        return ("claude", "auth", "status")
    raise ValueError(f"unsupported provider: {config.triage_agent.provider}")


def _check_lark(config: ProjectConfig, lark: LarkOpenApiMessengerClient) -> str:
    messages = lark.list_chat_messages(chat_id=config.lark.chat_id, limit=1)
    return f"read {len(messages)} recent messages"
