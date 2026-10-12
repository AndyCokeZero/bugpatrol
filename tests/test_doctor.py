from __future__ import annotations

import struct
import subprocess
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from bugpatrol.config import load_project_config
from bugpatrol.doctor import (
    _MEDIA_PROBE_TIMEOUT_SECONDS,
    _check_media_command,
    _probe_image_bytes,
    _triage_agent_auth_command,
    run_doctor,
)
from bugpatrol.github_fields import IssueField
from bugpatrol.resources import DESCRIPTION_UNAVAILABLE_PREFIX


class FakeGithub:
    def get_repository(self, *, repo: str) -> dict[str, object]:
        return {"full_name": repo, "private": True}

    def list_issue_types(self, *, repo: str) -> tuple[str, ...]:
        return ("Bug", "Feature", "Task")


class FakeIssueFields:
    def list_org_fields(self, *, org: str) -> dict[str, IssueField]:
        from bugpatrol.fields import default_field_specs

        return {
            name: IssueField(id=i, name=name, data_type="single_select", options=spec.values)
            for i, (name, spec) in enumerate(default_field_specs().items(), start=1)
        }


class FakeMediaDescriber:
    """Records what the doctor probe asked for and returns a canned answer.

    The real describer would call the vision endpoint; the unit suite must not.
    """

    response = "a red square"
    last: FakeMediaDescriber | None = None
    command: tuple[str, ...] = ()
    timeout_seconds: int = 0
    content: bytes = b""

    def __init__(self, *, command: tuple[str, ...], timeout_seconds: int = 300, **kwargs: object) -> None:
        self.command = tuple(command)
        self.timeout_seconds = timeout_seconds
        FakeMediaDescriber.last = self

    def describe(self, *, ref: object, resource: object) -> str:
        FakeMediaDescriber.content = getattr(resource, "content", b"")
        return self.response


class DoctorTest(unittest.TestCase):
    def test_run_doctor_reports_all_checks_ok(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        config = replace(config, triage_agent=replace(config.triage_agent, provider="codex"))

        with patch("bugpatrol.doctor.CommandResourceDescriber", FakeMediaDescriber):
            with patch("bugpatrol.doctor.shutil.which", return_value="/usr/bin/tool"):
                with patch("bugpatrol.doctor.subprocess.run") as run:
                    run.return_value = subprocess.CompletedProcess(
                        ["codex", "login", "status"],
                        0,
                        "Logged in using ChatGPT\n",
                        "",
                    )
                    checks = run_doctor(
                        config=config,
                        github=FakeGithub(),  # type: ignore[arg-type]
                        issue_fields=FakeIssueFields(),  # type: ignore[arg-type]
                    )

        self.assertTrue(all(check.ok for check in checks))
        self.assertEqual(
            [check.name for check in checks],
            [
                "config",
                "github_repo",
                "issue_types",
                "issue_fields",
                "asset_repo",
                "media_vision_command",
                "ffmpeg",
                "triage_agent",
                "triage_agent_auth",
            ],
        )
        run.assert_called_once_with(
            ["codex", "login", "status"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    def test_triage_agent_auth_command_matches_provider(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        codex = replace(config, triage_agent=replace(config.triage_agent, provider="codex"))
        claude = replace(config, triage_agent=replace(config.triage_agent, provider="claude"))

        self.assertEqual(_triage_agent_auth_command(codex), ("codex", "login", "status"))
        self.assertEqual(_triage_agent_auth_command(claude), ("claude", "auth", "status"))

    def test_doctor_reports_agent_auth_failure(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        config = replace(config, triage_agent=replace(config.triage_agent, provider="codex"))

        with patch("bugpatrol.doctor.CommandResourceDescriber", FakeMediaDescriber):
            with patch("bugpatrol.doctor.shutil.which", return_value="/usr/bin/tool"):
                with patch("bugpatrol.doctor.subprocess.run") as run:
                    run.return_value = subprocess.CompletedProcess(
                        ["codex", "login", "status"],
                        1,
                        "",
                        "not logged in",
                    )
                    checks = run_doctor(
                        config=config,
                        github=FakeGithub(),  # type: ignore[arg-type]
                        issue_fields=FakeIssueFields(),  # type: ignore[arg-type]
                    )

        auth = next(check for check in checks if check.name == "triage_agent_auth")
        self.assertFalse(auth.ok)
        self.assertIn("not logged in", auth.detail)

    def test_doctor_deepseek_auth_checks_api_key(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        config = replace(config, triage_agent=replace(config.triage_agent, provider="deepseek"))

        with patch("bugpatrol.doctor.CommandResourceDescriber", FakeMediaDescriber):
            with patch("bugpatrol.doctor.shutil.which", return_value="/usr/bin/claude"):
                with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}):
                    checks = run_doctor(
                        config=config,
                        github=FakeGithub(),  # type: ignore[arg-type]
                        issue_fields=FakeIssueFields(),  # type: ignore[arg-type]
                    )
        auth = next(check for check in checks if check.name == "triage_agent_auth")
        self.assertTrue(auth.ok)

        with patch("bugpatrol.doctor.CommandResourceDescriber", FakeMediaDescriber):
            with patch("bugpatrol.doctor.shutil.which", return_value="/usr/bin/claude"):
                with patch.dict("os.environ", {}, clear=True):
                    checks = run_doctor(
                        config=config,
                        github=FakeGithub(),  # type: ignore[arg-type]
                        issue_fields=FakeIssueFields(),  # type: ignore[arg-type]
                    )
        auth = next(check for check in checks if check.name == "triage_agent_auth")
        self.assertFalse(auth.ok)
        self.assertIn("DEEPSEEK_API_KEY", auth.detail)

    def test_media_probe_runs_the_configured_command_on_a_generated_png(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))

        # No shutil.which stub here on purpose: description_command starts with
        # "{python}", and the check has to resolve that to a real interpreter
        # before looking it up (it used to look up the literal `{python}` and
        # report a missing binary on every configured project).
        with patch("bugpatrol.doctor.CommandResourceDescriber", FakeMediaDescriber):
            detail = _check_media_command(config)

        self.assertIn("live probe ok", detail)
        probe = FakeMediaDescriber.last
        assert probe is not None
        # The probe hands the configured command (placeholders and all) to the
        # production describer: {python}/{path} substitution stays in one place.
        self.assertEqual(probe.command, config.media.description_command)
        self.assertEqual(probe.timeout_seconds, _MEDIA_PROBE_TIMEOUT_SECONDS)
        self.assertEqual(FakeMediaDescriber.content, _probe_image_bytes())

    def test_media_probe_fails_when_the_command_fails(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))

        class FailingDescriber(FakeMediaDescriber):
            response = f"{DESCRIPTION_UNAVAILABLE_PREFIX} HTTP 404: Model not exist."

        with patch("bugpatrol.doctor.CommandResourceDescriber", FailingDescriber):
            checks = run_doctor(
                config=config,
                github=FakeGithub(),  # type: ignore[arg-type]
                issue_fields=FakeIssueFields(),  # type: ignore[arg-type]
            )

        media = next(check for check in checks if check.name == "media_vision_command")
        self.assertFalse(media.ok)
        self.assertIn("Model not exist", media.detail)

    def test_probe_image_is_a_png(self) -> None:
        image = _probe_image_bytes(size=8)

        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(image.endswith(b"IEND\xaeB`\x82"))
        self.assertIn(b"IHDR", image)
        # IHDR payload starts 8 bytes in: length(4) + tag(4), then width/height.
        self.assertEqual(image[16:24], struct.pack(">II", 8, 8))


if __name__ == "__main__":
    unittest.main()
