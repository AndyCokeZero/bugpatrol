from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import subprocess
from pathlib import Path
from unittest.mock import patch

from bugpatrol.agents import AgentInvocation
from bugpatrol.clients import GitHubIssue, GitHubIssueComment
from bugpatrol.config import load_project_config
from bugpatrol.fields import default_field_specs
from bugpatrol.github_fields import IssueField
from bugpatrol.intake import IntakeRecord, render_issue_body
from bugpatrol.triage_context import ReferenceRepoContext
from bugpatrol.triage_runner import (
    TRIAGE_FAILED_HEADER,
    TRIAGE_SKIPPED_HEADER,
    TriageRunPlan,
    append_triage_run_metadata,
    build_assignee_roster,
    comment_ids,
    evaluate_followup_re_triage,
    execute_triage_run,
    list_known_assignees,
    prepare_triage_run,
    render_triage_failed_comment,
    report_workflow_failure,
    triage_run_in_flight,
    _write_turn_log,
)


class FakeGithub:
    def __init__(self, *, issue_body: str | None = None) -> None:
        self.comments: list[str] = ["Follow-up comment"]
        self.issue_types: list[str] = []
        self.assignees: list[str] = []
        self.updated_titles: list[str] = []
        self.issue_body = issue_body if issue_body is not None else managed_issue_body()

    def get_issue(self, *, repo: str, issue_number: int) -> GitHubIssue:
        return GitHubIssue(
            number=issue_number,
            url=f"https://github.test/{repo}/issues/{issue_number}",
            title="Todo empty state missing",
            body=self.issue_body,
        )

    def add_issue_comment(self, *, repo: str, issue_number: int, body: str) -> None:
        self.comments.append(body)

    def list_issue_comments(self, *, repo: str, issue_number: int) -> tuple[GitHubIssueComment, ...]:
        return tuple(
            GitHubIssueComment(id=str(index + 1), body=body)
            for index, body in enumerate(self.comments)
        )

    def set_issue_type(self, *, repo: str, issue_number: int, issue_type: str) -> None:
        self.issue_types.append(issue_type)

    def update_issue_title(self, *, repo: str, issue_number: int, title: str) -> None:
        self.updated_titles.append(title)

    def add_assignee(self, *, repo: str, issue_number: int, assignee: str) -> None:
        self.assignees.append(assignee)


class FakeIssueFields:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.writes: list[dict[str, object]] = []
        self.values = values or {}

    def get_issue_field_values(self, **kwargs: object) -> dict[str, str]:
        return dict(self.values)

    def add_issue_field_values(self, **kwargs: object) -> None:
        self.writes.append(kwargs)

    def list_org_fields(self, **kwargs: object) -> dict[str, IssueField]:
        return {
            name: IssueField(id=i, name=name, data_type="single_select", options=spec.values)
            for i, (name, spec) in enumerate(default_field_specs().items())
        }


class TriageRunnerTest(unittest.TestCase):
    def test_prepare_triage_run_writes_context_schema_and_command(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        config = replace(config, triage_agent=replace(config.triage_agent, provider="codex"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prd_root = root / config.prd.cache_path
            prd_root.mkdir(parents=True)
            (prd_root / "todo-list.md").write_text(
                "# Todo List PRD\n\nEmpty state appears after deleting all todos."
            )
            output_dir = root / "run"
            plan = prepare_triage_run(
                config=config,
                issue_number=7,
                repo_path=root,
                output_dir=output_dir,
                github=FakeGithub(),  # type: ignore[arg-type]
            )

            self.assertTrue(plan.context_path.exists())
            self.assertTrue(plan.schema_path.exists())
            # Paths handed to the agent are resolved to absolute (it runs with
            # cwd set to the checkout, not the runner workspace).
            self.assertEqual(plan.output_path, output_dir.resolve() / "triage-output.json")
            self.assertEqual(plan.agent_cwd, root.resolve())
            self.assertIn("Todo empty state missing", plan.context_path.read_text())
            self.assertIn("Follow-up comment", plan.context_path.read_text())
            self.assertIn("triage-context.md", plan.invocation.command[-1])

    def test_prepare_triage_run_surfaces_openspec_owner_without_whitelisting(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        # The shipped config's provider is deepseek, which refuses to build an
        # invocation without a key. The key is irrelevant here -- nothing runs
        # the agent -- so supply a dummy rather than requiring the ambient env
        # (CI has no DEEPSEEK_API_KEY, which had this test red since 2026-09-29).
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ", {"DEEPSEEK_API_KEY": "sk-test"}
        ):
            root = Path(tmp)
            prd_root = root / config.prd.cache_path
            change_dir = prd_root / "changes" / "todo-empty-state"
            change_dir.mkdir(parents=True)
            (change_dir / "tasks.md").write_text(
                "- [x] empty-state: 删除全部 todo 的空态 · @naohn"
            )
            # Baseline whitelist derived solely from the login sources
            # (CODEOWNERS + config), computed without any openspec input.
            baseline_assignees = tuple(list_known_assignees(root, config=config))
            plan = prepare_triage_run(
                config=config,
                issue_number=7,
                repo_path=root,
                output_dir=root / "run",
                github=FakeGithub(),  # type: ignore[arg-type]
            )

            context = plan.context_path.read_text()

        # The openspec owner nickname is surfaced in the context for the agent to
        # map via the roster, but is NOT injected into the assignee whitelist
        # (nicknames aren't GitHub logins — assigning one raw would fail).
        self.assertIn("## OpenSpec Owners", context)
        self.assertIn("@naohn", context)
        self.assertNotIn("naohn", plan.known_assignees)
        # Invariant: openspec owners add NOTHING to the schema-enum whitelist,
        # regardless of the token. This holds even if a change owner happened to
        # be a real login — guarding against re-introducing a union of external
        # tokens into known_assignees (the pre-ship bug this feature nearly shipped).
        self.assertEqual(tuple(plan.known_assignees), baseline_assignees)

    def test_prepare_triage_run_rejects_unmanaged_issue(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "run"

            with self.assertRaisesRegex(ValueError, "missing BUGPATROL_INTAKE_META"):
                prepare_triage_run(
                    config=config,
                    issue_number=7,
                    repo_path=root,
                    output_dir=output_dir,
                    github=FakeGithub(issue_body="legacy issue"),  # type: ignore[arg-type]
                )

            self.assertFalse(output_dir.exists())

    def test_prepare_triage_run_injects_reference_repo_context(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        config = replace(config, triage_agent=replace(config.triage_agent, provider="claude"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / config.prd.cache_path).mkdir(parents=True, exist_ok=True)
            ref_checkout = root / "weaver"
            ref_checkout.mkdir()
            output_dir = root / "run"
            plan = prepare_triage_run(
                config=config,
                issue_number=7,
                repo_path=root,
                output_dir=output_dir,
                github=FakeGithub(),  # type: ignore[arg-type]
                reference_repos=(
                    ReferenceRepoContext(
                        repo="org/weaver",
                        path=str(ref_checkout),
                        analyzed_branch="feature/live",
                        purpose="backend",
                    ),
                ),
            )

            context_text = plan.context_path.read_text()
            self.assertIn("## Reference Repos", context_text)
            self.assertIn("org/weaver", context_text)
            # The reference checkout is re-admitted for the agent via --add-dir.
            self.assertIn("--add-dir", plan.invocation.command)
            self.assertIn(str(ref_checkout.resolve()), plan.invocation.command)

    def test_prepare_triage_run_fails_loud_on_missing_reference_checkout(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / config.prd.cache_path).mkdir(parents=True, exist_ok=True)
            output_dir = root / "run"
            with self.assertRaisesRegex(FileNotFoundError, "org/weaver"):
                prepare_triage_run(
                    config=config,
                    issue_number=7,
                    repo_path=root,
                    output_dir=output_dir,
                    github=FakeGithub(),  # type: ignore[arg-type]
                    reference_repos=(
                        ReferenceRepoContext(
                            repo="org/weaver",
                            path=str(root / "does-not-exist"),
                            analyzed_branch="main",
                        ),
                    ),
                )

    def test_comment_ids_ignore_triage_run_metadata_comments(self) -> None:
        comments = (
            GitHubIssueComment(id="1", body="Reporter follow-up"),
            GitHubIssueComment(id="2", body=append_triage_run_metadata({"version": 1, "run_id": "run-1"})),
        )

        self.assertEqual(comment_ids(comments), ("1",))

    def test_comment_ids_ignore_triage_bookkeeping_comments(self) -> None:
        # A run yielding to a newer one posts the "skipped" note; counting it as
        # context made the newer run abort as stale, so both runs produced nothing.
        comments = (
            GitHubIssueComment(id="1", body="Reporter follow-up"),
            GitHubIssueComment(id="2", body=f"{TRIAGE_SKIPPED_HEADER}\n\nRun `x` was superseded"),
            GitHubIssueComment(id="3", body=f"{TRIAGE_FAILED_HEADER}\n\n分诊运行失败"),
        )

        self.assertEqual(comment_ids(comments), ("1",))

    def test_triage_run_in_flight_only_while_a_recent_run_has_not_reported(self) -> None:
        now = datetime(2026, 8, 3, 4, 0, tzinfo=timezone.utc)

        def marker(minutes_ago: int) -> GitHubIssueComment:
            started = (now - timedelta(minutes=minutes_ago)).isoformat()
            return GitHubIssueComment(
                id="m", body=append_triage_run_metadata({"version": 1, "run_id": "r", "started_at": started})
            )

        self.assertTrue(triage_run_in_flight((marker(2),), now=now))
        # Older than the workflow timeout: the run died without reporting.
        self.assertFalse(triage_run_in_flight((marker(90),), now=now))
        # The run already reported its outcome.
        self.assertFalse(
            triage_run_in_flight(
                (marker(2), GitHubIssueComment(id="s", body=f"{TRIAGE_SKIPPED_HEADER}\n\nsuperseded")),
                now=now,
            )
        )
        self.assertFalse(triage_run_in_flight((GitHubIssueComment(id="1", body="hi"),), now=now))

    def test_build_assignee_roster_uses_login_and_codeowners_name(self) -> None:
        roster = build_assignee_roster(
            ("naohn42", "garlanddiego"),
            codeowners_identities={"naohn42": ("Naohn",)},
        )

        by_login = {item.login: item.aliases for item in roster}
        # Login is always its own alias; the CODEOWNERS header adds the name.
        self.assertEqual(by_login["naohn42"], ("naohn42", "Naohn"))
        self.assertEqual(by_login["garlanddiego"], ("garlanddiego",))

    def test_execute_triage_run_marks_failed_when_agent_exits_nonzero(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        issue_fields = FakeIssueFields()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=root / "output.json",
                invocation=AgentInvocation(provider="codex", command=["false"]),
            )

            with patch("subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(["false"], 42)
                with self.assertRaisesRegex(RuntimeError, "exit 42"):
                    execute_triage_run(
                        config=config,
                        issue_number=7,
                        plan=plan,
                        github=github,  # type: ignore[arg-type]
                        issue_fields=issue_fields,  # type: ignore[arg-type]
                    )

        self.assertEqual(issue_fields.writes[0]["values"], {"Triage status": "Running"})
        self.assertEqual(issue_fields.writes[1]["values"], {"Triage status": "Failed"})
        self.assertIn("exited with code `42`", github.comments[-1])

    def test_execute_triage_run_surfaces_agent_stderr_on_failure(self) -> None:
        # The generic "exited with code N" gave zero clue why #4140 failed (a
        # runner-side github.com TLS timeout + keychain auth failure). Surface a
        # bounded, ANSI-stripped stderr tail in both the GitHub comment and Lark.
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        issue_fields = FakeIssueFields()
        stderr = (
            "\x1b[31mfatal: unable to access 'https://github.com/...': "
            "Failed to connect to github.com port 443 after 75002 ms\x1b[0m\n"
            "remote: Authentication failed"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=root / "output.json",
                invocation=AgentInvocation(provider="codex", command=["false"]),
            )

            with patch("subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(["false"], 1, "", stderr)
                with self.assertRaisesRegex(RuntimeError, "exit 1"):
                    execute_triage_run(
                        config=config,
                        issue_number=4140,
                        plan=plan,
                        github=github,  # type: ignore[arg-type]
                        issue_fields=issue_fields,  # type: ignore[arg-type]
                    )

        comment = github.comments[-1]
        self.assertIn("Agent stderr (tail):", comment)
        self.assertIn("Authentication failed", comment)
        self.assertIn("Failed to connect to github.com", comment)
        # ANSI escapes are stripped before surfacing.
        self.assertNotIn("\x1b[", comment)

    def test_execute_triage_run_surfaces_agent_stdout_api_error_on_failure(self) -> None:
        # #4145 (and #4139/#4141) failed in ~0.5s with DeepSeek 402 "Insufficient
        # Balance". The CLI reports that on stdout as a result event with
        # is_error:true while stderr is empty, so the earlier stderr-only surfacing
        # showed "No agent stderr was captured". Surface the stdout error too.
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        issue_fields = FakeIssueFields()
        api_error = (
            'API Error: 402 {"error":{"message":"Insufficient Balance",'
            '"type":"unknown_error","param":null,"code":"invalid_request_error"}}'
        )
        stdout = json.dumps(
            {"type": "result", "subtype": "success", "is_error": True, "result": api_error}
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=root / "output.json",
                invocation=AgentInvocation(provider="codex", command=["false"]),
            )

            with patch("subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(["false"], 1, stdout, "")
                with self.assertRaisesRegex(RuntimeError, "exit 1"):
                    execute_triage_run(
                        config=config,
                        issue_number=4145,
                        plan=plan,
                        github=github,  # type: ignore[arg-type]
                        issue_fields=issue_fields,  # type: ignore[arg-type]
                    )

        comment = github.comments[-1]
        self.assertIn("Agent error:", comment)
        self.assertIn("Insufficient Balance", comment)
        self.assertIn("402", comment)
        # No hollow "No agent stderr was captured" when the cause is known.
        self.assertNotIn("No agent stderr", comment)

    def test_execute_triage_run_marks_failed_when_agent_times_out(self) -> None:
        # #4938's agent hung for a full hour with no output and the job was
        # silently cancelled at the workflow's 60-minute timeout, wasting a
        # runner slot. A subprocess timeout must fail fast with a visible
        # Failed marker instead of letting the job burn to cancel.
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        issue_fields = FakeIssueFields()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=root / "output.json",
                invocation=AgentInvocation(provider="codex", command=["claude"]),
            )

            with patch("subprocess.run") as run:
                # Real TimeoutExpired carries raw bytes in .stdout/.stderr even
                # under text=True (CPython quirk); mirror that so the bytes-vs-str
                # crash that masked #5038's real cause stays covered.
                timeout = subprocess.TimeoutExpired(
                    cmd=["claude"],
                    timeout=1200,
                    output=b'{"partial": 1}',
                    stderr=b"",
                )
                # subprocess.run() populates `.stdout` when it raises the
                # exception; mirror that for the hand-built one.
                timeout.stdout = b'{"partial": 1}'
                run.side_effect = timeout
                with self.assertRaisesRegex(RuntimeError, "timed out after 1200s"):
                    execute_triage_run(
                        config=config,
                        issue_number=7,
                        plan=plan,
                        github=github,  # type: ignore[arg-type]
                        issue_fields=issue_fields,  # type: ignore[arg-type]
                    )

        self.assertEqual(issue_fields.writes[0]["values"], {"Triage status": "Running"})
        self.assertEqual(issue_fields.writes[1]["values"], {"Triage status": "Failed"})
        self.assertIn("no terminal output for 1200s", github.comments[-1])
        self.assertIn("## BugPatrol triage failed", github.comments[-1])

    def test_write_turn_log_accepts_bytes_like_timeoutexpired(self) -> None:
        # #5038: a timed-out `claude -p` carried bytes in .stdout even under
        # text=True, and write_text(bytes) crashed the failure path with
        # "data must be str, not bytes", masking the real timeout cause.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _write_turn_log(out, b'{"stream": "event"}\n', b"partial stderr\n")
            self.assertEqual((out / "agent-turns.jsonl").read_text(), '{"stream": "event"}\n')
            self.assertEqual((out / "agent-stderr.log").read_text(), "partial stderr\n")

    def test_execute_triage_run_retries_a_transient_api_crash_before_final_attempt(self) -> None:
        # Reproduces #6246: the gateway dropped the response mid-stream after
        # 133 turns and 7 minutes, and the run was thrown away and marked
        # Failed. A second attempt usually finishes, so it must not fail here.
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        issue_fields = FakeIssueFields()
        stdout = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": True,
                "result": (
                    "API Error: Connection closed mid-response. "
                    "The response above may be incomplete."
                ),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=root / "output.json",
                invocation=AgentInvocation(provider="deepseek", command=["claude"]),
                context_comment_ids=("1",),
            )

            with patch("subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(["claude"], 1, stdout, "")
                status = execute_triage_run(
                    config=config,
                    issue_number=6246,
                    plan=plan,
                    github=github,  # type: ignore[arg-type]
                    issue_fields=issue_fields,  # type: ignore[arg-type]
                    final_attempt=False,
                )

        self.assertEqual(status, "agent_crashed")
        # Not wedged: no Failed status, no failure comment, no Lark ping.
        self.assertEqual(issue_fields.writes[-1]["values"], {"Triage status": "Running"})
        self.assertNotIn("BugPatrol triage failed", "".join(github.comments))

    def test_execute_triage_run_marks_failed_on_transient_crash_final_attempt(self) -> None:
        # The retry is bounded: when every attempt drops, the operator still
        # gets the Failed marker and the real error, not a silent pass.
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        issue_fields = FakeIssueFields()
        stdout = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": True,
                "result": "API Error: Connection closed mid-response.",
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=root / "output.json",
                invocation=AgentInvocation(provider="deepseek", command=["claude"]),
                context_comment_ids=("1",),
            )

            with patch("subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(["claude"], 1, stdout, "")
                with self.assertRaisesRegex(RuntimeError, "exit 1"):
                    execute_triage_run(
                        config=config,
                        issue_number=6246,
                        plan=plan,
                        github=github,  # type: ignore[arg-type]
                        issue_fields=issue_fields,  # type: ignore[arg-type]
                        final_attempt=True,
                    )

        comment = github.comments[-1]
        self.assertIn("BugPatrol triage failed", comment)
        self.assertIn("Connection closed mid-response", comment)

    def test_execute_triage_run_fails_immediately_on_a_balance_error(self) -> None:
        # The #4145 lesson, kept: an exhausted account is identical on every
        # attempt, so retrying it only triples the cost of a human's problem.
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        issue_fields = FakeIssueFields()
        stdout = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": True,
                "result": "API Error: 402 Insufficient Balance",
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=root / "output.json",
                invocation=AgentInvocation(provider="deepseek", command=["claude"]),
                context_comment_ids=("1",),
            )

            with patch("subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(["claude"], 1, stdout, "")
                with self.assertRaisesRegex(RuntimeError, "exit 1"):
                    execute_triage_run(
                        config=config,
                        issue_number=4145,
                        plan=plan,
                        github=github,  # type: ignore[arg-type]
                        issue_fields=issue_fields,  # type: ignore[arg-type]
                        final_attempt=False,
                    )

        self.assertIn("Insufficient Balance", github.comments[-1])

    def test_execute_triage_run_rejects_unmanaged_issue_before_writes(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub(issue_body="legacy issue")
        issue_fields = FakeIssueFields()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=root / "output.json",
                invocation=AgentInvocation(provider="codex", command=["false"]),
            )

            with self.assertRaisesRegex(ValueError, "missing BUGPATROL_INTAKE_META"):
                execute_triage_run(
                    config=config,
                    issue_number=7,
                    plan=plan,
                    github=github,  # type: ignore[arg-type]
                    issue_fields=issue_fields,  # type: ignore[arg-type]
                )

        self.assertEqual(issue_fields.writes, [])
        self.assertEqual(github.comments, ["Follow-up comment"])

    def test_execute_triage_run_skips_when_already_triaged_without_new_material(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        github.comments.append(
            '<!-- BUGPATROL_TRIAGE_META\n{"result_fingerprint":"abc","version":1}\nBUGPATROL_TRIAGE_META -->'
        )
        issue_fields = FakeIssueFields()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=root / "output.json",
                invocation=AgentInvocation(provider="codex", command=["true"]),
            )

            with patch("subprocess.run") as run:
                status = execute_triage_run(
                    config=config,
                    issue_number=7,
                    plan=plan,
                    github=github,  # type: ignore[arg-type]
                    issue_fields=issue_fields,  # type: ignore[arg-type]
                )

        self.assertEqual(status, "already_triaged")
        run.assert_not_called()
        self.assertEqual(issue_fields.writes, [])
        self.assertEqual(len(github.comments), 2)

    def test_execute_triage_run_does_not_skip_when_new_material_follows_triage(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        github.comments.append(
            '<!-- BUGPATROL_TRIAGE_META\n{"result_fingerprint":"abc","version":1}\nBUGPATROL_TRIAGE_META -->'
        )
        github.comments.append("New material after the previous triage")
        issue_fields = FakeIssueFields()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output.json"
            output.write_text(valid_triage_output())
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=output,
                invocation=AgentInvocation(provider="codex", command=["true"]),
                context_comment_ids=("1", "2"),
            )

            with patch("subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(["true"], 0)
                status = execute_triage_run(
                    config=config,
                    issue_number=7,
                    plan=plan,
                    github=github,  # type: ignore[arg-type]
                    issue_fields=issue_fields,  # type: ignore[arg-type]
                )

        self.assertEqual(status, "stale_context")
        run.assert_called_once()
        self.assertEqual(issue_fields.writes[-1]["values"], {"Triage status": "Running"})

    def test_execute_triage_run_skips_agent_when_followup_does_not_change_triage(self) -> None:
        # A material followup that the cheap classifier judges as no-change must
        # not burn a full agent run. intake reset Done->Pending so the watcher
        # would re-dispatch; the skip restores the terminal status.
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        github.comments.append(
            '<!-- BUGPATROL_TRIAGE_META\n{"result_fingerprint":"abc","version":1,"triage_status":"Done"}\n'
            "BUGPATROL_TRIAGE_META -->"
        )
        github.comments.append("补充一点无关紧要的说明")
        issue_fields = FakeIssueFields()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=root / "output.json",
                invocation=AgentInvocation(
                    provider="deepseek",
                    command=["claude", "-p", "x"],
                    env={"ANTHROPIC_BASE_URL": "https://example.invalid"},
                    model="deepseek-v4-flash",
                ),
            )
            with patch("subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(["claude", "-p", "x"], 0, "no_change", "")
                status = execute_triage_run(
                    config=config,
                    issue_number=7,
                    plan=plan,
                    github=github,  # type: ignore[arg-type]
                    issue_fields=issue_fields,  # type: ignore[arg-type]
                )
        self.assertEqual(status, "no_change_followup")
        # Only the classifier ran; the expensive agent never did.
        self.assertEqual(run.call_count, 1)
        # The classifier env is the process env merged over the provider
        # endpoint override (replacing it wholesale would drop PATH/HOME).
        classifier_env = run.call_args.kwargs["env"]
        self.assertEqual(classifier_env["ANTHROPIC_BASE_URL"], "https://example.invalid")
        self.assertIn("HOME", classifier_env)
        self.assertEqual(issue_fields.writes[-1]["values"], {"Triage status": "Done"})
        self.assertEqual(len(github.comments), 3)

    def test_execute_triage_run_re_triages_when_followup_changes_triage(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        github.comments.append(
            '<!-- BUGPATROL_TRIAGE_META\n{"result_fingerprint":"abc","version":1,"triage_status":"Done"}\n'
            "BUGPATROL_TRIAGE_META -->"
        )
        github.comments.append("这个其实不是 bug，是网络问题")
        issue_fields = FakeIssueFields()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output.json"
            output.write_text(valid_triage_output())
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=output,
                invocation=AgentInvocation(
                    provider="deepseek",
                    command=["claude", "-p", "x"],
                    env={"ANTHROPIC_BASE_URL": "https://example.invalid"},
                    model="deepseek-v4-flash",
                ),
            )
            with patch("subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(["claude", "-p", "x"], 0, "change", "")
                status = execute_triage_run(
                    config=config,
                    issue_number=7,
                    plan=plan,
                    github=github,  # type: ignore[arg-type]
                    issue_fields=issue_fields,  # type: ignore[arg-type]
                )
        # Classifier said change -> the full agent ran after it.
        self.assertEqual(run.call_count, 2)
        self.assertEqual(issue_fields.writes[-1]["values"], {"Triage status": "Running"})
        self.assertIn(status, {"applied", "stale_context", "no_change_followup"})

    def test_execute_triage_run_re_triages_when_followup_classifier_fails(self) -> None:
        # A classifier outage must not silently skip a re-triage: fall back to
        # the expensive-but-correct full agent run.
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        github.comments.append(
            '<!-- BUGPATROL_TRIAGE_META\n{"result_fingerprint":"abc","version":1,"triage_status":"Done"}\n'
            "BUGPATROL_TRIAGE_META -->"
        )
        github.comments.append("补充细节")
        issue_fields = FakeIssueFields()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output.json"
            output.write_text(valid_triage_output())
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=output,
                invocation=AgentInvocation(
                    provider="deepseek",
                    command=["claude", "-p", "x"],
                    env={"ANTHROPIC_BASE_URL": "https://example.invalid"},
                    model="deepseek-v4-flash",
                ),
            )
            with patch("subprocess.run") as run:
                # Classifier call fails; the agent call that follows succeeds.
                run.side_effect = [
                    subprocess.CompletedProcess(["claude", "-p", "x"], 1, "", "boom"),
                    subprocess.CompletedProcess(["true"], 0, "", ""),
                ]
                status = execute_triage_run(
                    config=config,
                    issue_number=7,
                    plan=plan,
                    github=github,  # type: ignore[arg-type]
                    issue_fields=issue_fields,  # type: ignore[arg-type]
                )
        self.assertEqual(run.call_count, 2)
        self.assertEqual(issue_fields.writes[-1]["values"], {"Triage status": "Running"})
        self.assertIn(status, {"applied", "stale_context"})

    def test_evaluate_followup_re_triage_re_triages_on_new_media_without_classifier(self) -> None:
        # New screenshots/video are high-value evidence: always re-triage, never
        # spend a classifier call on them.
        github = FakeGithub()
        github.comments.append(
            '<!-- BUGPATROL_TRIAGE_META\n{"result_fingerprint":"abc","version":1,"triage_status":"Done"}\n'
            "BUGPATROL_TRIAGE_META -->"
        )
        github.comments.append(
            "## 附件\n\n![截图 1](https://github.com/TheCloverLab/fived-assets/blob/main/x/1.png)"
        )
        comments = github.list_issue_comments(repo="r", issue_number=7)
        invocation = AgentInvocation(
            provider="deepseek", command=["claude", "-p", "x"], model="deepseek-v4-flash"
        )
        with patch("subprocess.run") as run:
            review = evaluate_followup_re_triage(
                issue=github.get_issue(repo="r", issue_number=7),
                comments=comments,
                invocation=invocation,
            )
        self.assertIsNotNone(review)
        assert review is not None
        self.assertTrue(review.requires_re_triage)
        run.assert_not_called()

    def test_evaluate_followup_re_triage_re_triages_on_video_link_without_classifier(self) -> None:
        # Video attachments render as plain links (no `![]` embed); the extension
        # fallback must still catch them so media always re-triages.
        github = FakeGithub()
        github.comments.append(
            '<!-- BUGPATROL_TRIAGE_META\n{"result_fingerprint":"abc","version":1,"triage_status":"Done"}\n'
            "BUGPATROL_TRIAGE_META -->"
        )
        github.comments.append("## 附件\n\n- video: [查看视频](https://github.com/TheCloverLab/fived-assets/blob/main/x/clip.mp4)")
        comments = github.list_issue_comments(repo="r", issue_number=7)
        invocation = AgentInvocation(
            provider="deepseek", command=["claude", "-p", "x"], model="deepseek-v4-flash"
        )
        with patch("subprocess.run") as run:
            review = evaluate_followup_re_triage(
                issue=github.get_issue(repo="r", issue_number=7),
                comments=comments,
                invocation=invocation,
            )
        self.assertIsNotNone(review)
        assert review is not None
        self.assertTrue(review.requires_re_triage)
        run.assert_not_called()

    def test_evaluate_followup_re_triage_returns_none_without_prior_triage(self) -> None:
        github = FakeGithub()  # only the default "Follow-up comment", no triage meta
        comments = github.list_issue_comments(repo="r", issue_number=7)
        invocation = AgentInvocation(
            provider="deepseek", command=["claude", "-p", "x"], model="deepseek-v4-flash"
        )
        with patch("subprocess.run") as run:
            review = evaluate_followup_re_triage(
                issue=github.get_issue(repo="r", issue_number=7),
                comments=comments,
                invocation=invocation,
            )
        self.assertIsNone(review)
        run.assert_not_called()

    def test_execute_triage_run_reports_stale_context_when_comments_changed(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        issue_fields = FakeIssueFields()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output.json"
            output.write_text(
                """
                {
                  "issue_type": "Bug",
                  "priority": "High",
                  "triage_status": "Done",
                  "triage_verdict": "代码 Bug",
                  "platform": "Web",
                  "reproducibility": "必现",
                  "other_platforms": "未验证",
                  "capability": "Quest",
                  "evidence": "文字描述",
                  "prd_status": "已对齐",
                  "triage_confidence": "高",
                  "assignee": "octocat",
                  "owner_reason": "Manual",
                  "summary_cn": "空状态缺失",
                  "follow_up_questions": [],
                  "corrected_title": "删除全部 Todo 后缺少空状态",
                  "comment_markdown": "## Triage Analysis\\n\\nLooks like a code bug."
                }
                """
            )
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=output,
                invocation=AgentInvocation(provider="codex", command=["true"]),
                context_comment_ids=("1",),
            )
            github.comments.append("New material after context generation")

            with patch("subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(["true"], 0)
                status = execute_triage_run(
                    config=config,
                    issue_number=7,
                    plan=plan,
                    github=github,  # type: ignore[arg-type]
                    issue_fields=issue_fields,  # type: ignore[arg-type]
                )

        self.assertEqual(status, "stale_context")
        # Result is not applied: caller retries with fresh context instead.
        self.assertEqual(issue_fields.writes[-1]["values"], {"Triage status": "Running"})
        self.assertEqual(github.issue_types, [])
        # CI runners hand the agent a never-closing stdin pipe; claude -p
        # blocks on it forever unless stdin is DEVNULL.
        self.assertEqual(run.call_args.kwargs.get("stdin"), subprocess.DEVNULL)

    def test_execute_triage_run_applies_stale_context_on_final_attempt(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        issue_fields = FakeIssueFields()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output.json"
            output.write_text(valid_triage_output())
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=output,
                invocation=AgentInvocation(provider="codex", command=["true"]),
                context_comment_ids=("1",),
            )
            github.comments.append("New material after context generation")

            with patch("subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(["true"], 0)
                status = execute_triage_run(
                    config=config,
                    issue_number=7,
                    plan=plan,
                    github=github,  # type: ignore[arg-type]
                    issue_fields=issue_fields,  # type: ignore[arg-type]
                    accept_stale_context=True,
                )

        self.assertEqual(status, "applied")
        self.assertIn("new issue comments kept arriving", github.comments[-1])
        statuses = [w["values"].get("Triage status") for w in issue_fields.writes if "Triage status" in w["values"]]
        self.assertNotIn("Needs review", statuses)

    def test_execute_triage_run_skips_apply_when_superseded(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        issue_fields = FakeIssueFields()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output.json"
            output.write_text(valid_triage_output())
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=output,
                invocation=AgentInvocation(provider="codex", command=["true"]),
                context_comment_ids=("1",),
            )

            def run_agent(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
                github.comments.append(append_triage_run_metadata({"version": 1, "run_id": "run-new"}))
                return subprocess.CompletedProcess(["true"], 0)

            with patch("bugpatrol.triage_runner.uuid4", return_value="run-old"):
                with patch("subprocess.run", side_effect=run_agent):
                    status = execute_triage_run(
                        config=config,
                        issue_number=7,
                        plan=plan,
                        github=github,  # type: ignore[arg-type]
                        issue_fields=issue_fields,  # type: ignore[arg-type]
                    )

        self.assertEqual(status, "superseded")
        # The newer run owns the status field; the superseded run must not touch it.
        self.assertEqual(issue_fields.writes[-1]["values"], {"Triage status": "Running"})
        self.assertEqual(github.issue_types, [])
        self.assertEqual(github.assignees, [])
        self.assertIn("superseded", github.comments[-1])

    def test_execute_triage_run_fails_when_agent_produces_no_output(self) -> None:
        config = load_project_config(Path("projects/full.example.toml"))
        github = FakeGithub()
        issue_fields = FakeIssueFields()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=root / "output.json",
                invocation=AgentInvocation(provider="claude", command=["true"]),
                context_comment_ids=("1",),
            )

            with patch("subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(["true"], 0)
                with self.assertRaisesRegex(RuntimeError, "no output file"):
                    execute_triage_run(
                        config=config,
                        issue_number=7,
                        plan=plan,
                        github=github,  # type: ignore[arg-type]
                        issue_fields=issue_fields,  # type: ignore[arg-type]
                    )

        self.assertIn("BugPatrol triage failed", github.comments[-1])

    def test_execute_triage_run_retries_silently_missing_output_before_final_attempt(self) -> None:
        config = load_project_config(Path("projects/full.example.toml"))
        github = FakeGithub()
        issue_fields = FakeIssueFields()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=root / "output.json",
                invocation=AgentInvocation(provider="claude", command=["true"]),
                context_comment_ids=("1",),
            )

            with patch("subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(["true"], 0)
                status = execute_triage_run(
                    config=config,
                    issue_number=7,
                    plan=plan,
                    github=github,  # type: ignore[arg-type]
                    issue_fields=issue_fields,  # type: ignore[arg-type]
                    final_attempt=False,
                )

        self.assertEqual(status, "no_output")
        # No Failed status or failure comment yet: the caller retries.
        self.assertEqual(issue_fields.writes[-1]["values"], {"Triage status": "Running"})
        self.assertNotIn("BugPatrol triage failed", "".join(github.comments))

    def test_execute_triage_run_retries_on_invalid_output_before_final_attempt(self) -> None:
        # Reproduces #3987: agent wrote duplicate_of>0 without verdict 重复.
        # The strict parser used to raise unhandled, wedging Triage status at
        # "Running" forever; now it retries like no_output.
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        issue_fields = FakeIssueFields()
        bad = json.loads(valid_triage_output())
        bad["duplicate_of"] = 42
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output.json"
            output.write_text(json.dumps(bad))
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=output,
                invocation=AgentInvocation(provider="codex", command=["true"]),
                context_comment_ids=("1",),
            )

            with patch("subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(["true"], 0)
                status = execute_triage_run(
                    config=config,
                    issue_number=7,
                    plan=plan,
                    github=github,  # type: ignore[arg-type]
                    issue_fields=issue_fields,  # type: ignore[arg-type]
                    final_attempt=False,
                )

        self.assertEqual(status, "invalid_output")
        # Not wedged: no Failed status, no apply, caller retries.
        self.assertEqual(issue_fields.writes[-1]["values"], {"Triage status": "Running"})
        self.assertNotIn("BugPatrol triage failed", "".join(github.comments))
        self.assertEqual(github.issue_types, [])

    def test_execute_triage_run_marks_failed_on_invalid_output_final_attempt(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        issue_fields = FakeIssueFields()
        bad = json.loads(valid_triage_output())
        bad["duplicate_of"] = 42
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output.json"
            output.write_text(json.dumps(bad))
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=output,
                invocation=AgentInvocation(provider="codex", command=["true"]),
                context_comment_ids=("1",),
            )

            with patch("subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(["true"], 0)
                with self.assertRaisesRegex(RuntimeError, "failed validation"):
                    execute_triage_run(
                        config=config,
                        issue_number=7,
                        plan=plan,
                        github=github,  # type: ignore[arg-type]
                        issue_fields=issue_fields,  # type: ignore[arg-type]
                    )

        # Terminal Failed status clears the wedged "Running".
        self.assertEqual(issue_fields.writes[-1]["values"], {"Triage status": "Failed"})
        self.assertIn("BugPatrol triage failed", "".join(github.comments))

    def test_render_triage_failed_comment_is_actionable(self) -> None:
        comment = render_triage_failed_comment(exit_code=2)

        self.assertIn("BugPatrol triage failed", comment)
        self.assertIn("credentials", comment)

    def test_list_known_assignees_merges_codeowners_and_config(self) -> None:
        config = load_project_config(Path("projects/full.example.toml"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".github").mkdir()
            (root / ".github" / "CODEOWNERS").write_text(
                "# Andy (@AndyCokeZero) owns notifications\n"
                "/app/notifications.tsx @AndyCokeZero\n"
                "/app/* @garlanddiego @org/some-team\n"
            )

            assignees = list_known_assignees(root, config=config)

        self.assertIn("AndyCokeZero", assignees)
        self.assertIn("garlanddiego", assignees)
        # Team handles cannot be issue assignees.
        self.assertNotIn("org/some-team", assignees)
        # Display names from comments never leak in.
        self.assertNotIn("Andy", assignees)

    def test_execute_triage_run_fails_on_unknown_assignee(self) -> None:
        config = load_project_config(Path("projects/full.example.toml"))
        github = FakeGithub()
        issue_fields = FakeIssueFields()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output.json"
            output.write_text(valid_triage_output())  # assignee: octocat
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=output,
                invocation=AgentInvocation(provider="codex", command=["true"]),
                context_comment_ids=("1",),
                known_assignees=("AndyCokeZero", "garlanddiego"),
            )

            with patch("subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(["true"], 0)
                with self.assertRaisesRegex(RuntimeError, "unknown assignee 'octocat'"):
                    execute_triage_run(
                        config=config,
                        issue_number=7,
                        plan=plan,
                        github=github,  # type: ignore[arg-type]
                        issue_fields=issue_fields,  # type: ignore[arg-type]
                    )

        self.assertEqual(github.assignees, [])
        self.assertIn("octocat", github.comments[-1])
        self.assertIn("AndyCokeZero", github.comments[-1])
        self.assertEqual(issue_fields.writes[-1]["values"], {"Triage status": "Failed"})


    def test_execute_triage_run_notifies_lark_on_running_and_failed(self) -> None:
        from bugpatrol.testing.fakes import FakeLarkMessengerClient

        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        issue_fields = FakeIssueFields()
        lark = FakeLarkMessengerClient()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = TriageRunPlan(
                context_path=root / "context.md",
                schema_path=root / "schema.json",
                output_path=root / "output.json",
                invocation=AgentInvocation(provider="codex", command=["false"]),
            )

            with patch("subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(["false"], 42)
                with self.assertRaisesRegex(RuntimeError, "exit 42"):
                    execute_triage_run(
                        config=config,
                        issue_number=7,
                        plan=plan,
                        github=github,  # type: ignore[arg-type]
                        issue_fields=issue_fields,  # type: ignore[arg-type]
                        lark=lark,
                    )

        self.assertEqual(len(lark.replies), 2)
        self.assertIn("开始分诊", lark.replies[0].text)
        self.assertEqual(lark.replies[0].message_id, "om_1")
        self.assertIn("分诊失败", lark.replies[1].text)


class ReportWorkflowFailureTest(unittest.TestCase):
    def test_reports_when_bugpatrol_never_marked_the_issue_failed(self) -> None:
        from bugpatrol.testing.fakes import FakeLarkMessengerClient

        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        issue_fields = FakeIssueFields()
        lark = FakeLarkMessengerClient()

        outcome = report_workflow_failure(
            config=config,
            issue_number=7,
            job="triage",
            github=github,  # type: ignore[arg-type]
            issue_fields=issue_fields,  # type: ignore[arg-type]
            lark=lark,
            run_url="https://github.test/runs/1",
            detail="runner: minici16g-bugpatrol",
        )

        self.assertEqual(outcome, "reported")
        self.assertEqual(issue_fields.writes[-1]["values"], {"Triage status": "Failed"})
        self.assertIn("https://github.test/runs/1", github.comments[-1])
        self.assertIn("minici16g-bugpatrol", github.comments[-1])
        self.assertEqual(len(lark.replies), 1)
        self.assertIn("分诊失败", lark.replies[0].text)

    def test_skips_when_triage_already_reported_its_own_failure(self) -> None:
        from bugpatrol.testing.fakes import FakeLarkMessengerClient

        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        issue_fields = FakeIssueFields({"Triage status": "Failed"})
        lark = FakeLarkMessengerClient()
        before = list(github.comments)

        outcome = report_workflow_failure(
            config=config,
            issue_number=7,
            job="triage",
            github=github,  # type: ignore[arg-type]
            issue_fields=issue_fields,  # type: ignore[arg-type]
            lark=lark,
        )

        self.assertEqual(outcome, "already_reported")
        self.assertEqual(github.comments, before)
        self.assertEqual(issue_fields.writes, [])
        self.assertEqual(lark.replies, [])

    def test_non_triage_job_reports_without_touching_triage_status(self) -> None:
        from bugpatrol.testing.fakes import FakeLarkMessengerClient

        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        issue_fields = FakeIssueFields()
        lark = FakeLarkMessengerClient()

        outcome = report_workflow_failure(
            config=config,
            issue_number=7,
            job="fix",
            github=github,  # type: ignore[arg-type]
            issue_fields=issue_fields,  # type: ignore[arg-type]
            lark=lark,
            run_url="https://github.test/runs/2",
        )

        self.assertEqual(outcome, "reported")
        self.assertEqual(issue_fields.writes, [])
        self.assertIn("BugPatrol fix failed", github.comments[-1])
        self.assertIn("https://github.test/runs/2", github.comments[-1])
        self.assertEqual(len(lark.replies), 1)
        self.assertIn("自动修复运行失败", lark.replies[0].text)

    def test_dedupes_on_the_run_url_already_cited_in_a_comment(self) -> None:
        from bugpatrol.testing.fakes import FakeLarkMessengerClient

        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGithub()
        github.comments.append("## BugPatrol fix failed\n\nhttps://github.test/runs/2")
        issue_fields = FakeIssueFields()
        lark = FakeLarkMessengerClient()
        before = list(github.comments)

        outcome = report_workflow_failure(
            config=config,
            issue_number=7,
            job="fix",
            github=github,  # type: ignore[arg-type]
            issue_fields=issue_fields,  # type: ignore[arg-type]
            lark=lark,
            run_url="https://github.test/runs/2",
        )

        self.assertEqual(outcome, "already_reported")
        self.assertEqual(github.comments, before)
        self.assertEqual(lark.replies, [])


def valid_triage_output() -> str:
    return """
    {
      "issue_type": "Bug",
      "priority": "High",
      "triage_status": "Done",
      "triage_verdict": "代码 Bug",
      "platform": "Web",
      "reproducibility": "必现",
      "other_platforms": "未验证",
      "capability": "Quest",
      "evidence": "文字描述",
      "prd_status": "已对齐",
      "triage_confidence": "高",
      "assignee": "octocat",
      "owner_reason": "Manual",
      "summary_cn": "空状态缺失",
      "follow_up_questions": [],
      "corrected_title": "删除全部 Todo 后缺少空状态",
      "comment_markdown": "## Triage Analysis\\n\\nLooks like a code bug."
    }
    """


def managed_issue_body() -> str:
    return render_issue_body(
        IntakeRecord(
            reporter_name="Reporter",
            reporter_open_id="ou_1",
            created_at="2026-07-01T00:00:00Z",
            chat_id="oc_1",
            root_id="om_root",
            message_id="om_1",
            original_text="Deleting all todos does not show empty state.",
        )
    )


class TriageWorkflowTemplateTest(unittest.TestCase):
    def test_example_workflow_reports_job_failures(self) -> None:
        # A triage job that dies during provisioning reports nothing on its own,
        # leaving the issue silently untriaged. Guard that the shipped template
        # keeps the failure-reporting step, since deployments are copied from it.
        workflow = Path("examples/github-actions/bugpatrol-triage.yml").read_text(encoding="utf-8")
        self.assertIn("if: failure()", workflow)
        self.assertIn("report-job-failure", workflow)


class TriagePromptRubricTest(unittest.TestCase):
    def test_prompt_carries_priority_rubric(self) -> None:
        # Without an explicit rubric the agent guesses priority from the bare
        # enum and mislabels functional failures as Low. Guard that every
        # priority level plus the "don't default a functional failure to Low"
        # rule stays in the prompt so a future edit can't silently drop it.
        prompt = Path("prompts/triage.zh.md").read_text(encoding="utf-8")
        for level in ("Urgent", "High", "Medium", "Low"):
            self.assertIn(f"`{level}`", prompt)
        self.assertIn("关键按钮点击无效", prompt)
        self.assertIn("功能性失效", prompt)
        self.assertIn("不要判 `Low`", prompt)


if __name__ == "__main__":
    unittest.main()
