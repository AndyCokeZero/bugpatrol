from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from bugpatrol.agents import AgentInvocation
from bugpatrol.clients import (
    FailedRun,
    GitHubIssue,
    GitHubIssueComment,
    OpenPullRequest,
    ReviewComment,
    ReviewThread,
)
from bugpatrol.config import load_project_config
from bugpatrol.fix_runner import (
    FixRunPlan,
    _closing_issue_candidates,
    _run_fix_agent,
    execute_fix_revise,
    execute_fix_run,
    latest_reporter_correction,
    latest_triage_analysis,
    prepare_fix_run,
    read_triage_verdict,
    render_fix_context_markdown,
    run_ci_feedback,
    run_fix,
    run_fix_revise,
)
from bugpatrol.fix_result import append_ci_fix_metadata
from bugpatrol.intake import IntakeRecord, render_issue_body
from bugpatrol.intake_workflow import render_followup_comment
from bugpatrol.triage_result import append_triage_metadata


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


class FakeGithub:
    def __init__(
        self,
        *,
        comments=None,
        open_pr: str = "",
        assignees=(),
        open_pull_request: OpenPullRequest | None = None,
        review_threads=(),
        pr_comment_bodies=(),
        failed_runs=(),
        failed_logs=None,
        failed_check_runs=(),
        state: str = "open",
    ) -> None:
        self.state = state
        self.comments = list(comments or [])
        self.added_comments: list[str] = []
        self.open_pr = open_pr
        self.created_prs: list[dict] = []
        self.reviewers: list[str] = []
        self.assignees = tuple(assignees)
        self.open_pull_request = open_pull_request
        self.review_threads = tuple(review_threads)
        self.resolved_threads: list[str] = []
        self.pr_comments: list[dict] = []
        self.pr_comment_bodies = list(pr_comment_bodies)
        self.failed_runs = tuple(failed_runs)
        self.failed_logs = dict(failed_logs or {})
        self.failed_check_runs = tuple(failed_check_runs)

    def get_issue(self, *, repo: str, issue_number: int) -> GitHubIssue:
        return GitHubIssue(
            number=issue_number,
            url=f"https://github.test/{repo}/issues/{issue_number}",
            title="Todo empty state missing",
            body=managed_issue_body(),
            assignees=self.assignees,
            state=self.state,
        )

    def list_issue_comments(self, *, repo: str, issue_number: int):
        return tuple(GitHubIssueComment(id=str(i + 1), body=b) for i, b in enumerate(self.comments))

    def add_issue_comment(self, *, repo: str, issue_number: int, body: str) -> None:
        self.added_comments.append(body)

    def find_open_pull_request_by_head(self, *, repo: str, head: str) -> str:
        return self.open_pr

    def create_pull_request(self, *, repo, head, base, title, body) -> str:
        self.created_prs.append({"head": head, "base": base, "title": title, "body": body})
        return "https://github.test/o/r/pull/9"

    def add_pull_request_reviewer(self, *, repo, pr, reviewer) -> None:
        self.reviewers.append(reviewer)

    def get_open_pull_request_by_head(self, *, repo, head) -> OpenPullRequest | None:
        return self.open_pull_request

    def list_unresolved_review_threads(self, *, repo, pr_number):
        return self.review_threads

    def resolve_review_thread(self, *, thread_id) -> None:
        self.resolved_threads.append(thread_id)

    def add_pull_request_comment(self, *, repo, pr, body) -> None:
        self.pr_comments.append({"pr": pr, "body": body})
        # A posted meta marker becomes visible to a subsequent read (de-dupe).
        self.pr_comment_bodies.append(body)

    def list_pull_request_comments(self, *, repo, pr_number):
        return tuple(
            GitHubIssueComment(id=str(i + 1), body=b)
            for i, b in enumerate(self.pr_comment_bodies)
        )

    def list_failed_runs_for_sha(self, *, repo, head_sha):
        return self.failed_runs

    def list_failed_check_runs_for_sha(self, *, repo, head_sha):
        return self.failed_check_runs

    def get_run_failed_logs(self, *, repo, run_id):
        return self.failed_logs.get(run_id, "")


class FakeIssueFields:
    def __init__(self, verdict: str = "代码 Bug") -> None:
        self.verdict = verdict

    def get_issue_field_values(self, *, repo: str, issue_number: int) -> dict[str, str]:
        return {"Triage verdict": self.verdict}


def _sandbox_config():
    return load_project_config(Path("projects/todo-sandbox.toml"))


class PureHelpersTest(unittest.TestCase):
    def test_read_triage_verdict(self) -> None:
        config = _sandbox_config()
        self.assertEqual(
            read_triage_verdict(config=config, issue_number=7, issue_fields=FakeIssueFields("代码 Bug")),
            "代码 Bug",
        )

    def test_latest_triage_analysis_picks_marked_comment(self) -> None:
        plain = GitHubIssueComment(id="1", body="just chatter")
        triage = GitHubIssueComment(id="2", body=append_triage_metadata("## Triage\n根因分析", {"version": 1}))
        newer = GitHubIssueComment(id="3", body=append_triage_metadata("## Triage\n最新根因", {"version": 1}))
        self.assertIn("最新根因", latest_triage_analysis([plain, triage, newer]))

    def test_render_fix_context_markdown(self) -> None:
        issue = GitHubIssue(number=7, url="u", title="标题", body="正文")
        text = render_fix_context_markdown(
            issue=issue, verdict="代码 Bug", triage_analysis="根因是X", branch_note="目标分支：`feature-demo`"
        )
        self.assertIn("代码 Bug", text)
        self.assertIn("根因是X", text)
        self.assertIn("feature-demo", text)
        self.assertIn("正文", text)


class PrepareFixRunTest(unittest.TestCase):
    def test_writes_context_schema_and_absolute_paths(self) -> None:
        config = replace(_sandbox_config(), triage_agent=replace(_sandbox_config().triage_agent, provider="claude"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worktree = root / "wt"
            worktree.mkdir()
            output_dir = root / "out"
            plan = prepare_fix_run(
                config=config,
                issue_number=7,
                worktree_path=worktree,
                output_dir=output_dir,
                github=FakeGithub(comments=[append_triage_metadata("根因", {"version": 1})], assignees=("dev1",)),  # type: ignore[arg-type]
                issue_fields=FakeIssueFields(),  # type: ignore[arg-type]
                base_branch="feature-demo",
                head_branch="bugpatrol/fix-issue-7",
            )
            self.assertTrue(plan.context_path.exists())
            self.assertTrue(plan.schema_path.exists())
            self.assertEqual(plan.output_path, output_dir.resolve() / "fix-output.json")
            self.assertEqual(plan.agent_cwd, worktree.resolve())
            self.assertEqual(plan.verdict, "代码 Bug")
            self.assertEqual(plan.reviewer, "dev1")
            context = plan.context_path.read_text()
            self.assertIn("根因", context)
            # The agent is told the exact self-check commands (from [fix.verify]).
            self.assertIn("自检命令", context)
            self.assertIn("npm run typecheck", context)


class RunFixGuardsTest(unittest.TestCase):
    def test_not_fixable_verdict_posts_blocked(self) -> None:
        config = _sandbox_config()
        github = FakeGithub()
        with tempfile.TemporaryDirectory() as tmp:
            status = run_fix(
                config=config,
                issue_number=7,
                base_repo=Path(tmp),
                output_dir=Path(tmp) / "out",
                github=github,  # type: ignore[arg-type]
                issue_fields=FakeIssueFields("信息不足"),  # type: ignore[arg-type]
            )
        self.assertEqual(status, "not_fixable")
        self.assertTrue(github.added_comments)
        self.assertIn("跳过", github.added_comments[0])

    def test_already_open_pr_short_circuits(self) -> None:
        config = _sandbox_config()
        github = FakeGithub(open_pr="https://github.test/o/r/pull/1")
        with tempfile.TemporaryDirectory() as tmp:
            status = run_fix(
                config=config,
                issue_number=7,
                base_repo=Path(tmp),
                output_dir=Path(tmp) / "out",
                github=github,  # type: ignore[arg-type]
                issue_fields=FakeIssueFields("代码 Bug"),  # type: ignore[arg-type]
            )
        self.assertEqual(status, "already_open_pr")
        self.assertFalse(github.created_prs)

    def test_closed_issue_skips(self) -> None:
        config = _sandbox_config()
        github = FakeGithub(state="closed")
        with tempfile.TemporaryDirectory() as tmp:
            status = run_fix(
                config=config,
                issue_number=7,
                base_repo=Path(tmp),
                output_dir=Path(tmp) / "out",
                github=github,  # type: ignore[arg-type]
                issue_fields=FakeIssueFields("代码 Bug"),  # type: ignore[arg-type]
            )
        self.assertEqual(status, "issue_closed")
        self.assertFalse(github.created_prs)
        self.assertFalse(github.added_comments)


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t.test"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)
    (path / "src").mkdir()
    (path / "src" / "todo.ts").write_text("export const x = 1\n")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "init"], check=True)


def _add_worktree(root: Path) -> Path:
    worktree = root / ".bugpatrol-worktrees" / "wt"
    subprocess.run(
        ["git", "-C", str(root), "worktree", "add", "-b", "bugpatrol/fix-issue-7", str(worktree), "HEAD"],
        check=True,
        capture_output=True,
    )
    return worktree


class ExecuteFixRunTest(unittest.TestCase):
    def _plan(self, *, worktree: Path, output_dir: Path, verify: dict[str, str], command, setup=None):
        # A real agent command (never a subprocess.run patch): globally patching
        # subprocess.run would also stub the real git calls this test needs.
        # setup defaults to {} so tests don't shell out to a real `npm install`.
        config = _sandbox_config()
        config = replace(config, fix=replace(config.fix, verify=verify, setup=setup or {}))
        plan = FixRunPlan(
            context_path=output_dir / "fix-context.md",
            schema_path=output_dir / "fix.schema.json",
            output_path=output_dir / "fix-output.json",
            invocation=AgentInvocation(provider="deepseek", command=command, env={}, model="m"),
            agent_cwd=worktree,
            verdict="代码 Bug",
            base_branch="main",
            head_branch="bugpatrol/fix-issue-7",
            reviewer="dev1",
            reviewer_open_id="",
            branch_note="",
        )
        return config, plan

    @staticmethod
    def _edit_and_write_output(worktree: Path, output_path: Path) -> list[str]:
        payload = json.dumps(
            {
                "summary": "给空列表加空状态",
                "root_cause": "删除最后一条没渲染空态",
                "tests_added": True,
                "pr_title": "fix: 空状态",
                "pr_body": "修复空状态",
            }
        )
        script = (
            f"import json,pathlib;"
            f"pathlib.Path({str(worktree / 'src' / 'todo.ts')!r}).write_text('export const x = 2\\n');"
            f"pathlib.Path({str(output_path)!r}).write_text({payload!r})"
        )
        return ["python3", "-c", script]

    def test_blocked_when_no_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_repo(root)
            worktree = _add_worktree(root)
            output_dir = root / "out"
            output_dir.mkdir()
            # Agent that makes no edits at all.
            config, plan = self._plan(worktree=worktree, output_dir=output_dir, verify={"ok": "true"}, command=["true"])
            github = FakeGithub()
            status = execute_fix_run(
                config=config,
                issue=github.get_issue(repo=config.github_repo, issue_number=7),
                plan=plan,
                github=github,  # type: ignore[arg-type]
            )
            self.assertEqual(status, "no_changes")
            self.assertFalse(github.created_prs)

    def test_verify_failed_posts_and_opens_no_pr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_repo(root)
            worktree = _add_worktree(root)
            output_dir = root / "out"
            output_dir.mkdir()
            command = self._edit_and_write_output(worktree, output_dir / "fix-output.json")
            # The gate passes on the pristine baseline (`x = 1`) but fails once the
            # fix flips it to `x = 2` — so the failure is the fix's fault.
            config, plan = self._plan(
                worktree=worktree,
                output_dir=output_dir,
                verify={"guard": "grep -q 'x = 1' src/todo.ts"},
                command=command,
            )
            github = FakeGithub()
            status = execute_fix_run(
                config=config,
                issue=github.get_issue(repo=config.github_repo, issue_number=7),
                plan=plan,
                github=github,  # type: ignore[arg-type]
            )
            self.assertEqual(status, "verify_failed")
            self.assertFalse(github.created_prs)
            self.assertTrue(any("未通过验证" in c for c in github.added_comments))

    def test_baseline_broken_reports_and_does_not_blame_fix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_repo(root)
            worktree = _add_worktree(root)
            output_dir = root / "out"
            output_dir.mkdir()
            command = self._edit_and_write_output(worktree, output_dir / "fix-output.json")
            # A gate that fails regardless of the edit: after the post-fix run
            # fails, resetting to the pristine baseline and re-running still fails,
            # so the target branch itself is red — don't blame the fix.
            config, plan = self._plan(
                worktree=worktree,
                output_dir=output_dir,
                verify={"guard": "false"},
                command=command,
            )
            github = FakeGithub()
            status = execute_fix_run(
                config=config,
                issue=github.get_issue(repo=config.github_repo, issue_number=7),
                plan=plan,
                github=github,  # type: ignore[arg-type]
            )
            self.assertEqual(status, "baseline_broken")
            self.assertFalse(github.created_prs)
            self.assertTrue(any("baseline 本就红" in c for c in github.added_comments))
            self.assertFalse(any("未通过验证" in c for c in github.added_comments))
            # The worktree was reset back to the pristine baseline (`x = 1`).
            self.assertEqual((worktree / "src" / "todo.ts").read_text(), "export const x = 1\n")

    def test_opened_pr_on_clean_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_repo(root)
            worktree = _add_worktree(root)
            output_dir = root / "out"
            output_dir.mkdir()
            command = self._edit_and_write_output(worktree, output_dir / "fix-output.json")
            config, plan = self._plan(worktree=worktree, output_dir=output_dir, verify={"ok": "true"}, command=command)
            github = FakeGithub(assignees=("dev1",))
            with patch("bugpatrol.fix_runner.worktree_push_branch") as push:
                status = execute_fix_run(
                    config=config,
                    issue=github.get_issue(repo=config.github_repo, issue_number=7),
                    plan=plan,
                    github=github,  # type: ignore[arg-type]
                )
            self.assertEqual(status, "opened_pr")
            self.assertEqual(len(github.created_prs), 1)
            self.assertEqual(github.created_prs[0]["base"], "main")
            self.assertEqual(github.created_prs[0]["head"], "bugpatrol/fix-issue-7")
            self.assertIn("dev1", github.reviewers)
            push.assert_called_once()

    @staticmethod
    def _self_heal_command(worktree: Path, output_path: Path, counter_path: Path) -> list[str]:
        # Attempt 1 drops a `FAIL` marker so the verify guard `test ! -f FAIL`
        # fails (fix's fault; baseline is green). Attempt 2 omits it and passes.
        # The counter lives OUTSIDE the worktree so it survives the attribution
        # reset (reset --hard + clean -fd) between attempts.
        payload = json.dumps(
            {
                "summary": "修复",
                "root_cause": "根因",
                "tests_added": True,
                "pr_title": "fix: x",
                "pr_body": "body",
            }
        )
        script = (
            f"import json,pathlib;"
            f"c=pathlib.Path({str(counter_path)!r});"
            f"n=(int(c.read_text())+1) if c.exists() else 1;"
            f"c.write_text(str(n));"
            f"wt=pathlib.Path({str(worktree)!r});"
            f"(wt/'src'/'todo.ts').write_text('export const x = 2\\n');"
            f"(wt/'FAIL').write_text('x') if n==1 else None;"
            f"pathlib.Path({str(output_path)!r}).write_text({payload!r})"
        )
        return ["python3", "-c", script]

    def test_verify_self_heal_retries_then_opens_pr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_repo(root)
            worktree = _add_worktree(root)
            output_dir = root / "out"
            output_dir.mkdir()
            counter = output_dir / "attempts.txt"
            command = self._self_heal_command(worktree, output_dir / "fix-output.json", counter)
            config, plan = self._plan(
                worktree=worktree,
                output_dir=output_dir,
                verify={"guard": "test ! -f FAIL"},
                command=command,
            )
            github = FakeGithub(assignees=("dev1",))
            with patch("bugpatrol.fix_runner.worktree_push_branch") as push:
                status = execute_fix_run(
                    config=config,
                    issue=github.get_issue(repo=config.github_repo, issue_number=7),
                    plan=plan,
                    github=github,  # type: ignore[arg-type]
                )
            # First attempt failed verify; the second self-healed and opened a PR.
            self.assertEqual(status, "opened_pr")
            self.assertEqual(counter.read_text(), "2")
            self.assertEqual(len(github.created_prs), 1)
            push.assert_called_once()
            # The failure was fed back to the agent before the retry.
            self.assertIn("preflight", plan.context_path.read_text())
            # No premature verify_failed comment on the issue.
            self.assertFalse(any("未通过验证" in c for c in github.added_comments))

    def test_no_self_heal_when_attempts_is_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_repo(root)
            worktree = _add_worktree(root)
            output_dir = root / "out"
            output_dir.mkdir()
            counter = output_dir / "attempts.txt"
            command = self._self_heal_command(worktree, output_dir / "fix-output.json", counter)
            config, plan = self._plan(
                worktree=worktree,
                output_dir=output_dir,
                verify={"guard": "test ! -f FAIL"},
                command=command,
            )
            config = replace(config, fix=replace(config.fix, max_verify_fix_attempts=1))
            github = FakeGithub()
            status = execute_fix_run(
                config=config,
                issue=github.get_issue(repo=config.github_repo, issue_number=7),
                plan=plan,
                github=github,  # type: ignore[arg-type]
            )
            # Single-shot: no retry, so it dead-ends on verify_failed with no PR.
            self.assertEqual(status, "verify_failed")
            self.assertEqual(counter.read_text(), "1")
            self.assertFalse(github.created_prs)

    def test_setup_failure_reports_baseline_broken_and_skips_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_repo(root)
            worktree = _add_worktree(root)
            output_dir = root / "out"
            output_dir.mkdir()
            counter = output_dir / "attempts.txt"
            command = self._self_heal_command(worktree, output_dir / "fix-output.json", counter)
            # Setup (deps install) fails -> the base checkout can't be built, so
            # it's an environment/baseline problem: no PR, no agent turn.
            config, plan = self._plan(
                worktree=worktree,
                output_dir=output_dir,
                verify={"ok": "true"},
                command=command,
                setup={"install": "false"},
            )
            github = FakeGithub()
            status = execute_fix_run(
                config=config,
                issue=github.get_issue(repo=config.github_repo, issue_number=7),
                plan=plan,
                github=github,  # type: ignore[arg-type]
            )
            self.assertEqual(status, "setup_failed")
            self.assertFalse(github.created_prs)
            self.assertTrue(any("baseline 本就红" in c for c in github.added_comments))
            # The agent never ran (no counter written, no edit).
            self.assertFalse(counter.exists())
            self.assertEqual((worktree / "src" / "todo.ts").read_text(), "export const x = 1\n")

    def test_setup_runs_before_agent_then_opens_pr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_repo(root)
            worktree = _add_worktree(root)
            output_dir = root / "out"
            output_dir.mkdir()
            command = self._edit_and_write_output(worktree, output_dir / "fix-output.json")
            # Setup writes a marker the verify gate depends on: the gate only
            # passes because setup ran first, proving setup precedes verify.
            config, plan = self._plan(
                worktree=worktree,
                output_dir=output_dir,
                verify={"needs_setup": "test -f DEPS_READY"},
                command=command,
                setup={"install": "touch DEPS_READY"},
            )
            github = FakeGithub(assignees=("dev1",))
            with patch("bugpatrol.fix_runner.worktree_push_branch") as push:
                status = execute_fix_run(
                    config=config,
                    issue=github.get_issue(repo=config.github_repo, issue_number=7),
                    plan=plan,
                    github=github,  # type: ignore[arg-type]
                )
            self.assertEqual(status, "opened_pr")
            self.assertEqual(len(github.created_prs), 1)
            push.assert_called_once()


class RunFixReviseGuardsTest(unittest.TestCase):
    def test_no_open_pr(self) -> None:
        config = _sandbox_config()
        github = FakeGithub(open_pull_request=None)
        with tempfile.TemporaryDirectory() as tmp:
            status = run_fix_revise(
                config=config,
                issue_number=7,
                base_repo=Path(tmp),
                output_dir=Path(tmp) / "out",
                github=github,  # type: ignore[arg-type]
                issue_fields=FakeIssueFields("代码 Bug"),  # type: ignore[arg-type]
            )
        self.assertEqual(status, "no_open_pr")

    def test_no_feedback_when_no_unresolved_threads(self) -> None:
        config = _sandbox_config()
        github = FakeGithub(
            open_pull_request=OpenPullRequest(number=9, url="https://github.test/o/r/pull/9"),
            review_threads=(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            status = run_fix_revise(
                config=config,
                issue_number=7,
                base_repo=Path(tmp),
                output_dir=Path(tmp) / "out",
                github=github,  # type: ignore[arg-type]
                issue_fields=FakeIssueFields("代码 Bug"),  # type: ignore[arg-type]
            )
        self.assertEqual(status, "no_feedback")
        self.assertFalse(github.resolved_threads)

    def test_closed_issue_skips(self) -> None:
        config = _sandbox_config()
        github = FakeGithub(
            state="closed",
            open_pull_request=OpenPullRequest(number=9, url="https://github.test/o/r/pull/9"),
            review_threads=(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            status = run_fix_revise(
                config=config,
                issue_number=7,
                base_repo=Path(tmp),
                output_dir=Path(tmp) / "out",
                github=github,  # type: ignore[arg-type]
                issue_fields=FakeIssueFields("代码 Bug"),  # type: ignore[arg-type]
            )
        self.assertEqual(status, "issue_closed")
        self.assertFalse(github.resolved_threads)


class RunFixAgentTimeoutTest(unittest.TestCase):
    def test_run_fix_agent_fails_fast_when_llm_call_hangs(self) -> None:
        # Same latent gap as the triage agent (#4938): a hung provider call used
        # to burn the whole fix workflow job timeout silently. The subprocess
        # timeout must fail fast with a clear error instead.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = FixRunPlan(
                context_path=root / "fix-context.md",
                schema_path=root / "fix.schema.json",
                output_path=root / "fix-output.json",
                invocation=AgentInvocation(provider="deepseek", command=["claude"], env={}, model="m"),
                agent_cwd=root,
                verdict="代码 Bug",
                base_branch="main",
                head_branch="bugpatrol/fix-issue-7",
                reviewer="dev1",
                reviewer_open_id="",
                branch_note="",
            )
            # Real TimeoutExpired carries raw bytes even under text=True; the
            # turn-log write must survive that (bytes-vs-str crash, #5038).
            timeout = subprocess.TimeoutExpired(cmd=["claude"], timeout=1200, output=b"partial", stderr=b"")
            timeout.stdout = b"partial"
            with patch("bugpatrol.fix_runner.subprocess.run", side_effect=timeout):
                with self.assertRaisesRegex(RuntimeError, "fix agent timed out after 1200s"):
                    _run_fix_agent(plan)


def _followup_reply(text: str, reason: str, message_id: str = "om_x") -> str:
    record = IntakeRecord(
        reporter_name="Reporter",
        reporter_open_id="ou_1",
        created_at="2026-07-01T00:00:00Z",
        chat_id="oc_1",
        root_id="om_root",
        message_id=message_id,
        original_text=text,
    )
    return render_followup_comment(record, language="zh-CN", signal_reason=reason)


class LatestReporterCorrectionTest(unittest.TestCase):
    def test_picks_latest_material_and_strips_meta(self) -> None:
        comments = [
            GitHubIssueComment(id="1", body=_followup_reply("旧的补充", "material_followup", "om_a")),
            GitHubIssueComment(id="2", body=_followup_reply("收到，谢谢", "acknowledgement", "om_b")),
            GitHubIssueComment(
                id="3", body=_followup_reply("其实是标签上下位置不统一", "material_followup", "om_c")
            ),
        ]
        got = latest_reporter_correction(comments)
        self.assertIn("其实是标签上下位置不统一", got)
        # Only the newest material correction wins; older material is dropped.
        self.assertNotIn("旧的补充", got)
        # The machine-readable footer must not reach the agent.
        self.assertNotIn("BUGPATROL_INTAKE_REPLY_META", got)

    def test_ignores_acks_and_non_intake_comments(self) -> None:
        comments = [
            GitHubIssueComment(id="1", body=_followup_reply("收到", "acknowledgement", "om_a")),
            GitHubIssueComment(id="2", body="随手写的普通评论，不是 intake 回复"),
        ]
        self.assertEqual(latest_reporter_correction(comments), "")


class RunFixReviseReporterFeedbackTest(unittest.TestCase):
    def test_material_reporter_followup_triggers_revise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_repo = _make_fix_remote(root)
            github = FakeGithub(
                comments=[_followup_reply("其实是标签上下位置不统一", "material_followup")],
                open_pull_request=OpenPullRequest(
                    number=9,
                    url="https://github.test/o/r/pull/9",
                    base_ref="main",
                    mergeable="MERGEABLE",
                ),
                review_threads=(),
            )
            with patch("bugpatrol.fix_runner.prepare_fix_revise", return_value=object()) as prep:
                with patch(
                    "bugpatrol.fix_runner.execute_fix_revise", return_value="revised"
                ) as ex:
                    status = run_fix_revise(
                        config=_sandbox_config(),
                        issue_number=7,
                        base_repo=base_repo,
                        output_dir=root / "out",
                        github=github,  # type: ignore[arg-type]
                        issue_fields=FakeIssueFields("代码 Bug"),  # type: ignore[arg-type]
                    )
        self.assertEqual(status, "revised")
        # The reporter's stripped correction flows into both the context builder
        # and the executor (as the reporter_feedback signal).
        self.assertIn("其实是标签上下位置不统一", prep.call_args.kwargs["reporter_feedback"])
        self.assertTrue(ex.call_args.kwargs["reporter_feedback"])

    def test_ack_only_followup_is_no_feedback(self) -> None:
        github = FakeGithub(
            comments=[_followup_reply("收到，谢谢", "acknowledgement")],
            open_pull_request=OpenPullRequest(
                number=9,
                url="https://github.test/o/r/pull/9",
                base_ref="main",
                mergeable="MERGEABLE",
            ),
            review_threads=(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch("bugpatrol.fix_runner.execute_fix_revise") as ex:
                status = run_fix_revise(
                    config=_sandbox_config(),
                    issue_number=7,
                    base_repo=Path(tmp),
                    output_dir=Path(tmp) / "out",
                    github=github,  # type: ignore[arg-type]
                    issue_fields=FakeIssueFields("代码 Bug"),  # type: ignore[arg-type]
                )
        self.assertEqual(status, "no_feedback")
        ex.assert_not_called()


class ExecuteFixReviseTest(unittest.TestCase):
    def _plan(self, *, worktree: Path, output_dir: Path, verify: dict[str, str], command, setup=None):
        config = _sandbox_config()
        config = replace(config, fix=replace(config.fix, verify=verify, setup=setup or {}))
        plan = FixRunPlan(
            context_path=output_dir / "fix-context.md",
            schema_path=output_dir / "fix.schema.json",
            output_path=output_dir / "fix-output.json",
            invocation=AgentInvocation(provider="deepseek", command=command, env={}, model="m"),
            agent_cwd=worktree,
            verdict="代码 Bug",
            base_branch="main",
            head_branch="bugpatrol/fix-issue-7",
            reviewer="dev1",
            reviewer_open_id="",
            branch_note="",
        )
        return config, plan

    def _threads(self):
        return (
            ReviewThread(
                id="RT_1",
                comments=(ReviewComment(author="rev", body="这里改小一点", path="src/todo.ts", line=1),),
            ),
        )

    def test_revised_pushes_comments_and_resolves_threads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_repo(root)
            worktree = _add_worktree(root)
            output_dir = root / "out"
            output_dir.mkdir()
            command = ExecuteFixRunTest._edit_and_write_output(worktree, output_dir / "fix-output.json")
            config, plan = self._plan(worktree=worktree, output_dir=output_dir, verify={"ok": "true"}, command=command)
            threads = self._threads()
            pr = OpenPullRequest(number=9, url="https://github.test/o/r/pull/9")
            github = FakeGithub()
            with patch("bugpatrol.fix_runner.worktree_push_branch") as push:
                status = execute_fix_revise(
                    config=config,
                    issue=github.get_issue(repo=config.github_repo, issue_number=7),
                    plan=plan,
                    pr=pr,
                    threads=threads,
                    github=github,  # type: ignore[arg-type]
                )
            self.assertEqual(status, "revised")
            push.assert_called_once()
            # No new PR is created on revise; the existing branch is updated.
            self.assertFalse(github.created_prs)
            self.assertEqual(github.resolved_threads, ["RT_1"])
            self.assertTrue(any("已处理 1 条评审意见" in c["body"] for c in github.pr_comments))

    def test_verify_failed_does_not_resolve_threads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_repo(root)
            worktree = _add_worktree(root)
            output_dir = root / "out"
            output_dir.mkdir()
            command = ExecuteFixRunTest._edit_and_write_output(worktree, output_dir / "fix-output.json")
            config, plan = self._plan(worktree=worktree, output_dir=output_dir, verify={"bad": "false"}, command=command)
            github = FakeGithub()
            status = execute_fix_revise(
                config=config,
                issue=github.get_issue(repo=config.github_repo, issue_number=7),
                plan=plan,
                pr=OpenPullRequest(number=9, url="https://github.test/o/r/pull/9"),
                threads=self._threads(),
                github=github,  # type: ignore[arg-type]
            )
            self.assertEqual(status, "verify_failed")
            # Marker-last: a failed verify must leave threads unresolved for retry.
            self.assertFalse(github.resolved_threads)
            self.assertFalse(github.pr_comments)


class ExecuteFixReviseConflictTest(unittest.TestCase):
    @staticmethod
    def _write_marker_file_and_output(worktree: Path, output_path: Path) -> list[str]:
        """Agent that leaves conflict markers in place (a failed resolution)."""
        payload = json.dumps(
            {
                "summary": "s",
                "root_cause": "r",
                "tests_added": False,
                "pr_title": "t",
                "pr_body": "b",
            }
        )
        marker = "<<<<<<< HEAD\\nfix\\n=======\\nmoved\\n>>>>>>> origin/main\\n"
        script = (
            f"import pathlib;"
            f"pathlib.Path({str(worktree / 'src' / 'todo.ts')!r}).write_text({marker!r});"
            f"pathlib.Path({str(output_path)!r}).write_text({payload!r})"
        )
        return ["python3", "-c", script]

    def test_conflict_and_feedback_resolved_skips_gate_and_resolves_threads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_repo(root)
            worktree = _add_worktree(root)
            output_dir = root / "out"
            output_dir.mkdir()
            command = ExecuteFixRunTest._edit_and_write_output(worktree, output_dir / "fix-output.json")
            config, plan = ExecuteFixReviseTest()._plan(
                worktree=worktree, output_dir=output_dir, verify={"ok": "true"}, command=command
            )
            # A tiny max_diff would block a normal revise, but the conflict path
            # skips the diff gate — assert it does not trip here.
            config = replace(config, fix=replace(config.fix, max_diff_lines=1))
            threads = ExecuteFixReviseTest()._threads()
            pr = OpenPullRequest(number=9, url="https://github.test/o/r/pull/9", base_ref="main", mergeable="CONFLICTING")
            github = FakeGithub()
            with patch("bugpatrol.fix_runner.worktree_push_branch") as push:
                status = execute_fix_revise(
                    config=config,
                    issue=github.get_issue(repo=config.github_repo, issue_number=7),
                    plan=plan,
                    pr=pr,
                    threads=threads,
                    github=github,  # type: ignore[arg-type]
                    has_conflict=True,
                    conflict_files=("src/todo.ts",),
                )
            self.assertEqual(status, "revised")
            push.assert_called_once()
            self.assertEqual(github.resolved_threads, ["RT_1"])

    def test_conflict_only_resolved_returns_conflict_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_repo(root)
            worktree = _add_worktree(root)
            output_dir = root / "out"
            output_dir.mkdir()
            command = ExecuteFixRunTest._edit_and_write_output(worktree, output_dir / "fix-output.json")
            config, plan = ExecuteFixReviseTest()._plan(
                worktree=worktree, output_dir=output_dir, verify={"ok": "true"}, command=command
            )
            pr = OpenPullRequest(number=9, url="https://github.test/o/r/pull/9", base_ref="main", mergeable="CONFLICTING")
            github = FakeGithub()
            with patch("bugpatrol.fix_runner.worktree_push_branch") as push:
                status = execute_fix_revise(
                    config=config,
                    issue=github.get_issue(repo=config.github_repo, issue_number=7),
                    plan=plan,
                    pr=pr,
                    threads=(),
                    github=github,  # type: ignore[arg-type]
                    has_conflict=True,
                    conflict_files=("src/todo.ts",),
                )
            self.assertEqual(status, "conflict_resolved")
            push.assert_called_once()
            self.assertFalse(github.resolved_threads)
            self.assertTrue(any("解决冲突" in c["body"] for c in github.pr_comments))

    def test_leftover_conflict_markers_block_and_do_not_push(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_repo(root)
            worktree = _add_worktree(root)
            output_dir = root / "out"
            output_dir.mkdir()
            command = self._write_marker_file_and_output(worktree, output_dir / "fix-output.json")
            config, plan = ExecuteFixReviseTest()._plan(
                worktree=worktree, output_dir=output_dir, verify={"ok": "true"}, command=command
            )
            pr = OpenPullRequest(number=9, url="https://github.test/o/r/pull/9", base_ref="main", mergeable="CONFLICTING")
            github = FakeGithub()
            with patch("bugpatrol.fix_runner.worktree_push_branch") as push:
                status = execute_fix_revise(
                    config=config,
                    issue=github.get_issue(repo=config.github_repo, issue_number=7),
                    plan=plan,
                    pr=pr,
                    threads=(),
                    github=github,  # type: ignore[arg-type]
                    has_conflict=True,
                    conflict_files=("src/todo.ts",),
                )
            self.assertEqual(status, "conflict_unresolved")
            push.assert_not_called()
            self.assertTrue(any("冲突标记" in c for c in github.added_comments))


def _make_conflicting_fix_remote(root: Path) -> Path:
    """origin with a pushed fix branch that conflicts with an advanced main.

    Returns a `base_repo` clone (origin remote set) ready for fix_revise_worktree
    + worktree_merge_base to fetch and merge, producing a 2-file conflict.
    """
    origin = root / "origin"
    origin.mkdir()
    subprocess.run(["git", "-C", str(origin), "init", "-q", "-b", "main"], check=True)
    for k, v in (("user.email", "t@t.test"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(origin), "config", k, v], check=True)
    (origin / "a.txt").write_text("base\n")
    (origin / "b.txt").write_text("base\n")
    subprocess.run(["git", "-C", str(origin), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(origin), "commit", "-q", "-m", "init"], check=True)
    # fix branch edits both files
    subprocess.run(["git", "-C", str(origin), "checkout", "-q", "-b", "bugpatrol/fix-issue-7"], check=True)
    (origin / "a.txt").write_text("fix\n")
    (origin / "b.txt").write_text("fix\n")
    subprocess.run(["git", "-C", str(origin), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(origin), "commit", "-q", "-m", "fix"], check=True)
    # main advances on the SAME lines -> conflict on both files
    subprocess.run(["git", "-C", str(origin), "checkout", "-q", "main"], check=True)
    (origin / "a.txt").write_text("moved\n")
    (origin / "b.txt").write_text("moved\n")
    subprocess.run(["git", "-C", str(origin), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(origin), "commit", "-q", "-m", "advance"], check=True)

    base_repo = root / "base"
    subprocess.run(["git", "clone", "-q", str(origin), str(base_repo)], check=True, capture_output=True)
    for k, v in (("user.email", "t@t.test"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(base_repo), "config", k, v], check=True)
    return base_repo


class RunFixReviseConflictEscalationTest(unittest.TestCase):
    def test_too_many_conflict_files_escalates_to_human(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_repo = _make_conflicting_fix_remote(root)
            config = _sandbox_config()
            config = replace(config, fix=replace(config.fix, max_conflict_files=1))
            github = FakeGithub(
                open_pull_request=OpenPullRequest(
                    number=9,
                    url="https://github.test/o/r/pull/9",
                    base_ref="main",
                    mergeable="CONFLICTING",
                ),
                review_threads=(),
            )
            status = run_fix_revise(
                config=config,
                issue_number=7,
                base_repo=base_repo,
                output_dir=root / "out",
                github=github,  # type: ignore[arg-type]
                issue_fields=FakeIssueFields("代码 Bug"),  # type: ignore[arg-type]
            )
            self.assertEqual(status, "conflict_escalated")
            self.assertTrue(any("人工" in c["body"] for c in github.pr_comments))
            self.assertFalse(github.resolved_threads)


def _make_fix_remote(root: Path) -> Path:
    """origin with a pushed fix branch; returns a base_repo clone (origin set).

    Mirrors _make_conflicting_fix_remote but the fix branch does NOT conflict
    with main, so fix_revise_worktree checks out a clean tip the CI-fix agent
    can edit and push.
    """
    origin = root / "origin"
    origin.mkdir()
    subprocess.run(["git", "-C", str(origin), "init", "-q", "-b", "main"], check=True)
    for k, v in (("user.email", "t@t.test"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(origin), "config", k, v], check=True)
    (origin / "src").mkdir()
    (origin / "src" / "todo.ts").write_text("export const x = 1\n")
    subprocess.run(["git", "-C", str(origin), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(origin), "commit", "-q", "-m", "init"], check=True)
    subprocess.run(
        ["git", "-C", str(origin), "checkout", "-q", "-b", "bugpatrol/fix-issue-7"], check=True
    )
    (origin / "src" / "todo.ts").write_text("export const x = 1\n// fix\n")
    subprocess.run(["git", "-C", str(origin), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(origin), "commit", "-q", "-m", "fix"], check=True)
    subprocess.run(["git", "-C", str(origin), "checkout", "-q", "main"], check=True)

    base_repo = root / "base"
    subprocess.run(
        ["git", "clone", "-q", str(origin), str(base_repo)], check=True, capture_output=True
    )
    for k, v in (("user.email", "t@t.test"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(base_repo), "config", k, v], check=True)
    return base_repo


_BOT_BRANCH = "bugpatrol/fix-issue-7"


def _bot_pr() -> OpenPullRequest:
    # A real bugpatrol fix PR targets a feature branch, not the default branch,
    # so GitHub populates NO native closing reference -- the issue must resolve
    # from the bugpatrol/fix-issue-N head branch name.
    return OpenPullRequest(
        number=9,
        url="https://github.test/o/r/pull/9",
        head_ref=_BOT_BRANCH,
        closing_issue_numbers=(),
        body="Fixes #7",
    )


def _human_pr(head_ref: str = "feature/manual-fix") -> OpenPullRequest:
    # A human's manual-fix PR against a feature branch: no native closing
    # reference either, so the managed issue resolves from the body keyword.
    return OpenPullRequest(
        number=9,
        url="https://github.test/o/r/pull/9",
        head_ref=head_ref,
        closing_issue_numbers=(),
        body="Fixes #7",
    )


class ClosingIssueCandidatesTest(unittest.TestCase):
    def _fix(self):
        return _sandbox_config().fix

    def _pr(self, **kw) -> OpenPullRequest:
        base = dict(number=9, url="u", head_ref="", closing_issue_numbers=(), body="")
        base.update(kw)
        return OpenPullRequest(**base)  # type: ignore[arg-type]

    def test_bot_branch_name_is_first_candidate(self) -> None:
        pr = self._pr(head_ref="bugpatrol/fix-issue-42", body="Fixes #7")
        self.assertEqual(_closing_issue_candidates(pr, self._fix()), (42, 7))

    def test_empty_native_link_falls_back_to_body_keyword(self) -> None:
        pr = self._pr(head_ref="feature/x", body="Closes #4061 per PRD")
        self.assertEqual(_closing_issue_candidates(pr, self._fix()), (4061,))

    def test_keyword_variants_and_dedupe(self) -> None:
        pr = self._pr(
            head_ref="feature/x",
            closing_issue_numbers=(7,),
            body="fixes #7\nresolved #8\nCLOSED #9\nsee #10",
        )
        # #7 from native link (deduped with body), #8/#9 from keywords, #10 has no
        # keyword so it is ignored.
        self.assertEqual(_closing_issue_candidates(pr, self._fix()), (7, 8, 9))

    def test_no_candidates_when_nothing_cites_an_issue(self) -> None:
        pr = self._pr(head_ref="feature/x", body="just cleanup")
        self.assertEqual(_closing_issue_candidates(pr, self._fix()), ())


class RunCiFeedbackFailureTest(unittest.TestCase):
    def _run(self, github, base_repo, tmp, *, head_branch=_BOT_BRANCH, conclusion="failure"):
        return run_ci_feedback(
            config=_sandbox_config(),
            head_branch=head_branch,
            head_sha="deadbeef",
            conclusion=conclusion,
            base_repo=base_repo,
            output_dir=Path(tmp) / "out",
            github=github,  # type: ignore[arg-type]
            issue_fields=FakeIssueFields("代码 Bug"),  # type: ignore[arg-type]
        )

    def test_no_pr_when_no_open_pr(self) -> None:
        github = FakeGithub(open_pull_request=None)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._run(github, Path(tmp), tmp), "no_pr")

    def test_no_managed_issue_when_pr_closes_nothing_managed(self) -> None:
        # A human PR on a feature branch with no native closing reference and no
        # Fixes/#N keyword resolves nothing -> not ours to report.
        pr = OpenPullRequest(
            number=9,
            url="https://github.test/o/r/pull/9",
            head_ref="feature/unrelated",
            closing_issue_numbers=(),
            body="just some cleanup",
        )
        github = FakeGithub(open_pull_request=pr)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._run(github, Path(tmp), tmp), "no_managed_issue")

    def test_bot_pr_with_empty_native_link_resolves_via_branch_name(self) -> None:
        # Regression (#4074): a fix PR targets a feature branch, so GitHub's
        # closingIssuesReferences is empty; the issue must still resolve from the
        # bugpatrol/fix-issue-N head branch. Reaching "no_ci_failure" (rather than
        # "no_managed_issue") proves resolution got past the association gate.
        pr = OpenPullRequest(
            number=9,
            url="https://github.test/o/r/pull/9",
            head_ref=_BOT_BRANCH,
            closing_issue_numbers=(),
            body="",
        )
        github = FakeGithub(open_pull_request=pr, failed_runs=())
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._run(github, Path(tmp), tmp), "no_ci_failure")

    def test_closed_issue_skips(self) -> None:
        github = FakeGithub(
            state="closed",
            open_pull_request=_bot_pr(),
            failed_runs=(FailedRun(run_id=1, name="iOS Build", workflow_name="iOS Build"),),
        )
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._run(github, Path(tmp), tmp), "issue_closed")
        self.assertFalse(github.pr_comments)

    def test_sha_already_handled_short_circuits(self) -> None:
        github = FakeGithub(
            open_pull_request=_bot_pr(),
            pr_comment_bodies=[
                append_ci_fix_metadata("x", {"attempts": 1, "last_fixed_sha": "deadbeef"})
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._run(github, Path(tmp), tmp), "ci_already_handled")

    def test_no_ci_failure_when_no_failed_runs(self) -> None:
        github = FakeGithub(open_pull_request=_bot_pr(), failed_runs=())
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._run(github, Path(tmp), tmp), "no_ci_failure")

    def test_escalates_at_attempt_cap(self) -> None:
        github = FakeGithub(
            open_pull_request=_bot_pr(),
            failed_runs=(FailedRun(run_id=1, name="iOS Build", workflow_name="iOS Build"),),
            pr_comment_bodies=[
                append_ci_fix_metadata("x", {"attempts": 3, "last_fixed_sha": "older"})
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            status = self._run(github, Path(tmp), tmp)
        self.assertEqual(status, "ci_fix_escalated")
        # The escalation PR comment carries the de-dupe marker at this sha.
        self.assertTrue(any("人工" in c["body"] for c in github.pr_comments))
        meta = None
        from bugpatrol.fix_result import parse_ci_fix_metadata

        for c in github.pr_comments:
            meta = parse_ci_fix_metadata(c["body"]) or meta
        assert meta is not None
        self.assertEqual(meta["last_fixed_sha"], "deadbeef")

    def test_human_branch_failure_notifies_only_then_dedupes(self) -> None:
        # A human PR (not a bugpatrol fix branch) that fails CI: report to the
        # topic, never auto-revise. Repeated failure events for the same sha
        # de-dupe.
        github = FakeGithub(
            open_pull_request=_human_pr(),
            assignees=("dev1",),
            failed_runs=(FailedRun(run_id=1, name="iOS Build", workflow_name="iOS Build"),),
        )
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                self._run(github, Path(tmp), tmp, head_branch="feature/manual-fix"),
                "ci_failure_notified",
            )
        self.assertTrue(any("构建失败" in c for c in github.added_comments))
        self.assertFalse(github.created_prs)  # never revises a human branch
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                self._run(github, Path(tmp), tmp, head_branch="feature/manual-fix"),
                "ci_failure_already_notified",
            )

    def test_cancelled_aggregate_masking_failure_notifies(self) -> None:
        # #4074: fail-fast/concurrency cancels a sibling so the aggregate run
        # reports `cancelled`, but a job genuinely failed. The run-level failure
        # query is empty; the check-run surface exposes it -> notify (do NOT
        # auto-revise, a cancelled run has no clean per-run logs to feed).
        github = FakeGithub(
            open_pull_request=_bot_pr(),
            assignees=("dev1",),
            failed_runs=(),
            failed_check_runs=("web / test",),
        )
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                self._run(github, Path(tmp), tmp, conclusion="cancelled"),
                "ci_failure_notified",
            )
        self.assertTrue(any("web / test" in c for c in github.added_comments))
        self.assertFalse(github.created_prs)  # masked failure -> notify, not revise

    def test_cancelled_pure_supersede_is_noop(self) -> None:
        # A cancelled run with nothing actually failed (a newer push superseded
        # it) is a no-op, not a spurious notification.
        github = FakeGithub(
            open_pull_request=_bot_pr(), failed_runs=(), failed_check_runs=()
        )
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                self._run(github, Path(tmp), tmp, conclusion="cancelled"),
                "no_ci_failure",
            )
        self.assertFalse(github.added_comments)

    def test_cancelled_masked_failure_dedupes_across_events(self) -> None:
        # One push -> several cancelled aggregates (PR Checks + PR Builds) -> many
        # events for one sha; notify once.
        github = FakeGithub(
            open_pull_request=_bot_pr(),
            assignees=("dev1",),
            failed_check_runs=("web / test",),
        )
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                self._run(github, Path(tmp), tmp, conclusion="cancelled"),
                "ci_failure_notified",
            )
            self.assertEqual(
                self._run(github, Path(tmp), tmp, conclusion="cancelled"),
                "ci_failure_already_notified",
            )

    def test_cancelled_skipped_when_revise_already_handled_sha(self) -> None:
        # A real `failure` event already triggered auto-revise for this commit;
        # a cancelled sibling aggregate for the same sha must not also notify.
        github = FakeGithub(
            open_pull_request=_bot_pr(),
            failed_check_runs=("web / test",),
            pr_comment_bodies=[
                append_ci_fix_metadata("x", {"attempts": 1, "last_fixed_sha": "deadbeef"})
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                self._run(github, Path(tmp), tmp, conclusion="cancelled"),
                "ci_already_handled",
            )


class RunCiFixEndToEndTest(unittest.TestCase):
    def test_ci_fixed_edits_pushes_and_records_marker(self) -> None:
        # `prepare_fix_run` builds a deepseek invocation, which refuses to build
        # without a key. Nothing here runs the agent (the command is swapped out
        # below), so supply a dummy rather than requiring the ambient env (CI has
        # no DEEPSEEK_API_KEY, which had this test red since 2026-09-29).
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ", {"DEEPSEEK_API_KEY": "sk-test"}
        ):
            root = Path(tmp)
            base_repo = _make_fix_remote(root)
            config = replace(_sandbox_config(), fix=replace(_sandbox_config().fix, verify={"ok": "true"}, setup={}))
            github = FakeGithub(
                open_pull_request=_bot_pr(),
                assignees=("dev1",),
                failed_runs=(FailedRun(run_id=1, name="iOS Build", workflow_name="iOS Build"),),
                failed_logs={1: "error: boom"},
            )
            # Patch the agent invocation to a real command that edits + writes output.
            real_prepare = prepare_fix_run

            def fake_prepare(**kwargs):
                plan = real_prepare(**kwargs)
                command = ExecuteFixRunTest._edit_and_write_output(
                    plan.agent_cwd, plan.output_path
                )
                return replace(
                    plan,
                    invocation=replace(plan.invocation, command=command),
                )

            with patch("bugpatrol.fix_runner.prepare_fix_run", side_effect=fake_prepare), patch(
                "bugpatrol.fix_runner.worktree_push_branch"
            ) as push:
                status = run_ci_feedback(
                    config=config,
                    head_branch=_BOT_BRANCH,
                    head_sha="deadbeef",
                    conclusion="failure",
                    base_repo=base_repo,
                    output_dir=root / "out",
                    github=github,  # type: ignore[arg-type]
                    issue_fields=FakeIssueFields("代码 Bug"),  # type: ignore[arg-type]
                )
            self.assertEqual(status, "ci_fixed")
            push.assert_called_once()
            self.assertFalse(github.created_prs)
            from bugpatrol.fix_result import parse_ci_fix_metadata

            metas = [parse_ci_fix_metadata(c["body"]) for c in github.pr_comments]
            metas = [m for m in metas if m is not None]
            self.assertTrue(metas)
            self.assertEqual(metas[-1]["last_fixed_sha"], "deadbeef")
            self.assertEqual(metas[-1]["attempts"], 1)


def _config_with_build_link():
    from bugpatrol.config import BuildLinkPattern

    base = _sandbox_config()
    return replace(
        base,
        fix=replace(
            base.fix,
            build_link_patterns=(
                BuildLinkPattern(label="iOS 安装", pattern=r"iOS install: (https://\S+)"),
            ),
        ),
    )


class RunBuildReadyTest(unittest.TestCase):
    def _run(self, github, config=None, *, head_branch=_BOT_BRANCH):
        with tempfile.TemporaryDirectory() as tmp:
            return run_ci_feedback(
                config=config or _sandbox_config(),
                head_branch=head_branch,
                head_sha="deadbeef",
                conclusion="success",
                base_repo=Path(tmp),
                output_dir=Path(tmp) / "out",
                github=github,  # type: ignore[arg-type]
                issue_fields=FakeIssueFields("代码 Bug"),  # type: ignore[arg-type]
            )

    def test_no_pr(self) -> None:
        self.assertEqual(self._run(FakeGithub(open_pull_request=None)), "no_pr")

    def test_human_pr_build_ready_notifies(self) -> None:
        # A passing build on a human PR that closes a managed issue also surfaces.
        github = FakeGithub(open_pull_request=_human_pr(), assignees=("dev1",))
        self.assertEqual(self._run(github, head_branch="feature/manual-fix"), "build_notified")
        self.assertTrue(any("可测试" in c for c in github.added_comments))

    def test_already_notified_short_circuits(self) -> None:
        github = FakeGithub(
            open_pull_request=_bot_pr(),
            pr_comment_bodies=[
                append_ci_fix_metadata("x", {"last_notified_sha": "deadbeef"})
            ],
        )
        self.assertEqual(self._run(github), "build_already_notified")

    def test_notifies_once_and_records_marker(self) -> None:
        github = FakeGithub(open_pull_request=_bot_pr(), assignees=("dev1",))
        self.assertEqual(self._run(github), "build_notified")
        self.assertTrue(any("可测试" in c for c in github.added_comments))
        from bugpatrol.fix_result import parse_ci_fix_metadata

        metas = [parse_ci_fix_metadata(c["body"]) for c in github.pr_comments]
        metas = [m for m in metas if m is not None]
        self.assertTrue(metas)
        self.assertEqual(metas[-1]["last_notified_sha"], "deadbeef")
        # A second event for the same sha now de-dupes.
        self.assertEqual(self._run(github), "build_already_notified")

    def test_late_link_triggers_followup_then_dedupes(self) -> None:
        # The #4011 race: a fast build trips build-ready before the slow iOS build
        # posts its install link. The first ping carries no link; when the link
        # comment lands, the next event follows up with just that link, then stops.
        config = _config_with_build_link()
        github = FakeGithub(open_pull_request=_bot_pr(), assignees=("dev1",))

        # 1) First event: no link comment yet -> main "可测试" ping, no link.
        self.assertEqual(self._run(github, config), "build_notified")
        self.assertTrue(any("可测试" in c for c in github.added_comments))

        # 2) The slow iOS build now posts its install link on the PR.
        github.pr_comment_bodies.append("📱 iOS install: https://ota.test/i")

        # 3) Next event: same sha, but a new link appeared -> follow-up ping.
        self.assertEqual(self._run(github, config), "build_links_notified")
        self.assertTrue(any("已就绪" in c for c in github.added_comments))
        self.assertTrue(
            any("https://ota.test/i" in c for c in github.added_comments)
        )

        # 4) No further links -> de-dupes.
        self.assertEqual(self._run(github, config), "build_already_notified")

    def test_link_present_at_first_ping_not_repeated(self) -> None:
        config = _config_with_build_link()
        github = FakeGithub(
            open_pull_request=_bot_pr(),
            assignees=("dev1",),
            pr_comment_bodies=["📱 iOS install: https://ota.test/i"],
        )
        self.assertEqual(self._run(github, config), "build_notified")
        # The link was already surfaced in the main ping; no follow-up.
        self.assertEqual(self._run(github, config), "build_already_notified")


if __name__ == "__main__":
    unittest.main()
