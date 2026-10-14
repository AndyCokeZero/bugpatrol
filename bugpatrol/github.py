"""GitHub issue client backed by the GitHub CLI."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bugpatrol.clients import (
    FailedRun,
    GitHubIssue,
    GitHubIssueComment,
    GitHubPullRequest,
    OpenPullRequest,
    ReviewComment,
    ReviewThread,
    WorkflowRun,
)
from bugpatrol.gh_transient import run_gh_with_transient_retries
from bugpatrol.github_fields import GITHUB_API_VERSION, GitHubIssueFieldsClient

if TYPE_CHECKING:
    from bugpatrol.config import ProjectConfig


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str


class GitHubCliError(RuntimeError):
    pass


class GitHubCliIssuesClient:
    def __init__(
        self,
        *,
        gh: str = "gh",
        search_limit: int = 200,
        issue_fields: GitHubIssueFieldsClient | None = None,
        project_config: ProjectConfig | None = None,
        transient_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        env: Mapping[str, str] = os.environ,
    ) -> None:
        self._gh = gh
        self._search_limit = search_limit
        self._issue_fields = issue_fields
        self._project_config = project_config
        self._transient_retries = transient_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._sleep = sleep
        # Seam for the proxy→direct fallback: the environment the retry driver
        # reads to decide whether a no-proxy transport exists. Tests pin it; the
        # process environment is correct everywhere else.
        self._env = env

    def find_issue_by_intake_root(self, *, repo: str, chat_id: str, root_id: str) -> GitHubIssue | None:
        result = self._run(
            [
                "issue",
                "list",
                "--repo",
                repo,
                "--state",
                "all",
                "--limit",
                str(self._search_limit),
                "--json",
                "number,url,title,body,state,stateReason",
            ]
        )
        for item in json.loads(result.stdout):
            body = str(item.get("body") or "")
            if f'"chat_id":"{chat_id}"' in body and f'"root_id":"{root_id}"' in body:
                return GitHubIssue(
                    number=int(item["number"]),
                    url=str(item["url"]),
                    title=str(item["title"]),
                    body=body,
                    state=str(item.get("state") or "").lower(),
                    state_reason=str(item.get("stateReason") or ""),
                )
        return None

    def list_issues(self, *, repo: str, state: str = "open") -> tuple[GitHubIssue, ...]:
        result = self._run(
            [
                "issue",
                "list",
                "--repo",
                repo,
                "--state",
                state,
                "--limit",
                str(self._search_limit),
                "--json",
                "number,url,title,body,state,stateReason,closedAt",
            ]
        )
        return tuple(
            GitHubIssue(
                number=int(item["number"]),
                url=str(item["url"]),
                title=str(item["title"]),
                body=str(item.get("body") or ""),
                state=str(item.get("state") or ""),
                state_reason=str(item.get("stateReason") or ""),
                closed_at=str(item.get("closedAt") or ""),
            )
            for item in json.loads(result.stdout)
        )

    def create_issue(
        self,
        *,
        repo: str,
        title: str,
        body: str,
        issue_type: str,
        fields: dict[str, str],
    ) -> GitHubIssue:
        result = self._run(
            ["issue", "create", "--repo", repo, "--title", title, "--body-file", "-"],
            stdin=body,
        )
        url = result.stdout.strip().splitlines()[-1].strip()
        number = _issue_number_from_url(url)
        self.set_issue_type(repo=repo, issue_number=number, issue_type=issue_type)
        if self._issue_fields is not None:
            if self._project_config is None:
                raise GitHubCliError("project_config is required when issue_fields is configured")
            self._issue_fields.add_issue_field_values(
                repo=repo,
                issue_number=number,
                values=fields,
                config=self._project_config,
            )
        return GitHubIssue(number=number, url=url, title=title, body=body)

    def add_issue_comment(self, *, repo: str, issue_number: int, body: str) -> None:
        self._run(
            ["issue", "comment", str(issue_number), "--repo", repo, "--body-file", "-"],
            stdin=body,
        )

    def update_issue_body(self, *, repo: str, issue_number: int, body: str) -> None:
        """Replace an issue's body.

        Mail intake patches ``notify_anchor_message_id`` into the intake meta
        after posting the group receipt, and an issue body edit is the one
        update the issues API supports that touches the body text.
        """
        self._run(
            ["issue", "edit", str(issue_number), "--repo", repo, "--body-file", "-"],
            stdin=body,
        )

    def update_issue_title(self, *, repo: str, issue_number: int, title: str) -> None:
        """Replace an issue's title.

        Triage rewrites the intake auto-title (an 80-char prefix of the raw
        report) when the agent drafts a corrected_title that actually differs.
        """
        self._run(
            ["issue", "edit", str(issue_number), "--repo", repo, "--title", title],
        )

    def dispatch_triage_run(self, *, repo: str, issue_number: int) -> None:
        """Dispatch the standard bugpatrol-triage workflow for one issue.

        Each dispatched run mints its own fresh GitHub App token in-workflow, so
        a reconcile batch of any size stays within token lifetime (in-process
        triage under one job-lifetime token died on batches of more than ~7
        candidates — see reconcile_triage).
        """
        self._run(
            [
                "workflow",
                "run",
                "bugpatrol-triage.yml",
                "--repo",
                repo,
                "-f",
                f"issue_number={issue_number}",
            ]
        )

    def list_issue_comments(self, *, repo: str, issue_number: int) -> tuple[GitHubIssueComment, ...]:
        result = self._run(
            [
                "api",
                "-H",
                f"X-GitHub-Api-Version: {GITHUB_API_VERSION}",
                f"/repos/{repo}/issues/{issue_number}/comments",
            ]
        )
        data = json.loads(result.stdout)
        return tuple(
            GitHubIssueComment(id=str(item["id"]), body=str(item.get("body") or ""))
            for item in data
        )

    def set_issue_type(self, *, repo: str, issue_number: int, issue_type: str) -> None:
        self._run(
            [
                "api",
                "-X",
                "PATCH",
                "-H",
                f"X-GitHub-Api-Version: {GITHUB_API_VERSION}",
                f"/repos/{repo}/issues/{issue_number}",
                "-f",
                f"type={issue_type}",
            ]
        )

    def get_issue_type(self, *, repo: str, issue_number: int) -> str:
        result = self._run(
            [
                "api",
                "-H",
                f"X-GitHub-Api-Version: {GITHUB_API_VERSION}",
                f"/repos/{repo}/issues/{issue_number}",
            ]
        )
        data = json.loads(result.stdout)
        issue_type = data.get("type")
        if not isinstance(issue_type, dict) or not isinstance(issue_type.get("name"), str):
            return ""
        return str(issue_type["name"])

    def get_repository(self, *, repo: str) -> dict[str, object]:
        result = self._run(["api", f"/repos/{repo}"])
        data = json.loads(result.stdout)
        if not isinstance(data, dict):
            raise GitHubCliError(f"unexpected repository response: {data!r}")
        return data

    def get_issue(self, *, repo: str, issue_number: int) -> GitHubIssue:
        result = self._run(
            [
                "api",
                "-H",
                f"X-GitHub-Api-Version: {GITHUB_API_VERSION}",
                f"/repos/{repo}/issues/{issue_number}",
            ]
        )
        data = json.loads(result.stdout)
        return GitHubIssue(
            number=int(data["number"]),
            url=str(data["html_url"]),
            title=str(data["title"]),
            body=str(data.get("body") or ""),
            state=str(data.get("state") or ""),
            state_reason=str(data.get("state_reason") or ""),
            closed_at=str(data.get("closed_at") or ""),
            closed_by=str((data.get("closed_by") or {}).get("login") or "")
            if isinstance(data.get("closed_by"), dict)
            else "",
            assignees=tuple(
                str(item["login"])
                for item in data.get("assignees") or ()
                if isinstance(item, dict) and item.get("login")
            ),
        )

    def list_issue_timeline(self, *, repo: str, issue_number: int) -> tuple[dict, ...]:
        result = self._run(
            [
                "api",
                "-H",
                f"X-GitHub-Api-Version: {GITHUB_API_VERSION}",
                "-H",
                "Accept: application/vnd.github+json",
                f"/repos/{repo}/issues/{issue_number}/timeline?per_page=100",
            ]
        )
        data = json.loads(result.stdout)
        if not isinstance(data, list):
            raise GitHubCliError(f"unexpected timeline response for {repo}#{issue_number}")
        return tuple(item for item in data if isinstance(item, dict))

    def commit_exists(self, *, repo: str, sha: str) -> bool:
        """Whether `sha` resolves to a real commit in `repo`.

        GitHub resolves abbreviated SHAs, so a 7-char short hash works. A
        missing/ambiguous SHA makes `gh api` exit non-zero, which surfaces as a
        GitHubCliError -- reported here as "not a commit" rather than raised, so
        a bogus hex token in a comment simply isn't treated as fix evidence.
        """
        try:
            self._run(
                [
                    "api",
                    "-H",
                    f"X-GitHub-Api-Version: {GITHUB_API_VERSION}",
                    f"/repos/{repo}/commits/{sha}",
                    "--jq",
                    ".sha",
                ]
            )
        except GitHubCliError:
            return False
        return True

    def commit_referencing_issue(
        self, *, repo: str, branch: str, issue_number: int, max_commits: int = 200
    ) -> str:
        """Newest commit on `branch` whose message references `#issue_number`, else "".

        A fix committed straight to a feature branch (no PR, not on the default
        branch) produces an issue<->commit `referenced` timeline event that a
        GitHub App installation token cannot see, so close-audit's timeline path
        misses it -- and if the dev didn't paste the SHA in a comment either,
        the issue gets wrongly reopened. When the issue's declared target branch
        is known, reverse-look its recent commit messages for a `#N` reference.
        Bounded to `max_commits` (newest first) so an active branch can't fan
        out into unbounded API calls; a missing branch is treated as "no match".
        """
        pattern = re.compile(rf"#{issue_number}(?!\d)")
        per_page = 100
        scanned = 0
        page = 1
        while scanned < max_commits:
            try:
                result = self._run(
                    [
                        "api",
                        "-H",
                        f"X-GitHub-Api-Version: {GITHUB_API_VERSION}",
                        f"/repos/{repo}/commits?sha={branch}&per_page={per_page}&page={page}",
                    ]
                )
            except GitHubCliError:
                return ""
            data = json.loads(result.stdout)
            if not isinstance(data, list) or not data:
                return ""
            for item in data:
                if not isinstance(item, dict):
                    continue
                commit = item.get("commit")
                message = commit.get("message") if isinstance(commit, dict) else None
                if isinstance(message, str) and pattern.search(message):
                    sha = item.get("sha")
                    if isinstance(sha, str) and sha:
                        return sha
                scanned += 1
                if scanned >= max_commits:
                    break
            if len(data) < per_page:
                return ""
            page += 1
        return ""

    def pull_request_merged(self, *, repo: str, number: int) -> bool:
        """Whether `repo#number` is a pull request that has been merged.

        Used to verify a cross-repo fix PR a dev cites in a comment. A missing
        PR (or a plain issue number) makes `gh pr view` exit non-zero -> treated
        as "not a merged PR" rather than raised.
        """
        try:
            result = self._run(
                ["pr", "view", str(number), "--repo", repo, "--json", "mergedAt"]
            )
        except GitHubCliError:
            return False
        data = json.loads(result.stdout)
        return bool(data.get("mergedAt"))

    def get_pull_request(self, *, repo: str, pr: str) -> GitHubPullRequest:
        result = self._run(
            [
                "pr",
                "view",
                pr,
                "--repo",
                repo,
                "--json",
                "number,url,title,body,closingIssuesReferences,baseRefName,author",
            ]
        )
        data = json.loads(result.stdout)
        closing = data.get("closingIssuesReferences")
        closing_numbers: list[int] = []
        if isinstance(closing, list):
            for item in closing:
                if isinstance(item, dict) and isinstance(item.get("number"), int):
                    closing_numbers.append(int(item["number"]))
        return GitHubPullRequest(
            number=int(data["number"]),
            url=str(data["url"]),
            title=str(data.get("title") or ""),
            body=str(data.get("body") or ""),
            closing_issue_numbers=tuple(closing_numbers),
            timeline_issue_numbers=self._timeline_issue_numbers_for_pr(repo=repo, pr_number=int(data["number"])),
            base_ref=str(data.get("baseRefName") or ""),
            author=str((data.get("author") or {}).get("login") or "") if isinstance(data.get("author"), dict) else "",
        )

    def get_commit_author(self, *, repo: str, sha: str) -> str:
        """GitHub login of a commit's author (falls back to the git author name).

        Used to attribute a `commit_linked` fix to a person so the Lark
        notification can @-mention them. A missing/ambiguous SHA makes `gh api`
        exit non-zero, which surfaces as GitHubCliError -> reported here as no
        author (empty) rather than raised, so a bogus hex token can't abort the
        notification.
        """
        try:
            result = self._run(
                [
                    "api",
                    "-H",
                    f"X-GitHub-Api-Version: {GITHUB_API_VERSION}",
                    f"/repos/{repo}/commits/{sha}",
                    "--jq",
                    '.author.login // .commit.author.name // ""',
                ]
            )
        except GitHubCliError:
            return ""
        return result.stdout.strip()

    def list_merged_pull_requests(
        self, *, repo: str, limit: int = 30
    ) -> tuple[GitHubPullRequest, ...]:
        """Recent merged PRs with their closing-issue references.

        Bounded by `limit`. Timeline-linked issues are left empty here (one API
        call per PR would be wasteful for reconcile); reconcile resolves the
        associated issue from closingIssuesReferences plus title/body refs.
        """
        result = self._run(
            [
                "pr",
                "list",
                "--repo",
                repo,
                "--state",
                "merged",
                "--limit",
                str(limit),
                "--json",
                "number,url,title,body,closingIssuesReferences,mergedAt,mergeCommit,author",
            ]
        )
        pulls: list[GitHubPullRequest] = []
        for data in json.loads(result.stdout):
            closing = data.get("closingIssuesReferences")
            closing_numbers: list[int] = []
            if isinstance(closing, list):
                for item in closing:
                    if isinstance(item, dict) and isinstance(item.get("number"), int):
                        closing_numbers.append(int(item["number"]))
            merge_commit = data.get("mergeCommit")
            merge_commit_sha = str(merge_commit.get("oid") or "") if isinstance(merge_commit, dict) else ""
            pulls.append(
                GitHubPullRequest(
                    number=int(data["number"]),
                    url=str(data.get("url") or ""),
                    title=str(data.get("title") or ""),
                    body=str(data.get("body") or ""),
                    closing_issue_numbers=tuple(closing_numbers),
                    merged_at=str(data.get("mergedAt") or ""),
                    merge_commit_sha=merge_commit_sha,
                    author=str((data.get("author") or {}).get("login") or "") if isinstance(data.get("author"), dict) else "",
                )
            )
        return tuple(pulls)

    def _timeline_issue_numbers_for_pr(self, *, repo: str, pr_number: int) -> tuple[int, ...]:
        try:
            result = self._run(
                [
                    "api",
                    "-H",
                    f"X-GitHub-Api-Version: {GITHUB_API_VERSION}",
                    "-H",
                    "Accept: application/vnd.github+json",
                    f"/repos/{repo}/issues/{pr_number}/timeline",
                ]
            )
        except GitHubCliError:
            return ()
        data = json.loads(result.stdout)
        if not isinstance(data, list):
            return ()
        from bugpatrol.fix_notify import issue_numbers_from_timeline_events

        return issue_numbers_from_timeline_events(data, exclude=(pr_number,))

    def list_issue_types(self, *, repo: str) -> tuple[str, ...]:
        result = self._run(
            ["api", "-H", f"X-GitHub-Api-Version: {GITHUB_API_VERSION}", f"/repos/{repo}/issue-types"]
        )
        data = json.loads(result.stdout)
        return tuple(str(item["name"]) for item in data)

    def close_issue(self, *, repo: str, issue_number: int, reason: str = "not planned") -> None:
        self._run(
            [
                "issue",
                "close",
                str(issue_number),
                "--repo",
                repo,
                "--reason",
                reason,
                "--comment",
                "Closed automatically by bugpatrol live e2e cleanup.",
            ]
        )

    def close_issue_as_duplicate(self, *, repo: str, issue_number: int, duplicate_of: int) -> None:
        owner, name = repo.split("/", 1)
        result = self._run(
            [
                "api",
                "graphql",
                "-f",
                "query=query($owner: String!, $name: String!, $issue: Int!, $duplicate: Int!) {"
                " repository(owner: $owner, name: $name) {"
                " issue(number: $issue) { id }"
                " duplicate: issue(number: $duplicate) { id } } }",
                "-f",
                f"owner={owner}",
                "-f",
                f"name={name}",
                "-F",
                f"issue={issue_number}",
                "-F",
                f"duplicate={duplicate_of}",
            ]
        )
        data = json.loads(result.stdout)
        repository = data.get("data", {}).get("repository") or {}
        issue_id = (repository.get("issue") or {}).get("id")
        duplicate_id = (repository.get("duplicate") or {}).get("id")
        if not isinstance(issue_id, str) or not isinstance(duplicate_id, str):
            raise GitHubCliError(f"cannot resolve issue node ids for {repo}#{issue_number} / #{duplicate_of}")
        self._run(
            [
                "api",
                "graphql",
                "-f",
                "query=mutation($issue: ID!, $duplicate: ID!) {"
                " closeIssue(input: {issueId: $issue, stateReason: DUPLICATE, duplicateIssueId: $duplicate}) {"
                " issue { number state } } }",
                "-f",
                f"issue={issue_id}",
                "-f",
                f"duplicate={duplicate_id}",
            ]
        )

    def reopen_issue(self, *, repo: str, issue_number: int) -> None:
        self._run(["issue", "reopen", str(issue_number), "--repo", repo])

    def remote_branch_tip_sha(self, *, repo: str, branch: str) -> str:
        """Best-effort current tip SHA of a remote branch; "" if unavailable.

        Recorded at intake so a branch later merged and deleted can still be
        classified. Never raises: a missing branch just yields "".
        """
        try:
            result = self._run(
                [
                    "api",
                    f"repos/{repo}/git/ref/heads/{branch}",
                    "--jq",
                    ".object.sha",
                ]
            )
        except GitHubCliError:
            return ""
        return result.stdout.strip()

    def list_remote_branch_names(self, *, repo: str) -> tuple[str, ...]:
        """Every branch name on the remote repo.

        Anchors branch-chat discovery: a Lark group is only adopted as a branch
        topic when its name is an actually existing branch.
        """
        result = self._run(
            [
                "api",
                f"repos/{repo}/branches",
                "--paginate",
                "--jq",
                ".[].name",
            ]
        )
        return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())

    def find_open_pull_request_by_head(self, *, repo: str, head: str) -> str:
        """URL of an open PR whose head branch is `head`, or "" if none.

        Used for fix idempotency: a fix run must not open a second PR for an
        issue that already has an open one.
        """
        result = self._run(
            [
                "pr",
                "list",
                "--repo",
                repo,
                "--head",
                head,
                "--state",
                "open",
                "--json",
                "url",
            ]
        )
        data = json.loads(result.stdout)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return str(data[0].get("url") or "")
        return ""

    def get_open_pull_request_by_head(self, *, repo: str, head: str) -> OpenPullRequest | None:
        """Open PR whose head branch is `head`, or None.

        Like `find_open_pull_request_by_head` but returns the number, target
        branch, and mergeability too, which revise needs to read/resolve the
        PR's review threads and to detect/resolve a conflict with the target
        branch. Kept separate so the existing idempotency short-circuit's return
        type is unchanged.
        """
        result = self._run(
            [
                "pr",
                "list",
                "--repo",
                repo,
                "--head",
                head,
                "--state",
                "open",
                "--json",
                "number,url,baseRefName,mergeable,headRefName,closingIssuesReferences,body",
            ]
        )
        data = json.loads(result.stdout)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            number = data[0].get("number")
            url = data[0].get("url")
            if isinstance(number, int) and isinstance(url, str):
                closing = data[0].get("closingIssuesReferences")
                closing_numbers: list[int] = []
                if isinstance(closing, list):
                    for item in closing:
                        if isinstance(item, dict) and isinstance(item.get("number"), int):
                            closing_numbers.append(int(item["number"]))
                return OpenPullRequest(
                    number=number,
                    url=url,
                    base_ref=str(data[0].get("baseRefName") or ""),
                    mergeable=str(data[0].get("mergeable") or ""),
                    head_ref=str(data[0].get("headRefName") or ""),
                    closing_issue_numbers=tuple(closing_numbers),
                    body=str(data[0].get("body") or ""),
                )
        return None

    def list_unresolved_review_threads(self, *, repo: str, pr_number: int) -> tuple[ReviewThread, ...]:
        """Unresolved review threads on a PR (the revise work queue).

        State lives entirely in GitHub: any runner can read the same queue and,
        after addressing a thread, resolve it so it is not reprocessed.
        """
        owner, name = repo.split("/", 1)
        result = self._run(
            [
                "api",
                "graphql",
                "-f",
                "query=query($owner: String!, $name: String!, $pr: Int!) {"
                " repository(owner: $owner, name: $name) {"
                " pullRequest(number: $pr) {"
                " reviewThreads(first: 100) { nodes {"
                " id isResolved"
                " comments(first: 50) { nodes {"
                " body path line author { login } } } } } } } }",
                "-f",
                f"owner={owner}",
                "-f",
                f"name={name}",
                "-F",
                f"pr={pr_number}",
            ]
        )
        data = json.loads(result.stdout)
        pull = (
            (data.get("data", {}).get("repository") or {}).get("pullRequest") or {}
        )
        nodes = (pull.get("reviewThreads") or {}).get("nodes") or []
        threads: list[ReviewThread] = []
        for node in nodes:
            if not isinstance(node, dict) or node.get("isResolved"):
                continue
            thread_id = node.get("id")
            if not isinstance(thread_id, str):
                continue
            comments: list[ReviewComment] = []
            for raw in (node.get("comments") or {}).get("nodes") or []:
                if not isinstance(raw, dict):
                    continue
                author = ((raw.get("author") or {}).get("login")) or ""
                line = raw.get("line")
                comments.append(
                    ReviewComment(
                        author=str(author),
                        body=str(raw.get("body") or ""),
                        path=str(raw.get("path") or ""),
                        line=line if isinstance(line, int) else None,
                    )
                )
            threads.append(ReviewThread(id=thread_id, comments=tuple(comments)))
        return tuple(threads)

    def resolve_review_thread(self, *, thread_id: str) -> None:
        """Mark a review thread resolved once revise has addressed it."""
        self._run(
            [
                "api",
                "graphql",
                "-f",
                "query=mutation($thread: ID!) {"
                " resolveReviewThread(input: {threadId: $thread}) {"
                " thread { id isResolved } } }",
                "-f",
                f"thread={thread_id}",
            ]
        )

    def add_pull_request_comment(self, *, repo: str, pr: str, body: str) -> None:
        """Append a comment to a PR conversation (revise progress reply)."""
        self._run(
            [
                "pr",
                "comment",
                pr,
                "--repo",
                repo,
                "--body-file",
                "-",
            ],
            stdin=body,
        )

    def create_pull_request(
        self,
        *,
        repo: str,
        head: str,
        base: str,
        title: str,
        body: str,
    ) -> str:
        """Open a PR from `head` into `base`; return the PR URL."""
        result = self._run(
            [
                "pr",
                "create",
                "--repo",
                repo,
                "--head",
                head,
                "--base",
                base,
                "--title",
                title,
                "--body-file",
                "-",
            ],
            stdin=body,
        )
        return result.stdout.strip().splitlines()[-1].strip()

    def add_pull_request_reviewer(self, *, repo: str, pr: str, reviewer: str) -> None:
        """Request a review from `reviewer`; a non-collaborator reviewer or a
        self-review is not fatal to the fix run, so failures are surfaced by the
        caller rather than raised here."""
        self._run(
            [
                "pr",
                "edit",
                pr,
                "--repo",
                repo,
                "--add-reviewer",
                reviewer,
            ]
        )

    def list_pull_request_comments(
        self, *, repo: str, pr_number: int
    ) -> tuple[GitHubIssueComment, ...]:
        """Conversation comments on a PR (PRs share the issues comments API).

        Used by the CI-fix loop to read the ``BUGPATROL_CI_FIX_META`` marker and,
        for tier-2 build-ready links, the CI bot's install-link comments.
        """
        result = self._run(
            [
                "api",
                "-H",
                f"X-GitHub-Api-Version: {GITHUB_API_VERSION}",
                f"/repos/{repo}/issues/{pr_number}/comments",
            ]
        )
        data = json.loads(result.stdout)
        return tuple(
            GitHubIssueComment(id=str(item["id"]), body=str(item.get("body") or ""))
            for item in data
        )

    def list_failed_runs_for_sha(
        self, *, repo: str, head_sha: str
    ) -> tuple[FailedRun, ...]:
        """All CI workflow runs that concluded ``failure`` for a commit.

        One revise push can trigger several build workflows; gathering every
        failed run for the sha lets the CI-fix agent see the full failure context
        in a single turn (and de-dupe on the sha, not per-run).
        """
        result = self._run(
            [
                "run",
                "list",
                "--repo",
                repo,
                "--commit",
                head_sha,
                "--json",
                "databaseId,name,workflowName,conclusion",
                "--limit",
                "50",
            ]
        )
        data = json.loads(result.stdout)
        runs: list[FailedRun] = []
        for item in data:
            if not isinstance(item, dict) or item.get("conclusion") != "failure":
                continue
            run_id = item.get("databaseId")
            if not isinstance(run_id, int):
                continue
            runs.append(
                FailedRun(
                    run_id=run_id,
                    name=str(item.get("name") or ""),
                    workflow_name=str(item.get("workflowName") or ""),
                )
            )
        return tuple(runs)

    def list_recent_workflow_runs(
        self, *, repo: str, branch: str, limit: int = 20
    ) -> tuple[WorkflowRun, ...]:
        """Recent CI runs on one branch, newest first.

        The tool-CI watch needs a *branch's* state rather than one commit's runs
        (which is what ``list_failed_runs_for_sha`` answers): a red main is a
        standing condition, and only the newest commit says whether it still
        holds.
        """
        result = self._run(
            [
                "run",
                "list",
                "--repo",
                repo,
                "--branch",
                branch,
                "--json",
                "databaseId,workflowName,status,conclusion,headSha,url",
                "--limit",
                str(limit),
            ]
        )
        data = json.loads(result.stdout)
        runs: list[WorkflowRun] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            run_id = item.get("databaseId")
            if not isinstance(run_id, int):
                continue
            runs.append(
                WorkflowRun(
                    run_id=run_id,
                    workflow_name=str(item.get("workflowName") or ""),
                    status=str(item.get("status") or ""),
                    conclusion=str(item.get("conclusion") or ""),
                    head_sha=str(item.get("headSha") or ""),
                    url=str(item.get("url") or ""),
                )
            )
        return tuple(runs)

    def list_failed_check_runs_for_sha(
        self, *, repo: str, head_sha: str
    ) -> tuple[str, ...]:
        """Names of individual check-runs that concluded ``failure`` for a commit.

        A workflow_run whose aggregate conclusion is ``cancelled`` (a sibling job
        was cancelled by fail-fast or a concurrency lane) still reports genuinely
        failed jobs at the check-run surface, where the run-level conclusion hides
        them. This lets CI feedback tell a real-but-masked failure apart from a
        pure supersede (nothing failed, the run was just cancelled).
        """
        result = self._run(
            [
                "api",
                f"repos/{repo}/commits/{head_sha}/check-runs",
                "--paginate",
                "--jq",
                '.check_runs[] | select(.conclusion == "failure") | .name',
            ]
        )
        names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return tuple(names)

    def get_run_failed_logs(self, *, repo: str, run_id: int) -> str:
        """The failed-step logs of a workflow run, truncated to the tail.

        BugPatrol only reads the CI *result surface* (like ``[fix.verify]`` exit
        codes), never the project's build definition. The tail keeps the actual
        error region without flooding the agent's context.
        """
        result = self._run(
            [
                "run",
                "view",
                str(run_id),
                "--repo",
                repo,
                "--log-failed",
            ]
        )
        return _truncate_log_tail(result.stdout)

    def add_assignee(self, *, repo: str, issue_number: int, assignee: str) -> None:
        self._run(
            [
                "issue",
                "edit",
                str(issue_number),
                "--repo",
                repo,
                "--add-assignee",
                assignee,
            ]
        )

    def set_assignee(self, *, repo: str, issue_number: int, assignee: str) -> None:
        """Replace the whole assignee set with exactly one login.

        The issues PATCH endpoint treats `assignees` as a full replacement, so
        passing a single login drops any previously assigned people. Used by
        `/assign` to guarantee a sole assignee.
        """
        self._run(
            [
                "api",
                "-X",
                "PATCH",
                "-H",
                f"X-GitHub-Api-Version: {GITHUB_API_VERSION}",
                f"/repos/{repo}/issues/{issue_number}",
                "-f",
                f"assignees[]={assignee}",
            ]
        )

    def _run(self, args: Sequence[str], *, stdin: str | None = None) -> CommandResult:
        outcome = run_gh_with_transient_retries(
            [self._gh, *args],
            stdin=stdin,
            retries=self._transient_retries,
            backoff_seconds=self._retry_backoff_seconds,
            sleep=self._sleep,
            env=self._env,
        )
        completed = outcome.completed
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise GitHubCliError(
                f"gh {' '.join(args)} failed with exit {completed.returncode}: "
                f"{stderr}{outcome.transport_note}"
            )
        return CommandResult(stdout=completed.stdout, stderr=completed.stderr)


def _truncate_log_tail(text: str, *, max_lines: int = 200, max_chars: int = 8000) -> str:
    """Keep the tail of a CI log (where the error surfaces) within bounds."""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    tail = "\n".join(lines)
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def _issue_number_from_url(url: str) -> int:
    match = re.search(r"/issues/(\d+)$", url)
    if not match:
        raise GitHubCliError(f"cannot parse issue number from URL: {url}")
    return int(match.group(1))
