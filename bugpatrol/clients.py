"""Client protocols used by bugpatrol workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GitHubIssue:
    number: int
    url: str
    title: str
    body: str
    state: str = ""
    state_reason: str = ""
    closed_at: str = ""
    closed_by: str = ""
    assignees: tuple[str, ...] = ()


@dataclass(frozen=True)
class GitHubIssueComment:
    id: str
    body: str


@dataclass(frozen=True)
class OpenPullRequest:
    """Minimal identity of an open PR, enough to revise it from any runner.

    ``base_ref`` is the target branch (needed to merge it in when resolving a
    conflict); ``mergeable`` is GitHub's mergeability signal ("MERGEABLE" /
    "CONFLICTING" / "UNKNOWN"), where "CONFLICTING" means the PR conflicts with
    its target branch and revise should merge the target in before proceeding.
    ``head_ref`` is the PR's source branch (so CI feedback can tell a bugpatrol
    fix branch apart from a human branch); ``closing_issue_numbers`` are the
    issues the PR closes via GitHub's native linking, used to resolve which
    managed issue a passing/failing build should report to. ``body`` is the PR
    description, parsed for ``Fixes #N`` closing keywords when GitHub's native
    link is empty -- it is only populated for PRs targeting the default branch,
    so a fix PR opened against a feature branch has no native closing reference.
    """

    number: int
    url: str
    base_ref: str = ""
    mergeable: str = ""
    head_ref: str = ""
    closing_issue_numbers: tuple[int, ...] = ()
    body: str = ""


@dataclass(frozen=True)
class ReviewComment:
    author: str
    body: str
    path: str = ""
    line: int | None = None


@dataclass(frozen=True)
class ReviewThread:
    """An unresolved PR review thread (one or more inline/review comments).

    `id` is the GraphQL node id used to resolve the thread once addressed.
    """

    id: str
    comments: tuple[ReviewComment, ...] = ()


@dataclass(frozen=True)
class FailedRun:
    """A failed CI workflow run for a fix PR's head commit.

    ``run_id`` is the database id used to fetch the failed-step logs; the names
    identify which build broke so the CI-fix agent gets full context.
    """

    run_id: int
    name: str
    workflow_name: str = ""


@dataclass(frozen=True)
class WorkflowRun:
    """One CI workflow run on a branch, as listed by ``gh run list``.

    ``status`` is kept apart from ``conclusion`` because only the first says
    whether the run is over: a queued or in-flight run has no verdict yet, and
    reading its empty conclusion as "not a failure" would call a branch green
    while its build is still running.
    """

    run_id: int
    workflow_name: str
    status: str
    conclusion: str
    head_sha: str
    url: str


@dataclass(frozen=True)
class GitHubPullRequest:
    number: int
    url: str
    title: str
    body: str
    closing_issue_numbers: tuple[int, ...] = ()
    timeline_issue_numbers: tuple[int, ...] = ()
    base_ref: str = ""
    merged_at: str = ""
    merge_commit_sha: str = ""
    author: str = ""


class GitHubIssuesClient(Protocol):
    def find_issue_by_intake_root(self, *, repo: str, chat_id: str, root_id: str) -> GitHubIssue | None:
        """Return the issue already linked to a Lark topic root, if one exists."""

    def create_issue(
        self,
        *,
        repo: str,
        title: str,
        body: str,
        issue_type: str,
        fields: dict[str, str],
    ) -> GitHubIssue:
        """Create an issue and write initial deterministic metadata."""

    def add_issue_comment(self, *, repo: str, issue_number: int, body: str) -> None:
        """Append a comment to an existing issue."""

    def update_issue_body(self, *, repo: str, issue_number: int, body: str) -> None:
        """Replace an issue's body (used to patch intake meta after creation)."""

    def list_issue_comments(self, *, repo: str, issue_number: int) -> tuple[GitHubIssueComment, ...]:
        """Return comments for an issue."""


class LarkMessengerClient(Protocol):
    def reply_to_message(self, *, chat_id: str, message_id: str, text: str) -> None:
        """Reply in Lark so the reporter sees the GitHub backlink."""
