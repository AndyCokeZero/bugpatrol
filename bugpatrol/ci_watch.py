"""Alert when a repo's own CI goes red on its default branch.

BugPatrol's CI went red on 2026-09-29 and stayed red until 2026-10-14, because
nothing in the pipeline looks at it. The triage and fix runners ff-merge
`verneagent/bugpatrol`'s main on every run and then exercise one narrow path
through it, so a red suite -- tests broken, tool still working -- reaches
nothing downstream. The only surface that knew was the repo's Actions tab, and
nobody opens it.

A red main is therefore a *standing condition*, not an event. Each check reads
the newest settled commit on the branch -- settled meaning every run for it has
finished, since judging a commit from a queued run would flap the alert on every
check -- and alerts when it flips red, and again when it flips back to green.

That is what ``WatchState`` is for: the same red commit is the newest settled
commit on every check until a newer one is pushed, so a stateless "is main red?"
test would re-alert each time it ran. The state is a single sha, and losing it
is safe -- the worst case is one duplicate alert.

There is no issue to comment on, which is why the alert goes straight to a Lark
chat -- the same place, and for the same reason, as the reconcile failure steps.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bugpatrol.clients import WorkflowRun

# Conclusions that make a commit red. `cancelled` is deliberately absent: a
# superseded run reports cancelled while the sibling job that was actually
# cancelled carries the real result, the same distinction the CI-fix loop draws
# when it reads check-runs instead of the run-level conclusion.
FAILURE_CONCLUSIONS = frozenset({"failure", "timed_out", "startup_failure"})

# The run status that means the verdict is in. Anything else (queued,
# in_progress, waiting) has no conclusion yet.
SETTLED_STATUS = "completed"


class WorkflowRunLister(Protocol):
    def list_recent_workflow_runs(
        self, *, repo: str, branch: str, limit: int = 20
    ) -> tuple[WorkflowRun, ...]:
        """Recent runs on ``branch``, newest first."""


class ChatNotifier(Protocol):
    def send_chat_message(self, *, chat_id: str, text: str) -> object:
        """Post a message into a chat."""


@dataclass(frozen=True)
class BranchState:
    """What the newest settled commit on a branch says about that branch."""

    head_sha: str
    red: bool
    url: str
    failed_workflows: tuple[str, ...]


@dataclass(frozen=True)
class WatchState:
    """The red commit already alerted on; empty when nothing is outstanding."""

    alerted_sha: str = ""


@dataclass(frozen=True)
class Alert:
    kind: str
    text: str


@dataclass(frozen=True)
class WatchOutcome:
    repo: str
    branch: str
    head_sha: str
    red: bool
    alert: str
    delivered: bool


def settled_branch_state(runs: Sequence[WorkflowRun]) -> BranchState | None:
    """The newest commit on the branch whose runs have all finished.

    None when nothing has settled yet -- a brand-new branch, or every run still
    in flight.

    Runs arrive newest first but a commit normally has several, one per
    workflow, so they are grouped by commit and judged together: with two CI
    workflows finishing minutes apart, asking "was the latest run red?" would
    answer differently depending on which one finished last.
    """
    grouped: dict[str, list[WorkflowRun]] = {}
    order: list[str] = []
    for run in runs:
        if run.head_sha not in grouped:
            grouped[run.head_sha] = []
            order.append(run.head_sha)
        grouped[run.head_sha].append(run)
    for sha in order:
        commit_runs = grouped[sha]
        if any(run.status != SETTLED_STATUS for run in commit_runs):
            continue
        failed = [run for run in commit_runs if run.conclusion in FAILURE_CONCLUSIONS]
        reported = failed[0] if failed else commit_runs[0]
        return BranchState(
            head_sha=sha,
            red=bool(failed),
            url=reported.url,
            failed_workflows=tuple(sorted({run.workflow_name for run in failed})),
        )
    return None


def evaluate(
    *,
    previous: WatchState,
    current: BranchState | None,
    repo: str,
    branch: str,
) -> tuple[WatchState, Alert | None]:
    """The state to carry forward, and the alert to send if one is due.

    An unwatched repo that is already red does alert: an empty state means "not
    known to be red", not "assume red is old news", so a watch switched on
    during an outage still reports it.
    """
    if current is None:
        return previous, None
    short_sha = current.head_sha[:7]
    if current.red:
        if previous.alerted_sha == current.head_sha:
            return previous, None
        failing = "、".join(current.failed_workflows)
        detail = f"（{failing}）" if failing else ""
        return (
            WatchState(alerted_sha=current.head_sha),
            Alert(
                kind="broke",
                text=(
                    f"⚠️ CI 失败：{repo} @ {branch}\n"
                    f"提交：{short_sha}{detail}\n"
                    f"日志：{current.url}"
                ),
            ),
        )
    if previous.alerted_sha:
        return (
            WatchState(),
            Alert(
                kind="fixed",
                text=f"✅ CI 已恢复绿色：{repo} @ {branch}（提交 {short_sha}）",
            ),
        )
    return previous, None


def load_state(path: Path) -> WatchState:
    """Read the watch state; a missing file is an unwatched branch.

    A file that exists but cannot be understood raises rather than resetting:
    silently starting over would re-alert on a red commit every check, which is
    the exact noise the state exists to prevent.
    """
    if not path.exists():
        return WatchState()
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(payload).__name__}")
    alerted_sha = payload.get("alerted_sha", "")
    if not isinstance(alerted_sha, str):
        raise ValueError(f"{path}: alerted_sha must be a string")
    return WatchState(alerted_sha=alerted_sha)


def save_state(path: Path, state: WatchState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"alerted_sha": state.alerted_sha}) + "\n")


def watch_ci(
    *,
    repo: str,
    branch: str,
    chat_id: str,
    runs: WorkflowRunLister,
    lark: ChatNotifier | None,
    state_path: Path,
    dry_run: bool,
) -> WatchOutcome:
    """Judge one branch, alert on a flip, and persist the state.

    A dry run still reads and reports but neither sends nor saves: a dry run
    that recorded the state would swallow the real alert that follows it.
    """
    current = settled_branch_state(runs.list_recent_workflow_runs(repo=repo, branch=branch))
    next_state, alert = evaluate(
        previous=load_state(state_path),
        current=current,
        repo=repo,
        branch=branch,
    )
    delivered = False
    if alert is not None and not dry_run:
        if lark is None:
            raise ValueError("a Lark chat is required to deliver a CI alert")
        lark.send_chat_message(chat_id=chat_id, text=alert.text)
        save_state(state_path, next_state)
        delivered = True
    return WatchOutcome(
        repo=repo,
        branch=branch,
        head_sha=current.head_sha if current is not None else "",
        red=current.red if current is not None else False,
        alert=alert.kind if alert is not None else "",
        delivered=delivered,
    )
