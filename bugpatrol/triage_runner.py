"""Prepare and optionally execute a triage agent run."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from bugpatrol.agents import (
    AgentInvocation,
    agent_error_from_stdout,
    build_triage_agent_invocation,
    detect_sandbox_denial,
    is_retryable_agent_failure,
    load_agent_json,
    parse_claude_token_usage,
)
from bugpatrol.clients import GitHubIssue, GitHubIssueComment, LarkMessengerClient
from bugpatrol.config import ProjectConfig
from bugpatrol.fields import default_field_specs, triage_output_schema
from bugpatrol.github import GitHubCliIssuesClient
from bugpatrol.github_fields import (
    GitHubIssueFieldsClient,
    find_field_option_drift,
    split_repo,
)
from bugpatrol.intake import (
    branch_tip_sha_from_metadata,
    require_bugpatrol_managed_issue,
    target_branch_from_metadata,
)
from bugpatrol.fix_gate import strip_ansi
from bugpatrol.ownership import load_codeowners, load_codeowners_identities
from bugpatrol.worktree import (
    BranchResolution,
    GitDriver,
    SubprocessGitDriver,
    resolve_triage_branch,
)
from bugpatrol.media_localize import localize_media
from bugpatrol.openspec import load_openspec_changes
from bugpatrol.triage_context import (
    AssigneeIdentity,
    ReferenceRepoContext,
    build_triage_context,
    render_triage_context_markdown,
)
from bugpatrol.triage_result import (
    TriageResult,
    TriageRunStats,
    apply_triage_result,
    parse_triage_metadata,
    parse_triage_result,
    send_intake_topic_message,
    triage_runner_name,
)


TRIAGE_RUN_META_START = "<!-- BUGPATROL_TRIAGE_RUN_META"
TRIAGE_RUN_META_END = "BUGPATROL_TRIAGE_RUN_META -->"
TRIAGE_SKIPPED_HEADER = "## BugPatrol triage skipped"
TRIAGE_FAILED_HEADER = "## BugPatrol triage failed"
# A run marker older than the triage workflow's own timeout can only belong to a
# run that died without reporting (cancelled job, runner restart), so it must
# stop counting as in-flight or the issue would never be re-triaged.
TRIAGE_RUN_STALE_AFTER = timedelta(minutes=60)

# A hung LLM call used to burn the whole 60-minute triage job timeout with no
# output, wasting a runner slot and ending in a silent cancel (run 32022343306
# on #4938) rather than a visible failure. Normal triage is ~8 minutes; 20
# minutes fails fast while leaving room for slow runs and the workflow's own
# retries.
AGENT_TIMEOUT_SECONDS = 1200


@dataclass(frozen=True)
class TriageRunPlan:
    context_path: Path
    schema_path: Path
    output_path: Path
    invocation: AgentInvocation
    # Working directory for the agent subprocess: the repo checkout (main or a
    # branch worktree), so the agent reads CODEOWNERS/source/git of the right
    # branch in-place. None keeps the runner's own cwd (used by unit tests).
    agent_cwd: Path | None = None
    context_comment_ids: tuple[str, ...] = ()
    known_assignees: tuple[str, ...] = ()
    # Human-facing note about which branch was analyzed (empty for main), shown
    # in the triage comment header and the Lark notify.
    branch_note: str = ""


def prepare_triage_run(
    *,
    config: ProjectConfig,
    issue_number: int,
    repo_path: Path,
    output_dir: Path,
    github: GitHubCliIssuesClient,
    prompt_path: Path = Path("prompts/triage.zh.md"),
    branch_note: str = "",
    reference_repos: Sequence[ReferenceRepoContext] = (),
) -> TriageRunPlan:
    issue = github.get_issue(repo=config.github_repo, issue_number=issue_number)
    require_bugpatrol_managed_issue(issue)
    output_dir.mkdir(parents=True, exist_ok=True)
    # The agent runs with cwd=repo_path, so every path handed to it must be
    # absolute (a relative "prompts/triage.zh.md" would resolve against the
    # checkout, not the runner workspace, and vanish).
    output_dir = output_dir.resolve()
    prompt_path = prompt_path.resolve()
    repo_path = repo_path.resolve()
    # Resolve each reference-repo checkout to an absolute path and fail loud if
    # a declared one is missing — a silently absent ref would let the agent
    # cross-check nothing while still posting a confident triage.
    resolved_refs: list[ReferenceRepoContext] = []
    for ref in reference_repos:
        ref_path = Path(ref.path).resolve()
        if not ref_path.is_dir():
            raise FileNotFoundError(
                f"reference repo {ref.repo!r} checkout not found at {ref_path}"
            )
        resolved_refs.append(replace(ref, path=str(ref_path)))
    comments = github.list_issue_comments(repo=config.github_repo, issue_number=issue_number)
    known_assignees = list_known_assignees(repo_path, config=config)
    # OpenSpec records an owner per change (tasks.md `· @name` / `> **Owner**`).
    # These are display-name nicknames, not GitHub logins, so we surface them in
    # the context (not the assignee whitelist) and let the agent map a matched
    # change's owner to a real login via the Assignee Roster — preferring it over
    # CODEOWNERS inference, falling back when it cannot be mapped.
    openspec_changes = load_openspec_changes(repo_path / config.prd.cache_path)
    context = build_triage_context(
        issue=issue,
        comments=comments,
        prd_root=repo_path / config.prd.cache_path,
        prd_include_globs=config.prd.include_globs,
        roster=build_assignee_roster(
            known_assignees,
            codeowners_identities=load_codeowners_identities(repo_path),
        ),
        reference_repos=tuple(resolved_refs),
        openspec_changes=openspec_changes,
    )
    # Pull the attachments down ourselves, with the runner's own credentials,
    # and hand the agent file paths. The agent's unauthenticated fetch of a
    # private assets repo returns a 404 HTML page that it reads as an image —
    # the gateway then 400s and the entire run dies (issue #6124).
    context = replace(
        context,
        media=localize_media(
            context.media,
            dest_dir=output_dir / "attachments",
            assets_repo=config.assets.github_repo,
            branch=config.assets.branch,
            gh=config.github_cli,
        ),
    )
    context_path = output_dir / "triage-context.md"
    schema_path = output_dir / "triage.schema.json"
    output_path = output_dir / "triage-output.json"
    context_path.write_text(render_triage_context_markdown(context))
    schema_path.write_text(
        json.dumps(
            triage_output_schema(known_assignees=known_assignees),
            ensure_ascii=False,
            indent=2,
        )
    )
    invocation = build_triage_agent_invocation(
        config,
        issue_number=issue_number,
        prompt_path=prompt_path,
        schema_path=schema_path,
        output_path=output_path,
        context_path=context_path,
        # Re-admit the runner-side workspace (prompt + output dir) plus each
        # reference-repo checkout that all sit outside the checkout the agent
        # is cwd'd into.
        workspace_dirs=(
            prompt_path.parent,
            output_dir,
            *(Path(ref.path) for ref in resolved_refs),
        ),
    )
    return TriageRunPlan(
        context_path=context_path,
        schema_path=schema_path,
        output_path=output_path,
        agent_cwd=repo_path,
        context_comment_ids=comment_ids(comments),
        invocation=invocation,
        known_assignees=known_assignees,
        branch_note=branch_note,
    )


def list_known_assignees(repo_path: Path, *, config: ProjectConfig) -> tuple[str, ...]:
    """Return valid GitHub logins for triage assignment.

    Combines CODEOWNERS owners with the [owners] tables in the project config.
    Used both to constrain the agent output schema and to reject results that
    use display names ("Andy") instead of logins ("AndyCokeZero").
    """
    logins: set[str] = set()
    for rule in load_codeowners(repo_path):
        for owner in rule.owners:
            handle = owner.lstrip("@")
            # Team handles (org/team) cannot be issue assignees.
            if handle and "/" not in handle:
                logins.add(handle)
    for group in (config.owners.default, *config.owners.paths.values(), *config.owners.capabilities.values()):
        for owner in group:
            handle = owner.lstrip("@")
            if handle and "/" not in handle:
                logins.add(handle)
    return tuple(sorted(logins))


def build_assignee_roster(
    known_assignees: tuple[str, ...],
    *,
    codeowners_identities: dict[str, tuple[str, ...]] | None = None,
) -> tuple[AssigneeIdentity, ...]:
    """Pair each valid assignee login with its human aliases.

    Lets the triage agent map a free-form "assign to X" reference (an
    @mention, a typed Lark/GitHub name, or a short form) to a login. Aliases
    are the login itself plus the name documented in the CODEOWNERS header.
    """
    codeowners_identities = codeowners_identities or {}
    roster: list[AssigneeIdentity] = []
    for login in known_assignees:
        aliases = tuple(dict.fromkeys((login, *codeowners_identities.get(login, ()))))
        roster.append(AssigneeIdentity(login=login, aliases=aliases))
    return tuple(roster)


def _render_triage_start_message(*, issue_number: int, issue_url: str, branch_note: str = "") -> str:
    text = f"开始分诊，GitHub issue [#{issue_number}]({issue_url})"
    if branch_note:
        text += f"\n{branch_note}"
    runner = triage_runner_name()
    if runner:
        text += f"\n分诊执行机：{runner}"
    return text


def resolve_issue_branch(
    *,
    config: ProjectConfig,
    issue_number: int,
    base_repo: Path,
    github: GitHubCliIssuesClient,
    driver: GitDriver | None = None,
) -> BranchResolution:
    """Resolve which branch/ref a triage run should analyze for an issue.

    Reads the declared branch from the issue's BUGPATROL_INTAKE_META; legacy or
    main-branch issues resolve to main (no worktree needed).
    """
    issue = github.get_issue(repo=config.github_repo, issue_number=issue_number)
    metadata = require_bugpatrol_managed_issue(issue)
    driver = driver or SubprocessGitDriver(base_repo)
    return resolve_triage_branch(
        driver,
        target_branch=target_branch_from_metadata(metadata),
        branch_tip_sha=branch_tip_sha_from_metadata(metadata),
    )


def execute_triage_run(
    *,
    config: ProjectConfig,
    issue_number: int,
    plan: TriageRunPlan,
    github: GitHubCliIssuesClient,
    issue_fields: GitHubIssueFieldsClient,
    lark: LarkMessengerClient | None = None,
    accept_stale_context: bool = False,
    final_attempt: bool = True,
) -> str:
    issue = github.get_issue(repo=config.github_repo, issue_number=issue_number)
    require_bugpatrol_managed_issue(issue)
    existing_comments = github.list_issue_comments(repo=config.github_repo, issue_number=issue_number)
    if triage_already_applied_without_new_material(existing_comments):
        return "already_triaged"
    followup_review = evaluate_followup_re_triage(
        issue=issue,
        comments=existing_comments,
        invocation=plan.invocation,
    )
    if followup_review is not None and not followup_review.requires_re_triage:
        # A material followup that doesn't change the conclusion would otherwise
        # burn a full agent run (and intake reset the field to Pending, so the
        # watcher would keep re-dispatching). Skip the agent and restore the
        # prior terminal status the intake reset overwrote.
        issue_fields.add_issue_field_values(
            repo=config.github_repo,
            issue_number=issue_number,
            values={"Triage status": followup_review.prior_status or "Done"},
            config=config,
        )
        return "no_change_followup"
    # Fail fast if the canonical field specs declare an option the live org
    # field is missing. Without this, a triage would run the agent to completion
    # and only crash at write time (add_issue_field_values), leaving the issue
    # un-triaged after burning agent tokens — exactly the OpenSpec drift that
    # stalled a day of triage.
    live_fields = issue_fields.list_org_fields(org=split_repo(config.github_repo)[0])
    drift = find_field_option_drift(default_field_specs(), live_fields)
    if drift:
        raise RuntimeError(
            "Issue field options are out of sync with GitHub; fix the org "
            "field(s) before triaging:\n- " + "\n- ".join(drift)
        )
    run_id = str(uuid4())
    mark_triage_running(
        config=config,
        issue_number=issue_number,
        issue_fields=issue_fields,
    )
    if lark is not None:
        send_intake_topic_message(
            repo=config.github_repo,
            issue_number=issue_number,
            github=github,
            lark=lark,
            text=_render_triage_start_message(
                issue_number=issue_number,
                issue_url=issue.url,
                branch_note=plan.branch_note,
            ),
        )
    record_triage_run_start(
        config=config,
        issue_number=issue_number,
        plan=plan,
        run_id=run_id,
        github=github,
    )
    agent_env = {**os.environ, **plan.invocation.env} if plan.invocation.env else None
    # stdin must be closed: in CI runners stdin is a pipe that never reaches
    # EOF, and `claude -p` blocks reading it forever after finishing its work.
    started = time.monotonic()
    try:
        completed = subprocess.run(
            plan.invocation.command,
            check=False,
            env=agent_env,
            cwd=plan.agent_cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=AGENT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        duration_seconds = time.monotonic() - started
        stdout = _coerce_text(error.stdout)
        stderr = _coerce_text(error.stderr)
        _write_turn_log(plan.output_path.parent, stdout, stderr)
        mark_triage_failed(
            config=config,
            issue_number=issue_number,
            exit_code=124,
            github=github,
            issue_fields=issue_fields,
            lark=lark,
            reason=agent_failure_reason(
                f"The triage agent produced no terminal output for {AGENT_TIMEOUT_SECONDS}s "
                "and was killed (the LLM endpoint likely hung).",
                stderr=stderr,
                stdout=stdout,
            ),
        )
        raise RuntimeError(
            f"triage agent timed out after {AGENT_TIMEOUT_SECONDS}s "
            f"({duration_seconds:.0f}s elapsed with no terminal output)"
        ) from error
    duration_seconds = time.monotonic() - started
    # Persist the turn-by-turn stream so a failed or surprising run can be
    # analysed on the runner afterwards, and derive token usage from it.
    _write_turn_log(plan.output_path.parent, completed.stdout, completed.stderr)
    input_tokens, cached_input_tokens, output_tokens = parse_claude_token_usage(
        completed.stdout or ""
    )
    run_stats = TriageRunStats(
        duration_seconds=duration_seconds,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        model=plan.invocation.model,
    )
    if completed.returncode != 0:
        # The agent died mid-turn. Most non-zero exits like this are the LLM
        # endpoint dropping an in-flight response (#6246: `Connection closed
        # mid-response` after 133 turns and 7 minutes on minici32g, the same
        # issue having just died the same way on carbonci). Retrying here costs
        # one more agent run; failing throws that work away, marks the issue
        # Failed, pings the reporter, and leans on the watcher to start over
        # from scratch on another runner anyway. Credential/account errors and
        # a gateway-refused model reproduce on every attempt, so those keep
        # failing on the spot. A bare gateway 400 (`{"model": ...}`, no
        # diagnostic) is settled with a one-token probe: the model answering
        # means the run died to a hiccup and should retry.
        if not final_attempt and is_retryable_agent_failure(
            completed.stdout, plan.invocation
        ):
            return "agent_crashed"
        mark_triage_failed(
            config=config,
            issue_number=issue_number,
            exit_code=completed.returncode,
            github=github,
            issue_fields=issue_fields,
            lark=lark,
            reason=agent_failure_reason(
                f"The triage agent exited with code `{completed.returncode}`.",
                stderr=completed.stderr,
                stdout=completed.stdout,
            ),
        )
        raise RuntimeError(f"triage agent failed with exit {completed.returncode}")
    denial = detect_sandbox_denial(completed.stdout or "")
    if denial:
        # A sandbox/permission denial means the agent silently skipped
        # CODEOWNERS/repo/git/dup-search; retrying won't help (the invocation
        # is misconfigured), so fail loudly rather than posting a hollow triage.
        mark_triage_failed(
            config=config,
            issue_number=issue_number,
            exit_code=0,
            github=github,
            issue_fields=issue_fields,
            lark=lark,
            reason=f"Triage agent was blocked by a permission/sandbox denial: {denial}",
        )
        raise RuntimeError(f"triage agent blocked by sandbox/permission denial: {denial}")
    if not plan.output_path.exists():
        # Models occasionally end the turn silently without writing the
        # output file; retry instead of failing the run outright.
        if not final_attempt:
            return "no_output"
        mark_triage_failed(
            config=config,
            issue_number=issue_number,
            exit_code=0,
            github=github,
            issue_fields=issue_fields,
            lark=lark,
            reason=agent_failure_reason(
                "The triage agent exited 0 but wrote no output file.",
                stderr=completed.stderr,
                stdout=completed.stdout,
            ),
        )
        raise RuntimeError("triage agent exited 0 but produced no output file")
    try:
        result = parse_triage_result(load_agent_json(plan.output_path.read_text()))
    except (ValueError, json.JSONDecodeError) as error:
        # The agent wrote an output file, but it failed schema/consistency
        # validation (e.g. duplicate_of set without verdict 重复, or malformed
        # JSON). This is a stochastic agent mistake, not a misconfiguration, so
        # retry like no_output; only on the final attempt do we mark the issue
        # Failed. Crashing here (the old behavior) left Triage status stuck at
        # "Running" forever, which no re-scan would ever re-dispatch.
        if not final_attempt:
            return "invalid_output"
        mark_triage_failed(
            config=config,
            issue_number=issue_number,
            exit_code=0,
            github=github,
            issue_fields=issue_fields,
            lark=lark,
            reason=f"Triage agent output failed validation: {error}",
        )
        raise RuntimeError(f"triage agent output failed validation: {error}") from error
    if plan.known_assignees and result.assignee not in plan.known_assignees:
        mark_triage_failed(
            config=config,
            issue_number=issue_number,
            exit_code=0,
            github=github,
            issue_fields=issue_fields,
            lark=lark,
            reason=(
                f"Agent returned assignee `{result.assignee}`, which is not a known GitHub login. "
                f"Valid assignees: {', '.join(plan.known_assignees)}."
            ),
        )
        raise RuntimeError(f"triage agent returned unknown assignee {result.assignee!r}")
    current_comments = github.list_issue_comments(repo=config.github_repo, issue_number=issue_number)
    if latest_triage_run_id(current_comments) != run_id:
        mark_triage_superseded(
            config=config,
            issue_number=issue_number,
            run_id=run_id,
            github=github,
        )
        return "superseded"
    if comment_ids(current_comments) != plan.context_comment_ids:
        if not accept_stale_context:
            return "stale_context"
        result = annotate_result_stale_context(result)
    apply_triage_result(
        repo=config.github_repo,
        issue_number=issue_number,
        config=config,
        result=result,
        github=github,
        issue_fields=issue_fields,
        lark=lark,
        run_stats=run_stats,
        branch_note=plan.branch_note,
    )
    return "applied"


def _coerce_text(payload: str | bytes | None) -> str | None:
    """Decode a ``subprocess`` timeout payload to ``str``.

    ``TimeoutExpired`` keeps ``.stdout``/``.stderr`` as raw bytes even when the
    subprocess was spawned with ``text=True``, so any text-only sink (``write_text``,
    regex, ``json.loads``) must see decoded text first.
    """
    if payload is None:
        return None
    return payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else payload


def _write_turn_log(output_dir: Path, stdout: str | bytes | None, stderr: str | bytes | None) -> None:
    stdout = _coerce_text(stdout)
    stderr = _coerce_text(stderr)
    if stdout:
        _keep_previous(output_dir / "agent-turns.jsonl")
        (output_dir / "agent-turns.jsonl").write_text(stdout)
    if stderr:
        _keep_previous(output_dir / "agent-stderr.log")
        (output_dir / "agent-stderr.log").write_text(stderr)


def _keep_previous(path: Path) -> None:
    """Keep the prior attempt's log before this one replaces it.

    The run directory is keyed by issue number, so a GitHub re-run of the same
    issue writes over the failing attempt's log — exactly the evidence needed
    to diagnose it (2026-10-15: 14 of 16 failing logs were destroyed by the
    re-runs that were being used to rule the failure out). One previous
    attempt is enough to compare against, and it keeps the directory bounded.
    """
    if path.exists():
        path.replace(path.with_name(f"{path.stem}.previous{path.suffix}"))


def record_triage_run_start(
    *,
    config: ProjectConfig,
    issue_number: int,
    plan: TriageRunPlan,
    run_id: str,
    github: GitHubCliIssuesClient,
) -> None:
    metadata = {
        "version": 1,
        "issue": issue_number,
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "context_comment_ids": list(plan.context_comment_ids),
    }
    github.add_issue_comment(
        repo=config.github_repo,
        issue_number=issue_number,
        body=append_triage_run_metadata(metadata),
    )


def triage_already_applied_without_new_material(comments: tuple[GitHubIssueComment, ...]) -> bool:
    """True when the newest applied triage result has no later material comments."""
    latest_triage_index: int | None = None
    for index, comment in enumerate(comments):
        if parse_triage_metadata(comment.body) is not None:
            latest_triage_index = index
    if latest_triage_index is None:
        return False
    for comment in comments[latest_triage_index + 1 :]:
        if is_triage_bookkeeping_comment(comment.body):
            continue
        if parse_triage_metadata(comment.body) is not None:
            continue
        return False
    return True


# Cheap pre-gate before the expensive triage agent. When an already-triaged issue
# gets a material followup, a tiny classifier decides whether a full re-triage is
# warranted instead of blindly burning a full agent run (which nearly always
# reaches the same conclusion). Followups that add no new evidence skip the agent
# entirely. Media is treated as high-value evidence and always re-triages without
# even asking the classifier.
FOLLOWUP_RE_TRIAGE_PROVIDERS = ("deepseek", "claude")

FOLLOWUP_CLASSIFIER_PROMPT = """你是一个 bug 分诊结论变更预判器。一个 bug issue 已经完成过分诊，分诊后又来了补充消息。判断：是否需要重新完整分诊一次？

只输出一个词：change 或 no_change。不要输出任何其他内容。

判定标准：
- 补充消息提供了分诊需要的新信息（回答分诊提问、补充复现步骤/环境/平台/版本/日志/证据）、与现有结论冲突或修正、或附带新的截图/视频 → change
- 补充消息只是确认、寒暄、与结论一致的无信息量补充 → no_change
- 拿不准时输出 change（宁可多跑一次完整分诊，不可漏掉关键信息）

【issue 标题】
{title}

【issue 正文】
{body}

【已应用的分诊结论】
{conclusion}

【分诊后新增的补充消息】
{followups}"""

_MEDIA_EMBED_RE = re.compile(r"!\[[^\]]*\]\(https?://[^)]+\)")
_MEDIA_URL_RE = re.compile(
    r"https?://\S*\.(?:png|jpe?g|gif|webp|mp4|mov|m4v)(?:\?|#|&|\s|\)|$)", re.IGNORECASE
)


@dataclass(frozen=True)
class FollowupReTriage:
    requires_re_triage: bool
    # Terminal status the last triage wrote (Done/Skipped/Needs info); intake had
    # reset it to Pending so a no-change followup can put it back.
    prior_status: str = ""


def evaluate_followup_re_triage(
    *,
    issue: GitHubIssue,
    comments: tuple[GitHubIssueComment, ...],
    invocation: AgentInvocation,
) -> FollowupReTriage | None:
    """Decide whether a material followup needs a full re-triage run.

    Returns None when there is no prior applied triage (first run — always run
    the agent). Otherwise returns whether to re-triage, plus the prior terminal
    status so the no-change skip can restore it.
    """
    latest_triage_index: int | None = None
    for index, comment in enumerate(comments):
        if parse_triage_metadata(comment.body) is not None:
            latest_triage_index = index
    if latest_triage_index is None:
        return None
    prior_meta = parse_triage_metadata(comments[latest_triage_index].body) or {}
    prior_status = str(prior_meta.get("triage_status") or "")
    followups = tuple(
        comment
        for comment in comments[latest_triage_index + 1 :]
        if not is_triage_bookkeeping_comment(comment.body)
        and parse_triage_metadata(comment.body) is None
    )
    if not followups:
        # No material followup after the last triage; the already-triaged guard
        # normally returns before this, but stay safe and run the agent.
        return FollowupReTriage(requires_re_triage=True, prior_status=prior_status)
    if any(_comment_has_media(comment.body) for comment in followups):
        # New screenshots/video are the strongest new evidence: always re-triage.
        return FollowupReTriage(requires_re_triage=True, prior_status=prior_status)
    if invocation.provider not in FOLLOWUP_RE_TRIAGE_PROVIDERS:
        # Only deepseek/claude can drive the claude CLI classifier cheaply;
        # fall back to the expensive-but-correct full re-triage.
        return FollowupReTriage(requires_re_triage=True, prior_status=prior_status)
    decision = _run_followup_classifier(
        invocation=invocation,
        issue=issue,
        last_triage_comment=comments[latest_triage_index],
        followups=followups,
    )
    if decision == "no_change":
        return FollowupReTriage(requires_re_triage=False, prior_status=prior_status)
    # "change", or the classifier failed (None): be conservative and re-triage.
    return FollowupReTriage(requires_re_triage=True, prior_status=prior_status)


def _run_followup_classifier(
    *,
    invocation: AgentInvocation,
    issue: GitHubIssue,
    last_triage_comment: GitHubIssueComment,
    followups: tuple[GitHubIssueComment, ...],
) -> str | None:
    prompt = FOLLOWUP_CLASSIFIER_PROMPT.format(
        title=issue.title,
        body=_truncate(issue.body or "", 3000),
        conclusion=_truncate(last_triage_comment.body, 6000),
        followups="\n".join(f"- {_truncate(comment.body, 800)}" for comment in followups),
    )
    command = ["claude", "-p", prompt]
    if invocation.model:
        command += ["--model", invocation.model]
    # Same merge as the main agent: invocation.env carries only the provider
    # endpoint overrides; replacing the whole env would drop PATH/HOME and the
    # claude CLI would lose its config/cache.
    agent_env = {
        **os.environ,
        **(invocation.env or {}),
        # One-word completion; the unknown-model window enforcement only adds
        # startup noise to the classifier and never matters for it.
        "CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT": "1",
    }
    try:
        completed = subprocess.run(
            command,
            check=False,
            env=agent_env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            # The classifier is a tiny single-turn completion; seconds is plenty.
            # The 120s budget would only ever be consumed by provider retries on
            # a bad key, where failing fast is better than stalling the followup.
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        print(f"followup classifier failed: {error}", file=sys.stderr)
        return None
    if completed.returncode != 0:
        print(
            f"followup classifier exited {completed.returncode}: {(completed.stderr or '')[:500]}",
            file=sys.stderr,
        )
        return None
    return _classifier_decision(completed.stdout or "")


def _classifier_decision(output: str) -> str | None:
    lowered = output.lower()
    # "no_change" contains "change"; check the longer form first.
    if "no_change" in lowered or "no change" in lowered or "nochange" in lowered:
        return "no_change"
    if "change" in lowered:
        return "change"
    return None


def _comment_has_media(body: str) -> bool:
    return bool(_MEDIA_EMBED_RE.search(body) or _MEDIA_URL_RE.search(body))


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else f"{value[:limit]}…"


def mark_triage_running(
    *,
    config: ProjectConfig,
    issue_number: int,
    issue_fields: GitHubIssueFieldsClient,
) -> None:
    issue_fields.add_issue_field_values(
        repo=config.github_repo,
        issue_number=issue_number,
        values={"Triage status": "Running"},
        config=config,
    )


def mark_triage_superseded(
    *,
    config: ProjectConfig,
    issue_number: int,
    run_id: str,
    github: GitHubCliIssuesClient,
) -> None:
    # The newer run owns the status field; this run only leaves an audit trail.
    github.add_issue_comment(
        repo=config.github_repo,
        issue_number=issue_number,
        body=(
            f"{TRIAGE_SKIPPED_HEADER}\n\n"
            f"Run `{run_id}` was superseded by a newer triage run. Review the latest context before applying results."
        ),
    )


def mark_triage_failed(
    *,
    config: ProjectConfig,
    issue_number: int,
    exit_code: int,
    github: GitHubCliIssuesClient,
    issue_fields: GitHubIssueFieldsClient,
    lark: LarkMessengerClient | None = None,
    reason: str = "",
) -> None:
    issue_fields.add_issue_field_values(
        repo=config.github_repo,
        issue_number=issue_number,
        values={"Triage status": "Failed"},
        config=config,
    )
    github.add_issue_comment(
        repo=config.github_repo,
        issue_number=issue_number,
        body=render_triage_failed_comment(exit_code=exit_code, reason=reason),
    )
    if lark is not None:
        lines = [f"分诊失败，GitHub issue #{issue_number} 已标记 Failed，待重试或人工处理。"]
        if reason:
            lines.append(reason)
        send_intake_topic_message(
            repo=config.github_repo,
            issue_number=issue_number,
            github=github,
            lark=lark,
            text="\n".join(lines),
        )


JOB_LABELS = {
    "triage": "分诊",
    "fix": "自动修复",
    "fix-revise": "修复更新",
    "ci-fix": "CI 反馈",
    "close-audit": "关闭审计",
    "notify-fix": "修复通知",
}


def report_workflow_failure(
    *,
    config: ProjectConfig,
    issue_number: int,
    job: str,
    github: GitHubCliIssuesClient,
    issue_fields: GitHubIssueFieldsClient,
    lark: LarkMessengerClient | None = None,
    run_url: str = "",
    detail: str = "",
) -> str:
    """Report a bugpatrol job that died without reporting the failure itself.

    Any of these jobs can fail before bugpatrol starts (provisioning: tool clone
    fetch, venv install) or crash outside the paths that report on their own.
    Those failures leave no comment and no Lark ping, so the issue silently
    stalls. Dedup on the run URL cited in an existing comment, so re-running the
    reporter for one job is idempotent; triage additionally skips when it
    already marked the issue ``Failed`` on its own.
    """
    comments = github.list_issue_comments(repo=config.github_repo, issue_number=issue_number)
    if run_url and any(run_url in comment.body for comment in comments):
        return "already_reported"
    label = JOB_LABELS.get(job, job)
    lines = [f"{label}运行失败，本次没有产生结果。"]
    if detail:
        lines.append(detail)
    if run_url:
        lines.append(f"运行日志：[{run_url}]({run_url})")
    reason = "\n\n".join(lines)

    if job == "triage":
        # Triage owns the "Triage status" field, and the watcher re-dispatches
        # off it, so a failed triage must land there too.
        status = issue_fields.get_issue_field_values(
            repo=config.github_repo, issue_number=issue_number
        ).get(config.issue_field_names["Triage status"], "")
        if status == "Failed":
            return "already_reported"
        mark_triage_failed(
            config=config,
            issue_number=issue_number,
            exit_code=1,
            github=github,
            issue_fields=issue_fields,
            lark=lark,
            reason=reason,
        )
        return "reported"

    # Lark first, comment last: the comment is the dedup marker, so writing it
    # before a failed Lark send would suppress the ping forever.
    if lark is not None:
        send_intake_topic_message(
            repo=config.github_repo,
            issue_number=issue_number,
            github=github,
            lark=lark,
            text=f"{label}运行失败，GitHub issue #{issue_number} 需要人工跟进。\n{reason}",
        )
    github.add_issue_comment(
        repo=config.github_repo,
        issue_number=issue_number,
        body=f"## BugPatrol {job} failed\n\n{reason}",
    )
    return "reported"


def is_triage_bookkeeping_comment(body: str) -> bool:
    """True for comments bugpatrol writes about a triage run rather than about the bug.

    Run markers plus the "skipped"/"failed" audit notes carry no triage-relevant
    information, so they must not count as context: a run that yields to a newer
    one posts a skip note, and counting that note as new context made the newer
    run abort as stale — two runs, zero results.
    """
    if parse_triage_run_metadata(body) is not None:
        return True
    stripped = body.lstrip()
    return stripped.startswith(TRIAGE_SKIPPED_HEADER) or stripped.startswith(TRIAGE_FAILED_HEADER)


def comment_ids(comments: tuple[GitHubIssueComment, ...]) -> tuple[str, ...]:
    return tuple(comment.id for comment in comments if not is_triage_bookkeeping_comment(comment.body))


def triage_run_in_flight(
    comments: tuple[GitHubIssueComment, ...],
    *,
    now: datetime | None = None,
    stale_after: timedelta = TRIAGE_RUN_STALE_AFTER,
) -> bool:
    """True when a recent triage run claimed this issue and has not reported yet.

    Used to keep the scheduled reconcile from racing a triage the watcher just
    dispatched: both would run an agent, the older one would yield, and the
    newer one could still be dropped on stale context.
    """
    latest_started_at: datetime | None = None
    for comment in comments:
        metadata = parse_triage_run_metadata(comment.body)
        if metadata is not None:
            started_at = metadata.get("started_at")
            latest_started_at = _parse_timestamp(started_at) if isinstance(started_at, str) else None
            continue
        stripped = comment.body.lstrip()
        if stripped.startswith(TRIAGE_SKIPPED_HEADER) or stripped.startswith(TRIAGE_FAILED_HEADER):
            # The run that owned the newest marker already reported its outcome.
            latest_started_at = None
    if latest_started_at is None:
        return False
    return (now or datetime.now(timezone.utc)) - latest_started_at < stale_after


def _parse_timestamp(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def append_triage_run_metadata(metadata: dict[str, object]) -> str:
    return (
        f"{TRIAGE_RUN_META_START}\n"
        f"{json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2)}\n"
        f"{TRIAGE_RUN_META_END}"
    )


def parse_triage_run_metadata(comment_body: str) -> dict[str, object] | None:
    start = comment_body.find(TRIAGE_RUN_META_START)
    if start == -1:
        return None
    json_start = start + len(TRIAGE_RUN_META_START)
    end = comment_body.find(TRIAGE_RUN_META_END, json_start)
    if end == -1:
        return None
    data = json.loads(comment_body[json_start:end].strip())
    return data if isinstance(data, dict) else None


def latest_triage_run_id(comments: tuple[GitHubIssueComment, ...]) -> str:
    latest = ""
    for comment in comments:
        metadata = parse_triage_run_metadata(comment.body)
        if metadata is not None and isinstance(metadata.get("run_id"), str):
            latest = str(metadata["run_id"])
    return latest


def annotate_result_stale_context(result: TriageResult) -> TriageResult:
    comment = "\n".join(
        [
            result.comment_markdown.rstrip(),
            "",
            "> BugPatrol note: new issue comments kept arriving during triage retries; this result may not reflect the very latest comments.",
        ]
    )
    return replace(result, comment_markdown=comment)


def _bounded_log_tail(text: str | None, *, max_lines: int = 25, max_chars: int = 1500) -> str:
    """ANSI-stripped tail of an agent log, bounded for a comment / Lark message."""
    if not text:
        return ""
    lines = strip_ansi(text).rstrip().splitlines()
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def agent_failure_reason(
    headline: str, *, stderr: str | None, stdout: str | None = None
) -> str:
    """Build a failure reason that surfaces the agent's real error output.

    Without this the failure comment is just "exited with code 1", which forces
    an operator to SSH into whichever runner ran the job and read the turn log.
    Two sources carry the actionable detail:

    * ``stdout`` — provider/API errors (e.g. DeepSeek 402 "Insufficient
      Balance", auth failures) are emitted as a ``result`` event with
      ``is_error: true`` on stdout while stderr stays empty;
    * ``stderr`` — network/tooling failures (bounded, de-ANSI'd tail).

    Surface whichever are present directly in the issue comment and Lark ping.
    """
    agent_error = agent_error_from_stdout(stdout)
    stderr_tail = _bounded_log_tail(stderr)
    sections = [headline]
    if agent_error:
        sections.append(
            f"Agent error:\n```\n{_bounded_log_tail(agent_error)}\n```"
        )
    if stderr_tail:
        sections.append(f"Agent stderr (tail):\n```\n{stderr_tail}\n```")
    if not agent_error and not stderr_tail:
        sections.append(
            "No agent stderr or error output was captured; check the runner logs, "
            "credentials, prompt/schema files, and repository checkout."
        )
    return "\n\n".join(sections)


def render_triage_failed_comment(*, exit_code: int, reason: str = "") -> str:
    detail = reason or (
        f"The triage agent exited with code `{exit_code}`.\n"
        "Check the runner logs, credentials, prompt/schema files, and repository checkout."
    )
    return "\n".join(
        [
            TRIAGE_FAILED_HEADER,
            "",
            detail,
        ]
    )
