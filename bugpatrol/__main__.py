"""Command-line entry point for bugpatrol."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

from bugpatrol.agents import build_triage_agent_invocation
from bugpatrol.asset_cleanup import cleanup_asset_repo
from bugpatrol.backfill import run_lark_backfill
from bugpatrol.chat_discovery import (
    CachedBranchChatDiscoverer,
    apply_branch_chats,
    discover_branch_chats,
)
from bugpatrol.ci_watch import watch_ci
from bugpatrol.close_audit import audit_issue_close
from bugpatrol.config import load_project_config
from bugpatrol.doctor import run_doctor
from bugpatrol.event_watcher import iter_json_event_lines, run_lark_event_watcher
from bugpatrol.fields import TRIAGE_OUTPUT_SCHEMA, default_field_specs
from bugpatrol.fix_notify import (
    FIX_EVENTS,
    apply_fix_notification,
    collect_fix_candidates_from_github,
    fix_event_candidates_from_json,
    reconcile_fix_notifications,
    resolve_single_issue_from_pr,
)
from bugpatrol.fix_runner import (
    read_triage_verdict,
    run_ci_feedback,
    run_fix,
    run_fix_revise,
)
from bugpatrol.github import GitHubCliIssuesClient
from bugpatrol.github_fields import GitHubIssueFieldsClient
from bugpatrol.intake_workflow import IntakeWorkflow
from bugpatrol.lark import LarkOpenApiMessengerClient
from bugpatrol.mail import LarkMailClient
from bugpatrol.ownership import load_codeowners, resolve_configured_owners, resolve_owners
from bugpatrol.prd import load_prd_documents, search_prd_documents
from bugpatrol.reconcile_triage import reconcile_triage
from bugpatrol.resources import (
    CommandResourceDescriber,
    CommandResourceRedactor,
    CommandVideoFrameExtractor,
    CompositeResourceTransformer,
    FfprobeVideoDurationProbe,
    GitHubAssetRepoStore,
    ImageResourceResizer,
    ResourcePolicy,
    ResourceTransformer,
)
from bugpatrol.slash_commands import SlashCommandHandler, make_dispatch
from bugpatrol.triage_context import (
    ReferenceRepoContext,
    build_triage_context,
    render_triage_context_markdown,
)
from bugpatrol.triage_result import (
    apply_triage_result,
    build_triage_dry_run_report,
    parse_triage_result,
)
from bugpatrol.triage_runner import (
    JOB_LABELS,
    execute_triage_run,
    prepare_triage_run,
    report_workflow_failure,
    resolve_issue_branch,
)
from bugpatrol.watch_mail import MailIntakeWorkflow, run_mail_watcher
from bugpatrol.watcher import GitHubTriageStatusReader, run_polling_watcher
from bugpatrol.worktree import (
    SubprocessGitDriver,
    resolve_reference_branch,
    triage_worktree,
)


def media_resource_policy(config) -> ResourcePolicy:  # type: ignore[no-untyped-def]
    video_duration_probe = None
    if config.media.max_video_duration_seconds > 0:
        temp_dir = Path(config.media.description_temp_dir) if config.media.description_temp_dir else None
        video_duration_probe = FfprobeVideoDurationProbe(
            command=config.media.video_probe_command or FfprobeVideoDurationProbe.DEFAULT_COMMAND,
            timeout_seconds=config.media.video_probe_timeout_seconds,
            temp_dir=temp_dir,
        )
    return ResourcePolicy(
        max_image_bytes=config.media.max_image_bytes,
        max_video_bytes=config.media.max_video_bytes,
        max_file_bytes=config.media.max_file_bytes,
        max_video_duration_seconds=config.media.max_video_duration_seconds,
        video_duration_probe=video_duration_probe,
    )


def media_resource_describer(config) -> CommandResourceDescriber | None:  # type: ignore[no-untyped-def]
    if not config.media.description_command:
        return None
    temp_dir = Path(config.media.description_temp_dir) if config.media.description_temp_dir else None
    return CommandResourceDescriber(
        command=config.media.description_command,
        timeout_seconds=config.media.description_timeout_seconds,
        temp_dir=temp_dir,
        retries=config.media.description_retries,
        retry_backoff_seconds=config.media.description_retry_backoff_seconds,
    )


def media_resource_redactor(config) -> CommandResourceRedactor | None:  # type: ignore[no-untyped-def]
    if not config.media.redaction_command:
        return None
    temp_dir = Path(config.media.description_temp_dir) if config.media.description_temp_dir else None
    return CommandResourceRedactor(
        command=config.media.redaction_command,
        timeout_seconds=config.media.redaction_timeout_seconds,
        temp_dir=temp_dir,
    )


def media_resource_transformer(config) -> ResourceTransformer | None:  # type: ignore[no-untyped-def]
    transformers: list[ResourceTransformer] = []
    temp_dir = Path(config.media.description_temp_dir) if config.media.description_temp_dir else None
    if config.media.video_frame_command:
        duration_probe = None
        if config.media.video_frame_min_duration_seconds > 0:
            duration_probe = FfprobeVideoDurationProbe(
                command=config.media.video_probe_command or FfprobeVideoDurationProbe.DEFAULT_COMMAND,
                timeout_seconds=config.media.video_probe_timeout_seconds,
                temp_dir=temp_dir,
            )
        transformers.append(
            CommandVideoFrameExtractor(
                command=config.media.video_frame_command,
                timeout_seconds=config.media.video_frame_timeout_seconds,
                temp_dir=temp_dir,
                min_duration_seconds=config.media.video_frame_min_duration_seconds,
                duration_probe=duration_probe,
            )
        )
    if (
        config.media.resize_max_image_width
        or config.media.resize_max_image_height
        or config.media.convert_images_to_jpeg
    ):
        transformers.append(
            ImageResourceResizer(
                max_width=config.media.resize_max_image_width,
                max_height=config.media.resize_max_image_height,
                quality=config.media.resize_image_quality,
                convert_to_jpeg=config.media.convert_images_to_jpeg,
            )
        )
    if not transformers:
        return None
    if len(transformers) == 1:
        return transformers[0]
    return CompositeResourceTransformer(tuple(transformers))


def _optional_lark_client(config) -> LarkOpenApiMessengerClient | None:  # type: ignore[no-untyped-def]
    app_secret = os.environ.get(config.lark.app_secret_env)
    if not app_secret:
        return None
    return LarkOpenApiMessengerClient(
            app_id=config.lark.app_id,
            app_secret=app_secret,
            base_url=config.lark.api_base_url,
        )


def _parse_reference_repo_args(values: list[str] | None) -> dict[str, Path]:
    """Parse repeated ``--reference-repo REPO=PATH`` into a repo->checkout map."""
    checkouts: dict[str, Path] = {}
    for raw in values or ():
        repo, sep, path = raw.partition("=")
        if not sep or not repo or not path:
            raise ValueError(f"--reference-repo must be REPO=PATH, got {raw!r}")
        if repo in checkouts:
            raise ValueError(f"duplicate --reference-repo for {repo}")
        checkouts[repo] = Path(path)
    return checkouts


def _enter_reference_worktrees(
    stack: contextlib.ExitStack,
    *,
    reference_repos,  # type: ignore[no-untyped-def]
    checkouts: dict[str, Path],
    main_declared_branch: str,
) -> tuple[ReferenceRepoContext, ...]:
    """Resolve + check out each configured reference repo at the right branch.

    Bounded by the configured reference-repo list. A declared reference repo
    with no supplied checkout is a hard error (fail loud) — a silently missing
    sibling would let triage cross-check nothing.
    """
    contexts: list[ReferenceRepoContext] = []
    for ref in reference_repos:
        checkout = checkouts.get(ref.repo)
        if checkout is None:
            raise ValueError(
                f"reference repo {ref.repo!r} is configured but no checkout was "
                f"supplied; pass --reference-repo {ref.repo}=<path>"
            )
        checkout = checkout.resolve()
        if not checkout.is_dir():
            raise FileNotFoundError(
                f"reference repo {ref.repo!r} checkout not found at {checkout}"
            )
        ref_branch = ref.branch_for(main_declared_branch)
        resolution = resolve_reference_branch(
            SubprocessGitDriver(checkout),
            repo=ref.repo,
            branch=ref_branch,
        )
        worktree_path = stack.enter_context(
            triage_worktree(base_repo=checkout, ref=resolution.ref)
        )
        contexts.append(
            ReferenceRepoContext(
                repo=ref.repo,
                path=str(worktree_path),
                analyzed_branch=resolution.analyzed_branch,
                purpose=ref.purpose,
            )
        )
    return tuple(contexts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bugpatrol")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-config", help="validate a project TOML config")
    validate.add_argument("path", type=Path)
    validate.add_argument("--live", action="store_true", help="also run external GitHub/tool checks")

    schema = sub.add_parser("schema", help="print the triage JSON schema")
    schema.add_argument("--pretty", action="store_true")

    agent = sub.add_parser("agent-command", help="print the configured triage agent command")
    agent.add_argument("project_config", type=Path)
    agent.add_argument("--issue", type=int, required=True)
    agent.add_argument("--prompt", type=Path, default=Path("prompts/triage.zh.md"))
    agent.add_argument("--schema", type=Path, default=Path("triage.schema.json"))
    agent.add_argument("--output", type=Path, default=Path("triage-output.json"))
    agent.add_argument("--context", type=Path)

    backfill = sub.add_parser("backfill-lark", help="backfill recent Lark messages into GitHub")
    backfill.add_argument("project_config", type=Path)
    backfill.add_argument("--limit", type=int, default=20)
    backfill.add_argument("--write", action="store_true", help="perform writes; default is dry-run")
    backfill.add_argument("--resource-dir", type=Path, help="download Lark resources before writing issues")
    backfill.add_argument("--asset-repo", action="store_true", help="upload Lark resources to configured assets repo")
    backfill.add_argument("--root", action="append", default=[], help="only process messages in these topic root_ids (repeatable)")

    doctor = sub.add_parser("doctor", help="check project integration dependencies")
    doctor.add_argument("project_config", type=Path)
    doctor.add_argument("--with-lark", action="store_true")

    watch = sub.add_parser("watch-lark", help="poll Lark and mirror messages into GitHub")
    watch.add_argument("project_config", type=Path)
    watch.add_argument("--limit", type=int, default=20)
    watch.add_argument("--interval", type=float, default=30)
    watch.add_argument("--once", action="store_true")
    watch.add_argument("--dry-run", action="store_true", help="scan without GitHub writes")
    watch.add_argument("--resource-dir", type=Path, help="download Lark resources before writing issues")
    watch.add_argument("--asset-repo", action="store_true", help="upload Lark resources to configured assets repo")
    watch.add_argument("--event-log", type=Path, help="append structured watcher events to a JSONL file")
    watch.add_argument("--processed-ledger", type=Path, help="persist processed Lark message ids in this JSON file")
    watch.add_argument("--lease-file", type=Path, help="single-writer lease file for this watcher")
    watch.add_argument("--lease-ttl-seconds", type=float, default=120)
    watch.add_argument("--triage-queue", type=Path, help="persist debounced triage requests in this JSON file")
    watch.add_argument("--triage-quiet-seconds", type=float, default=60)
    watch.add_argument(
        "--triage-dispatch-command",
        nargs="+",
        help=(
            "command used for due triage requests; supports {issue_number}, "
            "{trigger_fingerprint}, and {reason}"
        ),
    )
    watch.add_argument(
        "--fix-dispatch-command",
        nargs="+",
        help=(
            "command run when a reporter posts `/fix` in a bug topic; supports "
            "{issue_number}. Without it, `/fix` replies that it is unconfigured."
        ),
    )
    watch.add_argument(
        "--retriage-dispatch-command",
        nargs="+",
        help=(
            "command run when a reporter posts `/retriage` in a bug topic; supports "
            "{issue_number}. Without it, `/retriage` replies that it is unconfigured."
        ),
    )
    watch.add_argument(
        "--parallel-topics",
        type=int,
        default=1,
        help="process up to N Lark topics concurrently (attachment download/vision/intake)",
    )

    event_watch = sub.add_parser("watch-lark-events", help="read Lark event NDJSON from stdin into GitHub")
    event_watch.add_argument("project_config", type=Path)
    event_watch.add_argument("--dry-run", action="store_true", help="scan without GitHub writes")
    event_watch.add_argument("--resource-dir", type=Path, help="download Lark resources before writing issues")
    event_watch.add_argument("--asset-repo", action="store_true", help="upload Lark resources to configured assets repo")
    event_watch.add_argument("--event-log", type=Path, help="append structured watcher events to a JSONL file")
    event_watch.add_argument("--processed-ledger", type=Path, help="persist processed Lark message ids in this JSON file")
    event_watch.add_argument("--triage-queue", type=Path, help="persist debounced triage requests in this JSON file")
    event_watch.add_argument("--triage-quiet-seconds", type=float, default=60)
    event_watch.add_argument(
        "--triage-dispatch-command",
        nargs="+",
        help=(
            "command used for due triage requests; supports {issue_number}, "
            "{trigger_fingerprint}, and {reason}"
        ),
    )

    watch_mail = sub.add_parser(
        "watch-mail",
        help="poll the public mailbox ([mail]) and mirror reports into GitHub",
    )
    watch_mail.add_argument("project_config", type=Path)
    watch_mail.add_argument("--limit", type=int, default=20)
    watch_mail.add_argument("--interval", type=float, default=60)
    watch_mail.add_argument("--once", action="store_true")
    watch_mail.add_argument("--dry-run", action="store_true", help="scan without GitHub writes")
    watch_mail.add_argument("--resource-dir", type=Path, help="download mail attachments before writing issues")
    watch_mail.add_argument("--asset-repo", action="store_true", help="upload mail attachments to configured assets repo")
    watch_mail.add_argument("--event-log", type=Path, help="append structured watcher events to a JSONL file")
    watch_mail.add_argument("--processed-ledger", type=Path, help="persist processed mail message ids in this JSON file")
    watch_mail.add_argument("--lease-file", type=Path, help="single-writer lease file for this watcher")
    watch_mail.add_argument("--lease-ttl-seconds", type=float, default=120)
    watch_mail.add_argument("--triage-queue", type=Path, help="persist debounced triage requests in this JSON file")
    watch_mail.add_argument("--triage-quiet-seconds", type=float, default=60)
    watch_mail.add_argument(
        "--triage-dispatch-command",
        nargs="+",
        help=(
            "command used for due triage requests; supports {issue_number}, "
            "{trigger_fingerprint}, and {reason}"
        ),
    )

    owner = sub.add_parser("resolve-owner", help="resolve owners for paths using CODEOWNERS")
    owner.add_argument("--project-config", type=Path)
    owner.add_argument("--capability", default="")
    owner.add_argument("repo_path", type=Path)
    owner.add_argument("paths", nargs="+")

    prd = sub.add_parser("search-prd", help="search local PRD markdown docs")
    prd.add_argument("root", type=Path)
    prd.add_argument("query")
    prd.add_argument("--limit", type=int, default=5)
    prd.add_argument("--include-glob", action="append", default=None)

    context = sub.add_parser("issue-context", help="build triage context markdown for an issue")
    context.add_argument("project_config", type=Path)
    context.add_argument("--issue", type=int, required=True)
    context.add_argument("--repo-path", type=Path, required=True)
    context.add_argument("--output", type=Path)

    apply_result = sub.add_parser("apply-triage-result", help="validate and apply triage JSON")
    apply_result.add_argument("project_config", type=Path)
    apply_result.add_argument("--issue", type=int, required=True)
    apply_result.add_argument("--input", type=Path, required=True)
    apply_result.add_argument("--dry-run", action="store_true", help="validate and report changes without writing")

    run_triage = sub.add_parser("run-triage", help="prepare and optionally execute triage")
    run_triage.add_argument("project_config", type=Path)
    run_triage.add_argument("--issue", type=int, required=True)
    run_triage.add_argument("--repo-path", type=Path, required=True)
    run_triage.add_argument("--output-dir", type=Path, default=Path(".bugpatrol/triage-run"))
    run_triage.add_argument("--execute", action="store_true")
    run_triage.add_argument(
        "--reference-repo",
        action="append",
        default=None,
        metavar="REPO=PATH",
        help=(
            "checkout path for a configured [[triage.reference_repos]] repo; "
            "repeatable, e.g. --reference-repo org/weaver=/cache/weaver"
        ),
    )
    run_triage.add_argument(
        "--fix-revise-dispatch-command",
        nargs="+",
        help=(
            "command run instead of triage when the issue already has an open "
            "bugpatrol fix PR, so a reporter's follow-up correction revises the PR "
            "rather than re-triaging to a no-op; supports {issue_number}."
        ),
    )

    report_failure = sub.add_parser(
        "report-job-failure",
        help="surface a bugpatrol job that died without reporting the failure itself",
    )
    report_failure.add_argument("project_config", type=Path)
    report_failure.add_argument("--issue", type=int, required=True)
    report_failure.add_argument("--job", default="triage", choices=sorted(JOB_LABELS))
    report_failure.add_argument("--run-url", default="", help="CI run URL to cite")
    report_failure.add_argument("--detail", default="", help="extra context, e.g. the runner name")

    run_fix_parser = sub.add_parser("run-fix", help="auto-fix a triaged code bug and open a PR")
    run_fix_parser.add_argument("project_config", type=Path)
    run_fix_parser.add_argument("--issue", type=int, required=True)
    run_fix_parser.add_argument("--repo-path", type=Path, required=True)
    run_fix_parser.add_argument("--output-dir", type=Path, default=Path(".bugpatrol/fix-run"))
    run_fix_parser.add_argument("--execute", action="store_true", help="actually run the agent and open a PR")

    run_fix_revise_parser = sub.add_parser(
        "run-fix-revise", help="address open-PR review feedback on an existing fix"
    )
    run_fix_revise_parser.add_argument("project_config", type=Path)
    run_fix_revise_parser.add_argument("--issue", type=int, required=True)
    run_fix_revise_parser.add_argument("--repo-path", type=Path, required=True)
    run_fix_revise_parser.add_argument("--output-dir", type=Path, default=Path(".bugpatrol/fix-revise"))
    run_fix_revise_parser.add_argument(
        "--execute", action="store_true", help="actually run the agent and push the update"
    )

    run_ci_feedback_parser = sub.add_parser(
        "run-ci-feedback",
        help="react to a project PR CI build finishing, reporting to the managed issue",
    )
    run_ci_feedback_parser.add_argument("project_config", type=Path)
    run_ci_feedback_parser.add_argument("--head-branch", required=True)
    run_ci_feedback_parser.add_argument("--head-sha", required=True)
    run_ci_feedback_parser.add_argument(
        "--conclusion", required=True, choices=["success", "failure"]
    )
    run_ci_feedback_parser.add_argument("--repo-path", type=Path, required=True)
    run_ci_feedback_parser.add_argument(
        "--output-dir", type=Path, default=Path(".bugpatrol/ci-fix")
    )
    run_ci_feedback_parser.add_argument(
        "--execute", action="store_true", help="actually react (notify / revise)"
    )

    reconcile_triage_parser = sub.add_parser(
        "reconcile-triage", help="triage intook issues that never got a triage result"
    )
    reconcile_triage_parser.add_argument("project_config", type=Path)
    # --repo-path / --output-dir are accepted for backward compatibility with
    # deployed workflows but no longer used: reconcile dispatches the triage
    # workflow per candidate instead of running triage in-process.
    reconcile_triage_parser.add_argument("--repo-path", type=Path)
    reconcile_triage_parser.add_argument("--output-dir", type=Path, default=Path(".bugpatrol/triage-run"))
    reconcile_triage_parser.add_argument("--execute", action="store_true")

    notify_fix = sub.add_parser("notify-fix", help="notify Lark about explicit fix progress")
    notify_fix.add_argument("project_config", type=Path)
    notify_fix.add_argument("--issue", type=int)
    notify_fix.add_argument("--event", choices=FIX_EVENTS, required=True)
    notify_fix.add_argument("--pr", default="")
    notify_fix.add_argument("--commit", default="")
    notify_fix.add_argument("--write", action="store_true", help="send Lark notification and write metadata")

    audit_close = sub.add_parser(
        "audit-issue-close", help="check that a closed-as-completed issue references a fix commit/PR"
    )
    audit_close.add_argument("project_config", type=Path)
    audit_close.add_argument("--issue", type=int, required=True)
    audit_close.add_argument("--write", action="store_true", help="post nag comment and Lark reply when evidence is missing")

    reconcile_fix = sub.add_parser("reconcile-fix-notifications", help="replay missed fix notification events")
    reconcile_fix.add_argument("project_config", type=Path)
    reconcile_fix.add_argument(
        "--input",
        type=Path,
        help="JSON array of fix notification events (omit with --from-github)",
    )
    reconcile_fix.add_argument(
        "--from-github",
        action="store_true",
        help="collect candidates from GitHub (merged PRs, closed managed issues, linked commits)",
    )
    reconcile_fix.add_argument("--pr-limit", type=int, default=30, help="merged PRs to scan with --from-github")
    reconcile_fix.add_argument(
        "--closed-issue-limit",
        type=int,
        default=100,
        help="closed issues to scan with --from-github",
    )
    reconcile_fix.add_argument(
        "--since-days",
        type=int,
        default=0,
        help="only backfill fix events within the last N days (0 = no window)",
    )
    reconcile_fix.add_argument(
        "--resend",
        action="store_true",
        help="re-deliver already-notified events in the current format without minting a new marker",
    )
    reconcile_fix.add_argument("--write", action="store_true", help="send Lark notifications and write metadata")

    cleanup_assets = sub.add_parser("cleanup-assets", help="dry-run or delete materialized asset repo files")
    cleanup_assets.add_argument("project_config", type=Path)
    cleanup_assets.add_argument("--message-id-prefix", default="", help="only match asset paths with this prefix")
    cleanup_assets.add_argument("--delete", action="store_true", help="delete matching files or directories")
    cleanup_assets.add_argument("--push", action="store_true", help="commit and push after --delete")

    ci_watch = sub.add_parser(
        "ci-watch",
        help="alert a Lark chat when a repo's CI goes red on a branch",
    )
    ci_watch.add_argument("project_config", type=Path, help="project config supplying the Lark app")
    ci_watch.add_argument("--repo", required=True, help="repo to watch, as owner/name")
    ci_watch.add_argument("--branch", default="main", help="branch whose newest settled commit is judged")
    ci_watch.add_argument("--chat-id", required=True, help="Lark chat to alert into")
    ci_watch.add_argument(
        "--state-path",
        type=Path,
        default=Path.home() / ".bugpatrol" / "ci-watch-state.json",
        help="file remembering the red commit already alerted on",
    )
    ci_watch.add_argument("--write", action="store_true", help="send the alert and persist state")

    args = parser.parse_args(argv)

    if args.command == "validate-config":
        config = load_project_config(args.path)
        config.validate_against(default_field_specs())
        if args.live:
            checks = run_doctor(
                config=config,
                github=GitHubCliIssuesClient(gh=config.github_cli),
                issue_fields=GitHubIssueFieldsClient(gh=config.github_cli),
                projects_dir=args.path.parent,
            )
            if not all(check.ok for check in checks):
                print(json.dumps([check.__dict__ for check in checks], ensure_ascii=False), file=sys.stderr)
                return 1
        print(f"ok: {config.project}")
        return 0

    if args.command == "schema":
        if args.pretty:
            print(json.dumps(TRIAGE_OUTPUT_SCHEMA, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(TRIAGE_OUTPUT_SCHEMA, ensure_ascii=False, separators=(",", ":")))
        return 0

    if args.command == "agent-command":
        config = load_project_config(args.project_config)
        invocation = build_triage_agent_invocation(
            config,
            issue_number=args.issue,
            prompt_path=args.prompt,
            schema_path=args.schema,
            output_path=args.output,
            context_path=args.context,
        )
        print(json.dumps({"provider": invocation.provider, "command": invocation.command}, ensure_ascii=False))
        return 0

    if args.command == "backfill-lark":
        config = load_project_config(args.project_config)
        app_secret = os.environ.get(config.lark.app_secret_env)
        if not app_secret:
            print(f"missing env: {config.lark.app_secret_env}", file=sys.stderr)
            return 2
        lark = LarkOpenApiMessengerClient(
            app_id=config.lark.app_id,
            app_secret=app_secret,
            base_url=config.lark.api_base_url,
        )
        issue_fields = GitHubIssueFieldsClient(gh=config.github_cli)
        github = GitHubCliIssuesClient(
            gh=config.github_cli,
            issue_fields=issue_fields,
            project_config=config,
        )
        # Same chat set the watcher scans, so a backfill can catch up a branch
        # topic group that was never listed in config.
        config = apply_branch_chats(
            config,
            discover_branch_chats(lark=lark, github=github, config=config).branch_chats,
        )
        workflow = IntakeWorkflow(config=config, github=github, lark=lark, issue_fields=issue_fields)
        if args.resource_dir and args.asset_repo:
            print("--resource-dir and --asset-repo are mutually exclusive", file=sys.stderr)
            return 2
        resource_store = None
        if args.asset_repo:
            if not config.assets.github_repo or not config.assets.checkout_path:
                print("missing [assets] github_repo or checkout_path", file=sys.stderr)
                return 2
            resource_store = GitHubAssetRepoStore(
                repo=config.assets.github_repo,
                checkout_path=Path(config.assets.checkout_path),
                base_path=config.assets.base_path,
                branch=config.assets.branch,
                remote_url=config.assets.remote_url,
            )
        resource_describer = media_resource_describer(config)
        resource_redactor = media_resource_redactor(config)
        resource_transformer = media_resource_transformer(config)
        resource_policy = media_resource_policy(config)
        result = run_lark_backfill(
            root_allowlist=tuple(args.root),
            config=config,
            lark=lark,
            workflow=workflow,
            limit=args.limit,
            dry_run=not args.write,
            resource_dir=args.resource_dir,
            resource_store=resource_store,
            resource_describer=resource_describer,
            resource_policy=resource_policy,
            resource_redactor=resource_redactor,
            resource_transformer=resource_transformer,
        )
        print(
            json.dumps(
                {
                    "dry_run": not args.write,
                    "scanned": result.scanned,
                    "processed": result.processed,
                    "skipped": result.skipped,
                    "issues": [
                        {
                            "action": outcome.action,
                            "number": outcome.issue.number,
                            "url": outcome.issue.url,
                        }
                        for outcome in result.outcomes
                    ],
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "doctor":
        config = load_project_config(args.project_config)
        lark = None
        if args.with_lark:
            app_secret = os.environ.get(config.lark.app_secret_env)
            if not app_secret:
                print(f"missing env: {config.lark.app_secret_env}", file=sys.stderr)
                return 2
            lark = LarkOpenApiMessengerClient(
                app_id=config.lark.app_id,
                app_secret=app_secret,
                base_url=config.lark.api_base_url,
            )
        checks = run_doctor(
            config=config,
            github=GitHubCliIssuesClient(gh=config.github_cli),
            issue_fields=GitHubIssueFieldsClient(gh=config.github_cli),
            lark=lark,
            projects_dir=args.project_config.parent,
        )
        print(json.dumps([check.__dict__ for check in checks], ensure_ascii=False))
        return 0 if all(check.ok for check in checks) else 1

    if args.command == "watch-lark":
        config = load_project_config(args.project_config)
        app_secret = os.environ.get(config.lark.app_secret_env)
        if not app_secret:
            print(f"missing env: {config.lark.app_secret_env}", file=sys.stderr)
            return 2
        lark = LarkOpenApiMessengerClient(
            app_id=config.lark.app_id,
            app_secret=app_secret,
            base_url=config.lark.api_base_url,
        )
        issue_fields = GitHubIssueFieldsClient(gh=config.github_cli)
        github = GitHubCliIssuesClient(
            gh=config.github_cli,
            issue_fields=issue_fields,
            project_config=config,
        )
        workflow = IntakeWorkflow(config=config, github=github, lark=lark, issue_fields=issue_fields)
        if args.resource_dir and args.asset_repo:
            print("--resource-dir and --asset-repo are mutually exclusive", file=sys.stderr)
            return 2
        resource_store = None
        if args.asset_repo:
            if not config.assets.github_repo or not config.assets.checkout_path:
                print("missing [assets] github_repo or checkout_path", file=sys.stderr)
                return 2
            resource_store = GitHubAssetRepoStore(
                repo=config.assets.github_repo,
                checkout_path=Path(config.assets.checkout_path),
                base_path=config.assets.base_path,
                branch=config.assets.branch,
                remote_url=config.assets.remote_url,
            )
        resource_describer = media_resource_describer(config)
        resource_redactor = media_resource_redactor(config)
        resource_transformer = media_resource_transformer(config)
        resource_policy = media_resource_policy(config)
        slash_handler = SlashCommandHandler(
            config=config,
            github=github,
            lark=lark,
            fix_dispatch=(
                make_dispatch(args.fix_dispatch_command) if args.fix_dispatch_command else None
            ),
            retriage_dispatch=(
                make_dispatch(args.retriage_dispatch_command)
                if args.retriage_dispatch_command
                else None
            ),
        )
        result = run_polling_watcher(
            config=config,
            lark=lark,
            workflow=workflow,
            limit=args.limit,
            interval_seconds=args.interval,
            once=args.once,
            dry_run=args.dry_run,
            resource_dir=args.resource_dir,
            resource_store=resource_store,
            resource_describer=resource_describer,
            resource_policy=resource_policy,
            resource_redactor=resource_redactor,
            resource_transformer=resource_transformer,
            event_log_path=args.event_log,
            processed_ledger_path=args.processed_ledger,
            lease_file=args.lease_file,
            lease_ttl_seconds=args.lease_ttl_seconds,
            triage_queue_path=args.triage_queue,
            triage_quiet_seconds=args.triage_quiet_seconds,
            triage_dispatch_command=args.triage_dispatch_command,
            reconcile_dispatch_command=config.reconcile.dispatch_command or None,
            reconcile_interval_seconds=config.reconcile.interval_seconds,
            triage_status_reader=GitHubTriageStatusReader(
                config=config,
                issue_fields=GitHubIssueFieldsClient(gh=config.github_cli),
            )
            if args.triage_dispatch_command
            else None,
            parallel_topics=args.parallel_topics,
            branch_chat_discoverer=CachedBranchChatDiscoverer(
                lark=lark,
                github=github,
                config=config,
            ),
            # Branch chats can now appear at runtime via discovery, so the
            # resolver is always wired: it is only called for a non-main chat.
            branch_tip_resolver=(
                lambda branch: github.remote_branch_tip_sha(repo=config.github_repo, branch=branch)
            ),
            slash_handler=slash_handler,
        )
        print(json.dumps(result.__dict__, ensure_ascii=False))
        return 0

    if args.command == "watch-mail":
        config = load_project_config(args.project_config)
        if config.mail is None:
            print("project config has no [mail] section", file=sys.stderr)
            return 2
        app_secret = os.environ.get(config.mail.app_secret_env)
        if not app_secret:
            print(f"missing env: {config.mail.app_secret_env}", file=sys.stderr)
            return 2
        mail = LarkMailClient(
            app_id=config.mail.app_id,
            app_secret=app_secret,
            base_url=config.lark.api_base_url,
        )
        lark_app_secret = os.environ.get(config.lark.app_secret_env)
        if not lark_app_secret:
            print(f"missing env: {config.lark.app_secret_env}", file=sys.stderr)
            return 2
        lark = LarkOpenApiMessengerClient(
            app_id=config.lark.app_id,
            app_secret=lark_app_secret,
            base_url=config.lark.api_base_url,
        )
        issue_fields = GitHubIssueFieldsClient(gh=config.github_cli)
        github = GitHubCliIssuesClient(
            gh=config.github_cli,
            issue_fields=issue_fields,
            project_config=config,
        )
        workflow = MailIntakeWorkflow(config=config, github=github, lark=lark, issue_fields=issue_fields)
        if args.resource_dir and args.asset_repo:
            print("--resource-dir and --asset-repo are mutually exclusive", file=sys.stderr)
            return 2
        resource_store = None
        if args.asset_repo:
            if not config.assets.github_repo or not config.assets.checkout_path:
                print("missing [assets] github_repo or checkout_path", file=sys.stderr)
                return 2
            resource_store = GitHubAssetRepoStore(
                repo=config.assets.github_repo,
                checkout_path=Path(config.assets.checkout_path),
                base_path=config.assets.base_path,
                branch=config.assets.branch,
                remote_url=config.assets.remote_url,
            )
        result = run_mail_watcher(
            config=config,
            mail=mail,
            workflow=workflow,
            limit=args.limit,
            interval_seconds=args.interval,
            once=args.once,
            dry_run=args.dry_run,
            resource_dir=args.resource_dir,
            resource_store=resource_store,
            resource_describer=media_resource_describer(config),
            resource_policy=media_resource_policy(config),
            resource_redactor=media_resource_redactor(config),
            resource_transformer=media_resource_transformer(config),
            event_log_path=args.event_log,
            processed_ledger_path=args.processed_ledger,
            lease_file=args.lease_file,
            lease_ttl_seconds=args.lease_ttl_seconds,
            triage_queue_path=args.triage_queue,
            triage_quiet_seconds=args.triage_quiet_seconds,
            triage_dispatch_command=args.triage_dispatch_command,
            triage_status_reader=GitHubTriageStatusReader(
                config=config,
                issue_fields=GitHubIssueFieldsClient(gh=config.github_cli),
            )
            if args.triage_dispatch_command
            else None,
        )
        print(json.dumps(result.__dict__, ensure_ascii=False))
        return 0

    if args.command == "watch-lark-events":
        config = load_project_config(args.project_config)
        app_secret = os.environ.get(config.lark.app_secret_env)
        if not app_secret:
            print(f"missing env: {config.lark.app_secret_env}", file=sys.stderr)
            return 2
        lark = LarkOpenApiMessengerClient(
            app_id=config.lark.app_id,
            app_secret=app_secret,
            base_url=config.lark.api_base_url,
        )
        issue_fields = GitHubIssueFieldsClient(gh=config.github_cli)
        github = GitHubCliIssuesClient(
            gh=config.github_cli,
            issue_fields=issue_fields,
            project_config=config,
        )
        workflow = IntakeWorkflow(config=config, github=github, lark=lark, issue_fields=issue_fields)
        if args.resource_dir and args.asset_repo:
            print("--resource-dir and --asset-repo are mutually exclusive", file=sys.stderr)
            return 2
        resource_store = None
        if args.asset_repo:
            if not config.assets.github_repo or not config.assets.checkout_path:
                print("missing [assets] github_repo or checkout_path", file=sys.stderr)
                return 2
            resource_store = GitHubAssetRepoStore(
                repo=config.assets.github_repo,
                checkout_path=Path(config.assets.checkout_path),
                base_path=config.assets.base_path,
                branch=config.assets.branch,
                remote_url=config.assets.remote_url,
            )
        resource_describer = media_resource_describer(config)
        resource_redactor = media_resource_redactor(config)
        resource_transformer = media_resource_transformer(config)
        resource_policy = media_resource_policy(config)
        result = run_lark_event_watcher(
            config=config,
            event_payloads=iter_json_event_lines(sys.stdin),
            lark=lark,
            workflow=workflow,
            dry_run=args.dry_run,
            resource_dir=args.resource_dir,
            resource_store=resource_store,
            resource_describer=resource_describer,
            resource_policy=resource_policy,
            resource_redactor=resource_redactor,
            resource_transformer=resource_transformer,
            event_log_path=args.event_log,
            processed_ledger_path=args.processed_ledger,
            triage_queue_path=args.triage_queue,
            triage_quiet_seconds=args.triage_quiet_seconds,
            triage_dispatch_command=args.triage_dispatch_command,
            triage_status_reader=GitHubTriageStatusReader(
                config=config,
                issue_fields=GitHubIssueFieldsClient(gh=config.github_cli),
            )
            if args.triage_dispatch_command
            else None,
        )
        print(
            json.dumps(
                {
                    "dry_run": args.dry_run,
                    "scanned": result.scanned,
                    "processed": result.processed,
                    "skipped": result.skipped,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "resolve-owner":
        rules = load_codeowners(args.repo_path)
        owner_config = load_project_config(args.project_config).owners if args.project_config else None
        print(
            json.dumps(
                {
                    path: list(
                        resolve_configured_owners(
                            path=path,
                            capability=args.capability,
                            owners=owner_config,
                        )
                        if owner_config is not None
                        else ()
                    )
                    or list(resolve_owners(path, rules))
                    for path in args.paths
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "search-prd":
        docs = load_prd_documents(args.root, include_globs=tuple(args.include_glob or ("**/*.md",)))
        hits = search_prd_documents(args.query, docs, limit=args.limit)
        print(json.dumps([hit.__dict__ for hit in hits], ensure_ascii=False))
        return 0

    if args.command == "issue-context":
        config = load_project_config(args.project_config)
        github = GitHubCliIssuesClient(gh=config.github_cli)
        issue = github.get_issue(repo=config.github_repo, issue_number=args.issue)
        comments = github.list_issue_comments(repo=config.github_repo, issue_number=args.issue)
        prd_root = args.repo_path / config.prd.cache_path
        context = build_triage_context(
            issue=issue,
            comments=comments,
            prd_root=prd_root,
            prd_include_globs=config.prd.include_globs,
        )
        markdown = render_triage_context_markdown(context)
        if args.output:
            args.output.write_text(markdown)
        else:
            print(markdown)
        return 0

    if args.command == "apply-triage-result":
        config = load_project_config(args.project_config)
        data = json.loads(args.input.read_text())
        result = parse_triage_result(data)
        if args.dry_run:
            report = build_triage_dry_run_report(
                repo=config.github_repo,
                issue_number=args.issue,
                config=config,
                result=result,
                issue_fields=GitHubIssueFieldsClient(gh=config.github_cli),
            )
            print(
                json.dumps(
                    {
                        "ok": True,
                        "dry_run": True,
                        "issue": args.issue,
                        "issue_type": report.issue_type,
                        "assignee": report.assignee,
                        "field_changes": [change.__dict__ for change in report.field_changes],
                        "comment_markdown": report.comment_markdown,
                        "result_fingerprint": report.result_fingerprint,
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        summary = apply_triage_result(
            repo=config.github_repo,
            issue_number=args.issue,
            config=config,
            result=result,
            github=GitHubCliIssuesClient(gh=config.github_cli),
            issue_fields=GitHubIssueFieldsClient(gh=config.github_cli),
            lark=_optional_lark_client(config),
        )
        print(json.dumps({"ok": True, "issue": args.issue, "summary": summary.__dict__}, ensure_ascii=False))
        return 0

    if args.command == "run-triage":
        config = load_project_config(args.project_config)
        github = GitHubCliIssuesClient(gh=config.github_cli, transient_retries=8)
        issue_fields = GitHubIssueFieldsClient(gh=config.github_cli)

        # When the issue already has an open bugpatrol fix PR, a reporter's
        # follow-up correction should revise that PR (the fix agent re-reads the
        # latest correction as feedback) instead of re-triaging to a no-op
        # "结论无变化" that silently drops the correction. Dispatch the separate
        # fix-revise workflow (its own concurrency group serializes with fix/
        # ci-fix) and skip triage.
        if config.fix is not None and args.fix_revise_dispatch_command:
            head = config.fix.branch_for_issue(args.issue)
            open_pr = github.get_open_pull_request_by_head(
                repo=config.github_repo, head=head
            )
            if open_pr is not None:
                make_dispatch(args.fix_revise_dispatch_command)(args.issue)
                print(
                    json.dumps(
                        {
                            "redirected": "fix-revise",
                            "issue": args.issue,
                            "pr": open_pr.number,
                            "pr_url": open_pr.url,
                        },
                        ensure_ascii=False,
                    )
                )
                return 0

        reference_checkouts = _parse_reference_repo_args(args.reference_repo)

        def _run_triage_attempts(*, repo_path, branch_note: str, reference_repos):
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                plan = prepare_triage_run(
                    config=config,
                    issue_number=args.issue,
                    repo_path=repo_path,
                    output_dir=args.output_dir,
                    github=github,
                    branch_note=branch_note,
                    reference_repos=reference_repos,
                )
                if not args.execute:
                    return plan
                status = execute_triage_run(
                    config=config,
                    issue_number=args.issue,
                    plan=plan,
                    github=github,
                    issue_fields=issue_fields,
                    lark=_optional_lark_client(config),
                    accept_stale_context=attempt == max_attempts,
                    final_attempt=attempt == max_attempts,
                )
                if status == "no_output":
                    print(
                        f"triage agent produced no output (attempt {attempt}); retrying",
                        file=sys.stderr,
                    )
                    continue
                if status == "invalid_output":
                    print(
                        f"triage agent output failed validation (attempt {attempt}); retrying",
                        file=sys.stderr,
                    )
                    continue
                if status == "agent_crashed":
                    # A transient API failure dropped the agent mid-turn (see
                    # is_transient_agent_failure). Re-run rather than posting a
                    # Failed marker for a run a second attempt usually finishes.
                    print(
                        f"triage agent died on a transient API error (attempt {attempt}); retrying",
                        file=sys.stderr,
                    )
                    continue
                if status != "stale_context":
                    return plan
                print(
                    f"new comments arrived during triage run (attempt {attempt}); retrying with fresh context",
                    file=sys.stderr,
                )
            return plan

        resolution = resolve_issue_branch(
            config=config,
            issue_number=args.issue,
            base_repo=args.repo_path,
            github=github,
        )
        # One ExitStack owns the (optional) main-repo worktree plus one nested
        # detached worktree per reference repo, so every worktree is torn down
        # even if a later step raises.
        with contextlib.ExitStack() as stack:
            if resolution.status == "main":
                repo_path = args.repo_path
                branch_note = ""
            else:
                repo_path = stack.enter_context(
                    triage_worktree(base_repo=args.repo_path, ref=resolution.ref)
                )
                branch_note = resolution.note
            reference_contexts = _enter_reference_worktrees(
                stack,
                reference_repos=config.reference_repos,
                checkouts=reference_checkouts,
                main_declared_branch=resolution.declared_branch,
            )
            plan = _run_triage_attempts(
                repo_path=repo_path,
                branch_note=branch_note,
                reference_repos=reference_contexts,
            )
        print(
            json.dumps(
                {
                    "execute": args.execute,
                    "analyzed_branch": resolution.analyzed_branch,
                    "declared_branch": resolution.declared_branch,
                    "branch_status": resolution.status,
                    "context": str(plan.context_path),
                    "schema": str(plan.schema_path),
                    "output": str(plan.output_path),
                    "provider": plan.invocation.provider,
                    "command": plan.invocation.command,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "report-job-failure":
        config = load_project_config(args.project_config)
        github = GitHubCliIssuesClient(gh=config.github_cli)
        issue_fields = GitHubIssueFieldsClient(gh=config.github_cli)
        outcome = report_workflow_failure(
            config=config,
            issue_number=args.issue,
            job=args.job,
            github=github,
            issue_fields=issue_fields,
            lark=_optional_lark_client(config),
            run_url=args.run_url,
            detail=args.detail,
        )
        print(json.dumps({"issue": args.issue, "outcome": outcome}, ensure_ascii=False))
        return 0

    if args.command == "run-fix":
        config = load_project_config(args.project_config)
        if config.fix is None:
            print("project config has no [fix] table; auto-fix is not enabled", file=sys.stderr)
            return 2
        github = GitHubCliIssuesClient(gh=config.github_cli)
        issue_fields = GitHubIssueFieldsClient(gh=config.github_cli)
        if not args.execute:
            # Dry run: report the verdict and gate readiness without touching code.
            verdict = read_triage_verdict(config=config, issue_number=args.issue, issue_fields=issue_fields)
            from bugpatrol.fix_gate import evaluate_triage_readiness

            readiness = evaluate_triage_readiness(verdict=verdict, fix=config.fix)
            print(
                json.dumps(
                    {
                        "execute": False,
                        "issue": args.issue,
                        "verdict": verdict,
                        "fixable": readiness.allowed,
                        "reason": readiness.reason,
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        status = run_fix(
            config=config,
            issue_number=args.issue,
            base_repo=args.repo_path,
            output_dir=args.output_dir,
            github=github,
            issue_fields=issue_fields,
            lark=_optional_lark_client(config),
        )
        print(json.dumps({"execute": True, "issue": args.issue, "status": status}, ensure_ascii=False))
        return 0

    if args.command == "run-fix-revise":
        config = load_project_config(args.project_config)
        if config.fix is None:
            print("project config has no [fix] table; auto-fix is not enabled", file=sys.stderr)
            return 2
        github = GitHubCliIssuesClient(gh=config.github_cli)
        issue_fields = GitHubIssueFieldsClient(gh=config.github_cli)
        if not args.execute:
            # Dry run: report whether an open fix PR exists, how much unresolved
            # review feedback is queued, and whether it conflicts with its target
            # branch — without touching code.
            head = config.fix.branch_for_issue(args.issue)
            pr = github.get_open_pull_request_by_head(repo=config.github_repo, head=head)
            unresolved = (
                len(github.list_unresolved_review_threads(repo=config.github_repo, pr_number=pr.number))
                if pr is not None
                else 0
            )
            print(
                json.dumps(
                    {
                        "execute": False,
                        "issue": args.issue,
                        "open_pr": pr.url if pr is not None else "",
                        "base_branch": pr.base_ref if pr is not None else "",
                        "conflicts_target": bool(pr is not None and pr.mergeable.upper() == "CONFLICTING"),
                        "unresolved_threads": unresolved,
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        status = run_fix_revise(
            config=config,
            issue_number=args.issue,
            base_repo=args.repo_path,
            output_dir=args.output_dir,
            github=github,
            issue_fields=issue_fields,
            lark=_optional_lark_client(config),
        )
        print(json.dumps({"execute": True, "issue": args.issue, "status": status}, ensure_ascii=False))
        return 0

    if args.command == "run-ci-feedback":
        config = load_project_config(args.project_config)
        if config.fix is None:
            print("project config has no [fix] table; auto-fix is not enabled", file=sys.stderr)
            return 2
        github = GitHubCliIssuesClient(gh=config.github_cli)
        issue_fields = GitHubIssueFieldsClient(gh=config.github_cli)
        if not args.execute:
            # Dry run: report the open PR for this branch and the issues it closes
            # (which resolve the managed issue to report to) — no notify / edit.
            pr = github.get_open_pull_request_by_head(
                repo=config.github_repo, head=args.head_branch
            )
            print(
                json.dumps(
                    {
                        "execute": False,
                        "head_branch": args.head_branch,
                        "head_sha": args.head_sha,
                        "conclusion": args.conclusion,
                        "open_pr": pr.url if pr is not None else "",
                        "closing_issues": list(pr.closing_issue_numbers) if pr is not None else [],
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        status = run_ci_feedback(
            config=config,
            head_branch=args.head_branch,
            head_sha=args.head_sha,
            conclusion=args.conclusion,
            base_repo=args.repo_path,
            output_dir=args.output_dir,
            github=github,
            issue_fields=issue_fields,
            lark=_optional_lark_client(config),
        )
        print(
            json.dumps(
                {
                    "execute": True,
                    "head_branch": args.head_branch,
                    "head_sha": args.head_sha,
                    "conclusion": args.conclusion,
                    "status": status,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "reconcile-triage":
        config = load_project_config(args.project_config)
        github = GitHubCliIssuesClient(gh=config.github_cli, transient_retries=8)
        result = reconcile_triage(
            config=config,
            github=github,
            execute=args.execute,
        )
        print(
            json.dumps(
                {
                    "execute": args.execute,
                    "scanned": result.scanned,
                    "candidates": [candidate.__dict__ for candidate in result.candidates],
                    "events": [event.__dict__ for event in result.events],
                },
                ensure_ascii=False,
            )
        )
        return 1 if result.failed else 0

    if args.command == "notify-fix":
        config = load_project_config(args.project_config)
        github = GitHubCliIssuesClient(gh=config.github_cli)
        lark = None
        if args.write:
            app_secret = os.environ.get(config.lark.app_secret_env)
            if not app_secret:
                print(f"missing env: {config.lark.app_secret_env}", file=sys.stderr)
                return 2
            lark = LarkOpenApiMessengerClient(
            app_id=config.lark.app_id,
            app_secret=app_secret,
            base_url=config.lark.api_base_url,
        )
        issue_number = args.issue
        if issue_number is None:
            if args.event not in ("pr_opened", "pr_merged") or not args.pr:
                print("--issue is required unless --event is pr_opened/pr_merged with --pr", file=sys.stderr)
                return 2
            try:
                issue_number = resolve_single_issue_from_pr(
                    github.get_pull_request(repo=config.github_repo, pr=args.pr)
                )
            except ValueError as error:
                print(str(error), file=sys.stderr)
                return 2
        try:
            summary = apply_fix_notification(
                repo=config.github_repo,
                issue_number=issue_number,
                event=args.event,
                pr=args.pr,
                commit=args.commit,
                dry_run=not args.write,
                github=github,
                lark=lark,
                user_open_ids=config.lark.user_open_ids,
            )
        except ValueError as error:
            if "BugPatrol Lark intake metadata" not in str(error):
                raise
            print(
                json.dumps(
                    {
                        "key": "",
                        "event": args.event,
                        "dry_run": not args.write,
                        "duplicate_skipped": False,
                        "lark_sent": False,
                        "metadata_written": False,
                        "skipped": True,
                        "error": str(error),
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        print(json.dumps(summary.__dict__, ensure_ascii=False))
        return 0

    if args.command == "audit-issue-close":
        config = load_project_config(args.project_config)
        github = GitHubCliIssuesClient(gh=config.github_cli)
        issue_fields = GitHubIssueFieldsClient(gh=config.github_cli)
        lark = None
        if args.write:
            app_secret = os.environ.get(config.lark.app_secret_env)
            if not app_secret:
                print(f"missing env: {config.lark.app_secret_env}", file=sys.stderr)
                return 2
            lark = LarkOpenApiMessengerClient(
                app_id=config.lark.app_id,
                app_secret=app_secret,
                base_url=config.lark.api_base_url,
            )
        summary = audit_issue_close(
            repo=config.github_repo,
            issue_number=args.issue,
            config=config,
            github=github,
            lark=lark,
            issue_fields=issue_fields,
            dry_run=not args.write,
        )
        print(json.dumps(summary.__dict__, ensure_ascii=False))
        return 0

    if args.command == "reconcile-fix-notifications":
        config = load_project_config(args.project_config)
        if bool(args.from_github) == bool(args.input):
            print("provide exactly one of --input or --from-github", file=sys.stderr)
            return 2
        github = GitHubCliIssuesClient(gh=config.github_cli, transient_retries=8)
        collection_errors: list[str] = []
        if args.from_github:
            candidates = collect_fix_candidates_from_github(
                repo=config.github_repo,
                github=github,
                pr_limit=args.pr_limit,
                closed_issue_limit=args.closed_issue_limit,
                since_days=args.since_days,
                errors=collection_errors,
            )
        else:
            candidates = fix_event_candidates_from_json(json.loads(args.input.read_text()))
        lark = None
        if args.write:
            app_secret = os.environ.get(config.lark.app_secret_env)
            if not app_secret:
                print(f"missing env: {config.lark.app_secret_env}", file=sys.stderr)
                return 2
            lark = LarkOpenApiMessengerClient(
            app_id=config.lark.app_id,
            app_secret=app_secret,
            base_url=config.lark.api_base_url,
        )
        result = reconcile_fix_notifications(
            repo=config.github_repo,
            candidates=candidates,
            github=github,
            lark=lark,
            dry_run=not args.write,
            resend=args.resend,
            user_open_ids=config.lark.user_open_ids,
        )
        print(
            json.dumps(
                {
                    "attempted": result.attempted,
                    "sent": result.sent,
                    "duplicate_skipped": result.duplicate_skipped,
                    "skipped": result.skipped,
                    "summaries": [summary.__dict__ for summary in result.summaries],
                    "errors": (*collection_errors, *result.errors),
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "ci-watch":
        config = load_project_config(args.project_config)
        github = GitHubCliIssuesClient(gh=config.github_cli, transient_retries=8)
        lark = None
        if args.write:
            app_secret = os.environ.get(config.lark.app_secret_env)
            if not app_secret:
                print(f"missing env: {config.lark.app_secret_env}", file=sys.stderr)
                return 2
            lark = LarkOpenApiMessengerClient(
                app_id=config.lark.app_id,
                app_secret=app_secret,
                base_url=config.lark.api_base_url,
            )
        outcome = watch_ci(
            repo=args.repo,
            branch=args.branch,
            chat_id=args.chat_id,
            runs=github,
            lark=lark,
            state_path=args.state_path,
            dry_run=not args.write,
        )
        print(json.dumps(outcome.__dict__, ensure_ascii=False))
        return 0

    if args.command == "cleanup-assets":
        config = load_project_config(args.project_config)
        if not config.assets.checkout_path:
            print("missing [assets] checkout_path", file=sys.stderr)
            return 2
        result = cleanup_asset_repo(
            checkout_path=Path(config.assets.checkout_path),
            base_path=config.assets.base_path,
            message_id_prefix=args.message_id_prefix,
            delete=args.delete,
            push=args.push,
            branch=config.assets.branch,
            remote_url=config.assets.remote_url or "origin",
        )
        print(json.dumps(result.__dict__, ensure_ascii=False))
        return 0

    parser.print_help(file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
