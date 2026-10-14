"""Project configuration for bugpatrol."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from bugpatrol.fields import FieldSpec

_LARK_PLATFORM_API_BASE_URLS = {
    "lark": "https://open.larksuite.com/open-apis",
    "feishu": "https://open.feishu.cn/open-apis",
}
_LARK_PLATFORM_APPLINK_DOMAINS = {
    "lark": "applink.larksuite.com",
    "feishu": "applink.feishu.cn",
}

# Separator between the owning-repo prefix and the branch in a topic group's
# name: "fived · feature-moments". Branch names cannot contain spaces, so the
# first occurrence splits unambiguously. Shared with chat discovery, which is
# why it lives here (the naming contract belongs to the config, not the
# discovery that reads it).
CHAT_NAME_SEPARATOR = " · "


@dataclass(frozen=True)
class LarkConfig:
    chat_id: str
    app_id: str
    app_secret_env: str
    bot_open_id: str
    sender_names: dict[str, str] | None = None
    # GitHub login -> Lark open_id, used to render real @mentions.
    user_open_ids: dict[str, str] | None = None
    message_url_template: str = ""
    # "lark" (international) or "feishu" (China); drives API base URL and
    # the default message link domain.
    platform: str = "lark"
    # Extra topic groups scoped to a feature branch: Lark chat_id -> branch.
    # The main `chat_id` above always maps to "main". A topic posted in one of
    # these groups declares (does not infer) that its build was the branch.
    branch_chats: dict[str, str] | None = None
    # Topic groups are named "<chat_prefix> · <branch>", the main group
    # "<chat_prefix> · main". The prefix names the owning repo, which is what
    # keeps a group whose name matches a branch in two repos from being claimed
    # by both projects (that is how one topic produced two issues). Defaults to
    # the repo's own name, so a new project needs no config.
    chat_prefix: str = ""

    @property
    def api_base_url(self) -> str:
        return _LARK_PLATFORM_API_BASE_URLS[self.platform]

    def branch_for_chat(self, chat_id: str) -> str:
        """Resolve the declared feature branch for a chat; main chat -> "main"."""
        if self.branch_chats and chat_id in self.branch_chats:
            return self.branch_chats[chat_id]
        return "main"

    def all_chat_ids(self) -> tuple[str, ...]:
        """Every chat the watcher scans: the main chat plus branch chats."""
        ids = [self.chat_id, *(self.branch_chats or {})]
        return tuple(dict.fromkeys(ids))


def _default_message_url_template(platform: str) -> str:
    domain = _LARK_PLATFORM_APPLINK_DOMAINS[platform]
    return f"https://{domain}/client/chat/open?openChatId={{chat_id}}&messageId={{message_id}}"


@dataclass(frozen=True)
class TriageAgentConfig:
    runner_labels: tuple[str, ...]
    provider: str = "codex"
    model: str = ""
    effort: str = ""


@dataclass(frozen=True)
class ReferenceRepo:
    """A sibling repo triage may read read-only to reason about cross-repo bugs.

    `path` is where the runner checks the repo out (relative to the runner
    workspace, or absolute); the workflow supplies the concrete checkout at
    run time. `branch_map` maps a main-repo branch to the reference-repo branch;
    unmapped branches default to the same name (and fall back to the reference
    repo's main when that branch does not exist).
    """

    repo: str
    path: str
    purpose: str = ""
    branch_map: dict[str, str] | None = None

    def branch_for(self, main_branch: str) -> str:
        if self.branch_map and main_branch in self.branch_map:
            return self.branch_map[main_branch]
        return main_branch


@dataclass(frozen=True)
class PrdConfig:
    root_wiki_node: str
    cache_path: str = ""
    include_globs: tuple[str, ...] = ("**/*.md",)


@dataclass(frozen=True)
class IntakeConfig:
    language: str
    since: str = ""
    skip_orphan_replies: bool = False

    def since_ms(self) -> int:
        if not self.since:
            return 0
        return int(datetime.fromisoformat(self.since).timestamp() * 1000)


@dataclass(frozen=True)
class AssetsConfig:
    github_repo: str = ""
    checkout_path: str = ""
    base_path: str = ".github/issue-assets"
    branch: str = "main"
    remote_url: str = ""


@dataclass(frozen=True)
class MediaConfig:
    description_command: tuple[str, ...] = ()
    description_timeout_seconds: int = 300
    description_temp_dir: str = ""
    redaction_command: tuple[str, ...] = ()
    redaction_timeout_seconds: int = 300
    resize_max_image_width: int = 0
    resize_max_image_height: int = 0
    resize_image_quality: int = 85
    convert_images_to_jpeg: bool = False
    max_image_bytes: int = 0
    max_video_bytes: int = 0
    max_file_bytes: int = 0
    max_video_duration_seconds: float = 0.0
    video_probe_command: tuple[str, ...] = ()
    video_probe_timeout_seconds: int = 30
    video_frame_command: tuple[str, ...] = ()
    video_frame_timeout_seconds: int = 300
    video_frame_min_duration_seconds: float = 0.0
    description_retries: int = 0
    description_retry_backoff_seconds: float = 1.0


@dataclass(frozen=True)
class OwnersConfig:
    default: tuple[str, ...] = ()
    paths: dict[str, tuple[str, ...]] | None = None
    capabilities: dict[str, tuple[str, ...]] | None = None


@dataclass(frozen=True)
class FollowupClassifierConfig:
    acknowledgement_texts: tuple[str, ...] = ()
    fix_status_keywords: tuple[str, ...] = ()


# Default paths a fix run must never touch. Editing CI/CD, lockfiles, secrets,
# or DB migrations from an auto-fix is high-blast-radius and must go through a
# human, so any diff touching these blocks the PR.
DEFAULT_FIX_PROTECTED_GLOBS: tuple[str, ...] = (
    ".github/**",
    "**/*.lock",
    "**/package-lock.json",
    "**/pnpm-lock.yaml",
    "**/yarn.lock",
    "**/Cargo.lock",
    "**/poetry.lock",
    "**/go.sum",
    "**/*.pem",
    "**/*.key",
    "**/.env",
    "**/.env.*",
    "**/migrations/**",
)


@dataclass(frozen=True)
class BuildLinkPattern:
    """One install/preview link the project's CI posts as a PR comment.

    ``pattern`` is a regex with a single capture group = the URL. The build-ready
    notification scans the fix PR's comments and, for each match, surfaces the
    real link (``label``) instead of vaguely telling the user to hunt through PR
    comments. Kept in project config so link formats stay per-project.
    """

    label: str
    pattern: str


@dataclass(frozen=True)
class FixConfig:
    """Configuration for the auto-fix runner.

    ``verify`` maps a human label (build/typecheck/test/lint or any name) to a
    shell command the project owns; BugPatrol only runs each and reads the exit
    code, so the toolchain stays fully decoupled from BugPatrol. The gate keeps
    fixes small and out of high-blast-radius paths; ``allowed_verdicts`` limits
    auto-fix to triage conclusions that are actually code fixes.

    ``setup`` maps a label to a side-effecting command run ONCE up front to
    prepare the worktree (e.g. ``npm ci``), before the agent turn and before the
    verify gate. Splitting it out lets the agent self-verify (typecheck/test) and
    iterate to green in its own turn with deps already present, and it persists
    across the baseline-attribution reset (which keeps gitignored files like
    node_modules), so we install once instead of on every verify. Empty for
    projects that need no setup.
    """

    verify: dict[str, str]
    setup: dict[str, str] = field(default_factory=dict)
    max_diff_lines: int = 800
    protected_globs: tuple[str, ...] = DEFAULT_FIX_PROTECTED_GLOBS
    allowed_verdicts: tuple[str, ...] = ("代码 Bug",)
    # When an open fix PR conflicts with its target branch, revise merges the
    # target in and lets the agent resolve the markers — but only up to this
    # many conflicted files. Beyond it the merge is aborted and the conflict is
    # escalated to a human (auto-resolving a sprawling conflict is unsafe).
    max_conflict_files: int = 5
    branch_prefix: str = "bugpatrol/fix-issue-"
    # 0 = unlimited; otherwise a device-level counting semaphore caps how many
    # heavy fix builds run at once across all runners sharing one machine.
    max_concurrent_per_device: int = 0
    # Interval (seconds) of the in-run progress heartbeat posted to the
    # reporter's Lark topic during a long fix run. 0 disables it. See
    # docs/PROGRESS-REPORTER.md.
    progress_heartbeat_seconds: int = 300
    # When a fix PR's CI build fails, the CI-fix loop feeds the failing logs back
    # to the revise agent and retries, but only up to this many attempts per PR
    # before escalating to a human. See docs/CI-FIX-FEEDBACK.md.
    max_ci_fix_attempts: int = 3
    # Pre-PR sibling of max_ci_fix_attempts: total agent attempts inside a single
    # fix run. When the fix's own edit fails the verify gate (preflight) — but the
    # baseline is green, so it's the fix's fault — feed the failure back and let
    # the agent try again, up to this many attempts, before giving up with
    # verify_failed and NO PR. 1 = no self-heal (the original single-shot).
    max_verify_fix_attempts: int = 2
    # Optional [fix.agent] override. When set it decides the fix agent's
    # provider/model independently of triage (e.g. triage on deepseek, fix on
    # claude). When None the fix runner inherits triage_agent. runner_labels are
    # unused for the fix agent (fix picks its runner via the workflow label var).
    agent: TriageAgentConfig | None = None
    # Install/preview link formats the project's CI posts as PR comments, so the
    # build-ready notification can surface the real links instead of pointing the
    # user at PR comments that may not exist (e.g. a test-only fix deploys no
    # artifact). Empty = the notification makes no install/preview claim.
    build_link_patterns: tuple[BuildLinkPattern, ...] = ()

    def branch_for_issue(self, issue_number: int) -> str:
        return f"{self.branch_prefix}{issue_number}"


@dataclass(frozen=True)
class CloseAuditConfig:
    # GitHub has no pre-close gate for issues, so "closed as completed without a
    # fix reference" can't be rejected at close time -- it can only be enforced
    # after the fact by reopening. Default off (nag only); a project that wants
    # the hard gate opts in.
    reopen_completed_without_evidence: bool = False


@dataclass(frozen=True)
class MailConfig:
    """Lark public-mailbox intake (`bug@fivedegrees.ai`).

    The mail watcher polls this mailbox with the bot's tenant token (zero user
    OAuth) and mirrors reports into the same GitHub issue pipeline. `chat_id` is
    the dedicated group where receipts and notifications are posted (never a
    reply to the customer). `user_emails` maps internal employee mail addresses
    to Lark open_ids so those reports can be @mentioned/assigned by person.
    """

    mailbox: str
    chat_id: str
    app_id: str
    app_secret_env: str
    user_emails: dict[str, str] | None = None


@dataclass(frozen=True)
class ReconcileConfig:
    """Periodic reconcile sweep, triggered by the watcher instead of GitHub cron.

    GitHub's `schedule` dispatch for bugpatrol-reconcile.yml proved unreliable:
    over late Aug 2026 its scheduled events arrived 40min->4h late, then stopped
    firing entirely (the 08-28 stall that left issue #5458 untriaged). The
    watcher already runs 24x7 on the relay and dispatches the triage workflow
    reliably, so it now dispatches the reconcile workflow on a timer too — the
    GitHub cron stays as a backup (reconcile is idempotent, so an overlap is a
    harmless no-op). Empty `dispatch_command` = reconcile sweep disabled.
    """

    # Command (argv, no shell) that dispatches the project's reconcile workflow.
    dispatch_command: tuple[str, ...] = ()
    # How often to dispatch it, seconds. Default 4h = a tighter safety net than
    # the 6h cron, without monopolizing the shared self-hosted runner queue.
    interval_seconds: int = 4 * 3600


@dataclass(frozen=True)
class ProjectConfig:
    github_repo: str
    github_cli: str
    lark: LarkConfig
    triage_agent: TriageAgentConfig
    prd: PrdConfig
    intake: IntakeConfig
    assets: AssetsConfig
    media: MediaConfig
    owners: OwnersConfig
    followup_classifier: FollowupClassifierConfig
    issue_field_names: dict[str, str]
    reference_repos: tuple[ReferenceRepo, ...] = ()
    fix: FixConfig | None = None
    close_audit: CloseAuditConfig = CloseAuditConfig()
    # Optional [mail] public-mailbox intake. None = no mail watcher for the project.
    mail: MailConfig | None = None
    # Periodic reconcile sweep driven by the watcher. Empty dispatch_command =
    # no reconcile (the GitHub cron alone is then the safety net).
    reconcile: ReconcileConfig = ReconcileConfig()

    @property
    def project(self) -> str:
        return self.github_repo.rsplit("/", 1)[-1]

    @property
    def fix_agent(self) -> TriageAgentConfig:
        """Effective agent config for the fix runner.

        A `[fix.agent]` override lets fix use a different provider/model than
        triage; each unset override field inherits from triage_agent so a
        partial override (e.g. provider only) still works.
        """
        base = self.triage_agent
        override = self.fix.agent if self.fix else None
        if override is None:
            return base
        return TriageAgentConfig(
            runner_labels=base.runner_labels,
            provider=override.provider or base.provider,
            model=override.model or base.model,
            effort=override.effort or base.effort,
        )

    def validate_against(self, field_specs: dict[str, FieldSpec]) -> None:
        missing = [name for name in field_specs if name not in self.issue_field_names]
        if missing:
            raise ValueError(f"missing issue field mapping: {', '.join(sorted(missing))}")

    def validate_github_field_options(
        self,
        github_options: dict[str, tuple[str, ...]],
        field_specs: dict[str, FieldSpec],
    ) -> None:
        """Validate live GitHub Issue Field options against bugpatrol semantics.

        `github_options` is keyed by actual GitHub field display name. The
        project config maps bugpatrol's canonical logical field names to those
        display names, so projects can rename fields without changing code.
        """
        self.validate_against(field_specs)
        for logical_name, spec in field_specs.items():
            github_name = self.issue_field_names[logical_name]
            configured = set(github_options.get(github_name, ()))
            expected = set(spec.values)
            if configured != expected:
                extra = sorted(configured - expected)
                absent = sorted(expected - configured)
                parts = []
                if extra:
                    parts.append(f"extra={extra}")
                if absent:
                    parts.append(f"missing={absent}")
                raise ValueError(f"GitHub issue field {github_name!r} values mismatch: {' '.join(parts)}")


def load_project_config(path: Path) -> ProjectConfig:
    data = tomllib.loads(path.read_text())
    return parse_project_config(data)


def parse_project_config(data: dict[str, Any]) -> ProjectConfig:
    lark = _required_table(data, "lark")
    triage_agent = _required_table(data, "triage_agent")
    prd = _required_table(data, "prd")
    intake = _required_table(data, "intake")
    assets = data.get("assets") or {}
    if not isinstance(assets, dict):
        raise ValueError("[assets] must be a table")
    media = data.get("media") or {}
    if not isinstance(media, dict):
        raise ValueError("[media] must be a table")
    owners = data.get("owners") or {}
    if not isinstance(owners, dict):
        raise ValueError("[owners] must be a table")
    followup_classifier = data.get("followup_classifier") or {}
    if not isinstance(followup_classifier, dict):
        raise ValueError("[followup_classifier] must be a table")
    field_names = _required_table(data, "issue_field_names")
    language = _required_str(intake, "language")
    if language not in {"zh-CN", "en-US"}:
        raise ValueError("intake.language must be one of: zh-CN, en-US")

    include_globs = prd.get("include_globs") or ("**/*.md",)
    if not isinstance(include_globs, (list, tuple)) or not all(isinstance(item, str) for item in include_globs):
        raise ValueError("prd.include_globs must be a string list")
    sender_names = lark.get("sender_names") or {}
    if not isinstance(sender_names, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in sender_names.items()
    ):
        raise ValueError("lark.sender_names must be a string map")
    user_open_ids = lark.get("user_open_ids") or {}
    if not isinstance(user_open_ids, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in user_open_ids.items()
    ):
        raise ValueError("lark.user_open_ids must be a string map")
    platform = str(lark.get("platform") or "lark")
    if platform not in _LARK_PLATFORM_API_BASE_URLS:
        raise ValueError('lark.platform must be "lark" or "feishu"')
    branch_chats = _parse_branch_chats(lark, main_chat_id=_required_str(lark, "chat_id"))
    github_repo = _required_str(data, "github_repo")
    chat_prefix = _parse_chat_prefix(lark, github_repo=github_repo)

    return ProjectConfig(
        github_repo=github_repo,
        # expanduser so one config works across machines with different homes.
        # BUGPATROL_GITHUB_CLI overrides the config value at runtime so one
        # shared config serves two runtimes: the watcher keeps the config's
        # self-refreshing bot wrapper (mints from an on-disk key), while a
        # triage runner sets BUGPATROL_GITHUB_CLI=gh and passes a workflow-minted
        # GH_TOKEN, keeping no credential on the box.
        github_cli=str(
            Path(
                os.environ.get("BUGPATROL_GITHUB_CLI") or str(data.get("github_cli") or "gh")
            ).expanduser()
        ),
        lark=LarkConfig(
            chat_id=_required_str(lark, "chat_id"),
            app_id=_required_str(lark, "app_id"),
            app_secret_env=_required_str(lark, "app_secret_env"),
            bot_open_id=_required_str(lark, "bot_open_id"),
            sender_names=dict(sender_names),
            user_open_ids=dict(user_open_ids),
            message_url_template=str(lark.get("message_url_template") or "")
            or _default_message_url_template(platform),
            platform=platform,
            branch_chats=branch_chats or None,
            chat_prefix=chat_prefix,
        ),
        triage_agent=TriageAgentConfig(
            runner_labels=tuple(_required_list(triage_agent, "runner_labels")),
            provider=str(triage_agent.get("provider") or "codex"),
            model=str(triage_agent.get("model") or ""),
            effort=str(triage_agent.get("effort") or ""),
        ),
        prd=PrdConfig(
            root_wiki_node=_required_str(prd, "root_wiki_node"),
            cache_path=str(prd.get("cache_path") or ""),
            include_globs=tuple(include_globs),
        ),
        intake=_parse_intake(language=language, intake=intake),
        assets=AssetsConfig(
            github_repo=str(assets.get("github_repo") or ""),
            checkout_path=str(assets.get("checkout_path") or ""),
            base_path=str(assets.get("base_path") or ".github/issue-assets"),
            branch=str(assets.get("branch") or "main"),
            remote_url=str(assets.get("remote_url") or ""),
        ),
        media=MediaConfig(
            description_command=tuple(_optional_str_list(media, "description_command")),
            description_timeout_seconds=_num(media, "description_timeout_seconds", 300),
            description_temp_dir=str(media.get("description_temp_dir") or ""),
            redaction_command=tuple(_optional_str_list(media, "redaction_command")),
            redaction_timeout_seconds=_num(media, "redaction_timeout_seconds", 300),
            resize_max_image_width=_num(media, "resize_max_image_width", 0),
            resize_max_image_height=_num(media, "resize_max_image_height", 0),
            resize_image_quality=_num(media, "resize_image_quality", 85),
            convert_images_to_jpeg=bool(media.get("convert_images_to_jpeg") or False),
            max_image_bytes=_num(media, "max_image_bytes", 0),
            max_video_bytes=_num(media, "max_video_bytes", 0),
            max_file_bytes=_num(media, "max_file_bytes", 0),
            max_video_duration_seconds=_num(media, "max_video_duration_seconds", 0.0, cast=float),
            video_probe_command=tuple(_optional_str_list(media, "video_probe_command")),
            video_probe_timeout_seconds=_num(media, "video_probe_timeout_seconds", 30),
            video_frame_command=tuple(_optional_str_list(media, "video_frame_command")),
            video_frame_timeout_seconds=_num(media, "video_frame_timeout_seconds", 300),
            video_frame_min_duration_seconds=_num(media, "video_frame_min_duration_seconds", 0.0, cast=float),
            description_retries=_num(media, "description_retries", 0),
            description_retry_backoff_seconds=_num(media, "description_retry_backoff_seconds", 1.0, cast=float),
        ),
        owners=OwnersConfig(
            default=tuple(_optional_str_list(owners, "default")),
            paths=_optional_owner_map(owners, "paths"),
            capabilities=_optional_owner_map(owners, "capabilities"),
        ),
        followup_classifier=FollowupClassifierConfig(
            acknowledgement_texts=tuple(_optional_str_list(followup_classifier, "acknowledgement_texts")),
            fix_status_keywords=tuple(_optional_str_list(followup_classifier, "fix_status_keywords")),
        ),
        issue_field_names={
            str(name): str(value)
            for name, value in field_names.items()
            if isinstance(value, str)
        },
        reference_repos=_parse_reference_repos(data),
        fix=_parse_fix(data),
        close_audit=_parse_close_audit(data),
        mail=_parse_mail(data),
        reconcile=_parse_reconcile(data),
    )


def _parse_mail(data: dict[str, Any]) -> MailConfig | None:
    """Parse the optional `[mail]` table (public-mailbox intake).

    Absent `[mail]` -> None (the project has no mail watcher). When present,
    mailbox/chat_id/app credentials are required so a half-written config is
    rejected loudly rather than silently polling nothing.
    """
    table = data.get("mail")
    if table is None:
        return None
    if not isinstance(table, dict):
        raise ValueError("[mail] must be a table")
    user_emails = table.get("user_emails") or {}
    if not isinstance(user_emails, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in user_emails.items()
    ):
        raise ValueError("[mail.user_emails] must be a string map")
    return MailConfig(
        mailbox=_required_str(table, "mailbox"),
        chat_id=_required_str(table, "chat_id"),
        app_id=_required_str(table, "app_id"),
        app_secret_env=_required_str(table, "app_secret_env"),
        user_emails=dict(user_emails),
    )


def _parse_reconcile(data: dict[str, Any]) -> ReconcileConfig:
    """Parse the optional `[reconcile]` table (watcher-driven reconcile sweep).

    Absent/empty `[reconcile]` -> disabled (GitHub cron alone is the safety
    net). A dispatch_command with no arguments is rejected loudly rather than
    silently dispatching nothing every interval.
    """
    table = data.get("reconcile")
    if table is None:
        return ReconcileConfig()
    if not isinstance(table, dict):
        raise ValueError("[reconcile] must be a table")
    dispatch = tuple(_optional_str_list(table, "dispatch_command"))
    interval = _num(table, "interval_seconds", ReconcileConfig.interval_seconds)
    if interval <= 0:
        raise ValueError("reconcile.interval_seconds must be > 0")
    if dispatch and len(dispatch) == 1 and not dispatch[0]:
        raise ValueError("reconcile.dispatch_command must not be empty")
    return ReconcileConfig(dispatch_command=dispatch, interval_seconds=int(interval))


def _parse_close_audit(data: dict[str, Any]) -> CloseAuditConfig:
    table = data.get("close_audit") or {}
    if not isinstance(table, dict):
        raise ValueError("[close_audit] must be a table")
    return CloseAuditConfig(
        reopen_completed_without_evidence=bool(
            table.get("reopen_completed_without_evidence") or False
        ),
    )


def _parse_fix(data: dict[str, Any]) -> FixConfig | None:
    """Parse the optional `[fix]` table (auto-fix runner).

    Absent `[fix]` -> None (the project has no fix runner). When present, at
    least one `[fix.verify]` command must be non-empty: an auto-fix with no
    verification is not shippable, so a config that would skip every gate is
    rejected loudly rather than opening unverified PRs.
    """
    fix = data.get("fix")
    if fix is None:
        return None
    if not isinstance(fix, dict):
        raise ValueError("[fix] must be a table")
    verify_raw = fix.get("verify") or {}
    if not isinstance(verify_raw, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in verify_raw.items()
    ):
        raise ValueError("[fix.verify] must be a table of string commands")
    verify = {key: value for key, value in verify_raw.items() if value.strip()}
    if not verify:
        raise ValueError(
            "[fix.verify] must define at least one non-empty command; "
            "an auto-fix with no verification is not shippable"
        )
    setup_raw = fix.get("setup") or {}
    if not isinstance(setup_raw, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in setup_raw.items()
    ):
        raise ValueError("[fix.setup] must be a table of string commands")
    setup = {key: value for key, value in setup_raw.items() if value.strip()}
    gate = fix.get("gate") or {}
    if not isinstance(gate, dict):
        raise ValueError("[fix.gate] must be a table")
    max_diff_lines = _num(gate, "max_diff_lines", 800)
    if max_diff_lines <= 0:
        raise ValueError("fix.gate.max_diff_lines must be positive")
    max_conflict_files = _num(gate, "max_conflict_files", 5)
    if max_conflict_files <= 0:
        raise ValueError("fix.gate.max_conflict_files must be positive")
    max_ci_fix_attempts = _num(gate, "max_ci_fix_attempts", 3)
    if max_ci_fix_attempts <= 0:
        raise ValueError("fix.gate.max_ci_fix_attempts must be positive")
    max_verify_fix_attempts = _num(gate, "max_verify_fix_attempts", 2)
    if max_verify_fix_attempts <= 0:
        raise ValueError("fix.gate.max_verify_fix_attempts must be positive")
    protected_globs = gate.get("protected_globs")
    if protected_globs is None:
        protected = DEFAULT_FIX_PROTECTED_GLOBS
    else:
        if not isinstance(protected_globs, list) or not all(
            isinstance(item, str) for item in protected_globs
        ):
            raise ValueError("fix.gate.protected_globs must be a string list")
        protected = tuple(protected_globs)
    allowed_verdicts = gate.get("allowed_verdicts")
    if allowed_verdicts is None:
        verdicts = ("代码 Bug",)
    else:
        if not isinstance(allowed_verdicts, list) or not all(
            isinstance(item, str) and item for item in allowed_verdicts
        ):
            raise ValueError("fix.gate.allowed_verdicts must be a non-empty string list")
        verdicts = tuple(allowed_verdicts)
    runner = fix.get("runner") or {}
    if not isinstance(runner, dict):
        raise ValueError("[fix.runner] must be a table")
    max_concurrent = _num(runner, "max_concurrent_per_device", 0)
    if max_concurrent < 0:
        raise ValueError("fix.runner.max_concurrent_per_device must be >= 0")
    progress = fix.get("progress") or {}
    if not isinstance(progress, dict):
        raise ValueError("[fix.progress] must be a table")
    progress_heartbeat = _num(progress, "heartbeat_seconds", 300)
    if progress_heartbeat < 0:
        raise ValueError("fix.progress.heartbeat_seconds must be >= 0")
    branch_prefix = str(fix.get("branch_prefix") or "bugpatrol/fix-issue-")
    build_links_raw = fix.get("build_links") or []
    if not isinstance(build_links_raw, list):
        raise ValueError("[[fix.build_links]] must be an array of tables")
    build_link_patterns: list[BuildLinkPattern] = []
    for entry in build_links_raw:
        if not isinstance(entry, dict):
            raise ValueError("[[fix.build_links]] entries must be tables")
        label = str(entry.get("label") or "")
        pattern = str(entry.get("pattern") or "")
        if not label or not pattern:
            raise ValueError("[[fix.build_links]] entries need non-empty label and pattern")
        try:
            compiled = re.compile(pattern)
        except re.error as error:
            raise ValueError(f"[[fix.build_links]] pattern for {label!r} is invalid: {error}") from error
        if compiled.groups != 1:
            raise ValueError(
                f"[[fix.build_links]] pattern for {label!r} must have exactly one capture group (the URL)"
            )
        build_link_patterns.append(BuildLinkPattern(label=label, pattern=pattern))
    agent_raw = fix.get("agent")
    if agent_raw is None:
        agent = None
    else:
        if not isinstance(agent_raw, dict):
            raise ValueError("[fix.agent] must be a table")
        agent = TriageAgentConfig(
            runner_labels=(),
            provider=str(agent_raw.get("provider") or ""),
            model=str(agent_raw.get("model") or ""),
            effort=str(agent_raw.get("effort") or ""),
        )
    return FixConfig(
        verify=verify,
        setup=setup,
        max_diff_lines=max_diff_lines,
        protected_globs=protected,
        allowed_verdicts=verdicts,
        max_conflict_files=max_conflict_files,
        branch_prefix=branch_prefix,
        max_concurrent_per_device=max_concurrent,
        progress_heartbeat_seconds=progress_heartbeat,
        max_ci_fix_attempts=max_ci_fix_attempts,
        max_verify_fix_attempts=max_verify_fix_attempts,
        agent=agent,
        build_link_patterns=tuple(build_link_patterns),
    )


def _parse_intake(*, language: str, intake: dict[str, Any]) -> IntakeConfig:
    since = str(intake.get("since") or "")
    if since:
        datetime.fromisoformat(since)  # validate eagerly; raises ValueError on bad input
    skip_orphan = bool(intake.get("skip_orphan_replies", False))
    return IntakeConfig(language=language, since=since, skip_orphan_replies=skip_orphan)


def _required_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"missing table [{key}]")
    return value


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing string field {key!r}")
    return value


def _num(table: dict[str, Any], key: str, default: float, *, cast: Any = int) -> Any:
    """Default a numeric config field ONLY when the key is missing.

    Never use ``table.get(key) or default``: ``or`` also fires on a legitimately
    configured ``0`` / ``0.0``, silently replacing it with the default (and
    skipping any downstream ``<= 0`` validation). Default on ``None`` alone so a
    configured zero is honored or rejected, never silently coerced.
    """
    raw = table.get(key)
    return default if raw is None else cast(raw)


def _required_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"missing string list field {key!r}")
    return value


def _optional_str_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key!r} must be a string list")
    return value


def _parse_chat_prefix(lark: dict[str, Any], *, github_repo: str) -> str:
    """The name prefix that marks a topic group as this project's.

    Defaults to the repo's own name ("TheCloverLab/fived" -> "fived"), which is
    what makes the naming contract work without config. Projects whose groups
    predate the contract (or that want a label other than the repo name) set
    `lark.chat_prefix` explicitly.
    """
    value = lark.get("chat_prefix")
    if value is None:
        return github_repo.rsplit("/", 1)[-1]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("lark.chat_prefix must be a non-empty string")
    if CHAT_NAME_SEPARATOR in value:
        raise ValueError(f"lark.chat_prefix must not contain {CHAT_NAME_SEPARATOR!r}")
    return value.strip()


def _parse_branch_chats(lark: dict[str, Any], *, main_chat_id: str) -> dict[str, str]:
    value = lark.get("branch_chats")
    if value is None:
        return {}
    if not isinstance(value, list):
        raise ValueError("lark.branch_chats must be an array of tables")
    result: dict[str, str] = {}
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError("each [[lark.branch_chats]] entry must be a table")
        chat_id = _required_str(entry, "chat_id")
        branch = _required_str(entry, "branch")
        if chat_id == main_chat_id:
            raise ValueError("lark.branch_chats cannot re-map the main lark.chat_id")
        if chat_id in result:
            raise ValueError(f"duplicate lark.branch_chats chat_id: {chat_id}")
        result[chat_id] = branch
    return result


def _parse_reference_repos(data: dict[str, Any]) -> tuple[ReferenceRepo, ...]:
    """Parse optional `[[triage.reference_repos]]` sibling-repo declarations.

    Bounded by the config's array length; each entry is a table with `repo`,
    `path`, optional `purpose`, and an optional `branch_map` (main-repo branch
    -> reference-repo branch). Duplicate `repo` values are rejected.
    """
    triage = data.get("triage")
    if triage is None:
        return ()
    if not isinstance(triage, dict):
        raise ValueError("[triage] must be a table")
    entries = triage.get("reference_repos")
    if entries is None:
        return ()
    if not isinstance(entries, list):
        raise ValueError("triage.reference_repos must be an array of tables")
    result: list[ReferenceRepo] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each [[triage.reference_repos]] entry must be a table")
        repo = _required_str(entry, "repo")
        path = _required_str(entry, "path")
        if repo in seen:
            raise ValueError(f"duplicate triage.reference_repos repo: {repo}")
        seen.add(repo)
        branch_map_raw = entry.get("branch_map")
        branch_map: dict[str, str] | None = None
        if branch_map_raw is not None:
            if not isinstance(branch_map_raw, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in branch_map_raw.items()
            ):
                raise ValueError(
                    "triage.reference_repos branch_map must be a string map"
                )
            branch_map = dict(branch_map_raw)
        result.append(
            ReferenceRepo(
                repo=repo,
                path=path,
                purpose=str(entry.get("purpose") or ""),
                branch_map=branch_map,
            )
        )
    return tuple(result)


def _optional_owner_map(data: dict[str, Any], key: str) -> dict[str, tuple[str, ...]]:
    value = data.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"owners.{key} must be a table")
    result: dict[str, tuple[str, ...]] = {}
    for name, owners in value.items():
        if not isinstance(name, str):
            raise ValueError(f"owners.{key} keys must be strings")
        if not isinstance(owners, list) or not all(isinstance(item, str) for item in owners):
            raise ValueError(f"owners.{key}.{name} must be a string list")
        result[name] = tuple(owners)
    return result
