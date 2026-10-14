"""Polling Lark watcher."""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from bugpatrol.backfill import (
    BranchTipResolver,
    ScanResult,
    TopicResult,
    process_topic_batch,
    scan_topic_batches,
)
from bugpatrol.chat_discovery import BranchChatDiscovery, apply_branch_chats
from bugpatrol.config import CHAT_NAME_SEPARATOR, ProjectConfig
from bugpatrol.event_log import JsonlEventLog
from bugpatrol.github_fields import GitHubIssueFieldsClient, GitHubIssueFieldsError
from bugpatrol.intake_workflow import IntakeWorkflow
from bugpatrol.ledger import JsonMessageLedger, MessageLedger
from bugpatrol.lease import FileLease
from bugpatrol.lark import LarkOpenApiError, LarkOpenApiMessengerClient
from bugpatrol.slash_commands import SlashCommandHandler
from bugpatrol.resources import (
    LocalResourceStore,
    ResourceDescriber,
    ResourcePolicy,
    ResourceRedactor,
    ResourceStore,
    ResourceTransformer,
)
from bugpatrol.triage_queue import CommandTriageDispatcher, TriageRequest, TriageRequestQueue


# Transient Lark scan failures tolerated before the watcher gives up and
# crashes (surfacing a persistent outage to launchd/operators).
MAX_CONSECUTIVE_SCAN_FAILURES = 10

# Consecutive polls where a topic keeps failing (e.g. the fived-assets push 403
# that silently retried forever, dropping issues) before we ping Lark once. A
# failed topic is never ledgered, so it re-processes every poll; without this an
# outage is invisible until someone notices missing issues. No-Silent-Failures:
# admit the repeated failure instead of retrying quietly. The streak is counted
# per topic and cleared only when that topic succeeds, so a 15-hour outage is
# one ping, not one per burst.
TOPIC_FAILURE_ALERT_THRESHOLD = 3

# Topics crossing the threshold in one poll before the alert is also broadcast
# to the group chat: a single stuck topic belongs in that topic, but a
# fleet-wide outage would otherwise only be visible to whoever happens to open
# each topic.
TOPIC_OUTAGE_CHAT_SUMMARY_TOPICS = 3

# Backoff before retrying a failed reconcile dispatch. GitHub cron for
# bugpatrol-reconcile.yml silently stopped firing (08-28), so the watcher is
# now the primary trigger; a transient dispatch failure must retry soon (not
# wait a full interval) while never crashing the poll loop.
RECONCILE_DISPATCH_RETRY_SECONDS = 600


@dataclass(frozen=True)
class WatchResult:
    iterations: int
    scanned: int
    processed: int
    skipped: int
    queued_triage: int = 0
    dispatched_triage: int = 0
    dispatched_reconcile: int = 0


class TriageDispatcher(Protocol):
    def dispatch(self, request: TriageRequest) -> object:
        """Dispatch a due triage request."""


class BranchChatDiscoverer(Protocol):
    def resolve(self) -> BranchChatDiscovery:
        """Return the branch topic chats currently discoverable."""


def _apply_discovered_chats(
    *,
    config: ProjectConfig,
    discoverer: BranchChatDiscoverer,
    base_config: ProjectConfig,
    workflow: IntakeWorkflow,
    iteration: int,
    logger: JsonlEventLog | None,
    unmatched_seen: set[tuple[str, ...]],
) -> ProjectConfig:
    """Refresh the scanned chat set from discovery, keeping the last one on error.

    A discovery outage must not silently shrink the watcher's scan set (bugs
    reported in a branch group would just vanish), so the previous mapping is
    kept and the failure is reported rather than swallowed.
    """
    try:
        discovery = discoverer.resolve()
    except Exception as error:  # noqa: BLE001 - reported, never swallowed
        print(
            f"watch-lark: branch chat discovery failed, keeping previous chats: {error}",
            file=sys.stderr,
        )
        if logger is not None:
            logger.write(
                {
                    "event": "watch_chat_discovery_error",
                    "iteration": iteration,
                    "error": str(error),
                }
            )
        return config
    updated = apply_branch_chats(base_config, discovery.branch_chats)
    # A group named after one of this project's branches but carrying no
    # "<prefix> · <branch>" name is not scanned, and silence there is
    # indistinguishable from "no bugs were reported". Report it once per
    # distinct set -- the refresh runs every poll, and a watcher that logs the
    # same list 2880 times a day is a watcher nobody reads.
    report = discovery.unmatched_chats
    if report and report not in unmatched_seen:
        unmatched_seen.add(report)
        print(
            f"watch-lark: {len(report)} chat(s) are named after branches of "
            f"{base_config.github_repo} but carry no "
            f"'{base_config.lark.chat_prefix}{CHAT_NAME_SEPARATOR}<branch>' name, so they are "
            f"NOT scanned -- a topic group is only adopted when its name says which project "
            f"owns it (they may well belong to another project): {list(report)}",
            file=sys.stderr,
        )
        if logger is not None:
            logger.write(
                {
                    "event": "watch_chat_unmatched",
                    "iteration": iteration,
                    "chat_prefix": base_config.lark.chat_prefix,
                    "unmatched_chats": list(report),
                }
            )
    if updated.lark.branch_chats != config.lark.branch_chats:
        workflow.set_config(updated)
        print(
            f"watch-lark: branch chats now {updated.lark.branch_chats or {}}",
            file=sys.stderr,
        )
        if logger is not None:
            logger.write(
                {
                    "event": "watch_branch_chats_changed",
                    "iteration": iteration,
                    "branch_chats": updated.lark.branch_chats or {},
                }
            )
    return updated


class TriageStatusReader(Protocol):
    def issue_state(self, issue_number: int) -> str:
        """Return the GitHub issue state, "open" or "closed"."""

    def triage_status(self, issue_number: int) -> str:
        """Return the current triage status for an issue."""


class GitHubTriageStatusReader:
    def __init__(self, *, config: ProjectConfig, issue_fields: GitHubIssueFieldsClient) -> None:
        self._config = config
        self._issue_fields = issue_fields

    def issue_state(self, issue_number: int) -> str:
        return self._issue_fields.get_issue_state(
            repo=self._config.github_repo,
            issue_number=issue_number,
        )

    def triage_status(self, issue_number: int) -> str:
        values = self._issue_fields.get_issue_field_values(
            repo=self._config.github_repo,
            issue_number=issue_number,
        )
        github_name = self._config.issue_field_names["Triage status"]
        return values.get(github_name, "")


STALE_QUEUE_TERMINAL_TRIAGE_STATUSES = frozenset({"Done", "Skipped", "Needs info"})


def run_polling_watcher(
    *,
    config: ProjectConfig,
    lark: LarkOpenApiMessengerClient,
    workflow: IntakeWorkflow,
    limit: int = 20,
    interval_seconds: float = 30,
    once: bool = False,
    dry_run: bool = False,
    max_iterations: int | None = None,
    resource_dir: Path | None = None,
    resource_store: ResourceStore | None = None,
    resource_describer: ResourceDescriber | None = None,
    resource_policy: ResourcePolicy | None = None,
    resource_redactor: ResourceRedactor | None = None,
    resource_transformer: ResourceTransformer | None = None,
    event_log_path: Path | None = None,
    event_log: JsonlEventLog | None = None,
    processed_ledger_path: Path | None = None,
    processed_ledger: MessageLedger | None = None,
    lease_file: Path | None = None,
    lease_ttl_seconds: float = 120,
    triage_queue_path: Path | None = None,
    triage_queue: TriageRequestQueue | None = None,
    triage_quiet_seconds: float = 60,
    triage_dispatch_command: str | Sequence[str] | None = None,
    triage_dispatcher: TriageDispatcher | None = None,
    triage_status_reader: TriageStatusReader | None = None,
    # Periodic reconcile sweep. GitHub's `schedule` cron for the reconcile
    # workflow proved unreliable (silently stopped firing 08-28), so the watcher
    # dispatches it on a timer as the primary safety net; empty command = off.
    reconcile_dispatch_command: str | Sequence[str] | None = None,
    reconcile_interval_seconds: float = 4 * 3600,
    parallel_topics: int = 1,
    branch_tip_resolver: BranchTipResolver | None = None,
    slash_handler: SlashCommandHandler | None = None,
    topic_failure_alert_threshold: int = TOPIC_FAILURE_ALERT_THRESHOLD,
    branch_chat_discoverer: BranchChatDiscoverer | None = None,
) -> WatchResult:
    # Discovery is re-applied to the pristine config every poll, so a group that
    # stops matching (renamed, or its branch deleted) actually drops out instead
    # of sticking around from an earlier iteration.
    base_config = config
    if event_log is not None and event_log_path is not None:
        raise ValueError("event_log and event_log_path are mutually exclusive")
    if processed_ledger is not None and processed_ledger_path is not None:
        raise ValueError("processed_ledger and processed_ledger_path are mutually exclusive")
    if triage_queue is not None and triage_queue_path is not None:
        raise ValueError("triage_queue and triage_queue_path are mutually exclusive")
    if triage_dispatcher is not None and triage_dispatch_command is not None:
        raise ValueError("triage_dispatcher and triage_dispatch_command are mutually exclusive")
    queue = triage_queue
    if queue is None and triage_queue_path is not None:
        queue = TriageRequestQueue.load(triage_queue_path)
    dispatcher = triage_dispatcher
    if dispatcher is None and triage_dispatch_command is not None:
        dispatcher = CommandTriageDispatcher(triage_dispatch_command)
    ledger = processed_ledger
    if ledger is None and processed_ledger_path is not None:
        ledger = JsonMessageLedger.load(processed_ledger_path)
    logger = event_log
    if logger is None and event_log_path is not None:
        logger = JsonlEventLog(event_log_path)
    lease = FileLease(lease_file, ttl_seconds=lease_ttl_seconds) if lease_file is not None else None

    if parallel_topics < 1:
        raise ValueError("parallel_topics must be >= 1")
    store = resource_store
    if store is None and resource_dir is not None:
        store = LocalResourceStore(resource_dir)

    if reconcile_dispatch_command is not None and reconcile_interval_seconds <= 0:
        raise ValueError("reconcile_interval_seconds must be > 0 when a reconcile dispatch command is set")
    if isinstance(reconcile_dispatch_command, str):
        reconcile_dispatch_command = tuple(shlex.split(reconcile_dispatch_command))
    elif reconcile_dispatch_command is not None:
        reconcile_dispatch_command = tuple(reconcile_dispatch_command)
    next_reconcile_at = (
        time.monotonic() + reconcile_interval_seconds if reconcile_dispatch_command else None
    )

    iterations = 0
    scanned = 0
    processed = 0
    skipped = 0
    queued_triage = 0
    dispatched_triage = 0
    dispatched_reconcile = 0
    in_flight: dict[str, Future[TopicResult]] = {}
    if lease is not None:
        lease.acquire()
    try:
        with ThreadPoolExecutor(max_workers=parallel_topics) as executor:
            consecutive_scan_failures = 0
            # Outage bookkeeping is per topic, not per poll: a poll that fails
            # one topic and succeeds another must not reset the failing topic's
            # streak (and re-arm its alert) just because some other topic was
            # healthy. Otherwise a topic stuck for hours — e.g. the relay's
            # proxy EOFing every `gh` call — re-pings its reporter every time a
            # clean topic passes in between, which is exactly the repeated-alert
            # spam this state exists to prevent.
            topic_failures: dict[str, int] = {}
            topic_outage_alerted: set[str] = set()
            logged_skips: set[tuple[str, str]] = set()
            unmatched_seen: set[tuple[str, ...]] = set()
            while True:
                iterations += 1
                if branch_chat_discoverer is not None:
                    config = _apply_discovered_chats(
                        config=config,
                        discoverer=branch_chat_discoverer,
                        base_config=base_config,
                        workflow=workflow,
                        iteration=iterations,
                        logger=logger,
                        unmatched_seen=unmatched_seen,
                    )
                try:
                    scan = _scan_all_chats(
                        config=config,
                        lark=lark,
                        limit=limit,
                        processed_ledger=ledger,
                        exclude_roots=frozenset(in_flight),
                    )
                except LarkOpenApiError as error:
                    # Transient Lark/network failures (timeouts, expired
                    # tokens) must not kill the watcher; retry next poll.
                    # Persistent failures still crash so launchd/operators see
                    # them instead of an eternally silent loop.
                    consecutive_scan_failures += 1
                    if consecutive_scan_failures >= MAX_CONSECUTIVE_SCAN_FAILURES:
                        raise
                    print(
                        f"watch-lark: scan failed ({consecutive_scan_failures}/"
                        f"{MAX_CONSECUTIVE_SCAN_FAILURES}), retrying next poll: {error}",
                        file=sys.stderr,
                    )
                    if logger is not None:
                        logger.write(
                            {
                                "event": "watch_scan_error",
                                "iteration": iterations,
                                "error": str(error),
                            }
                        )
                    if lease is not None:
                        lease.refresh()
                    if once or (max_iterations is not None and iterations >= max_iterations):
                        raise
                    time.sleep(interval_seconds)
                    continue
                consecutive_scan_failures = 0
                for batch in scan.topics:
                    in_flight[batch.root_key] = executor.submit(
                        process_topic_batch,
                        batch,
                        config=config,
                        lark=lark,
                        workflow=workflow,
                        dry_run=dry_run,
                        store=store,
                        resource_describer=resource_describer,
                        resource_policy=resource_policy,
                        resource_redactor=resource_redactor,
                        resource_transformer=resource_transformer,
                        branch_tip_resolver=branch_tip_resolver,
                        slash_handler=slash_handler,
                    )
                final_iteration = once or (max_iterations is not None and iterations >= max_iterations)
                results = _harvest_topic_results(in_flight, wait_all=final_iteration)
                iteration_events = list(scan.skipped_events)
                iteration_outcomes = []
                for result in results:
                    if ledger is not None:
                        for message_id in result.processed_message_ids:
                            ledger.mark_processed(message_id)
                    iteration_events.extend(result.events)
                    iteration_outcomes.extend(result.outcomes)
                for result in results:
                    if result.error:
                        topic_failures[result.root_key] = topic_failures.get(result.root_key, 0) + 1
                        continue
                    # A topic that finally went through ends its streak and
                    # re-arms the alert, so a genuinely new outage still pings.
                    topic_failures.pop(result.root_key, None)
                    topic_outage_alerted.discard(result.root_key)
                # Only the topics that just crossed the threshold are alerted:
                # a topic already alerted for this streak stays silent until it
                # recovers, which is what "ping Lark once" actually means.
                newly_failing = [
                    result
                    for result in results
                    if result.error
                    and topic_failures.get(result.root_key, 0) >= topic_failure_alert_threshold
                    and result.root_key not in topic_outage_alerted
                ]
                if newly_failing and not dry_run:
                    _alert_topic_outage(
                        lark=lark,
                        config=config,
                        errored_results=newly_failing,
                        consecutive_iterations=max(
                            topic_failures[result.root_key] for result in newly_failing
                        ),
                        logger=logger,
                    )
                    topic_outage_alerted.update(result.root_key for result in newly_failing)
                iteration_skipped = sum(1 for event in iteration_events if event.action == "skipped")
                scanned += scan.scanned
                processed += len(iteration_outcomes)
                skipped += iteration_skipped
                if logger is not None:
                    logger.write(
                        {
                            "event": "watch_scan",
                            "iteration": iterations,
                            "scanned": scan.scanned,
                            "processed": len(iteration_outcomes),
                            "skipped": iteration_skipped,
                        }
                    )
                    write_backfill_events(
                        logger=logger,
                        iteration=iterations,
                        events=iteration_events,
                        logged_skips=logged_skips,
                    )
                if queue is not None:
                    queued_triage += enqueue_triage_outcomes(
                        outcomes=iteration_outcomes,
                        queue=queue,
                        triage_quiet_seconds=triage_quiet_seconds,
                    )
                    if dispatcher is not None:
                        dispatched_triage += dispatch_due_triage(
                            queue=queue,
                            dispatcher=dispatcher,
                            triage_quiet_seconds=triage_quiet_seconds,
                            status_reader=triage_status_reader,
                        )
                if (
                    reconcile_dispatch_command is not None
                    and not final_iteration
                    and time.monotonic() >= next_reconcile_at
                ):
                    if _dispatch_reconcile(reconcile_dispatch_command, logger=logger, iteration=iterations):
                        next_reconcile_at = time.monotonic() + reconcile_interval_seconds
                        dispatched_reconcile += 1
                    else:
                        # Transient failure: retry on the backoff, not the full
                        # interval, so a down network doesn't silently extend the
                        # safety-net gap (No-Silent-Failures: it is logged too).
                        next_reconcile_at = time.monotonic() + RECONCILE_DISPATCH_RETRY_SECONDS
                if lease is not None:
                    lease.refresh()
                if final_iteration:
                    return WatchResult(
                        iterations=iterations,
                        scanned=scanned,
                        processed=processed,
                        skipped=skipped,
                        queued_triage=queued_triage,
                        dispatched_triage=dispatched_triage,
                        dispatched_reconcile=dispatched_reconcile,
                    )
                time.sleep(interval_seconds)
    finally:
        if lease is not None:
            lease.release()


def _scan_all_chats(
    *,
    config: ProjectConfig,
    lark: LarkOpenApiMessengerClient,
    limit: int,
    processed_ledger: MessageLedger | None,
    exclude_roots: frozenset[str],
) -> ScanResult:
    """Scan the main chat plus every branch chat in one pass.

    Topic root keys are globally unique Lark message ids, so batches from
    different chats never collide and the existing `exclude_roots` dedup keeps
    each group isolated. Each batch's messages carry their own chat_id, which
    downstream resolves to the declared branch.
    """
    scanned = 0
    skipped_events: list = []
    topics: list = []
    for chat_id in config.lark.all_chat_ids():
        result = scan_topic_batches(
            config=config,
            lark=lark,
            limit=limit,
            processed_ledger=processed_ledger,
            exclude_roots=exclude_roots,
            chat_id=chat_id,
        )
        scanned += result.scanned
        skipped_events.extend(result.skipped_events)
        topics.extend(result.topics)
    return ScanResult(
        scanned=scanned,
        skipped_events=tuple(skipped_events),
        topics=tuple(topics),
    )


def _harvest_topic_results(
    in_flight: dict[str, "Future[TopicResult]"],
    *,
    wait_all: bool,
) -> list[TopicResult]:
    """Collect finished topic results; optionally block for the stragglers.

    A topic processor is supposed to report its error in the result and never
    raise, but one that does must not kill the whole watcher: fold the exception
    into an errored TopicResult so the topic retries next scan and the outage
    alerting still fires.
    """
    if wait_all:
        for future in list(in_flight.values()):
            future.exception()  # block until done
    results: list[TopicResult] = []
    for root_key, future in list(in_flight.items()):
        if not future.done():
            continue
        try:
            results.append(future.result())
        except Exception as exc:  # noqa: BLE001 - a raising topic must never kill the watcher
            results.append(
                TopicResult(
                    root_key=root_key,
                    outcomes=(),
                    events=(),
                    processed_message_ids=(),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        del in_flight[root_key]
    return results


def write_backfill_events(  # type: ignore[no-untyped-def]
    *,
    logger: JsonlEventLog,
    iteration: int,
    events,
    logged_skips: set[tuple[str, str]] | None = None,
) -> None:
    """Write one `lark_message` event per message outcome.

    Each poll re-reads the same recent messages, and a message that was skipped
    stays skipped, so without `logged_skips` every skip is re-written each round
    forever -- that is what grew the fived event log to 2GB of near-duplicates.
    When the caller passes a `logged_skips` set, a skip is written only the first
    time that (message_id, reason) is seen; a message whose reason changes is
    still written, and `processed`/`error` are always written. Per-round volume
    stays visible in the `watch_scan` event, which counts the skips.

    The set is replaced with the skips seen this round, so it stays the size of
    the poll window instead of growing for the life of the process.
    """
    current_skips: set[tuple[str, str]] = set()
    for event in events:
        if logged_skips is not None and event.action == "skipped":
            key = (event.message_id, event.reason or "")
            current_skips.add(key)
            if key in logged_skips:
                continue
        logger.write(
            {
                "event": "lark_message",
                "iteration": iteration,
                "message_id": event.message_id,
                "action": event.action,
                "reason": event.reason,
                "issue_number": event.issue_number,
            }
        )
    if logged_skips is not None:
        logged_skips.clear()
        logged_skips.update(current_skips)


def enqueue_triage_outcomes(
    *,
    outcomes,
    queue: TriageRequestQueue,
    triage_quiet_seconds: float,
) -> int:  # type: ignore[no-untyped-def]
    queued = 0
    for outcome in outcomes:
        request = queue.enqueue(
            issue_number=outcome.issue.number,
            signal=outcome.triage_signal,
            quiet_seconds=triage_quiet_seconds,
        )
        if request is not None:
            queued += 1
    return queued


def dispatch_due_triage(
    *,
    queue: TriageRequestQueue,
    dispatcher: TriageDispatcher,
    triage_quiet_seconds: float = 60,
    status_reader: TriageStatusReader | None = None,
) -> int:
    dispatched = 0
    for request in queue.due_requests():
        if status_reader is not None:
            try:
                state = status_reader.issue_state(request.issue_number)
            except GitHubIssueFieldsError as error:
                # The state probe can transiently fail (e.g. api.github.com
                # EOF) even after its own bounded retries. Crash here and the
                # whole watcher dies with every blip; defer the request instead
                # and retry when it comes due again.
                print(
                    f"dispatch_due_triage: state check failed for #{request.issue_number}: {error}",
                    file=sys.stderr,
                )
                queue.mark_pending_review(request=request, quiet_seconds=triage_quiet_seconds)
                continue
            if state == "closed":
                # A closed issue never needs triage. Drop the stale queue entry
                # (its field may be stuck on "Running" from an old run) instead
                # of deferring it forever — otherwise the watcher cycles it every
                # poll and a later reopen would never re-dispatch.
                queue.discard(request)
                continue
            try:
                status = status_reader.triage_status(request.issue_number)
            except GitHubIssueFieldsError as error:
                # The field-values probe can transiently fail (e.g. api.github.com
                # EOF) even after its own bounded retries. Crash here and the whole
                # watcher dies with every blip; defer the request instead and retry
                # when it comes due again.
                print(
                    f"dispatch_due_triage: field-values check failed for #{request.issue_number}: {error}",
                    file=sys.stderr,
                )
                queue.mark_pending_review(request=request, quiet_seconds=triage_quiet_seconds)
                continue
            if status == "Running":
                queue.mark_pending_review(request=request, quiet_seconds=triage_quiet_seconds)
                continue
            if status in STALE_QUEUE_TERMINAL_TRIAGE_STATUSES:
                queue.discard(request)
                continue
        try:
            dispatcher.dispatch(request)
        except RuntimeError as error:
            # The dispatch command (gh workflow run) hits the same flaky network;
            # leave the request queued and defer instead of killing the watcher.
            print(
                f"dispatch_due_triage: dispatch failed for #{request.issue_number}: {error}",
                file=sys.stderr,
            )
            queue.mark_pending_review(request=request, quiet_seconds=triage_quiet_seconds)
            continue
        queue.mark_dispatched(request)
        dispatched += 1
    return dispatched


def _dispatch_reconcile(
    command: Sequence[str],
    *,
    logger: JsonlEventLog | None,
    iteration: int,
) -> bool:
    """Run one reconcile dispatch command; never raises.

    Reconcile is the safety net that re-dispatches triage for intook-but-untriaged
    issues. It runs on a timer in the watcher (GitHub cron is only a backup), so a
    failure must not crash the loop — it is logged and retried on a short backoff.
    """
    completed = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        if logger is not None:
            logger.write(
                {
                    "event": "watch_reconcile_dispatch",
                    "iteration": iteration,
                    "command": " ".join(command),
                }
            )
        return True
    print(
        f"watch-lark: reconcile dispatch failed (exit {completed.returncode}): "
        f"{completed.stderr.strip() or completed.stdout.strip()}",
        file=sys.stderr,
    )
    if logger is not None:
        logger.write(
            {
                "event": "watch_reconcile_dispatch_error",
                "iteration": iteration,
                "command": " ".join(command),
                "returncode": completed.returncode,
                "stderr": completed.stderr.strip()[-500:],
            }
        )
    return False


def render_topic_outage_reply(*, error: str, consecutive_iterations: int) -> str:
    return (
        f"⚠️ 这条上报连续 {consecutive_iterations} 轮未能建成 GitHub issue，"
        "watcher 正在自动重试，维护者需检查凭据/网络。\n"
        f"失败原因：{error}"
    )


def render_topic_outage_alert(
    *,
    errored_results: Sequence[TopicResult],
    consecutive_iterations: int,
    max_topics: int = 5,
) -> str:
    lines = [
        f"⚠️ BugPatrol 连续 {consecutive_iterations} 轮处理话题失败：这些消息未能建成 "
        "GitHub issue，watcher 正在自动重试。请检查 watcher 日志与凭据/网络：",
    ]
    for result in errored_results[:max_topics]:
        lines.append(f"· `{result.root_key}` — {result.error}")
    remaining = len(errored_results) - max_topics
    if remaining > 0:
        lines.append(f"…以及另外 {remaining} 个话题")
    return "\n".join(lines)


def _alert_topic_outage(
    *,
    lark: LarkOpenApiMessengerClient,
    config: ProjectConfig,
    errored_results: Sequence[TopicResult],
    consecutive_iterations: int,
    logger: JsonlEventLog | None,
) -> None:
    # Tell each reporter inside their own topic: that is where they are waiting
    # for an intake reply. The chat-level summary is reserved for a fleet-wide
    # outage (many topics failing) or as a fallback when replying failed.
    undelivered: list[TopicResult] = []
    for result in errored_results:
        try:
            lark.reply_to_message(
                chat_id=config.lark.chat_id,
                message_id=result.root_key,
                text=render_topic_outage_reply(
                    error=result.error,
                    consecutive_iterations=consecutive_iterations,
                ),
            )
        except Exception as error:  # noqa: BLE001 — alerting must never crash the watcher
            print(
                f"watch-lark: failed to reply topic-outage alert to {result.root_key}: {error}",
                file=sys.stderr,
            )
            undelivered.append(result)
    chat_results = (
        errored_results
        if len(errored_results) >= TOPIC_OUTAGE_CHAT_SUMMARY_TOPICS
        else undelivered
    )
    if chat_results:
        text = render_topic_outage_alert(
            errored_results=chat_results,
            consecutive_iterations=consecutive_iterations,
        )
        try:
            lark.send_chat_message(chat_id=config.lark.chat_id, text=text)
        except Exception as error:  # noqa: BLE001 — alerting must never crash the watcher
            # Best-effort: log so the failed alert is visible, but keep polling.
            print(f"watch-lark: failed to send topic-outage alert: {error}", file=sys.stderr)
    if logger is not None:
        logger.write(
            {
                "event": "watch_topic_outage_alert",
                "consecutive_iterations": consecutive_iterations,
                "failing_topics": [result.root_key for result in errored_results],
                "undelivered_topics": [result.root_key for result in undelivered],
                "chat_summary": bool(chat_results),
            }
        )
