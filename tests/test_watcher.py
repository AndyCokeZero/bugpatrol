from __future__ import annotations

import concurrent.futures
import io
import json
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from bugpatrol.backfill import BackfillEvent, TopicResult
from bugpatrol.chat_discovery import BranchChatDiscovery
from bugpatrol.config import load_project_config
from bugpatrol.github_fields import GitHubIssueFieldsError
from bugpatrol.intake_workflow import IntakeWorkflow
from bugpatrol.lease import FileLease, LeaseHeldError
from bugpatrol.lark import LarkMessage, LarkOpenApiError
from bugpatrol.testing.fakes import FakeGitHubIssuesClient, FakeLarkMessengerClient
from bugpatrol.triage_queue import TriageRequest, TriageRequestQueue
from bugpatrol.watcher import (
    MAX_CONSECUTIVE_SCAN_FAILURES,
    TOPIC_OUTAGE_CHAT_SUMMARY_TOPICS,
    _dispatch_reconcile,
    _harvest_topic_results,
    dispatch_due_triage,
    render_topic_outage_alert,
    render_topic_outage_reply,
    run_polling_watcher,
    write_backfill_events,
)


MAIN_CHAT_ID = "oc_d371f022f168b567a141ced142691894"


class FakeHistoryLark(FakeLarkMessengerClient):
    def list_chat_messages(self, *, chat_id: str, limit: int = 20) -> list[LarkMessage]:
        if chat_id != MAIN_CHAT_ID:
            return []
        return [
            LarkMessage(
                message_id="om_1",
                chat_id=chat_id,
                root_id="om_1",
                sender_open_id="ou_user",
                sender_type="user",
                create_time="1000",
                msg_type="text",
                text="Todo 空状态不显示",
            )
        ]


class FakeTwoTopicLark(FakeLarkMessengerClient):
    def list_chat_messages(self, *, chat_id: str, limit: int = 20) -> list[LarkMessage]:
        if chat_id != MAIN_CHAT_ID:
            return []
        return [
            LarkMessage(
                message_id=message_id,
                chat_id=chat_id,
                root_id=message_id,
                sender_open_id="ou_user",
                sender_type="user",
                create_time="1000",
                msg_type="text",
                text=f"bug report {message_id}",
            )
            for message_id in ("om_t1", "om_t2")
        ]


class UnreplyableTopicLark(FakeHistoryLark):
    def __init__(self, *, unreplyable: str) -> None:
        super().__init__()
        self._unreplyable = unreplyable

    def reply_to_message(self, *, chat_id: str, message_id: str, text: str) -> None:
        if message_id == self._unreplyable:
            raise LarkOpenApiError("Lark request failed: message not found")
        super().reply_to_message(chat_id=chat_id, message_id=message_id, text=text)


def _outage_alert_replies(lark: FakeHistoryLark) -> list:
    return [reply for reply in lark.replies if "未能建成 GitHub issue" in reply.text]


class FlakyThenHealthyLark(FakeHistoryLark):
    def __init__(self, *, failures: int) -> None:
        super().__init__()
        self._failures = failures
        self.scan_calls = 0

    def list_chat_messages(self, *, chat_id: str, limit: int = 20) -> list[LarkMessage]:
        self.scan_calls += 1
        if self.scan_calls <= self._failures:
            raise LarkOpenApiError("Lark request failed: <urlopen error [Errno 60] Operation timed out>")
        return super().list_chat_messages(chat_id=chat_id, limit=limit)


class DiscoveredChatLark(FakeLarkMessengerClient):
    """Reports a bug only in a chat that config never listed."""

    def list_chat_messages(self, *, chat_id: str, limit: int = 20) -> list[LarkMessage]:
        if chat_id != "oc_discovered":
            return []
        return [
            LarkMessage(
                message_id="om_disc",
                chat_id=chat_id,
                root_id="om_disc",
                sender_open_id="ou_user",
                sender_type="user",
                create_time="1000",
                msg_type="text",
                text="分支群里报的 bug",
            )
        ]


class StaticDiscoverer:
    def __init__(self, branch_chats: dict[str, str]) -> None:
        self.branch_chats = branch_chats

    def resolve(self) -> BranchChatDiscovery:
        return BranchChatDiscovery(
            branch_chats=dict(self.branch_chats), unmatched_chats=(), archived_chats=()
        )


class ExplodingDiscoverer:
    def resolve(self) -> BranchChatDiscovery:
        raise RuntimeError("gh api boom")


class BranchChatDiscoveryWatcherTest(unittest.TestCase):
    def test_discovered_chat_is_scanned_and_accepted_by_intake(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGitHubIssuesClient()
        lark = DiscoveredChatLark()
        workflow = IntakeWorkflow(config=config, github=github, lark=lark)

        result = run_polling_watcher(
            config=config,
            lark=lark,  # type: ignore[arg-type]
            workflow=workflow,
            once=True,
            interval_seconds=0,
            branch_chat_discoverer=StaticDiscoverer({"oc_discovered": "feature-moments"}),
            branch_tip_resolver=lambda branch: "deadbeef",
        )

        self.assertEqual(result.processed, 1)
        self.assertEqual(len(github.created), 1)

    def test_discovery_failure_keeps_scanning_the_configured_chats(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGitHubIssuesClient()
        lark = FakeHistoryLark()
        workflow = IntakeWorkflow(config=config, github=github, lark=lark)

        result = run_polling_watcher(
            config=config,
            lark=lark,  # type: ignore[arg-type]
            workflow=workflow,
            once=True,
            interval_seconds=0,
            branch_chat_discoverer=ExplodingDiscoverer(),
        )

        self.assertEqual(result.processed, 1)


class WatcherTest(unittest.TestCase):
    def test_run_polling_watcher_survives_transient_lark_scan_errors(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGitHubIssuesClient()
        lark = FlakyThenHealthyLark(failures=2)
        workflow = IntakeWorkflow(config=config, github=github, lark=lark)

        result = run_polling_watcher(
            config=config,
            lark=lark,  # type: ignore[arg-type]
            workflow=workflow,
            max_iterations=3,
            interval_seconds=0,
        )

        self.assertEqual(result.iterations, 3)
        self.assertEqual(result.processed, 1)
        self.assertEqual(len(github.created), 1)

    def test_run_polling_watcher_gives_up_after_persistent_scan_errors(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGitHubIssuesClient()
        lark = FlakyThenHealthyLark(failures=99)
        workflow = IntakeWorkflow(config=config, github=github, lark=lark)

        with self.assertRaises(LarkOpenApiError):
            run_polling_watcher(
                config=config,
                lark=lark,  # type: ignore[arg-type]
                workflow=workflow,
                max_iterations=99,
                interval_seconds=0,
            )
        self.assertEqual(lark.scan_calls, MAX_CONSECUTIVE_SCAN_FAILURES)

    def test_run_polling_watcher_once_processes_one_scan(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGitHubIssuesClient()
        lark = FakeHistoryLark()
        workflow = IntakeWorkflow(config=config, github=github, lark=lark)

        result = run_polling_watcher(
            config=config,
            lark=lark,  # type: ignore[arg-type]
            workflow=workflow,
            once=True,
            interval_seconds=0,
        )

        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.scanned, 1)
        self.assertEqual(result.processed, 1)
        self.assertEqual(len(github.created), 1)

    def test_run_polling_watcher_passes_resource_options_to_backfill(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGitHubIssuesClient()
        lark = FakeHistoryLark()
        workflow = IntakeWorkflow(config=config, github=github, lark=lark)
        store = object()
        describer = object()
        transformer = object()

        with patch("bugpatrol.watcher.process_topic_batch") as process:
            process.return_value = TopicResult(
                root_key="om_1",
                outcomes=(),
                events=(BackfillEvent(message_id="om_1", action="skipped", reason="dry_run"),),
                processed_message_ids=(),
            )
            result = run_polling_watcher(
                config=config,
                lark=lark,  # type: ignore[arg-type]
                workflow=workflow,
                once=True,
                interval_seconds=0,
                dry_run=True,
                resource_store=store,  # type: ignore[arg-type]
                resource_describer=describer,  # type: ignore[arg-type]
                resource_transformer=transformer,  # type: ignore[arg-type]
            )

        self.assertEqual(result.skipped, 1)
        self.assertIs(process.call_args.kwargs["store"], store)
        self.assertIs(process.call_args.kwargs["resource_describer"], describer)
        self.assertIs(process.call_args.kwargs["resource_transformer"], transformer)

    def test_run_polling_watcher_alerts_lark_after_repeated_topic_failures(self) -> None:
        # A topic that keeps failing (e.g. the fived-assets push 403) is never
        # ledgered, so it silently re-processes every poll. After the threshold
        # the watcher must ping Lark once instead of retrying quietly forever.
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGitHubIssuesClient()
        lark = FakeHistoryLark()
        workflow = IntakeWorkflow(config=config, github=github, lark=lark)

        errored = TopicResult(
            root_key="om_1",
            outcomes=(),
            events=(BackfillEvent(message_id="om_1", action="error", reason="push 403"),),
            processed_message_ids=(),
            error="GitHubCliError: fived-assets push failed with 403",
        )
        # Patch the harvest step so each poll yields the failing topic
        # deterministically (real futures complete across polls, not within one).
        with patch("bugpatrol.watcher._harvest_topic_results", return_value=[errored]):
            run_polling_watcher(
                config=config,
                lark=lark,  # type: ignore[arg-type]
                workflow=workflow,
                max_iterations=3,
                interval_seconds=0,
                topic_failure_alert_threshold=3,
            )

        # Exactly one alert for the whole outage, delivered inside the failing
        # topic (where the reporter is waiting), not broadcast to the chat.
        alerts = _outage_alert_replies(lark)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].message_id, "om_1")
        self.assertIn("fived-assets push failed with 403", alerts[0].text)
        self.assertEqual(lark.chat_messages, [])

    def test_run_polling_watcher_broadcasts_chat_summary_for_fleet_wide_outage(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGitHubIssuesClient()
        lark = FakeHistoryLark()
        workflow = IntakeWorkflow(config=config, github=github, lark=lark)

        errored = [
            TopicResult(
                root_key=f"om_{index}",
                outcomes=(),
                events=(BackfillEvent(message_id=f"om_{index}", action="error", reason="boom"),),
                processed_message_ids=(),
                error="boom",
            )
            for index in range(TOPIC_OUTAGE_CHAT_SUMMARY_TOPICS)
        ]
        with patch("bugpatrol.watcher._harvest_topic_results", return_value=errored):
            run_polling_watcher(
                config=config,
                lark=lark,  # type: ignore[arg-type]
                workflow=workflow,
                max_iterations=3,
                interval_seconds=0,
                topic_failure_alert_threshold=3,
            )

        self.assertEqual(len(_outage_alert_replies(lark)), TOPIC_OUTAGE_CHAT_SUMMARY_TOPICS)
        self.assertEqual(len(lark.chat_messages), 1)
        self.assertEqual(lark.chat_messages[0].chat_id, MAIN_CHAT_ID)

    def test_run_polling_watcher_falls_back_to_chat_when_topic_reply_fails(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGitHubIssuesClient()
        lark = UnreplyableTopicLark(unreplyable="om_1")
        workflow = IntakeWorkflow(config=config, github=github, lark=lark)

        errored = TopicResult(
            root_key="om_1",
            outcomes=(),
            events=(BackfillEvent(message_id="om_1", action="error", reason="boom"),),
            processed_message_ids=(),
            error="boom",
        )
        with patch("bugpatrol.watcher._harvest_topic_results", return_value=[errored]):
            run_polling_watcher(
                config=config,
                lark=lark,  # type: ignore[arg-type]
                workflow=workflow,
                max_iterations=3,
                interval_seconds=0,
                topic_failure_alert_threshold=3,
            )

        # The reply could not be delivered, so the alert must not be lost.
        self.assertEqual(len(lark.chat_messages), 1)
        self.assertIn("om_1", lark.chat_messages[0].text)

    def test_stuck_topic_alerts_once_while_healthy_topics_keep_passing(self) -> None:
        # Production shape (relay, 2026-09-22): the proxy EOFed every `gh` call,
        # so ONE topic failed on every poll for 15 hours while other topics in
        # the same poll went through. A poll-level streak resets on those clean
        # neighbours and re-arms the alert, which posted the identical warning
        # into the reporter's topic 11 times. The streak must be per topic.
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGitHubIssuesClient()
        lark = FakeHistoryLark()
        workflow = IntakeWorkflow(config=config, github=github, lark=lark)

        stuck = TopicResult(
            root_key="om_stuck",
            outcomes=(),
            events=(BackfillEvent(message_id="om_stuck", action="error", reason="EOF"),),
            processed_message_ids=(),
            error="GitHubCliError: gh issue list ... EOF",
        )
        healthy = TopicResult(
            root_key="om_healthy",
            outcomes=(),
            events=(BackfillEvent(message_id="om_healthy", action="processed", reason=""),),
            processed_message_ids=("om_healthy",),
        )
        # A poll that harvests ONLY clean topics resets the old poll-level
        # streak, so every fresh burst of three failing polls re-armed and
        # re-fired the alert (3 alerts below). The stuck topic never succeeds,
        # so its own streak must survive those clean polls: one alert.
        harvests = [
            [stuck],
            [stuck],
            [stuck],
            [healthy],
            [stuck],
            [stuck],
            [stuck],
            [healthy],
            [stuck],
            [stuck],
            [stuck],
            [healthy],
        ]
        with patch("bugpatrol.watcher._harvest_topic_results", side_effect=harvests):
            run_polling_watcher(
                config=config,
                lark=lark,  # type: ignore[arg-type]
                workflow=workflow,
                max_iterations=len(harvests),
                interval_seconds=0,
                topic_failure_alert_threshold=3,
            )

        alerts = _outage_alert_replies(lark)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].message_id, "om_stuck")

    def test_stuck_topic_realerts_after_it_recovers_and_breaks_again(self) -> None:
        # The flip side: once the topic actually succeeds, the alert re-arms, so
        # a genuinely new outage is not swallowed by the earlier one.
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGitHubIssuesClient()
        lark = FakeHistoryLark()
        workflow = IntakeWorkflow(config=config, github=github, lark=lark)

        stuck = TopicResult(
            root_key="om_stuck",
            outcomes=(),
            events=(BackfillEvent(message_id="om_stuck", action="error", reason="EOF"),),
            processed_message_ids=(),
            error="GitHubCliError: gh issue list ... EOF",
        )
        recovered = TopicResult(
            root_key="om_stuck",
            outcomes=(),
            events=(BackfillEvent(message_id="om_stuck", action="processed", reason=""),),
            processed_message_ids=("om_stuck",),
        )
        healthy = TopicResult(
            root_key="om_healthy",
            outcomes=(),
            events=(BackfillEvent(message_id="om_healthy", action="processed", reason=""),),
            processed_message_ids=("om_healthy",),
        )
        harvests = [
            [stuck, healthy],
            [stuck, healthy],
            [stuck, healthy],
            [recovered, healthy],
            [stuck, healthy],
            [stuck, healthy],
            [stuck, healthy],
        ]
        with patch("bugpatrol.watcher._harvest_topic_results", side_effect=harvests):
            run_polling_watcher(
                config=config,
                lark=lark,  # type: ignore[arg-type]
                workflow=workflow,
                max_iterations=len(harvests),
                interval_seconds=0,
                topic_failure_alert_threshold=3,
            )

        self.assertEqual(len(_outage_alert_replies(lark)), 2)

    def test_run_polling_watcher_resets_topic_alert_after_recovery(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGitHubIssuesClient()
        lark = FakeHistoryLark()
        workflow = IntakeWorkflow(config=config, github=github, lark=lark)

        errored = TopicResult(
            root_key="om_1",
            outcomes=(),
            events=(BackfillEvent(message_id="om_1", action="error", reason="boom"),),
            processed_message_ids=(),
            error="boom",
        )
        healthy = TopicResult(
            root_key="om_1",
            outcomes=(),
            events=(BackfillEvent(message_id="om_1", action="processed", reason=""),),
            processed_message_ids=("om_1",),
        )
        # fail x3 (alert once) -> recover (reset) -> fail x3 (alert again).
        harvests = [[errored], [errored], [errored], [healthy], [errored], [errored], [errored]]
        with patch("bugpatrol.watcher._harvest_topic_results", side_effect=harvests):
            run_polling_watcher(
                config=config,
                lark=lark,  # type: ignore[arg-type]
                workflow=workflow,
                max_iterations=len(harvests),
                interval_seconds=0,
                topic_failure_alert_threshold=3,
            )

        self.assertEqual(len(_outage_alert_replies(lark)), 2)

    def test_run_polling_watcher_does_not_alert_in_dry_run(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGitHubIssuesClient()
        lark = FakeHistoryLark()
        workflow = IntakeWorkflow(config=config, github=github, lark=lark)

        errored = TopicResult(
            root_key="om_1",
            outcomes=(),
            events=(BackfillEvent(message_id="om_1", action="error", reason="boom"),),
            processed_message_ids=(),
            error="boom",
        )
        with patch("bugpatrol.watcher._harvest_topic_results", return_value=[errored]):
            run_polling_watcher(
                config=config,
                lark=lark,  # type: ignore[arg-type]
                workflow=workflow,
                max_iterations=5,
                interval_seconds=0,
                dry_run=True,
                topic_failure_alert_threshold=3,
            )

        self.assertEqual(lark.chat_messages, [])
        self.assertEqual(_outage_alert_replies(lark), [])

    def test_render_topic_outage_reply_states_error_and_retry(self) -> None:
        text = render_topic_outage_reply(error="boom: push 403", consecutive_iterations=3)
        self.assertIn("连续 3 轮未能建成 GitHub issue", text)
        self.assertIn("boom: push 403", text)

    def test_render_topic_outage_alert_truncates_topic_list(self) -> None:
        results = tuple(
            TopicResult(root_key=f"om_{i}", outcomes=(), events=(), processed_message_ids=(), error=f"err{i}")
            for i in range(8)
        )
        text = render_topic_outage_alert(errored_results=results, consecutive_iterations=4, max_topics=5)
        self.assertIn("连续 4 轮", text)
        self.assertIn("om_0", text)
        self.assertIn("om_4", text)
        self.assertNotIn("om_5", text)
        self.assertIn("另外 3 个", text)

    def test_harvest_topic_results_folds_raising_future_into_error_result(self) -> None:
        # A topic processor that raises (instead of reporting its error in the
        # result) must never kill the watcher: fold the exception into an errored
        # TopicResult so the topic retries and the outage alerting still fires.
        def _boom() -> TopicResult:
            raise RuntimeError("boom")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            in_flight = {"om_a1": pool.submit(_boom)}
            results = _harvest_topic_results(in_flight, wait_all=True)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].root_key, "om_a1")
        self.assertEqual(results[0].outcomes, ())
        self.assertEqual(results[0].events, ())
        self.assertEqual(results[0].processed_message_ids, ())
        self.assertIn("RuntimeError: boom", results[0].error)
        self.assertEqual(in_flight, {})

    def test_run_polling_watcher_can_enqueue_and_dispatch_triage(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGitHubIssuesClient()
        lark = FakeHistoryLark()
        workflow = IntakeWorkflow(config=config, github=github, lark=lark)
        with tempfile.TemporaryDirectory() as temp:
            queue = TriageRequestQueue(Path(temp) / "triage-queue.json")
            dispatcher = RecordingDispatcher()

            result = run_polling_watcher(
                config=config,
                lark=lark,  # type: ignore[arg-type]
                workflow=workflow,
                once=True,
                interval_seconds=0,
                triage_queue=queue,
                triage_quiet_seconds=0,
                triage_dispatcher=dispatcher,
            )

        self.assertEqual(result.queued_triage, 1)
        self.assertEqual(result.dispatched_triage, 1)
        self.assertEqual(dispatcher.requests[0].issue_number, 1)
        self.assertEqual(queue.due_requests(now=999999), ())

    def test_run_polling_watcher_runs_topics_concurrently(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGitHubIssuesClient()
        lark = FakeTwoTopicLark()
        workflow = IntakeWorkflow(config=config, github=github, lark=lark)
        # Both topic workers must be inside process_topic_batch at the same
        # time, otherwise the barrier times out and the test fails.
        barrier = threading.Barrier(2)

        def concurrent_process(batch, **kwargs):  # type: ignore[no-untyped-def]
            barrier.wait(timeout=10)
            return TopicResult(
                root_key=batch.root_key,
                outcomes=(),
                events=(),
                processed_message_ids=tuple(m.message_id for m in batch.messages),
            )

        with tempfile.TemporaryDirectory() as temp:
            ledger_path = Path(temp) / "ledger.json"
            from bugpatrol.ledger import JsonMessageLedger

            ledger = JsonMessageLedger.load(ledger_path)
            with patch("bugpatrol.watcher.process_topic_batch", side_effect=concurrent_process):
                result = run_polling_watcher(
                    config=config,
                    lark=lark,  # type: ignore[arg-type]
                    workflow=workflow,
                    once=True,
                    interval_seconds=0,
                    parallel_topics=2,
                    processed_ledger=ledger,
                )

            self.assertEqual(result.iterations, 1)
            self.assertTrue(ledger.is_processed("om_t1"))
            self.assertTrue(ledger.is_processed("om_t2"))

    def test_run_polling_watcher_releases_lease_after_once(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGitHubIssuesClient()
        lark = FakeHistoryLark()
        workflow = IntakeWorkflow(config=config, github=github, lark=lark)

        with tempfile.TemporaryDirectory() as temp:
            lease_file = Path(temp) / "watch.lock"
            result = run_polling_watcher(
                config=config,
                lark=lark,  # type: ignore[arg-type]
                workflow=workflow,
                once=True,
                interval_seconds=0,
                lease_file=lease_file,
            )

            self.assertEqual(result.iterations, 1)
            self.assertFalse(lease_file.exists())

    def test_run_polling_watcher_rejects_active_lease(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGitHubIssuesClient()
        lark = FakeHistoryLark()
        workflow = IntakeWorkflow(config=config, github=github, lark=lark)

        with tempfile.TemporaryDirectory() as temp:
            lease_file = Path(temp) / "watch.lock"
            FileLease(lease_file, ttl_seconds=120, owner="other").acquire()

            with self.assertRaises(LeaseHeldError):
                run_polling_watcher(
                    config=config,
                    lark=lark,  # type: ignore[arg-type]
                    workflow=workflow,
                    once=True,
                    interval_seconds=0,
                    lease_file=lease_file,
                )

    def test_run_polling_watcher_writes_event_log(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGitHubIssuesClient()
        lark = FakeHistoryLark()
        workflow = IntakeWorkflow(config=config, github=github, lark=lark)

        with tempfile.TemporaryDirectory() as temp:
            event_log = Path(temp) / "watch-events.jsonl"
            run_polling_watcher(
                config=config,
                lark=lark,  # type: ignore[arg-type]
                workflow=workflow,
                once=True,
                interval_seconds=0,
                event_log_path=event_log,
            )
            events = [json.loads(line) for line in event_log.read_text().splitlines()]

        self.assertEqual(events[0]["event"], "watch_scan")
        self.assertEqual(events[0]["processed"], 1)
        self.assertEqual(events[1]["event"], "lark_message")
        self.assertEqual(events[1]["action"], "processed")
        self.assertEqual(events[1]["reason"], "created")

    def test_repeated_skips_are_written_once_per_poll_window(self) -> None:
        """A poll re-reads the same messages, so re-logging every skip each
        round buried the event log in near-duplicates."""

        class RecordingLogger:
            def __init__(self) -> None:
                self.written: list[dict] = []

            def write(self, event: dict) -> None:
                self.written.append(event)

        logger = RecordingLogger()
        logged_skips: set[tuple[str, str]] = set()
        skip = BackfillEvent(message_id="om_1", action="skipped", reason="bot_message")

        for iteration in (1, 2, 3):
            write_backfill_events(
                logger=logger,  # type: ignore[arg-type]
                iteration=iteration,
                events=[skip],
                logged_skips=logged_skips,
            )

        self.assertEqual(len(logger.written), 1)
        self.assertEqual(logger.written[0]["message_id"], "om_1")

        # A skip that stops being reported drops out of the window, so if the
        # message comes back it is reported again rather than staying invisible.
        write_backfill_events(
            logger=logger,  # type: ignore[arg-type]
            iteration=4,
            events=[],
            logged_skips=logged_skips,
        )
        write_backfill_events(
            logger=logger,  # type: ignore[arg-type]
            iteration=5,
            events=[skip],
            logged_skips=logged_skips,
        )
        self.assertEqual(len(logger.written), 2)

    def test_a_skip_whose_reason_changes_is_written_again(self) -> None:
        class RecordingLogger:
            def __init__(self) -> None:
                self.written: list[dict] = []

            def write(self, event: dict) -> None:
                self.written.append(event)

        logger = RecordingLogger()
        logged_skips: set[tuple[str, str]] = set()

        for iteration, reason in ((1, "bot_message"), (2, "processed_ledger")):
            write_backfill_events(
                logger=logger,  # type: ignore[arg-type]
                iteration=iteration,
                events=[BackfillEvent(message_id="om_1", action="skipped", reason=reason)],
                logged_skips=logged_skips,
            )

        self.assertEqual([event["reason"] for event in logger.written], ["bot_message", "processed_ledger"])

    def test_processed_and_error_events_are_never_deduped(self) -> None:
        class RecordingLogger:
            def __init__(self) -> None:
                self.written: list[dict] = []

            def write(self, event: dict) -> None:
                self.written.append(event)

        logger = RecordingLogger()
        logged_skips: set[tuple[str, str]] = set()
        events = [
            BackfillEvent(message_id="om_1", action="processed", reason="created", issue_number=1),
            BackfillEvent(message_id="om_2", action="error", reason="boom"),
        ]

        for iteration in (1, 2):
            write_backfill_events(
                logger=logger,  # type: ignore[arg-type]
                iteration=iteration,
                events=events,
                logged_skips=logged_skips,
            )

        self.assertEqual([event["action"] for event in logger.written], ["processed", "error", "processed", "error"])

    def test_dispatch_due_triage_defers_when_issue_is_running(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            queue = TriageRequestQueue(Path(temp) / "triage-queue.json")
            request = queue.enqueue(
                issue_number=7,
                signal=FakeSignal(),
                quiet_seconds=0,
                now=100,
            )
            self.assertIsNotNone(request)
            dispatcher = RecordingDispatcher()

            dispatched = dispatch_due_triage(
                queue=queue,
                dispatcher=dispatcher,
                triage_quiet_seconds=60,
                status_reader=StaticStatusReader("Running"),
            )

            due = queue.due_requests(now=10**20)
            self.assertEqual(dispatched, 0)
            self.assertEqual(dispatcher.requests, [])
            self.assertTrue(due[0].pending_review)
            self.assertIn("pending_review_running", due[0].reasons)

    def test_dispatch_due_triage_discards_stale_request_when_issue_is_done(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            queue = TriageRequestQueue(Path(temp) / "triage-queue.json")
            request = queue.enqueue(
                issue_number=7,
                signal=FakeSignal(),
                quiet_seconds=0,
                now=100,
            )
            self.assertIsNotNone(request)
            dispatcher = RecordingDispatcher()

            dispatched = dispatch_due_triage(
                queue=queue,
                dispatcher=dispatcher,
                triage_quiet_seconds=60,
                status_reader=StaticStatusReader("Done"),
            )

            self.assertEqual(dispatched, 0)
            self.assertEqual(dispatcher.requests, [])
            self.assertEqual(queue.due_requests(now=10**20), ())

    def test_dispatch_due_triage_defers_when_field_values_probe_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            queue = TriageRequestQueue(Path(temp) / "triage-queue.json")
            request = queue.enqueue(
                issue_number=7,
                signal=FakeSignal(),
                quiet_seconds=0,
                now=100,
            )
            self.assertIsNotNone(request)
            dispatcher = RecordingDispatcher()

            dispatched = dispatch_due_triage(
                queue=queue,
                dispatcher=dispatcher,
                triage_quiet_seconds=60,
                status_reader=RaisingStatusReader(GitHubIssueFieldsError("api.github.com EOF")),
            )

            due = queue.due_requests(now=10**20)
            self.assertEqual(dispatched, 0)
            self.assertEqual(dispatcher.requests, [])
            self.assertTrue(due[0].pending_review)
            self.assertIn("pending_review_running", due[0].reasons)

    def test_dispatch_due_triage_discards_closed_issue(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            queue = TriageRequestQueue(Path(temp) / "triage-queue.json")
            request = queue.enqueue(
                issue_number=7,
                signal=FakeSignal(),
                quiet_seconds=0,
                now=100,
            )
            self.assertIsNotNone(request)
            dispatcher = RecordingDispatcher()

            # A closed issue must be dropped from the queue even though its
            # field reads "Running" (stale from an old run) — otherwise it
            # would defer forever and never re-dispatch if reopened.
            dispatched = dispatch_due_triage(
                queue=queue,
                dispatcher=dispatcher,
                triage_quiet_seconds=60,
                status_reader=StaticStatusReader("Running", state="closed"),
            )

            self.assertEqual(dispatched, 0)
            self.assertEqual(dispatcher.requests, [])
            self.assertEqual(queue.due_requests(now=10**20), ())

    def test_dispatch_due_triage_defers_when_state_probe_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            queue = TriageRequestQueue(Path(temp) / "triage-queue.json")
            request = queue.enqueue(
                issue_number=7,
                signal=FakeSignal(),
                quiet_seconds=0,
                now=100,
            )
            self.assertIsNotNone(request)
            dispatcher = RecordingDispatcher()

            dispatched = dispatch_due_triage(
                queue=queue,
                dispatcher=dispatcher,
                triage_quiet_seconds=60,
                status_reader=RaisingStateReader(GitHubIssueFieldsError("api.github.com EOF")),
            )

            due = queue.due_requests(now=10**20)
            self.assertEqual(dispatched, 0)
            self.assertEqual(dispatcher.requests, [])
            self.assertTrue(due[0].pending_review)
            self.assertIn("pending_review_running", due[0].reasons)

    def test_dispatch_due_triage_defers_when_dispatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            queue = TriageRequestQueue(Path(temp) / "triage-queue.json")
            request = queue.enqueue(
                issue_number=7,
                signal=FakeSignal(),
                quiet_seconds=0,
                now=100,
            )
            self.assertIsNotNone(request)
            dispatcher = RaisingDispatcher(RuntimeError("gh workflow run failed"))

            dispatched = dispatch_due_triage(
                queue=queue,
                dispatcher=dispatcher,
                triage_quiet_seconds=60,
            )

            due = queue.due_requests(now=10**20)
            self.assertEqual(dispatched, 0)
            # The dispatcher was invoked (the failure is transient, after it
            # started), so the request was recorded — but it must be deferred,
            # not counted as dispatched.
            self.assertEqual(len(dispatcher.requests), 1)
            self.assertTrue(due[0].pending_review)
            self.assertIn("pending_review_running", due[0].reasons)

    def test_dispatch_reconcile_runs_command_and_writes_success_event(self) -> None:
        logger = RecordingLogger()

        ok = _dispatch_reconcile([sys.executable, "-c", "pass"], logger=logger, iteration=3)

        self.assertTrue(ok)
        self.assertEqual(len(logger.written), 1)
        event = logger.written[0]
        self.assertEqual(event["event"], "watch_reconcile_dispatch")
        self.assertEqual(event["iteration"], 3)

    def test_dispatch_reconcile_returns_false_and_logs_on_failure(self) -> None:
        logger = RecordingLogger()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            ok = _dispatch_reconcile(
                [sys.executable, "-c", "raise SystemExit(3)"],
                logger=logger,
                iteration=3,
            )

        self.assertFalse(ok)
        self.assertIn("reconcile dispatch failed", stderr.getvalue())
        self.assertEqual(len(logger.written), 1)
        event = logger.written[0]
        self.assertEqual(event["event"], "watch_reconcile_dispatch_error")
        self.assertEqual(event["returncode"], 3)

    def test_run_polling_watcher_dispatches_reconcile_when_due(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGitHubIssuesClient()
        lark = FakeHistoryLark()
        workflow = IntakeWorkflow(config=config, github=github, lark=lark)

        dispatched: list[tuple] = []

        def fake_dispatch(command: object, *, logger: object, iteration: int) -> bool:
            dispatched.append(command)  # type: ignore[arg-type]
            return True

        with patch(
            "bugpatrol.watcher.time.monotonic",
            side_effect=iter(range(100, 1000)).__next__,
        ), patch(
            "bugpatrol.watcher._dispatch_reconcile",
            side_effect=fake_dispatch,
        ):
            result = run_polling_watcher(
                config=config,
                lark=lark,  # type: ignore[arg-type]
                workflow=workflow,
                max_iterations=2,
                interval_seconds=0,
                reconcile_dispatch_command=["true"],
                reconcile_interval_seconds=0.5,
            )

        self.assertEqual(result.dispatched_reconcile, 1)
        self.assertEqual(dispatched, [("true",)])

    def test_run_polling_watcher_failed_reconcile_is_retried_not_crashed(self) -> None:
        config = load_project_config(Path("projects/todo-sandbox.toml"))
        github = FakeGitHubIssuesClient()
        lark = FakeHistoryLark()
        workflow = IntakeWorkflow(config=config, github=github, lark=lark)

        calls: list[int] = []

        def failing_dispatch(command: object, *, logger: object, iteration: int) -> bool:
            calls.append(iteration)
            return False

        with patch(
            "bugpatrol.watcher.time.monotonic",
            side_effect=iter(range(100, 1000)).__next__,
        ), patch(
            "bugpatrol.watcher._dispatch_reconcile",
            side_effect=failing_dispatch,
        ):
            result = run_polling_watcher(
                config=config,
                lark=lark,  # type: ignore[arg-type]
                workflow=workflow,
                max_iterations=3,
                interval_seconds=0,
                reconcile_dispatch_command=["true"],
                reconcile_interval_seconds=0.5,
            )

        # Failure was logged+backed off, not counted as dispatched, and did not
        # crash the loop (iteration 2 re-checks but is past the backoff window).
        self.assertEqual(result.dispatched_reconcile, 0)
        self.assertEqual(calls, [1])


class RecordingLogger:
    def __init__(self) -> None:
        self.written: list[dict] = []

    def write(self, event: dict) -> None:
        self.written.append(event)


class RecordingDispatcher:
    def __init__(self) -> None:
        self.requests: list[TriageRequest] = []

    def dispatch(self, request: TriageRequest) -> object:
        self.requests.append(request)
        return object()


class FakeSignal:
    should_enqueue = True
    reason = "material_followup"
    material_message_ids = ("om_1",)
    asset_urls = ()


class StaticStatusReader:
    def __init__(self, status: str, *, state: str = "open") -> None:
        self._status = status
        self._state = state

    def issue_state(self, issue_number: int) -> str:
        return self._state

    def triage_status(self, issue_number: int) -> str:
        return self._status


class RaisingStatusReader:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def issue_state(self, issue_number: int) -> str:
        return "open"

    def triage_status(self, issue_number: int) -> str:
        raise self._error


class RaisingStateReader:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def issue_state(self, issue_number: int) -> str:
        raise self._error

    def triage_status(self, issue_number: int) -> str:
        return "Pending"


class RaisingDispatcher:
    def __init__(self, error: Exception) -> None:
        self._error = error
        self.requests: list[TriageRequest] = []

    def dispatch(self, request: TriageRequest) -> object:
        self.requests.append(request)
        raise self._error


if __name__ == "__main__":
    unittest.main()
