from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bugpatrol.ci_watch import (
    BranchState,
    WatchState,
    evaluate,
    load_state,
    save_state,
    settled_branch_state,
    watch_ci,
)
from bugpatrol.clients import WorkflowRun

REPO = "verneagent/bugpatrol"
BRANCH = "main"
CHAT_ID = "oc_fived"
RED_SHA = "aaaa111122223333444455556666777788889999"
GREEN_SHA = "bbbb111122223333444455556666777788889999"


def run(
    *,
    sha: str,
    conclusion: str,
    status: str = "completed",
    workflow: str = "CI",
    run_id: int = 1,
    url: str = "https://example.test/run/1",
) -> WorkflowRun:
    return WorkflowRun(
        run_id=run_id,
        workflow_name=workflow,
        status=status,
        conclusion=conclusion,
        head_sha=sha,
        url=url,
    )


class FakeRunLister:
    def __init__(self, runs: tuple[WorkflowRun, ...]) -> None:
        self.runs = runs
        self.calls = 0

    def list_recent_workflow_runs(
        self, *, repo: str, branch: str, limit: int = 20
    ) -> tuple[WorkflowRun, ...]:
        del repo, branch, limit
        self.calls += 1
        return self.runs


class FakeNotifier:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send_chat_message(self, *, chat_id: str, text: str) -> object:
        self.sent.append((chat_id, text))
        return object()


class SettledBranchStateTest(unittest.TestCase):
    def test_green_commit_is_not_red(self) -> None:
        state = settled_branch_state((run(sha=GREEN_SHA, conclusion="success"),))

        assert state is not None
        self.assertEqual(state.head_sha, GREEN_SHA)
        self.assertFalse(state.red)

    def test_a_cancelled_run_does_not_make_the_commit_red(self) -> None:
        # A superseded run reports cancelled while the real result lives in the
        # sibling job, so counting it would alert on every force-push.
        state = settled_branch_state((run(sha=GREEN_SHA, conclusion="cancelled"),))

        assert state is not None
        self.assertFalse(state.red)

    def test_a_commit_is_judged_across_all_of_its_workflows(self) -> None:
        # One commit, two workflows, only one of them red. Which one finished
        # last must not change the verdict.
        state = settled_branch_state(
            (
                run(sha=RED_SHA, conclusion="success", workflow="api", run_id=2),
                run(sha=RED_SHA, conclusion="failure", workflow="web", run_id=1),
            )
        )

        assert state is not None
        self.assertTrue(state.red)
        self.assertEqual(state.failed_workflows, ("web",))

    def test_a_commit_still_running_is_skipped_for_the_one_before_it(self) -> None:
        state = settled_branch_state(
            (
                run(sha=GREEN_SHA, conclusion="", status="in_progress", run_id=2),
                run(sha=RED_SHA, conclusion="failure", run_id=1),
            )
        )

        assert state is not None
        self.assertEqual(state.head_sha, RED_SHA)

    def test_a_commit_with_a_queued_sibling_workflow_is_not_judged_yet(self) -> None:
        state = settled_branch_state(
            (
                run(sha=GREEN_SHA, conclusion="success", workflow="api", run_id=2),
                run(sha=GREEN_SHA, conclusion="", status="queued", workflow="web", run_id=1),
            )
        )

        self.assertIsNone(state)

    def test_nothing_settled_yet_is_no_state(self) -> None:
        self.assertIsNone(settled_branch_state((run(sha=GREEN_SHA, conclusion="", status="queued"),)))
        self.assertIsNone(settled_branch_state(()))


class EvaluateTest(unittest.TestCase):
    def _red(self) -> BranchState:
        return BranchState(
            head_sha=RED_SHA,
            red=True,
            url="https://example.test/run/9",
            failed_workflows=("CI",),
        )

    def test_a_freshly_red_commit_alerts_and_is_remembered(self) -> None:
        state, alert = evaluate(previous=WatchState(), current=self._red(), repo=REPO, branch=BRANCH)

        assert alert is not None
        self.assertEqual(alert.kind, "broke")
        self.assertEqual(state.alerted_sha, RED_SHA)
        self.assertIn(RED_SHA[:7], alert.text)
        self.assertIn(REPO, alert.text)

    def test_the_same_red_commit_does_not_alert_twice(self) -> None:
        # The point of the state: every later check sees the same red commit
        # until a newer one is pushed.
        state, alert = evaluate(
            previous=WatchState(alerted_sha=RED_SHA),
            current=self._red(),
            repo=REPO,
            branch=BRANCH,
        )

        self.assertIsNone(alert)
        self.assertEqual(state.alerted_sha, RED_SHA)

    def test_a_newer_red_commit_alerts_again(self) -> None:
        _, alert = evaluate(
            previous=WatchState(alerted_sha="cccc0000000000000000000000000000000000000"),
            current=self._red(),
            repo=REPO,
            branch=BRANCH,
        )

        assert alert is not None
        self.assertEqual(alert.kind, "broke")

    def test_recovery_alerts_once_and_clears_the_state(self) -> None:
        current = BranchState(head_sha=GREEN_SHA, red=False, url="", failed_workflows=())

        state, alert = evaluate(
            previous=WatchState(alerted_sha=RED_SHA), current=current, repo=REPO, branch=BRANCH
        )

        assert alert is not None
        self.assertEqual(alert.kind, "fixed")
        self.assertEqual(state.alerted_sha, "")

        _, again = evaluate(previous=state, current=current, repo=REPO, branch=BRANCH)
        self.assertIsNone(again)

    def test_a_branch_that_was_never_red_stays_quiet(self) -> None:
        current = BranchState(head_sha=GREEN_SHA, red=False, url="", failed_workflows=())

        state, alert = evaluate(previous=WatchState(), current=current, repo=REPO, branch=BRANCH)

        self.assertIsNone(alert)
        self.assertEqual(state.alerted_sha, "")

    def test_an_unsettled_branch_says_nothing(self) -> None:
        _, alert = evaluate(previous=WatchState(), current=None, repo=REPO, branch=BRANCH)

        self.assertIsNone(alert)


class StateFileTest(unittest.TestCase):
    def test_missing_file_is_an_unwatched_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_state(Path(tmp) / "absent.json"), WatchState())

    def test_state_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "state.json"

            save_state(path, WatchState(alerted_sha=RED_SHA))

            self.assertEqual(load_state(path).alerted_sha, RED_SHA)

    def test_a_corrupt_state_file_raises_rather_than_resetting(self) -> None:
        # Resetting silently would re-alert on the same red commit on every
        # check, which is the noise the state exists to prevent.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text("[]")

            with self.assertRaises(ValueError):
                load_state(path)

    def test_a_non_string_sha_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text(json.dumps({"alerted_sha": 7}))

            with self.assertRaises(ValueError):
                load_state(path)


class WatchCiTest(unittest.TestCase):
    def _watch(self, tmp: str, lister: FakeRunLister, notifier: FakeNotifier | None, *, dry_run: bool):
        return watch_ci(
            repo=REPO,
            branch=BRANCH,
            chat_id=CHAT_ID,
            runs=lister,
            lark=notifier,
            state_path=Path(tmp) / "state.json",
            dry_run=dry_run,
        )

    def test_a_red_branch_is_delivered_and_remembered(self) -> None:
        notifier = FakeNotifier()
        with tempfile.TemporaryDirectory() as tmp:
            outcome = self._watch(
                tmp, FakeRunLister((run(sha=RED_SHA, conclusion="failure"),)), notifier, dry_run=False
            )

            self.assertTrue(outcome.delivered)
            self.assertEqual(outcome.alert, "broke")
            self.assertTrue(outcome.red)
            self.assertEqual(len(notifier.sent), 1)
            self.assertEqual(notifier.sent[0][0], CHAT_ID)
            self.assertEqual(load_state(Path(tmp) / "state.json").alerted_sha, RED_SHA)

    def test_a_second_check_of_the_same_red_commit_sends_nothing(self) -> None:
        notifier = FakeNotifier()
        with tempfile.TemporaryDirectory() as tmp:
            lister = FakeRunLister((run(sha=RED_SHA, conclusion="failure"),))

            self._watch(tmp, lister, notifier, dry_run=False)
            second = self._watch(tmp, lister, notifier, dry_run=False)

            self.assertFalse(second.delivered)
            self.assertEqual(second.alert, "")
            self.assertEqual(len(notifier.sent), 1)

    def test_a_dry_run_neither_sends_nor_persists(self) -> None:
        # Persisting here would swallow the real alert that follows.
        notifier = FakeNotifier()
        with tempfile.TemporaryDirectory() as tmp:
            outcome = self._watch(
                tmp, FakeRunLister((run(sha=RED_SHA, conclusion="failure"),)), notifier, dry_run=True
            )

            self.assertFalse(outcome.delivered)
            self.assertEqual(outcome.alert, "broke")
            self.assertEqual(notifier.sent, [])
            self.assertFalse((Path(tmp) / "state.json").exists())

    def test_delivering_without_a_chat_is_an_error_not_a_silent_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self._watch(
                    tmp, FakeRunLister((run(sha=RED_SHA, conclusion="failure"),)), None, dry_run=False
                )

    def test_recovery_is_delivered_after_a_red_alert(self) -> None:
        notifier = FakeNotifier()
        with tempfile.TemporaryDirectory() as tmp:
            self._watch(
                tmp, FakeRunLister((run(sha=RED_SHA, conclusion="failure"),)), notifier, dry_run=False
            )
            outcome = self._watch(
                tmp, FakeRunLister((run(sha=GREEN_SHA, conclusion="success"),)), notifier, dry_run=False
            )

            self.assertEqual(outcome.alert, "fixed")
            self.assertFalse(outcome.red)
            self.assertEqual(len(notifier.sent), 2)


if __name__ == "__main__":
    unittest.main()
