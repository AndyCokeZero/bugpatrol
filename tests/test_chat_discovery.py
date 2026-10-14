from __future__ import annotations

import unittest
from pathlib import Path

from bugpatrol.chat_discovery import (
    CachedBranchChatDiscoverer,
    apply_branch_chats,
    discover_branch_chats,
)
from bugpatrol.config import load_project_config


CONFIG = load_project_config(Path("projects/todo-sandbox.toml"))
MAIN_CHAT_ID = CONFIG.lark.chat_id
PREFIX = CONFIG.lark.chat_prefix
# Two real projects that share one Lark app, and whose repos share branch names.
FIVED = load_project_config(Path("projects/fived.toml"))
WEAVER = load_project_config(Path("projects/weaver.toml"))


class FakeChatLister:
    def __init__(self, chats: tuple[tuple[str, str], ...]) -> None:
        self.chats = chats
        self.calls = 0

    def list_bot_chats(self) -> tuple[tuple[str, str], ...]:
        self.calls += 1
        return self.chats


class FakeBranchLister:
    def __init__(self, branches: tuple[str, ...]) -> None:
        self.branches = branches

    def list_remote_branch_names(self, *, repo: str) -> tuple[str, ...]:
        del repo
        return self.branches


class DiscoverBranchChatsTest(unittest.TestCase):
    def test_prefixed_chat_named_after_a_live_branch_is_adopted(self) -> None:
        discovery = discover_branch_chats(
            lark=FakeChatLister((("oc_moments", f"{PREFIX} · feature-moments"),)),
            github=FakeBranchLister(("main", "feature-moments")),
            config=CONFIG,
        )

        self.assertEqual(discovery.branch_chats, {"oc_moments": "feature-moments"})

    def test_unprefixed_branch_named_chat_is_reported_not_adopted(self) -> None:
        # The duplicate-issue bug: a bare branch name is a live branch here, but
        # it is equally a live branch in another repo, so claiming it would hand
        # the same group to two watchers. Report it instead of guessing.
        discovery = discover_branch_chats(
            lark=FakeChatLister((("oc_legacy", "feature-moments"),)),
            github=FakeBranchLister(("main", "feature-moments")),
            config=CONFIG,
        )

        self.assertEqual(discovery.branch_chats, {})
        self.assertEqual(discovery.unmatched_chats, ("feature-moments",))

    def test_another_projects_prefix_is_ignored_entirely(self) -> None:
        # Not ours is not the same as unmatched: another project's group is not
        # this project's to report on.
        discovery = discover_branch_chats(
            lark=FakeChatLister((("oc_theirs", "fived · feature-moments"),)),
            github=FakeBranchLister(("main", "feature-moments")),
            config=CONFIG,
        )

        self.assertEqual(discovery.branch_chats, {})
        self.assertEqual(discovery.unmatched_chats, ())

    def test_chat_whose_prefixed_branch_is_gone_is_reported_as_archived(self) -> None:
        # The name is right and the branch is not, so this is not the same
        # problem as an un-prefixed name: renaming is the fix for that one,
        # archiving the group is the fix for this one. Keeping them in one list
        # made the watcher describe a correctly-named group as mis-named.
        discovery = discover_branch_chats(
            lark=FakeChatLister((("oc_stale", f"{PREFIX} · feature-deleted"),)),
            github=FakeBranchLister(("main", "feature-moments")),
            config=CONFIG,
        )

        self.assertEqual(discovery.branch_chats, {})
        self.assertEqual(discovery.archived_chats, (f"{PREFIX} · feature-deleted",))
        self.assertEqual(discovery.unmatched_chats, ())

    def test_the_two_unclaimed_kinds_are_never_mixed_together(self) -> None:
        discovery = discover_branch_chats(
            lark=FakeChatLister(
                (
                    ("oc_legacy", "feature-moments"),
                    ("oc_stale", f"{PREFIX} · feature-deleted"),
                )
            ),
            github=FakeBranchLister(("main", "feature-moments")),
            config=CONFIG,
        )

        self.assertEqual(discovery.unmatched_chats, ("feature-moments",))
        self.assertEqual(discovery.archived_chats, (f"{PREFIX} · feature-deleted",))

    def test_main_group_name_maps_to_the_main_branch(self) -> None:
        discovery = discover_branch_chats(
            lark=FakeChatLister((("oc_renamed_main", f"{PREFIX} · main"),)),
            github=FakeBranchLister(("main", "feature-moments")),
            config=CONFIG,
        )

        self.assertEqual(discovery.branch_chats, {"oc_renamed_main": "main"})

    def test_unrelated_chats_and_the_main_chat_are_never_adopted(self) -> None:
        pinned = next(iter(CONFIG.lark.branch_chats))
        discovery = discover_branch_chats(
            lark=FakeChatLister(
                (
                    (MAIN_CHAT_ID, "main"),
                    ("oc_random", "Agent 体验反馈"),
                    ("oc_mainish", "main"),
                    (pinned, f"{PREFIX} · feature-moments"),
                )
            ),
            github=FakeBranchLister(("main", "feature-moments")),
            config=CONFIG,
        )

        # The chat pinned in `[[lark.branch_chats]]` is skipped: its mapping is
        # declared, so discovery has nothing to add and nothing to report.
        self.assertEqual(discovery.branch_chats, {})
        self.assertEqual(discovery.unmatched_chats, ())

    def test_prefix_is_matched_after_stripping_whitespace(self) -> None:
        discovery = discover_branch_chats(
            lark=FakeChatLister((("oc_moments", f"  {PREFIX} ·  feature-moments  "),)),
            github=FakeBranchLister(("feature-moments",)),
            config=CONFIG,
        )

        self.assertEqual(discovery.branch_chats, {"oc_moments": "feature-moments"})

    def test_a_group_shared_by_two_repos_is_claimed_by_prefix_only(self) -> None:
        # Both repos have `feature-october-impr`, and one Lark app serves both
        # projects, so both watchers see this exact group. The prefix is the
        # only thing that says whose it is.
        chats = (("oc_october", "fived · feature-october-impr"),)
        branches = FakeBranchLister(("main", "feature-october-impr"))

        fived = discover_branch_chats(lark=FakeChatLister(chats), github=branches, config=FIVED)
        weaver = discover_branch_chats(lark=FakeChatLister(chats), github=branches, config=WEAVER)

        self.assertEqual(fived.branch_chats, {"oc_october": "feature-october-impr"})
        self.assertEqual(weaver.branch_chats, {})
        self.assertEqual(weaver.unmatched_chats, ())

    def test_default_prefix_is_the_repo_name(self) -> None:
        self.assertEqual(FIVED.lark.chat_prefix, "fived")
        self.assertEqual(WEAVER.lark.chat_prefix, "weaver")


class ApplyBranchChatsTest(unittest.TestCase):
    def test_discovered_chats_are_added_to_the_scanned_set(self) -> None:
        updated = apply_branch_chats(CONFIG, {"oc_moments": "feature-moments"})

        self.assertIn("oc_moments", updated.lark.all_chat_ids())
        self.assertEqual(updated.lark.branch_for_chat("oc_moments"), "feature-moments")
        self.assertEqual(CONFIG.lark.branch_for_chat("oc_moments"), "main")

    def test_configured_override_wins_over_discovery(self) -> None:
        configured_chat = next(iter(CONFIG.lark.branch_chats))

        updated = apply_branch_chats(CONFIG, {configured_chat: "feature-guessed"})

        self.assertEqual(
            updated.lark.branch_for_chat(configured_chat),
            CONFIG.lark.branch_for_chat(configured_chat),
        )

    def test_applying_to_the_base_config_drops_chats_that_stopped_matching(self) -> None:
        with_chat = apply_branch_chats(CONFIG, {"oc_moments": "feature-moments"})

        without_chat = apply_branch_chats(CONFIG, {})

        self.assertIn("oc_moments", with_chat.lark.all_chat_ids())
        self.assertNotIn("oc_moments", without_chat.lark.all_chat_ids())


class CachedBranchChatDiscovererTest(unittest.TestCase):
    def test_refetches_only_after_the_ttl_expires(self) -> None:
        lark = FakeChatLister((("oc_moments", "feature-moments"),))
        now = [0.0]
        discoverer = CachedBranchChatDiscoverer(
            lark=lark,
            github=FakeBranchLister(("feature-moments",)),
            config=CONFIG,
            ttl_seconds=300,
            clock=lambda: now[0],
        )

        discoverer.resolve()
        now[0] = 299
        discoverer.resolve()
        self.assertEqual(lark.calls, 1)

        now[0] = 301
        discoverer.resolve()
        self.assertEqual(lark.calls, 2)


if __name__ == "__main__":
    unittest.main()
