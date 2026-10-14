"""Discover branch-scoped topic chats by convention instead of config.

A project's CI creates one Lark topic group per feature branch (see fived
`.github/workflows/bugpatrol-feature-topic.yml`). Listing those groups in
`[[lark.branch_chats]]` meant every new branch needed a config edit plus a
watcher restart before its group was scanned at all -- until then bugs reported
there were silently dropped.

The group name carries the owning repo and the branch -- "fived ·
feature-moments", main group "fived · main" -- so the mapping can be derived:
take the chats the bot belongs to, keep the ones prefixed with this project's
`lark.chat_prefix` whose branch is live on the remote, and treat that branch as
the chat's target branch. Config entries stay supported as overrides for groups
that predate (or don't follow) the convention.

The prefix is what keeps projects apart. One Lark app serves every project, so
`list_bot_chats` returns every project's groups, and matching on the bare branch
name picked up a group whose branch happened to exist in two repos -- the same
topic then became an issue in both (23 pairs before this was caught). A name
that carries no prefix is reported as unmatched rather than guessed at.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable, Protocol

from bugpatrol.config import CHAT_NAME_SEPARATOR, ProjectConfig


class ChatLister(Protocol):
    def list_bot_chats(self) -> tuple[tuple[str, str], ...]:
        """Return (chat_id, name) for every chat the bot belongs to."""


class BranchLister(Protocol):
    def list_remote_branch_names(self, *, repo: str) -> tuple[str, ...]:
        """Return every branch name on the remote repo."""


@dataclass(frozen=True)
class BranchChatDiscovery:
    """Chat id -> branch, plus the groups that were seen and deliberately not claimed.

    Neither tuple is a dump of every group the bot sits in: groups belonging to
    another project are silently skipped, since they are not this project's to
    report on. What is left is two groups an operator may need to act on, and
    the two need *different* actions -- which is why they are not one list:

    `unmatched_chats` -- un-prefixed groups named after one of this project's
    live branches (the pre-contract naming). The name is the problem, so the fix
    is to rename the group.
    `archived_chats` -- groups carrying this project's prefix whose branch is
    gone. The name is right and the branch is not, so renaming would not help:
    the group needs archiving.
    """

    branch_chats: dict[str, str]
    unmatched_chats: tuple[str, ...]
    archived_chats: tuple[str, ...]


def _split_chat_name(name: str) -> tuple[str, str] | None:
    """Split "<prefix> · <branch>"; None when the name carries no prefix."""
    prefix, separator, branch = name.partition(CHAT_NAME_SEPARATOR)
    if not separator:
        return None
    return prefix.strip(), branch.strip()


def discover_branch_chats(
    *,
    lark: ChatLister,
    github: BranchLister,
    config: ProjectConfig,
) -> BranchChatDiscovery:
    """Map this project's topic chats to their branch, by name.

    A chat is claimed only when its name is prefixed with this project's
    `chat_prefix` and its branch token is either "main" or live on the remote.
    Everything else is left alone: another project's prefix means the group is
    theirs, and an un-prefixed name is never guessed at -- guessing is what let
    a group whose branch exists in two repos be claimed by both.

    The main watcher chat, and any chat already pinned in `[[lark.branch_chats]]`,
    are skipped: their mapping is declared, so discovery has nothing to add and
    should not report them as unmatched.
    """
    branches = set(github.list_remote_branch_names(repo=config.github_repo))
    branches.discard("main")
    prefix = config.lark.chat_prefix
    if not prefix:
        raise ValueError("config.lark.chat_prefix is empty; topic groups cannot be attributed")
    declared = set(config.lark.branch_chats or {})
    matched: dict[str, str] = {}
    unmatched: list[str] = []
    archived: list[str] = []
    for chat_id, name in lark.list_bot_chats():
        if chat_id == config.lark.chat_id or chat_id in declared:
            continue
        raw = name.strip()
        split = _split_chat_name(raw)
        if split is None:
            # Pre-contract naming. Report it if it looks like ours rather than
            # claiming it: the same name may be a live branch in another repo.
            if raw and raw in branches:
                unmatched.append(raw)
            continue
        chat_prefix, branch = split
        if chat_prefix != prefix:
            continue
        if branch == "main" or branch in branches:
            matched[chat_id] = branch
        elif branch:
            archived.append(raw)
    return BranchChatDiscovery(
        branch_chats=matched,
        unmatched_chats=tuple(unmatched),
        archived_chats=tuple(archived),
    )


def apply_branch_chats(config: ProjectConfig, branch_chats: dict[str, str]) -> ProjectConfig:
    """Return `config` with discovered chats merged in; config entries win.

    An explicit `[[lark.branch_chats]]` entry is a deliberate override (e.g.
    the `2026/chat-live` group, whose name does not match its branch), so it
    takes precedence over whatever discovery inferred for the same chat.
    """
    merged = {**branch_chats, **(config.lark.branch_chats or {})}
    return replace(config, lark=replace(config.lark, branch_chats=merged or None))


class CachedBranchChatDiscoverer:
    """Re-runs discovery at most once per `ttl_seconds`.

    The watcher polls every ~30s but branches and groups change on a human
    timescale, so refreshing on a slower clock keeps the poll loop cheap while
    still picking up a new feature branch without a restart.
    """

    def __init__(
        self,
        *,
        lark: ChatLister,
        github: BranchLister,
        config: ProjectConfig,
        ttl_seconds: float = 300,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._lark = lark
        self._github = github
        self._config = config
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._cached: BranchChatDiscovery | None = None
        self._fetched_at = 0.0

    def resolve(self) -> BranchChatDiscovery:
        now = self._clock()
        if self._cached is not None and now - self._fetched_at < self._ttl_seconds:
            return self._cached
        discovery = discover_branch_chats(
            lark=self._lark,
            github=self._github,
            config=self._config,
        )
        self._cached = discovery
        self._fetched_at = now
        return discovery
