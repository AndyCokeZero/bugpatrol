"""Attachments must reach the agent as verified local files, never as a URL.

Regression cover for issue #6124: the agent fetched a private assets-repo URL,
got GitHub's 404 HTML page, saved it as `s1.jpg`, and the model gateway
answered `400 {"model":"deepseek-v4-flash"}`, aborting the run.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bugpatrol.media_localize import localize_media, parse_asset_url, sniff_media
from bugpatrol.triage_context import MediaEvidence, render_triage_context_markdown

ASSETS_REPO = "TheCloverLab/fived-assets"
ASSET_URL = (
    "https://github.com/TheCloverLab/fived-assets/raw/main/"
    ".github/issue-assets/om_x1/img_v3_abc.jpg"
)
CONTENTS_ENDPOINT = (
    "/repos/TheCloverLab/fived-assets/contents/.github/issue-assets/om_x1/img_v3_abc.jpg?ref=main"
)
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
# The shape of the actual failure: a full GitHub 404 page, saved as `.jpg`.
HTML_PAGE = b"\n\n\n<!DOCTYPE html>\n<html lang=\"en\">\n<title>Page not found</title>"


class _FakeGh:
    """Stand-in for the `gh` CLI: records calls, replays scripted responses."""

    def __init__(self, responses: dict[str, subprocess.CompletedProcess[bytes]]) -> None:
        self.responses = responses
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        # args = [<gh>, "api", <endpoint>, "-H", "Accept: ..."].
        self.calls.append(list(args))
        return self.responses[args[2]]


def _install_gh(fake: _FakeGh):
    return mock.patch("bugpatrol.media_localize.subprocess.run", fake)


def _ok(body: bytes) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=body, stderr=b"")


def _localize_one(
    dest_dir: Path,
    *,
    kind: str = "image",
    assets_repo: str = ASSETS_REPO,
) -> MediaEvidence:
    (item,) = localize_media(
        (MediaEvidence(kind=kind, url=ASSET_URL),),
        dest_dir=dest_dir,
        assets_repo=assets_repo,
        gh="gh",
    )
    return item


class MediaLocalizeTest(unittest.TestCase):
    def test_parse_asset_url_maps_to_contents_api(self) -> None:
        self.assertEqual(
            parse_asset_url(ASSET_URL, assets_repo=ASSETS_REPO, branch="main"),
            CONTENTS_ENDPOINT,
        )

    def test_parse_asset_url_ignores_foreign_urls(self) -> None:
        # A Lark resource URL, another repo, and a non-raw link all stay untouched.
        for url in (
            "https://open.larksuite.com/open-apis/x",
            ASSET_URL.replace("fived-assets", "fived"),
            ASSET_URL.replace("/raw/", "/blob/"),
        ):
            with self.subTest(url=url):
                self.assertIsNone(parse_asset_url(url, assets_repo=ASSETS_REPO, branch="main"))
        # A raw link on someone else's branch is not ours either.
        self.assertIsNone(parse_asset_url(ASSET_URL, assets_repo=ASSETS_REPO, branch="release"))

    def test_sniff_media(self) -> None:
        cases: tuple[tuple[bytes, tuple[str, str] | None], ...] = (
            (b"\xff\xd8\xff\xe0", ("image", ".jpg")),
            (b"\x89PNG\r\n\x1a\n", ("image", ".png")),
            (b"GIF89a", ("image", ".gif")),
            (b"RIFF\x00\x00\x00\x00WEBP", ("image", ".webp")),
            (b"\x00\x00\x00\x18ftypmp42", ("video", ".mp4")),
            (b"\x00\x00\x00\x14ftypqt  ", ("video", ".mov")),
            (b"\x1aE\xdf\xa3", ("video", ".webm")),
            (HTML_PAGE[:12], None),
            (b"", None),
        )
        for head, expected in cases:
            with self.subTest(head=head):
                self.assertEqual(sniff_media(head), expected)

    def test_localize_writes_a_verified_local_copy(self) -> None:
        fake = _FakeGh({CONTENTS_ENDPOINT: _ok(JPEG)})
        with tempfile.TemporaryDirectory() as tmp:
            dest_dir = Path(tmp) / "attachments"
            with _install_gh(fake):
                item = _localize_one(dest_dir)

            self.assertTrue(item.local_path)
            self.assertEqual(Path(item.local_path).read_bytes(), JPEG)
            self.assertEqual(Path(item.local_path).name, "media-1.jpg")
            self.assertEqual(item.local_status, "")
        # Fetched with the runner's credentials, not the agent's anonymous curl.
        self.assertEqual(fake.calls[0][:3], ["gh", "api", CONTENTS_ENDPOINT])

    def test_html_error_page_is_rejected_and_leaves_no_file(self) -> None:
        """The exact #6124 failure: a 404 page must never become a readable `.jpg`."""
        fake = _FakeGh({CONTENTS_ENDPOINT: _ok(HTML_PAGE)})
        with tempfile.TemporaryDirectory() as tmp:
            dest_dir = Path(tmp) / "attachments"
            with _install_gh(fake):
                item = _localize_one(dest_dir)

            self.assertEqual(item.local_path, "")
            self.assertIn("not a media file", item.local_status)
            # No stray file for the agent to trip over.
            self.assertEqual(list(dest_dir.iterdir()), [])

    def test_failed_download_reports_status_without_raising(self) -> None:
        failure = subprocess.CompletedProcess(
            args=[], returncode=1, stdout=b"", stderr=b"gh: Not Found (HTTP 404)"
        )
        fake = _FakeGh({CONTENTS_ENDPOINT: failure})
        with tempfile.TemporaryDirectory() as tmp:
            with _install_gh(fake):
                item = _localize_one(Path(tmp) / "attachments")

        self.assertEqual(item.local_path, "")
        self.assertIn("Not Found", item.local_status)

    def test_kind_mismatch_is_reported(self) -> None:
        fake = _FakeGh({CONTENTS_ENDPOINT: _ok(JPEG)})
        with tempfile.TemporaryDirectory() as tmp:
            with _install_gh(fake):
                item = _localize_one(Path(tmp) / "attachments", kind="video")

        self.assertEqual(item.local_path, "")
        self.assertIn("expected video, got image", item.local_status)

    def test_localize_is_a_noop_without_an_assets_repo(self) -> None:
        fake = _FakeGh({})
        with tempfile.TemporaryDirectory() as tmp:
            dest_dir = Path(tmp) / "attachments"
            with _install_gh(fake):
                item = _localize_one(dest_dir, assets_repo="")

            self.assertEqual(item, MediaEvidence(kind="image", url=ASSET_URL))
            self.assertEqual(fake.calls, [])
            self.assertFalse(dest_dir.exists())


class TriageContextMediaRenderTest(unittest.TestCase):
    def test_context_renders_local_path_instead_of_a_fetch_invitation(self) -> None:
        context = _context_with(
            MediaEvidence(
                kind="image",
                url=ASSET_URL,
                description="闪退截图",
                local_path="/w/attachments/media-1.jpg",
            )
        )
        markdown = render_triage_context_markdown(context)

        self.assertIn("- Local file: /w/attachments/media-1.jpg", markdown)
        self.assertIn("do **not** fetch the `url`", markdown)

    def test_context_renders_a_failed_copy_explicitly(self) -> None:
        context = _context_with(
            MediaEvidence(
                kind="image",
                url=ASSET_URL,
                description="闪退截图",
                local_status="not fetched: gh exited 1",
            )
        )
        markdown = render_triage_context_markdown(context)

        self.assertIn("- Local copy: not fetched: gh exited 1", markdown)


def _context_with(*media: MediaEvidence):
    from bugpatrol.clients import GitHubIssue
    from bugpatrol.triage_context import TriageContext

    return TriageContext(
        issue=GitHubIssue(number=1, url="https://example.test/1", title="t", body="b"),
        comments=(),
        prd_hits=(),
        media=media,
    )
