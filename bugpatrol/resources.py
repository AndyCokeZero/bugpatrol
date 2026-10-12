"""Local resource materialization for intake attachments."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Protocol

from bugpatrol.intake import Attachment, IntakeRecord
from bugpatrol.lark import DownloadedLarkResource

LARK_RESOURCE_RE = re.compile(r"^lark://message/([^/]+)/([^/]+)/([^/]+)$")

# Git subcommands that talk to the remote, and therefore can fail on a transport
# blip that a bounded retry fixes. Local-only commands (add/commit) are excluded
# so a real failure there is never retried.
_NETWORK_GIT_SUBCOMMANDS = frozenset({"clone", "fetch", "pull", "push"})

# Transport-layer failures of the git remote helper (TLS/SSH/HTTP), e.g. the
# `LibreSSL SSL_connect: SSL_ERROR_SYSCALL` that broke an asset push mid-poll and
# stalled intake. These fail before or while carrying the request, so retrying is
# safe; auth/permission failures do not match and still fail immediately.
_TRANSIENT_GIT_RE = re.compile(
    r"SSL_ERROR_SYSCALL"
    r"|SSL_connect"
    r"|GnuTLS recv error"
    r"|Connection reset by peer"
    r"|Connection timed out"
    r"|Operation timed out"
    r"|Could not resolve host"
    r"|Failed to connect to"
    r"|Empty reply from server"
    r"|The requested URL returned error: 5\d\d"
    r"|RPC failed"
    r"|early EOF"
    r"|unexpected disconnect"
    r"|remote end hung up unexpectedly"
    r"|kex_exchange_identification"
    r"|Broken pipe",
    re.IGNORECASE,
)


def is_transient_git_error(stderr: str) -> bool:
    return bool(_TRANSIENT_GIT_RE.search(stderr))


class LarkResourceDownloader(Protocol):
    def download_message_resource(
        self,
        *,
        message_id: str,
        resource_key: str,
        resource_type: str = "",
    ) -> DownloadedLarkResource:
        """Download a Lark message resource."""


class ResourceStore(Protocol):
    def write(self, *, ref: LarkResourceRef, resource: DownloadedLarkResource) -> Path | str:
        """Persist a downloaded resource and return the URL/path to put in intake."""


class ResourceDescriber(Protocol):
    def describe(self, *, ref: LarkResourceRef, resource: DownloadedLarkResource) -> str:
        """Return a textual media description for triage."""


class ResourceRedactor(Protocol):
    def redact(self, *, ref: LarkResourceRef, resource: DownloadedLarkResource) -> DownloadedLarkResource:
        """Return a redacted resource before upload or vision."""


class ResourceTransformer(Protocol):
    def transform(self, *, ref: LarkResourceRef, resource: DownloadedLarkResource) -> DownloadedLarkResource:
        """Return a normalized resource before policy, upload, and vision."""


class VideoDurationProbe(Protocol):
    def duration_seconds(self, *, ref: LarkResourceRef, resource: DownloadedLarkResource) -> float:
        """Return video duration in seconds."""


@dataclass(frozen=True)
class ResourcePolicy:
    max_image_bytes: int = 0
    max_video_bytes: int = 0
    max_file_bytes: int = 0
    max_video_duration_seconds: float = 0.0
    video_duration_probe: VideoDurationProbe | None = None

    def rejection_reason(self, *, ref: LarkResourceRef, resource: DownloadedLarkResource) -> str:
        limit = self._limit_for(ref=ref, resource=resource)
        if limit > 0 and len(resource.content) > limit:
            return f"resource skipped: {ref.kind} is {len(resource.content)} bytes, limit is {limit} bytes"
        if self.max_video_duration_seconds > 0 and _is_video_resource(ref=ref, resource=resource):
            try:
                duration = self._duration_probe().duration_seconds(ref=ref, resource=resource)
            except RuntimeError as exc:
                return f"resource skipped: video duration unavailable: {exc}"
            if duration > self.max_video_duration_seconds:
                return (
                    f"resource skipped: video duration is {duration:.1f}s, "
                    f"limit is {self.max_video_duration_seconds:.1f}s"
                )
        return ""

    def _limit_for(self, *, ref: LarkResourceRef, resource: DownloadedLarkResource) -> int:
        content_type = resource.content_type.split(";", 1)[0].strip().lower()
        if ref.kind == "image" or content_type.startswith("image/"):
            return self.max_image_bytes
        if ref.kind in {"video", "media"} or content_type.startswith("video/"):
            return self.max_video_bytes
        return self.max_file_bytes

    def _duration_probe(self) -> VideoDurationProbe:
        return self.video_duration_probe or FfprobeVideoDurationProbe()


class CompositeResourceTransformer:
    def __init__(self, transformers: tuple[ResourceTransformer, ...]) -> None:
        self._transformers = transformers

    def transform(self, *, ref: LarkResourceRef, resource: DownloadedLarkResource) -> DownloadedLarkResource:
        transformed = resource
        for transformer in self._transformers:
            transformed = transformer.transform(ref=ref, resource=transformed)
        return transformed


@dataclass(frozen=True)
class LarkResourceRef:
    message_id: str
    kind: str
    resource_key: str


class LocalResourceStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def write(self, *, ref: LarkResourceRef, resource: DownloadedLarkResource) -> Path:
        directory = self._root / _safe_segment(ref.message_id)
        directory.mkdir(parents=True, exist_ok=True)
        filename = _resource_filename(resource=resource, fallback=ref.resource_key)
        path = directory / filename
        path.write_bytes(resource.content)
        return path


class GitHubAssetRepoStore:
    def __init__(
        self,
        *,
        repo: str,
        checkout_path: Path,
        base_path: str = ".github/issue-assets",
        branch: str = "main",
        remote_url: str = "",
        git: str = "git",
        transient_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._repo = repo
        self._checkout_path = checkout_path.expanduser()
        self._base_path = base_path.strip("/")
        self._branch = branch
        self._remote_url = remote_url
        self._git = git
        self._transient_retries = transient_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._sleep = sleep
        # Parallel topic workers share one checkout; git mutations must not interleave.
        self._lock = threading.Lock()

    def write(self, *, ref: LarkResourceRef, resource: DownloadedLarkResource) -> str:
        with self._lock:
            return self._write_locked(ref=ref, resource=resource)

    def _write_locked(self, *, ref: LarkResourceRef, resource: DownloadedLarkResource) -> str:
        self._ensure_checkout()
        rel_path = Path(self._base_path) / _safe_segment(ref.message_id) / _resource_filename(
            resource=resource,
            fallback=ref.resource_key,
        )
        path = self._checkout_path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resource.content)
        self._run(["-C", str(self._checkout_path), "add", str(rel_path)])
        self._run(
            [
                "-C",
                str(self._checkout_path),
                "commit",
                "--no-verify",
                "-m",
                f"add: bug attachment {ref.message_id}",
            ],
            allow_no_changes=True,
        )
        self._run(["-C", str(self._checkout_path), "push", "--no-verify", self._existing_remote(), self._branch])
        return f"https://github.com/{self._repo}/raw/{self._branch}/{rel_path.as_posix()}"

    def _ensure_checkout(self) -> None:
        if (self._checkout_path / ".git").exists():
            self._run(["-C", str(self._checkout_path), "pull", "--quiet", self._existing_remote(), self._branch])
            return
        self._checkout_path.parent.mkdir(parents=True, exist_ok=True)
        self._run(
            [
                "clone",
                "--branch",
                self._branch,
                self._clone_remote(),
                str(self._checkout_path),
            ]
        )

    def _existing_remote(self) -> str:
        return self._remote_url or "origin"

    def _clone_remote(self) -> str:
        return self._remote_url or f"git@github.com:{self._repo}.git"

    def _run(self, args: list[str], *, allow_no_changes: bool = False) -> None:
        retries = self._transient_retries if self._is_network_command(args) else 1
        for attempt in range(1, retries + 1):
            completed = subprocess.run(
                [self._git, *args],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode == 0:
                return
            combined = f"{completed.stdout}\n{completed.stderr}"
            if allow_no_changes and "nothing to commit" in combined:
                return
            stderr = completed.stderr.strip()
            if attempt < retries and is_transient_git_error(combined):
                print(
                    f"git {' '.join(args)} hit a transient failure "
                    f"({attempt}/{retries}), retrying: {stderr}",
                    file=sys.stderr,
                )
                self._sleep(self._retry_backoff_seconds * attempt)
                continue
            raise RuntimeError(f"git {' '.join(args)} failed: {stderr}")

    def _is_network_command(self, args: list[str]) -> bool:
        index = 0
        while index < len(args) and args[index] == "-C":
            index += 2
        return index < len(args) and args[index] in _NETWORK_GIT_SUBCOMMANDS


# A description that failed comes back as text rather than an exception, so the
# caller can keep the run going with a placeholder. Anything that has to *tell a
# description from a failure* — bugpatrol/doctor.py's live probe — matches this
# prefix instead of re-deriving it.
DESCRIPTION_UNAVAILABLE_PREFIX = "vision description unavailable:"


class CommandResourceDescriber:
    def __init__(
        self,
        *,
        command: tuple[str, ...],
        timeout_seconds: int = 300,
        temp_dir: Path | None = None,
        retries: int = 0,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self._command = command
        self._timeout_seconds = timeout_seconds
        self._temp_dir = temp_dir
        self._retries = max(0, retries)
        self._retry_backoff_seconds = max(0.0, retry_backoff_seconds)

    def describe(self, *, ref: LarkResourceRef, resource: DownloadedLarkResource) -> str:
        if not self._command:
            return ""
        temp_root = self._temp_dir
        if temp_root is not None:
            temp_root.mkdir(parents=True, exist_ok=True)
        filename = _resource_filename(resource=resource, fallback=ref.resource_key)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            path = Path(tmp) / filename
            path.write_bytes(resource.content)
            command = tuple(
                _format_command_part(part, path=path, ref=ref, resource=resource)
                for part in self._command
            )
            completed = None
            for attempt in range(self._retries + 1):
                try:
                    completed = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=self._timeout_seconds,
                    )
                except subprocess.TimeoutExpired:
                    if attempt >= self._retries:
                        return (
                            f"{DESCRIPTION_UNAVAILABLE_PREFIX} "
                            f"timed out after {self._timeout_seconds}s"
                        )
                    time.sleep(self._retry_backoff_seconds)
                    continue
                if completed.returncode == 0 and completed.stdout.strip():
                    break
                if attempt < self._retries:
                    time.sleep(self._retry_backoff_seconds)
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip()
        detail = (completed.stderr or completed.stdout or f"exit {completed.returncode}").strip()
        return f"{DESCRIPTION_UNAVAILABLE_PREFIX} {detail[:300]}"


class CommandResourceRedactor:
    def __init__(
        self,
        *,
        command: tuple[str, ...],
        timeout_seconds: int = 300,
        temp_dir: Path | None = None,
    ) -> None:
        self._command = command
        self._timeout_seconds = timeout_seconds
        self._temp_dir = temp_dir

    def redact(self, *, ref: LarkResourceRef, resource: DownloadedLarkResource) -> DownloadedLarkResource:
        if not self._command:
            return resource
        temp_root = self._temp_dir
        if temp_root is not None:
            temp_root.mkdir(parents=True, exist_ok=True)
        filename = _resource_filename(resource=resource, fallback=ref.resource_key)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            path = Path(tmp) / filename
            path.write_bytes(resource.content)
            command = tuple(
                _format_command_part(part, path=path, ref=ref, resource=resource)
                for part in self._command
            )
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=self._timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"redaction command timed out after {self._timeout_seconds}s") from exc
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or f"exit {completed.returncode}").strip()
                raise RuntimeError(f"redaction command failed: {detail[:300]}")
            redacted = path.read_bytes()
        return DownloadedLarkResource(
            content=redacted,
            content_type=resource.content_type,
            filename=resource.filename,
        )


class FfprobeVideoDurationProbe:
    DEFAULT_COMMAND = (
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        "{path}",
    )

    def __init__(
        self,
        *,
        command: tuple[str, ...] = DEFAULT_COMMAND,
        timeout_seconds: int = 30,
        temp_dir: Path | None = None,
    ) -> None:
        self._command = command
        self._timeout_seconds = timeout_seconds
        self._temp_dir = temp_dir

    def duration_seconds(self, *, ref: LarkResourceRef, resource: DownloadedLarkResource) -> float:
        temp_root = self._temp_dir
        if temp_root is not None:
            temp_root.mkdir(parents=True, exist_ok=True)
        filename = _resource_filename(resource=resource, fallback=ref.resource_key)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            path = Path(tmp) / filename
            path.write_bytes(resource.content)
            command = tuple(
                _format_command_part(part, path=path, ref=ref, resource=resource)
                for part in self._command
            )
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=self._timeout_seconds,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(f"video probe command not found: {command[0]}") from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"video probe timed out after {self._timeout_seconds}s") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or f"exit {completed.returncode}").strip()
            raise RuntimeError(f"video probe failed: {detail[:300]}")
        match = re.search(r"[-+]?\d+(?:\.\d+)?", completed.stdout)
        if match is None:
            raise RuntimeError("video probe returned no duration")
        return float(match.group(0))


class CommandVideoFrameExtractor:
    def __init__(
        self,
        *,
        command: tuple[str, ...],
        timeout_seconds: int = 300,
        temp_dir: Path | None = None,
        min_duration_seconds: float = 0.0,
        duration_probe: VideoDurationProbe | None = None,
    ) -> None:
        self._command = command
        self._timeout_seconds = timeout_seconds
        self._temp_dir = temp_dir
        self._min_duration_seconds = max(0.0, min_duration_seconds)
        self._duration_probe = duration_probe

    def transform(self, *, ref: LarkResourceRef, resource: DownloadedLarkResource) -> DownloadedLarkResource:
        if not self._command or not _is_video_resource(ref=ref, resource=resource):
            return resource
        if self._min_duration_seconds > 0 and not self._meets_min_duration(ref=ref, resource=resource):
            return resource

        temp_root = self._temp_dir
        if temp_root is not None:
            temp_root.mkdir(parents=True, exist_ok=True)
        filename = _resource_filename(resource=resource, fallback=ref.resource_key)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            path = Path(tmp) / filename
            output_path = Path(tmp) / _frame_filename(filename=filename)
            path.write_bytes(resource.content)
            command = tuple(
                _format_command_part(part, path=path, output_path=output_path, ref=ref, resource=resource)
                for part in self._command
            )
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=self._timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"video frame command timed out after {self._timeout_seconds}s") from exc
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or f"exit {completed.returncode}").strip()
                raise RuntimeError(f"video frame command failed: {detail[:300]}")
            if not output_path.is_file():
                raise RuntimeError(f"video frame command did not write output: {output_path}")
            frames = output_path.read_bytes()
        return DownloadedLarkResource(
            content=frames,
            content_type="image/png",
            filename=_frame_filename(filename=filename),
        )

    def _meets_min_duration(self, *, ref: LarkResourceRef, resource: DownloadedLarkResource) -> bool:
        probe = self._duration_probe or FfprobeVideoDurationProbe(temp_dir=self._temp_dir)
        try:
            return probe.duration_seconds(ref=ref, resource=resource) >= self._min_duration_seconds
        except RuntimeError:
            return True


class ImageResourceResizer:
    def __init__(
        self,
        *,
        max_width: int = 0,
        max_height: int = 0,
        quality: int = 85,
        convert_to_jpeg: bool = False,
    ) -> None:
        self._max_width = max(0, max_width)
        self._max_height = max(0, max_height)
        self._quality = min(100, max(1, quality))
        self._convert_to_jpeg = convert_to_jpeg

    def transform(self, *, ref: LarkResourceRef, resource: DownloadedLarkResource) -> DownloadedLarkResource:
        if not self._max_width and not self._max_height and not self._convert_to_jpeg:
            return resource
        if not _is_image_resource(ref=ref, resource=resource):
            return resource

        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - dependency is declared, but keep runner error explicit.
            raise RuntimeError("image resizing requires Pillow to be installed") from exc

        with Image.open(BytesIO(resource.content)) as image:
            source_format = image.format or _pillow_format_for_resource(resource)
            target = _scaled_image_size(
                width=image.width,
                height=image.height,
                max_width=self._max_width,
                max_height=self._max_height,
            )
            convert = self._convert_to_jpeg and source_format != "JPEG"
            if target == (image.width, image.height) and not convert:
                return resource
            format_name = "JPEG" if self._convert_to_jpeg else source_format
            output_image = image
            if target != (image.width, image.height):
                output_image = output_image.resize(target, Image.Resampling.LANCZOS)
            if format_name == "JPEG" and output_image.mode not in {"RGB", "L"}:
                # Flatten transparency onto white instead of the default black.
                rgba = output_image.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.getchannel("A"))
                output_image = background
            output = BytesIO()
            output_image.save(output, format=format_name, quality=self._quality, optimize=True)
        content_type = "image/jpeg" if format_name == "JPEG" else resource.content_type
        filename = _jpeg_filename(resource.filename) if format_name == "JPEG" else resource.filename
        return DownloadedLarkResource(
            content=output.getvalue(),
            content_type=content_type,
            filename=filename,
        )


def _jpeg_filename(filename: str) -> str:
    if not filename:
        return filename
    stem, _, ext = filename.rpartition(".")
    if not stem:
        return f"{filename}.jpg"
    if ext.lower() in {"jpg", "jpeg"}:
        return filename
    return f"{stem}.jpg"


def materialize_lark_attachments(
    *,
    record: IntakeRecord,
    lark: LarkResourceDownloader,
    store: ResourceStore,
    describer: ResourceDescriber | None = None,
    policy: ResourcePolicy | None = None,
    redactor: ResourceRedactor | None = None,
    transformer: ResourceTransformer | None = None,
) -> IntakeRecord:
    attachments = tuple(
        materialize_attachment(
            attachment=attachment,
            lark=lark,
            store=store,
            describer=describer,
            policy=policy,
            redactor=redactor,
            transformer=transformer,
        )
        for attachment in record.attachments
    )
    return replace(record, attachments=attachments)


def materialize_attachment(
    *,
    attachment: Attachment,
    lark: LarkResourceDownloader,
    store: ResourceStore,
    describer: ResourceDescriber | None = None,
    policy: ResourcePolicy | None = None,
    redactor: ResourceRedactor | None = None,
    transformer: ResourceTransformer | None = None,
) -> Attachment:
    ref = parse_lark_resource_url(attachment.url)
    if ref is None:
        return attachment
    resource = lark.download_message_resource(
        message_id=ref.message_id,
        resource_key=ref.resource_key,
        resource_type=_download_resource_type(ref.kind),
    )
    if redactor is not None:
        resource = redactor.redact(ref=ref, resource=resource)
    if transformer is not None:
        resource = transformer.transform(ref=ref, resource=resource)
    if policy is not None:
        rejection = policy.rejection_reason(ref=ref, resource=resource)
        if rejection:
            return Attachment(kind=attachment.kind, url=attachment.url, description=rejection)
    path = store.write(ref=ref, resource=resource)
    description = attachment.description or resource.filename
    if describer is not None:
        generated = describer.describe(ref=ref, resource=resource).strip()
        if generated:
            description = generated
    return Attachment(
        kind=attachment.kind,
        url=str(path),
        description=description,
    )


def parse_lark_resource_url(url: str) -> LarkResourceRef | None:
    match = LARK_RESOURCE_RE.match(url)
    if not match:
        return None
    return LarkResourceRef(
        message_id=match.group(1),
        kind=match.group(2),
        resource_key=match.group(3),
    )


def _safe_segment(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "resource"


def _resource_filename(*, resource: DownloadedLarkResource, fallback: str) -> str:
    filename = _safe_segment(resource.filename or fallback)
    if Path(filename).suffix:
        return filename
    extension = _extension_for_content_type(resource.content_type)
    return f"{filename}{extension}" if extension else filename


def _frame_filename(*, filename: str) -> str:
    stem = Path(filename).stem or "video"
    return f"{_safe_segment(stem)}.frames.png"


def _extension_for_content_type(content_type: str) -> str:
    media_type = content_type.split(";", 1)[0].strip().lower()
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
    }.get(media_type, "")


def _is_image_resource(*, ref: LarkResourceRef, resource: DownloadedLarkResource) -> bool:
    content_type = resource.content_type.split(";", 1)[0].strip().lower()
    return ref.kind == "image" or content_type.startswith("image/")


def _is_video_resource(*, ref: LarkResourceRef, resource: DownloadedLarkResource) -> bool:
    content_type = resource.content_type.split(";", 1)[0].strip().lower()
    return ref.kind in {"video", "media"} or content_type.startswith("video/")


def _scaled_image_size(
    *,
    width: int,
    height: int,
    max_width: int,
    max_height: int,
) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        return width, height
    width_ratio = max_width / width if max_width > 0 else 1.0
    height_ratio = max_height / height if max_height > 0 else 1.0
    ratio = min(1.0, width_ratio, height_ratio)
    return max(1, round(width * ratio)), max(1, round(height * ratio))


def _pillow_format_for_resource(resource: DownloadedLarkResource) -> str:
    media_type = resource.content_type.split(";", 1)[0].strip().lower()
    if media_type in {"image/jpeg", "image/jpg"}:
        return "JPEG"
    if media_type == "image/webp":
        return "WEBP"
    if media_type == "image/gif":
        return "GIF"
    return "PNG"


def _download_resource_type(kind: str) -> str:
    if kind == "image":
        return "image"
    if kind in {"file", "video", "media"}:
        return "file"
    return kind


def _format_command_part(
    part: str,
    *,
    path: Path,
    output_path: Path | None = None,
    ref: LarkResourceRef,
    resource: DownloadedLarkResource,
) -> str:
    formatted = part.format(
        python=sys.executable,
        path=str(path),
        output_path=str(output_path or ""),
        kind=ref.kind,
        message_id=ref.message_id,
        resource_key=ref.resource_key,
        filename=resource.filename,
        content_type=resource.content_type,
    )
    return Path(formatted).expanduser().as_posix() if formatted.startswith("~") else formatted
