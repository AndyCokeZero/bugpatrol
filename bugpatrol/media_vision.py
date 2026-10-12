"""Describe local image/video files with an OpenAI-compatible multimodal API."""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-vl-plus"
DEFAULT_QUESTION = (
    "详细描述这个媒体文件的内容。若画面出现悬浮的调试信息浮层（dev overlay，"
    "如 FPS、网络、内存、设备信息等实时数据），这是开发/测试环境正常组件，不是 bug，"
    "其数据仅作分析参考。"
)
DEFAULT_TIMEOUT_SECONDS = 300
CONFIG_PATHS = (
    Path("~/.bugpatrol/vision.json").expanduser(),
    Path("~/.lark-bug-watcher/vision.json").expanduser(),
)
ENV_REF = re.compile(r"^\{env:([A-Za-z_][A-Za-z0-9_]*)\}$")
IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".heic": "image/heic",
    ".heif": "image/heif",
}
VIDEO_MIME = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".m4v": "video/x-m4v",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}


def describe(path: Path, *, question: str = DEFAULT_QUESTION) -> str:
    base_url, api_key, model = load_config()
    if not api_key:
        raise RuntimeError("no API key; set BUGPATROL_VISION_API_KEY, BAILIAN_API_KEY, or apiKey in vision.json")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [_media_block(path), {"type": "text", "text": question}],
            }
        ],
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
        body = json.loads(response.read().decode())
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"no choices in response: {body!r}")
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"empty content in response: {body!r}")
    return content.strip()


def load_config() -> tuple[str, str, str]:
    raw: dict[str, object] = {}
    explicit_config = os.environ.get("BUGPATROL_VISION_CONFIG", "").strip()
    paths = (Path(explicit_config).expanduser(),) if explicit_config else CONFIG_PATHS
    for path in paths:
        try:
            loaded = json.loads(path.read_text())
        except FileNotFoundError:
            continue
        if isinstance(loaded, dict):
            raw = loaded
            break
    base_url = _str(raw.get("baseURL")) or os.environ.get("BUGPATROL_VISION_BASE_URL") or DEFAULT_BASE_URL
    model = _str(raw.get("model")) or os.environ.get("BUGPATROL_VISION_MODEL") or DEFAULT_MODEL
    api_key = (
        os.environ.get("BUGPATROL_VISION_API_KEY", "").strip()
        or _resolve_key(raw.get("apiKey"))
        or os.environ.get("BAILIAN_API_KEY", "").strip()
    )
    return base_url.rstrip("/"), api_key, model


def _media_block(path: Path) -> dict[str, object]:
    extension = path.suffix.lower()
    data = base64.b64encode(path.read_bytes()).decode()
    if extension in IMAGE_MIME:
        return {"type": "image_url", "image_url": {"url": f"data:{IMAGE_MIME[extension]};base64,{data}"}}
    if extension in VIDEO_MIME:
        return {"type": "video_url", "video_url": {"url": f"data:{VIDEO_MIME[extension]};base64,{data}"}}
    raise RuntimeError(f"unsupported media extension: {extension or '(none)'}")


def _resolve_key(raw: object) -> str:
    value = _str(raw)
    if not value:
        return ""
    match = ENV_REF.match(value)
    return os.environ.get(match.group(1), "").strip() if match else value


def _str(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args or args[0] in {"-h", "--help"}:
        print("usage: python -m bugpatrol.media_vision <image-or-video-path> [question]", file=sys.stderr)
        return 2
    path = Path(args[0]).expanduser()
    if not path.is_file():
        print(f"media_vision: file not found: {path}", file=sys.stderr)
        return 2
    question = " ".join(args[1:]).strip() or DEFAULT_QUESTION
    try:
        print(describe(path, question=question))
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:500]
        print(f"media_vision: HTTP {error.code}: {detail}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError, urllib.error.URLError) as error:
        print(f"media_vision: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
