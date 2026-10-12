from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from bugpatrol.media_vision import describe, load_config


class MediaVisionTest(unittest.TestCase):
    def test_load_config_uses_env_key_and_default_model(self) -> None:
        with patch.dict(os.environ, {"BUGPATROL_VISION_API_KEY": "key"}, clear=True):
            base_url, api_key, model = load_config()

        self.assertEqual(base_url, "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(api_key, "key")
        self.assertEqual(model, "qwen3-vl-plus")

    def test_describe_sends_image_data_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "bug.png"
            image.write_bytes(b"png-bytes")
            with patch.dict(os.environ, {"BUGPATROL_VISION_API_KEY": "key"}, clear=True):
                with patch("urllib.request.urlopen") as urlopen:
                    response = MagicMock()
                    response.__enter__.return_value.read.return_value = json.dumps(
                        {"choices": [{"message": {"content": "red screen"}}]}
                    ).encode()
                    urlopen.return_value = response

                    text = describe(image, question="describe")

        self.assertEqual(text, "red screen")
        request = urlopen.call_args.args[0]
        body = json.loads(request.data.decode())
        content = body["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertIn("data:image/png;base64", content[0]["image_url"]["url"])
        self.assertEqual(content[1]["text"], "describe")


if __name__ == "__main__":
    unittest.main()
