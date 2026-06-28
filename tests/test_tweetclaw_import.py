"""Tests for TweetClaw export loading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.tweetclaw_import import extract_tweetclaw_texts, load_tweetclaw_export


class TestTweetClawImport(unittest.TestCase):
    def test_jsonl_export_extracts_text_fields(self) -> None:
        content = "\n".join(
            [
                json.dumps({"text": "First post about a launch"}),
                json.dumps({"full_text": "Second post &amp; follow-up"}),
            ],
        )

        comments = extract_tweetclaw_texts(content, suffix=".jsonl", target=5)

        self.assertEqual(
            comments,
            ["First post about a launch", "Second post & follow-up"],
        )

    def test_wrapped_json_export_respects_target(self) -> None:
        content = json.dumps(
            {
                "results": [
                    {"tweet": "Keep this one"},
                    {"tweet": "Skip due to target"},
                ],
            },
        )

        comments = extract_tweetclaw_texts(content, suffix=".json", target=1)

        self.assertEqual(comments, ["Keep this one"])

    def test_csv_export_loads_from_file(self) -> None:
        path = self._write_file("text,created_at\nCSV imported comment,2026-06-20\n", ".csv")

        comments = load_tweetclaw_export(path)

        self.assertEqual(comments, ["CSV imported comment"])

    def _write_file(self, content: str, suffix: str) -> Path:
        handle = tempfile.NamedTemporaryFile("w", delete=False, suffix=suffix, encoding="utf-8")
        with handle:
            handle.write(content)
        return Path(handle.name)


if __name__ == "__main__":
    unittest.main()
