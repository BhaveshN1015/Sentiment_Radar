"""Load saved TweetClaw exports as Sentiment Radar comment text."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

TEXT_FIELDS = ("text", "tweet_text", "tweet", "full_text", "content", "body")


def load_tweetclaw_export(export_path: str | Path, target: int = 50) -> list[str]:
    """Return up to target comment texts from a TweetClaw JSON, JSONL, NDJSON, or CSV export."""
    path = Path(export_path)
    content = path.read_text(encoding="utf-8-sig").strip()
    if not content:
        return []
    return extract_tweetclaw_texts(content, suffix=path.suffix.lower(), target=target)


def extract_tweetclaw_texts(content: str, suffix: str = "", target: int = 50) -> list[str]:
    """Extract clean text values from TweetClaw export content."""
    rows = _parse_rows(content.strip(), suffix)
    comments: list[str] = []
    for row in rows:
        text = _first_text(row)
        if text:
            comments.append(_clean(text))
        if len(comments) >= target:
            break
    return comments


def _parse_rows(content: str, suffix: str) -> list[dict[str, Any]]:
    if not content:
        return []
    if suffix == ".csv":
        return _read_csv_rows(content)
    if suffix in {".jsonl", ".ndjson"}:
        return _read_json_lines(content)
    if content[0] in "[{":
        payload = json.loads(content)
        return _rows_from_payload(payload)
    return _read_json_lines(content)


def _read_csv_rows(content: str) -> list[dict[str, Any]]:
    first_line = content.splitlines()[0]
    delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
    return [dict(row) for row in csv.DictReader(content.splitlines(), delimiter=delimiter)]


def _read_json_lines(content: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("results", "tweets", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [payload]
    return []


def _first_text(row: dict[str, Any]) -> str:
    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _clean(text: str, max_length: int = 400) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    for source, replacement in (
        ("&#x27;", "'"),
        ("&amp;", "&"),
        ("&gt;", ">"),
        ("&lt;", "<"),
        ("&quot;", '"'),
        ("&#39;", "'"),
    ):
        text = text.replace(source, replacement)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_length] + "..." if len(text) > max_length else text
