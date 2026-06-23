"""
scraper.py — Sentiment Radar
------------------------------
Reddit      → Official OAuth API (REDDIT_CLIENT_ID + SECRET in .env)
              Fallback → Pullpush.io archive (no key needed)
HackerNews  → Algolia public API (no key)
Dev.to      → Public REST API (no key)
YouTube     → Data API v3 (YOUTUBE_API_KEY in .env, never from UI)
Bluesky     → Public ATP API (no key, replaces dead Twitter/X)
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from app.tweetclaw_import import load_tweetclaw_export

logger = logging.getLogger(__name__)
NUM_COMMENTS = 50

_UA = os.getenv("REDDIT_USER_AGENT", "SentimentRadar/1.0")
_BASE_H = {"User-Agent": _UA}


# ── Load .env automatically if present ───────────────────────
def _load_env():
    """Load .env from project root without requiring python-dotenv."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:   # don't override real env vars
                os.environ[key] = val

_load_env()


# ── Generic HTTP helpers ──────────────────────────────────────

def _get(url: str, headers: Optional[dict] = None, timeout: int = 12) -> dict:
    req = urllib.request.Request(url, headers={**_BASE_H, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _get_retry(url: str, headers: Optional[dict] = None,
               retries: int = 3, backoff: float = 2.0) -> dict:
    last = None
    for i in range(retries):
        try:
            return _get(url, headers)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                time.sleep(backoff ** i); last = e
            else:
                raise
        except (urllib.error.URLError, TimeoutError) as e:
            time.sleep(backoff ** i); last = e
    raise last


def _clean(text: str, max_len: int = 400) -> str:
    text = re.sub(r"<[^>]+>", "", str(text))
    for h, r in [("&#x27;","'"),("&amp;","&"),("&gt;",">"),
                 ("&lt;","<"),("&quot;",'"'),("&#39;","'")]:
        text = text.replace(h, r)
    text = re.sub(r"\s+", " ", text).strip()
    return (text[:max_len] + "…") if len(text) > max_len else text


# ── Reddit OAuth token ────────────────────────────────────────

_r_token: Optional[str] = None
_r_expiry: float = 0.0


def _reddit_token() -> Optional[str]:
    global _r_token, _r_expiry
    cid = os.getenv("REDDIT_CLIENT_ID", "").strip()
    cs  = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
    if not cid or not cs:
        return None
    if _r_token and time.time() < _r_expiry - 60:
        return _r_token
    creds = base64.b64encode(f"{cid}:{cs}".encode()).decode()
    req = urllib.request.Request(
        "https://www.reddit.com/api/v1/access_token",
        data=urllib.parse.urlencode({"grant_type": "client_credentials"}).encode(),
        method="POST",
        headers={"Authorization": f"Basic {creds}", "User-Agent": _UA,
                 "Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.loads(r.read().decode())
            _r_token  = d["access_token"]
            _r_expiry = time.time() + d.get("expires_in", 3600)
            logger.info("[Reddit] OAuth token obtained.")
            return _r_token
    except Exception as e:
        logger.error("[Reddit] OAuth failed: %s", e)
        return None


def _reddit_oauth(topic: str, target: int) -> list:
    """Fetch via official Reddit OAuth API."""
    token = _reddit_token()
    if not token:
        return []
    ah = {"Authorization": f"bearer {token}", "User-Agent": _UA}
    try:
        data  = _get_retry(
            "https://oauth.reddit.com/search?" + urllib.parse.urlencode(
                {"q": topic, "sort": "top", "t": "month",
                 "limit": 10, "type": "link"}), headers=ah)
        posts = data.get("data", {}).get("children", [])
    except Exception as e:
        logger.warning("[Reddit OAuth] Search failed: %s", e)
        return []

    comments = []
    for post in posts:
        if len(comments) >= target: break
        pd = post.get("data", {})
        sr, pid = pd.get("subreddit",""), pd.get("id","")
        if not sr or not pid: continue
        try:
            cd = _get_retry(
                f"https://oauth.reddit.com/r/{sr}/comments/{pid}"
                f"?limit=50&depth=1", headers=ah)
            for c in cd[1].get("data", {}).get("children", []):
                body = c.get("data", {}).get("body", "").strip()
                if body and body not in ("[deleted]","[removed]") and len(body) > 10:
                    comments.append(_clean(body))
                if len(comments) >= target: break
        except Exception as e:
            logger.warning("[Reddit OAuth] Post %s failed: %s", pid, e)
        time.sleep(0.5)
    return comments


def _reddit_pullpush(topic: str, target: int) -> list:
    """
    Fallback: Pullpush.io — free Reddit archive API, no auth needed.
    Mirrors Reddit comments with a slight delay but very reliable.
    """
    comments = []
    try:
        url = (
            "https://api.pullpush.io/reddit/search/comment/?" +
            urllib.parse.urlencode({
                "q": topic, "size": target, "sort": "desc",
                "sort_type": "score",
            })
        )
        data = _get_retry(url)
        for item in data.get("data", []):
            body = item.get("body", "").strip()
            if body and body not in ("[deleted]","[removed]") and len(body) > 10:
                comments.append(_clean(body))
            if len(comments) >= target:
                break
    except Exception as e:
        logger.warning("[Reddit Pullpush] Failed: %s", e)
    return comments


def scrape_reddit(topic: str, target: int = NUM_COMMENTS) -> list:
    """Try OAuth first, fall back to Pullpush archive."""
    # Try official API first if credentials exist
    comments = _reddit_oauth(topic, target)
    if comments:
        logger.info("[Reddit] %d comments via OAuth.", len(comments))
        return comments[:target]

    # Fallback — no credentials needed
    logger.info("[Reddit] No OAuth credentials, using Pullpush fallback...")
    comments = _reddit_pullpush(topic, target)
    logger.info("[Reddit] %d comments via Pullpush.", len(comments))
    return comments[:target]


# ── HackerNews ────────────────────────────────────────────────

def scrape_hackernews(topic: str, target: int = NUM_COMMENTS) -> list:
    comments, page = [], 0
    while len(comments) < target:
        try:
            data = _get_retry(
                "https://hn.algolia.com/api/v1/search?" + urllib.parse.urlencode(
                    {"query": topic, "tags": "comment",
                     "hitsPerPage": 50, "page": page}))
        except Exception as e:
            logger.error("[HN] Failed: %s", e); break
        hits = data.get("hits", [])
        if not hits: break
        for h in hits:
            t = _clean(h.get("comment_text") or "")
            if t and len(t) > 10:
                comments.append(t)
            if len(comments) >= target: break
        page += 1
        if page >= data.get("nbPages", 1): break
        time.sleep(0.3)
    logger.info("[HackerNews] %d comments fetched.", len(comments))
    return comments[:target]


# ── Dev.to ────────────────────────────────────────────────────

def scrape_devto(topic: str, target: int = NUM_COMMENTS) -> list:
    comments = []
    try:
        articles = _get_retry(
            "https://dev.to/api/articles?" + urllib.parse.urlencode(
                {"q": topic, "per_page": 12}))
    except Exception as e:
        logger.error("[Dev.to] Failed: %s", e); return []

    def _extract(nodes):
        for n in nodes:
            b = _clean(n.get("body_html", ""))
            if b and len(b) > 10: yield b
            yield from _extract(n.get("children", []))

    for article in articles:
        if len(comments) >= target: break
        aid = article.get("id")
        if not aid: continue
        try:
            raw = _get_retry(f"https://dev.to/api/comments?a_id={aid}")
            comments.extend(list(_extract(raw)))
        except Exception as e:
            logger.warning("[Dev.to] Article %s failed: %s", aid, e)
        time.sleep(0.3)
    logger.info("[Dev.to] %d comments fetched.", len(comments))
    return comments[:target]


# ── YouTube ───────────────────────────────────────────────────

def scrape_youtube(topic: str, target: int = NUM_COMMENTS) -> list:
    """
    Key read from YOUTUBE_API_KEY env var (loaded from .env automatically).
    Never passed from the UI. Returns [] silently if key not set.
    """
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        logger.warning("[YouTube] YOUTUBE_API_KEY not set in .env — skipping.")
        return []
    comments = []
    try:
        search = _get_retry(
            "https://www.googleapis.com/youtube/v3/search?" +
            urllib.parse.urlencode({
                "part": "id", "q": topic, "type": "video",
                "maxResults": 5, "order": "relevance", "key": api_key,
            }))
    except Exception as e:
        logger.error("[YouTube] Search failed: %s", e)
        return []

    video_ids = [
        i["id"]["videoId"] for i in search.get("items", [])
        if i.get("id", {}).get("videoId")
    ]
    for vid in video_ids:
        if len(comments) >= target: break
        try:
            ct = _get_retry(
                "https://www.googleapis.com/youtube/v3/commentThreads?" +
                urllib.parse.urlencode({
                    "part": "snippet", "videoId": vid, "maxResults": 50,
                    "order": "relevance", "textFormat": "plainText",
                    "key": api_key,
                }))
            for item in ct.get("items", []):
                text = (item.get("snippet", {})
                            .get("topLevelComment", {})
                            .get("snippet", {})
                            .get("textDisplay", "").strip())
                if text and len(text) > 5:
                    comments.append(_clean(text))
                if len(comments) >= target: break
        except Exception as e:
            logger.warning("[YouTube] Video %s failed: %s", vid, e)
    logger.info("[YouTube] %d comments fetched.", len(comments))
    return comments[:target]


# ── Bluesky (replaces dead Twitter/X) ────────────────────────

def scrape_bluesky(topic: str, target: int = NUM_COMMENTS) -> list:
    """
    Bluesky public search API — completely free, no key needed.
    Returns posts matching the topic from the AT Protocol network.
    """
    comments = []
    try:
        url = (
            "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts?" +
            urllib.parse.urlencode({
                "q": topic, "limit": min(target, 100), "sort": "top",
            })
        )
        data = _get_retry(url, headers={
            **_BASE_H, "Accept": "application/json"
        })
        for post in data.get("posts", []):
            text = post.get("record", {}).get("text", "").strip()
            if text and len(text) > 5:
                comments.append(_clean(text))
            if len(comments) >= target:
                break
    except Exception as e:
        logger.error("[Bluesky] Failed: %s", e)

    logger.info("[Bluesky] %d posts fetched.", len(comments))
    return comments[:target]


# ── Orchestrator ──────────────────────────────────────────────

def scrape_all(
    topic: str,
    comments_per_platform: int = NUM_COMMENTS,
    youtube_api_key: Optional[str] = None,    # kept for API compat, ignored
    include_twitter: bool = False,             # now controls Bluesky
    tweetclaw_export_path: Optional[str] = None,
) -> dict:
    """
    Scrape all platforms.
    - Reddit: OAuth if keys set in .env, else Pullpush fallback (always works)
    - YouTube: auto-enabled when YOUTUBE_API_KEY is in .env
    - Bluesky: enabled when include_twitter=True (replaces dead Twitter/X)
    """
    yt_key = os.getenv("YOUTUBE_API_KEY", "").strip()

    scrapers: dict = {
        "Reddit":     lambda: scrape_reddit(topic, comments_per_platform),
        "HackerNews": lambda: scrape_hackernews(topic, comments_per_platform),
        "Dev.to":     lambda: scrape_devto(topic, comments_per_platform),
    }
    if yt_key:
        scrapers["YouTube"] = lambda: scrape_youtube(topic, comments_per_platform)
    if include_twitter:
        scrapers["Bluesky"] = lambda: scrape_bluesky(topic, comments_per_platform)
    if tweetclaw_export_path:
        scrapers["TweetClaw"] = lambda: load_tweetclaw_export(
            tweetclaw_export_path,
            comments_per_platform,
        )

    results = {}
    for platform, fn in scrapers.items():
        logger.info("[%s] Scraping '%s' ...", platform, topic)
        try:
            results[platform] = fn()
        except Exception as e:
            logger.error("[%s] Unhandled: %s", platform, e)
            results[platform] = []

    total = sum(len(v) for v in results.values())
    logger.info("Done. Total: %d comments", total)
    return results
