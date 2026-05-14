from __future__ import annotations

import hashlib
import html
import re
from datetime import datetime, timezone
from urllib.parse import urldefrag, urlparse, urlunparse

import feedparser
import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..models import Item, SourceSpec


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=False,
)
def _http_get_bytes(url: str) -> bytes:
    """Fetch a feed body via httpx so we get retry control over feedparser."""
    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        resp = client.get(url, headers={"User-Agent": "dailydigest/0.1"})
        resp.raise_for_status()
        return resp.content


_HTML_TAG = re.compile(r"<[^>]+>")
_TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content"}


def _strip_html(s: str) -> str:
    return html.unescape(_HTML_TAG.sub("", s or "")).strip()


def canonicalize_url(url: str) -> str:
    if not url:
        return url
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    parsed = parsed._replace(scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower())
    # drop tracking query params
    if parsed.query:
        kept = [
            kv
            for kv in parsed.query.split("&")
            if kv and kv.split("=")[0].lower() not in _TRACKING_PARAMS
        ]
        parsed = parsed._replace(query="&".join(kept))
    return urlunparse(parsed).rstrip("/")


def _parsed_dt(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        val = entry.get(key)
        if val:
            try:
                return datetime(*val[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return None


def _external_id(entry, url: str) -> str:
    raw = entry.get("id") or entry.get("guid") or url
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _extract_abstract(entry) -> str:
    """Extract abstract text, preferring content:encoded (CDATA) over summary.

    Nature, Cell, ACS, and other publishers put full abstracts in
    ``content[0].value`` (the ``content:encoded`` element) while ``summary``
    contains only a short teaser or is empty.  Without this fallback, items
    from top journals embed only the title, which badly hurts ranking quality.
    """
    content = entry.get("content")
    if content and isinstance(content, list):
        value = content[0].get("value") if content[0] else None
        if value:
            return _strip_html(value)
    return _strip_html(entry.get("summary", entry.get("description", "")))


class RSSSource:
    def fetch(self, spec: SourceSpec) -> list[Item]:
        if not spec.url:
            return []
        try:
            content = _http_get_bytes(spec.url)
        except Exception:
            return []
        if not content:
            return []
        feed = feedparser.parse(content)
        out: list[Item] = []
        for entry in feed.entries[:200]:
            url = canonicalize_url(entry.get("link", ""))
            if not url:
                continue
            title = _strip_html(entry.get("title", "")).strip()
            if not title:
                continue
            abstract = _extract_abstract(entry)
            authors = ""
            if entry.get("authors"):
                authors = ", ".join(a.get("name", "") for a in entry["authors"] if a.get("name"))
            elif entry.get("author"):
                authors = entry["author"]
            out.append(
                Item(
                    source=spec.name,
                    section=spec.section,
                    external_id=_external_id(entry, url),
                    url=url,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    published_at=_parsed_dt(entry),
                )
            )
        return out
