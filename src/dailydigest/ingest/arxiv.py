from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import datetime, timezone
from threading import Lock

import feedparser
import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..models import Item, SourceSpec

logger = logging.getLogger(__name__)

# arXiv Terms of Use require >= 3 seconds between automated requests.
_ARXIV_LOCK = Lock()
_ARXIV_LAST_REQUEST: float = 0.0
_ARXIV_POLITE_DELAY = 4.0


def _arxiv_polite_wait() -> None:
    global _ARXIV_LAST_REQUEST
    with _ARXIV_LOCK:
        elapsed = time.monotonic() - _ARXIV_LAST_REQUEST
        if elapsed < _ARXIV_POLITE_DELAY:
            time.sleep(_ARXIV_POLITE_DELAY - elapsed)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=False,
)
def _http_get_text(url: str, params: dict[str, str]) -> str:
    global _ARXIV_LAST_REQUEST
    _arxiv_polite_wait()
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp.text
    finally:
        with _ARXIV_LOCK:
            _ARXIV_LAST_REQUEST = time.monotonic()


class ArxivSource:
    """Pulls recent papers from the arXiv Atom API.

    API: http://export.arxiv.org/api/query?search_query=cat:{cat}
            &sortBy=submittedDate&sortOrder=descending&max_results=50
    Spec field: ``category`` (e.g. ``q-bio.QM``, ``cs.LG`` or compound
    expressions like ``q-bio.GN+OR+q-bio.QM``).
    """

    BASE = "https://export.arxiv.org/api/query"
    MAX_RESULTS = 50

    def fetch(self, spec: SourceSpec) -> list[Item]:
        category = spec.category or "q-bio.QM"
        params = {
            "search_query": f"cat:{category}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": str(self.MAX_RESULTS),
        }
        out: list[Item] = []
        try:
            body = _http_get_text(self.BASE, params)
        except Exception as e:
            logger.warning("%s fetch failed: %s: %s", getattr(spec, "name", "ArxivSource"), type(e).__name__, str(e)[:200])
            return out
        if not body:
            return out

        feed = feedparser.parse(body)
        if len(feed.entries) >= self.MAX_RESULTS:
            logger.debug("%s: received %d entries (at cap); some papers may be missed", getattr(spec, "name", "ArxivSource"), len(feed.entries))
        for entry in feed.entries:
            raw_id = entry.get("id", "") or ""
            # arXiv IDs come back as the abs URL; keep only the trailing arxiv id.
            arxiv_id = raw_id.rsplit("/abs/", 1)[-1] if "/abs/" in raw_id else raw_id
            arxiv_id = re.sub(r"^(?:oai:arxiv\.org:|arxiv:)", "", arxiv_id, flags=re.IGNORECASE)
            arxiv_id = arxiv_id.strip()
            arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
            if not arxiv_id:
                continue
            link = entry.get("link", "") or raw_id
            title = (entry.get("title", "") or "").strip().replace("\n", " ")
            if not title:
                continue
            abstract = (entry.get("summary", "") or "").strip().replace("\n", " ")
            authors = ""
            if entry.get("authors"):
                authors = ", ".join(
                    a.get("name", "") for a in entry["authors"] if a.get("name")
                )
            elif entry.get("author"):
                authors = entry["author"]

            pub_dt: datetime | None = None
            for key in ("updated_parsed", "published_parsed"):
                val = entry.get(key)
                if val:
                    try:
                        pub_dt = datetime(*val[:6], tzinfo=timezone.utc)
                        break
                    except Exception:
                        pass

            ext = arxiv_id or hashlib.sha1(link.encode("utf-8")).hexdigest()[:16]
            out.append(
                Item(
                    source=spec.name,
                    section=spec.section,
                    external_id=ext,
                    url=link,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    published_at=pub_dt,
                )
            )
        return out
