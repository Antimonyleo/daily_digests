from __future__ import annotations

import hashlib
from datetime import datetime, timezone

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
def _http_get_text(url: str, params: dict[str, str]) -> str:
    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.text


class ArxivSource:
    """Pulls recent papers from the arXiv Atom API.

    API: http://export.arxiv.org/api/query?search_query=cat:{cat}
            &sortBy=submittedDate&sortOrder=descending&max_results=50
    Spec field: ``category`` (e.g. ``q-bio.QM``, ``cs.LG`` or compound
    expressions like ``q-bio.GN+OR+q-bio.QM``).
    """

    BASE = "http://export.arxiv.org/api/query"
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
        except Exception:
            return out
        if not body:
            return out

        feed = feedparser.parse(body)
        for entry in feed.entries[:100]:
            raw_id = entry.get("id", "") or ""
            # arXiv IDs come back as the abs URL; keep only the trailing arxiv id.
            arxiv_id = raw_id.rsplit("/abs/", 1)[-1] if "/abs/" in raw_id else raw_id
            arxiv_id = arxiv_id.strip()
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
            for key in ("published_parsed", "updated_parsed"):
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
