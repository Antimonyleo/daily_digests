from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any

import feedparser
import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ..models import Item, SourceSpec
from ..dedupe import _canonical_doi
from ._retry import transient_http_error

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
    retry=retry_if_exception(transient_http_error),
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
    MAX_ITEMS = 400
    MAX_PAGES = 8

    def __init__(self) -> None:
        self.last_fetch_diagnostics: dict[str, Any] = {'queries': []}

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        category = spec.category or "q-bio.QM"
        # SourceSpec categories historically used URL-style +OR+ separators.
        # HTTPX encodes parameter values, so provide real spaces and scope each
        # category separately instead of double-encoding literal plus signs.
        parts = re.split(r'\s+(OR|AND|ANDNOT)\s+', category.replace('+', ' ').strip())
        query_parts = []
        for index, part in enumerate(parts):
            if index % 2:
                query_parts.append(part)
                continue
            value = part.removeprefix('cat:')
            if not re.fullmatch(r'[A-Za-z][A-Za-z0-9.-]*', value):
                raise ValueError(f'Invalid arXiv category expression: {category}')
            query_parts.append(f'cat:{value}')
        query = ' '.join(query_parts)
        page_size = min(self.MAX_RESULTS * max(1, math.ceil(max(1, days) / 2)), self.MAX_ITEMS)
        params = {
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        diagnostic: dict[str, Any] = {
            'query': query, 'attempted': True, 'request_count': 0,
            'request_count_includes_retries': False, 'fetched_count': 0,
            'returned_count': 0, 'total_available': None, 'exhausted': False,
            'truncated': False, 'stop_reason': None, 'failure': None,
            'item_cap': self.MAX_ITEMS, 'page_budget': self.MAX_PAGES,
        }
        self.last_fetch_diagnostics = {'queries': [diagnostic], 'returned_count': 0}
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, days))
        entries = []
        seen_ids = set()
        while diagnostic['fetched_count'] < self.MAX_ITEMS and diagnostic['request_count'] < self.MAX_PAGES:
            remaining = self.MAX_ITEMS - diagnostic['fetched_count']
            requested = min(page_size, remaining)
            params.update(start=str(diagnostic['fetched_count']), max_results=str(requested))
            diagnostic['request_count'] += 1
            try:
                body = _http_get_text(self.BASE, params)
                feed = feedparser.parse(body)
                if feed.bozo or not feed.version:
                    raise ValueError('Invalid arXiv Atom response')
            except Exception as e:
                diagnostic.update(stop_reason='failure', failure=type(e).__name__, truncated=True)
                raise RuntimeError(
                    f'{spec.name} fetch failed: {type(e).__name__}: {str(e)[:200]}'
                ) from e
            total = feed.feed.get('opensearch_totalresults')
            if total is not None:
                try:
                    diagnostic['total_available'] = int(total)
                except (TypeError, ValueError):
                    pass
            page = feed.entries
            diagnostic['fetched_count'] += len(page)
            if not page:
                if diagnostic['total_available'] is not None and diagnostic['total_available'] > len(entries):
                    diagnostic.update(truncated=True, stop_reason='inconsistent_empty_page')
                else:
                    diagnostic.update(exhausted=True, stop_reason='exhausted')
                break
            new_entries = [entry for entry in page if entry.get('id') not in seen_ids]
            if not new_entries:
                diagnostic.update(truncated=True, stop_reason='repeated_page')
                break
            entries.extend(new_entries)
            seen_ids.update(entry.get('id') for entry in new_entries)
            # Results are sorted by original submission date. Keep the boundary
            # page for the common downstream candidate-window filter to trim.
            dates = [datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                     for entry in page if entry.get('published_parsed')]
            if dates and min(dates) < cutoff:
                diagnostic.update(exhausted=True, stop_reason='window_exhausted')
                break
            if diagnostic['total_available'] is not None and diagnostic['fetched_count'] >= diagnostic['total_available']:
                diagnostic.update(exhausted=True, stop_reason='exhausted')
                break
            if len(page) < requested and diagnostic['total_available'] is None:
                diagnostic.update(exhausted=True, stop_reason='exhausted')
                break
        if diagnostic['stop_reason'] is None:
            diagnostic.update(truncated=True, stop_reason=(
                'item_cap' if diagnostic['fetched_count'] >= self.MAX_ITEMS else 'request_budget'))

        out: list[Item] = []
        for entry in entries:
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
            for key in ("published_parsed", "updated_parsed"):
                val = entry.get(key)
                if val:
                    try:
                        pub_dt = datetime(*val[:6], tzinfo=timezone.utc)
                        break
                    except Exception:
                        pass

            ext = arxiv_id or hashlib.sha1(link.encode("utf-8")).hexdigest()[:16]
            # These optional Atom fields describe this paper's publication,
            # not works cited in the summary. Keep the arXiv identity/source:
            # a supplied journal reference is not a venue-quality upgrade.
            metadata = {}
            doi = entry.get("arxiv_doi")
            published_doi = _canonical_doi(doi) if isinstance(doi, str) else ""
            if re.fullmatch(r"10\.\d{4,9}/\S+", published_doi):
                metadata["published_doi"] = published_doi
            journal_ref = entry.get("arxiv_journal_ref")
            if isinstance(journal_ref, str) and journal_ref.strip():
                metadata["journal_reference"] = " ".join(journal_ref.split())
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
                    metadata=metadata,
                )
            )
        diagnostic['returned_count'] = len(out)
        diagnostic['skipped_invalid'] = len(entries) - len(out)
        if diagnostic['skipped_invalid']:
            diagnostic.update(truncated=True, exhausted=False)
        self.last_fetch_diagnostics['returned_count'] = len(out)
        return out
