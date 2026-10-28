from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..dedupe import _canonical_doi
from ..models import Item, SourceSpec

logger = logging.getLogger(__name__)

# bioRxiv posts ~500 preprints/day, so a 500-row ceiling silently truncated a
# normal two-day window (observed: "hit pagination cap (510/525)"). The API
# pages 100 at a time, so this is ~20 requests worst case.
_MAX_ROWS = 2000


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=False,
)
def _get_json(client: httpx.Client, url: str) -> dict:
    resp = client.get(url)
    resp.raise_for_status()
    return resp.json()


def request_window(days: int, today=None) -> tuple[str, str]:
    """Inclusive ``(from, to)`` posting-date interval that ``fetch`` requests."""
    today = today or datetime.now(timezone.utc).date()
    return (today - timedelta(days=max(1, days))).isoformat(), today.isoformat()


def entry_date(entry: dict) -> str | None:
    """The posting date ``fetch`` assigns to an API entry, or None if unparseable."""
    pub = entry.get("date") or ""
    try:
        y, m, d = (int(x) for x in pub.split("-"))
        return datetime(y, m, d, tzinfo=timezone.utc).date().isoformat()
    except Exception:
        return None


def current_versions(items: list[Item], withdrawals: list[Item]) -> list[Item]:
    """Keep the newest revision per preprint and drop withdrawn ones.

    URL deduplication later keeps the first item. Choose the authoritative
    revision here so upstream ordering cannot discard a newer abstract.
    """
    withdrawn_ids = {item.external_id for item in withdrawals}
    newest: dict[str, Item] = {}
    for item in items:
        if item.external_id in withdrawn_ids:
            continue
        prior = newest.get(item.external_id)
        version = str(item.metadata.get("preprint_version") or "")
        prior_version = str(prior.metadata.get("preprint_version") or "") if prior else ""
        if prior is None or (version.isdecimal() and
                             (not prior_version.isdecimal() or int(version) > int(prior_version))):
            newest[item.external_id] = item
    return list(newest.values())


class BiorxivSource:
    """Pulls the most recent 2 days from the bioRxiv/medRxiv details API.

    API: https://api.biorxiv.org/details/[server]/[from]/[to]/[cursor]
    """

    BASE = "https://api.biorxiv.org/details"

    def __init__(self) -> None:
        self.last_fetch_diagnostics: dict = {"queries": []}
        self.last_fetch_withdrawals: list[Item] = []
        # Public-snapshot collection only: every parsed revision before
        # ``current_versions``, posting dates for each withdrawal notice, and
        # per-date counts of raw API entries. Native brews ignore these.
        self.last_fetch_versions: list[Item] = []
        self.last_fetch_withdrawal_dates: list[str | None] = []
        self.last_fetch_entry_dates: dict[str | None, dict[str, int]] = {}
        # None keeps the native module cap; the public collector raises it.
        self.max_rows: int | None = None

    def _count_entry(self, entry: dict, outcome: str) -> None:
        counts = self.last_fetch_entry_dates.setdefault(
            entry_date(entry), {"entries": 0, "withdrawn": 0, "invalid": 0})
        counts["entries"] += 1
        if outcome != "entries":
            counts[outcome] += 1

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        self.last_fetch_withdrawals = []
        self.last_fetch_versions = []
        self.last_fetch_withdrawal_dates = []
        self.last_fetch_entry_dates = {}
        max_rows = _MAX_ROWS if self.max_rows is None else self.max_rows
        server = (spec.server or "biorxiv").lower()
        frm, to = request_window(days)
        diagnostic = {
            "query": f"{server}:{frm}/{to}", "attempted": True,
            "request_count": 0, "request_count_includes_retries": False,
            "fetched_count": 0, "returned_count": 0, "unique_yield": 0,
            "total_available": None, "item_cap": max_rows,
            "exhausted": False, "truncated": False, "stop_reason": None,
            "failure": None, "skipped_withdrawn": 0, "skipped_invalid": 0,
            "oldest_publication_date": None, "newest_publication_date": None,
            "indexing_delay": None,
        }
        self.last_fetch_diagnostics = {"queries": [diagnostic], "returned_count": 0}
        out: list[Item] = []
        cursor = 0
        with httpx.Client(timeout=20.0, follow_redirects=True, headers={"User-Agent": "dailydigest/0.1"}) as client:
            while True:
                url = f"{self.BASE}/{server}/{frm}/{to}/{cursor}"
                diagnostic["request_count"] += 1
                try:
                    data = _get_json(client, url)
                except Exception as e:
                    diagnostic.update(stop_reason="failure", failure=type(e).__name__, truncated=True)
                    name = getattr(spec, "name", "BiorxivSource")
                    raise RuntimeError(
                        f"{name} fetch failed: {type(e).__name__}: {str(e)[:200]}"
                    ) from e
                if (not isinstance(data, dict) or not isinstance(data.get("collection"), list)
                        or any(not isinstance(entry, dict) for entry in data["collection"])):
                    diagnostic.update(stop_reason="failure", failure="invalid_response", truncated=True)
                    raise RuntimeError(f"{spec.name} returned an invalid bioRxiv response")
                collection = data["collection"]
                msg = data.get("messages") or [{}]
                try:
                    total = int(msg[0].get("total"))
                    if total < 0:
                        total = None
                except (TypeError, ValueError, IndexError, AttributeError, KeyError):
                    total = None
                if total is not None:
                    diagnostic["total_available"] = total
                if not collection:
                    known_total = diagnostic["total_available"]
                    if known_total is not None and known_total > cursor:
                        diagnostic.update(stop_reason="failure", failure="inconsistent_empty_page", truncated=True)
                        raise RuntimeError(f"{spec.name} returned an unexpectedly empty bioRxiv page")
                    diagnostic.update(exhausted=known_total is not None,
                                      truncated=known_total is None,
                                      stop_reason="exhausted" if known_total is not None else "unknown_total")
                    break
                diagnostic["fetched_count"] += len(collection)
                for entry in collection:
                    doi = entry.get("doi") or ""
                    title = (entry.get("title") or "").strip()
                    if not title:
                        diagnostic["skipped_invalid"] += 1
                        self._count_entry(entry, "invalid")
                        continue
                    # bioRxiv marks retracted/withdrawn work in-band and keeps
                    # serving it in the daily feed. Keep notices separate from
                    # live supply so ingestion can invalidate cached copies even
                    # when the remaining research coverage cannot support a brew.
                    entry_type = str(entry.get("type") or "").strip().lower()
                    if entry_type == "withdrawn" or title.upper().startswith("WITHDRAWN"):
                        diagnostic["skipped_withdrawn"] += 1
                        self._count_entry(entry, "withdrawn")
                        canonical_doi = _canonical_doi(doi)
                        if canonical_doi:
                            self.last_fetch_withdrawal_dates.append(entry_date(entry))
                            self.last_fetch_withdrawals.append(Item(
                                source=spec.name, section=spec.section,
                                external_id=canonical_doi, url=f"https://doi.org/{canonical_doi}",
                                title=title, metadata={
                                    "doi": canonical_doi, "is_retracted": True,
                                    "preprint_type": "withdrawn",
                                    "preprint_version": str(entry.get("version") or ""),
                                },
                            ))
                        logger.info("biorxiv: skipping withdrawn preprint %s", doi or title[:60])
                        continue
                    link = f"https://doi.org/{doi}" if doi else entry.get("link", "")
                    if not link:
                        diagnostic["skipped_invalid"] += 1
                        self._count_entry(entry, "invalid")
                        continue
                    self._count_entry(entry, "entries")
                    abstract = (entry.get("abstract") or "").strip()
                    authors = entry.get("authors") or ""
                    pub = entry.get("date") or ""
                    try:
                        if pub:
                            # Parse as date-only and anchor to noon UTC so we don't
                            # accidentally mark a US-Pacific date as midnight UTC
                            # (which would push it to the previous calendar day).
                            y, m, d = (int(x) for x in pub.split("-"))
                            pub_dt = datetime(y, m, d, 12, 0, 0, tzinfo=timezone.utc)
                        else:
                            pub_dt = None
                    except Exception:
                        pub_dt = None
                    ext = _canonical_doi(doi) or hashlib.sha1(link.encode()).hexdigest()[:16]
                    published = entry.get("published")
                    published_doi = _canonical_doi(published) if isinstance(published, str) else ""
                    if not re.fullmatch(r"10\.\d{4,9}/\S+", published_doi):
                        published_doi = ""
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
                            # Kept so preprint-quality questions are answerable
                            # later without re-fetching: version (has it been
                            # revised), category, corresponding institution, and
                            # the entry type. Nothing ranks on these yet.
                            metadata={
                                "preprint_version": str(entry.get("version") or ""),
                                "preprint_category": str(entry.get("category") or ""),
                                "preprint_type": entry_type,
                                "corresponding_institution": str(
                                    entry.get("author_corresponding_institution") or ""
                                ),
                                "doi": doi,
                                **({"published_doi": published_doi} if published_doi else {}),
                            },
                        )
                    )
                # paginate
                diagnostic.update(returned_count=len(out), unique_yield=len({item.external_id for item in out}))
                self.last_fetch_diagnostics["returned_count"] = len(out)
                if total is None or total == 0:
                    diagnostic.update(truncated=True, stop_reason="unknown_total")
                    logger.warning(
                        "biorxiv %s: API total is unavailable but page has %d items; coverage is unknown",
                        server, len(collection)
                    )
                    break  # treat as single-page; data is captured, just can't paginate
                cursor += len(collection)
                if cursor >= total:
                    diagnostic.update(exhausted=True, stop_reason="exhausted")
                    break
                if cursor >= max_rows:
                    diagnostic.update(truncated=True, stop_reason="item_cap")
                    logger.warning(
                        "biorxiv %s hit pagination cap (%d/%d); some items may be dropped",
                        server, cursor, total,
                    )
                    break
        self.last_fetch_versions = list(out)
        out = current_versions(out, self.last_fetch_withdrawals)
        diagnostic.update(returned_count=len(out), unique_yield=len(out))
        self.last_fetch_diagnostics["returned_count"] = len(out)
        dates = [item.published_at.date().isoformat() for item in out if item.published_at]
        if dates:
            diagnostic.update(oldest_publication_date=min(dates), newest_publication_date=max(dates))
        return out
