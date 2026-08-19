from __future__ import annotations

import hashlib
import logging
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


class BiorxivSource:
    """Pulls the most recent 2 days from the bioRxiv/medRxiv details API.

    API: https://api.biorxiv.org/details/[server]/[from]/[to]/[cursor]
    """

    BASE = "https://api.biorxiv.org/details"

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        server = (spec.server or "biorxiv").lower()
        today = datetime.now(timezone.utc).date()
        frm = (today - timedelta(days=max(1, days))).isoformat()
        to = today.isoformat()
        out: list[Item] = []
        cursor = 0
        with httpx.Client(timeout=20.0, follow_redirects=True, headers={"User-Agent": "dailydigest/0.1"}) as client:
            while True:
                url = f"{self.BASE}/{server}/{frm}/{to}/{cursor}"
                try:
                    data = _get_json(client, url)
                except Exception as e:
                    name = getattr(spec, "name", "BiorxivSource")
                    if out:
                        logger.warning(
                            "%s pagination stopped after %d items: %s: %s",
                            name,
                            len(out),
                            type(e).__name__,
                            str(e)[:200],
                        )
                        break
                    raise RuntimeError(
                        f"{name} fetch failed: {type(e).__name__}: {str(e)[:200]}"
                    ) from e
                if not data:
                    break
                collection = data.get("collection") or []
                if not collection:
                    break
                for entry in collection:
                    doi = entry.get("doi") or ""
                    title = (entry.get("title") or "").strip()
                    if not title:
                        continue
                    # bioRxiv marks retracted/withdrawn work in-band and keeps
                    # serving it in the daily feed. Recommending a withdrawn
                    # preprint is never right, so drop it at ingest.
                    entry_type = str(entry.get("type") or "").strip().lower()
                    if entry_type == "withdrawn" or title.upper().startswith("WITHDRAWN"):
                        logger.info("biorxiv: skipping withdrawn preprint %s", doi or title[:60])
                        continue
                    link = f"https://doi.org/{doi}" if doi else entry.get("link", "")
                    if not link:
                        continue
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
                            },
                        )
                    )
                # paginate
                msg = data.get("messages") or [{}]
                try:
                    total = int(msg[0].get("total", 0))
                except (TypeError, ValueError):
                    total = 0
                if total == 0 and len(collection) > 0:
                    logger.warning(
                        "biorxiv %s: API returned total=0 but page has %d items; treating as single-page response",
                        server, len(collection)
                    )
                    break  # treat as single-page; data is captured, just can't paginate
                cursor += len(collection)
                if cursor >= total:
                    break
                if cursor >= _MAX_ROWS:
                    logger.warning(
                        "biorxiv %s hit pagination cap (%d/%d); some items may be dropped",
                        server, cursor, total,
                    )
                    break
        return out
