from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

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
def _get_json(client: httpx.Client, url: str) -> dict:
    resp = client.get(url)
    resp.raise_for_status()
    return resp.json()


class BiorxivSource:
    """Pulls the most recent 2 days from the bioRxiv/medRxiv details API.

    API: https://api.biorxiv.org/details/[server]/[from]/[to]/[cursor]
    """

    BASE = "https://api.biorxiv.org/details"

    def fetch(self, spec: SourceSpec) -> list[Item]:
        server = (spec.server or "biorxiv").lower()
        today = datetime.now(timezone.utc).date()
        frm = (today - timedelta(days=2)).isoformat()
        to = today.isoformat()
        out: list[Item] = []
        cursor = 0
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            while True:
                url = f"{self.BASE}/{server}/{frm}/{to}/{cursor}"
                try:
                    data = _get_json(client, url)
                except Exception:
                    break
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
                    ext = doi or hashlib.sha1(link.encode()).hexdigest()[:16]
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
                # paginate
                msg = data.get("messages") or [{}]
                try:
                    total = int(msg[0].get("total", 0))
                except (TypeError, ValueError):
                    total = 0
                cursor += len(collection)
                if cursor >= total or cursor >= 200:  # cap at 200/run for safety
                    break
        return out
