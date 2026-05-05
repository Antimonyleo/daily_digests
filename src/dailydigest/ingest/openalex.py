from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

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
def _get_json(
    url: str, params: dict[str, str], headers: dict[str, str]
) -> dict:
    with httpx.Client(timeout=20.0, follow_redirects=True, headers=headers) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def _reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str:
    """Reconstruct the abstract from OpenAlex's inverted index."""
    if not inverted:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    except Exception:
        return None


class OpenAlexSource:
    """Queries the OpenAlex Works endpoint for fresh articles.

    API: https://api.openalex.org/works?filter=from_publication_date:{yesterday},
        type:article&search={query}&per-page=25
    Spec fields: ``query`` (search term), optional ``polite_email``.
    """

    BASE = "https://api.openalex.org/works"
    PER_PAGE = 25
    MAX_ITEMS = 100

    def fetch(self, spec: SourceSpec) -> list[Item]:
        query = spec.query or ""
        yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
        params: dict[str, str] = {
            "filter": f"from_publication_date:{yesterday},type:article",
            "per-page": str(self.PER_PAGE),
        }
        if query:
            params["search"] = query

        ua = "dailydigest/0.1"
        if spec.polite_email:
            ua = f"dailydigest/0.1 (mailto:{spec.polite_email})"
        headers = {"User-Agent": ua}

        out: list[Item] = []
        try:
            payload = _get_json(self.BASE, params, headers)
        except Exception:
            return out
        if not payload:
            return out

        results: list[dict[str, Any]] = payload.get("results") or []
        for work in results[: self.MAX_ITEMS]:
            title = (work.get("title") or work.get("display_name") or "").strip()
            if not title:
                continue
            url = (
                work.get("doi")
                or (work.get("primary_location") or {}).get("landing_page_url")
                or work.get("id")
                or ""
            )
            if not url:
                continue
            abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
            authorships = work.get("authorships") or []
            authors = ", ".join(
                (a.get("author") or {}).get("display_name", "")
                for a in authorships
                if (a.get("author") or {}).get("display_name")
            )
            pub = _parse_date(work.get("publication_date"))
            ext = (
                work.get("doi")
                or work.get("id")
                or hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
            )
            out.append(
                Item(
                    source=spec.name,
                    section=spec.section,
                    external_id=str(ext),
                    url=url,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    published_at=pub,
                )
            )
        return out
