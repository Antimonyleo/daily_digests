from __future__ import annotations

import hashlib
import html
import logging
import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..dedupe import _canonical_doi
from ..models import Item, SourceSpec
from ._terms import profile_search_term_groups, watched_author_names

logger = logging.getLogger(__name__)

_CROSSREF_ACS_WORKS = "https://api.crossref.org/prefixes/10.1021/works"
_HTML_TAG_RE = re.compile(r"<[^>]+>")


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


def _get_crossref_json(
    url: str, params: dict[str, str], headers: dict[str, str]
) -> dict:
    """Separate seam for Crossref title metadata; transport retries via `_get_json`."""
    return _get_json(url, params, headers)


def _plain_title(value: str) -> str:
    value = _HTML_TAG_RE.sub(" ", str(value or ""))
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


@lru_cache(maxsize=8)
def _recent_acs_titles(
    window_start: str, window_end: str, polite_email: str
) -> dict[str, str]:
    """Load recent authoritative ACS titles once per brew window.

    OpenAlex occasionally collapses Crossref title line breaks and inline markup,
    yielding strings such as ``FluorescentSelf-Assembled``. A single bounded
    prefix query avoids one HTTP request per paper.
    """
    params = {
        "filter": f"from-pub-date:{window_start},until-pub-date:{window_end}",
        "select": "DOI,title",
        "rows": "1000",
        "sort": "published",
        "order": "desc",
    }
    if polite_email:
        params["mailto"] = polite_email
    headers = {"User-Agent": "dailydigest/0.1"}
    if polite_email:
        headers["User-Agent"] += f" (mailto:{polite_email})"
    try:
        payload = _get_crossref_json(_CROSSREF_ACS_WORKS, params, headers)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Crossref ACS title cleanup unavailable: %s", exc)
        return {}

    titles: dict[str, str] = {}
    for item in (payload.get("message") or {}).get("items") or []:
        doi = _canonical_doi(str(item.get("DOI") or ""))
        raw_titles = item.get("title") or []
        raw_title = raw_titles[0] if isinstance(raw_titles, list) and raw_titles else ""
        title = _plain_title(str(raw_title))
        if doi and title:
            titles[doi] = title
    return titles


def _repair_acs_title(
    title: str,
    doi: str,
    *,
    window_start: str,
    window_end: str,
    polite_email: str,
) -> str:
    if not doi.startswith("10.1021/"):
        return title
    return _recent_acs_titles(window_start, window_end, polite_email).get(doi, title)


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
        dt = datetime.fromisoformat(str(value).split("T")[0])
        return dt.replace(hour=12, minute=0, second=0, tzinfo=timezone.utc)
    except Exception:
        return None


class OpenAlexSource:
    """Queries the OpenAlex Works endpoint for fresh articles.

    API: https://api.openalex.org/works?filter=from_publication_date:{yesterday},
        type:article&search={query}&per-page=25&cursor=*
    Spec fields: ``query`` (search term), optional ``polite_email``.
    """

    BASE = "https://api.openalex.org/works"
    PER_PAGE = 25
    MAX_ITEMS = 100
    # Profile-driven mode: one search per keyword, bounded so ~a dozen keyword
    # queries don't balloon the ingest / embedding cost.
    PROFILE_MAX_TERMS = 12
    PROFILE_PER_TERM = 20
    PROFILE_MAX_ITEMS = 180

    def __init__(self) -> None:
        self.last_fetch_diagnostics: dict[str, Any] = {"queries": []}

    def _headers(self, spec: SourceSpec) -> dict[str, str]:
        ua = "dailydigest/0.1"
        if spec.polite_email:
            ua = f"dailydigest/0.1 (mailto:{spec.polite_email})"
        return {"User-Agent": ua}

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        self.last_fetch_diagnostics = {"queries": [], "returned_count": 0}
        headers = self._headers(spec)
        if spec.profile_driven:
            core, context = profile_search_term_groups(
                self.PROFILE_MAX_TERMS, canonical_order=True
            )
            if not core and not context and spec.query:
                core = [spec.query]
            terms = core + context
            # One bounded page per term. All core queries share the item budget
            # before context gets a turn. More than the soft limit of core terms
            # requires one request per core, matching the all-core contract.
            self.last_fetch_diagnostics.update(
                item_budget=self.PROFILE_MAX_ITEMS,
                request_budget=max(self.PROFILE_MAX_TERMS, len(core)),
                planned_terms=terms,
                core_term_count=len(core),
                # Transport retries are independently bounded by _get_json.
                request_count_includes_retries=False,
            )
            if len(core) > self.PROFILE_MAX_ITEMS:
                raise RuntimeError(
                    f"{spec.name}: {len(core)} core queries exceed the "
                    f"{self.PROFILE_MAX_ITEMS}-item retrieval budget"
                )
            out: list[Item] = []
            seen: set[str] = set()
            remaining = self.PROFILE_MAX_ITEMS
            for index, term in enumerate(terms):
                group_remaining = (len(core) if index < len(core) else len(terms)) - index
                cap = min(self.PROFILE_PER_TERM, remaining // group_remaining)
                if remaining % group_remaining:
                    cap = min(self.PROFILE_PER_TERM, cap + 1)
                if not cap:
                    self.last_fetch_diagnostics["queries"].append({
                        "query": term, "attempted": False, "request_count": 0,
                        "fetched_count": 0, "returned_count": 0, "unique_yield": 0,
                        "exhausted": False, "truncated": True,
                        "stop_reason": "item_budget", "failure": None,
                    })
                    continue
                fetched = self._fetch_works(
                    spec, days, headers, cap=cap, query=term,
                    upgrade_venue=True, max_pages=1,
                )
                diagnostic = self.last_fetch_diagnostics["queries"][-1]
                # Count raw rows, including duplicates/invalid records: otherwise
                # duplicate-heavy queries could silently expand API work.
                remaining -= min(cap, diagnostic["fetched_count"])
                before = len(out)
                for item in fetched:
                    if item.external_id not in seen:
                        seen.add(item.external_id)
                        out.append(item)
                diagnostic["unique_yield"] = len(out) - before
                self.last_fetch_diagnostics["returned_count"] = len(out)
            logger.debug("%s: profile-driven fetched %d items over %d terms", spec.name, len(out), len(terms))
            return out
        out = self._fetch_works(spec, days, headers, cap=self.MAX_ITEMS, query=spec.query or "")
        self.last_fetch_diagnostics["returned_count"] = len(out)
        return out

    def _fetch_works(
        self,
        spec: SourceSpec,
        days: int,
        headers: dict[str, str],
        *,
        cap: int,
        query: str = "",
        extra_filter: str = "",
        use_venue_source: bool = False,
        upgrade_venue: bool = False,
        max_pages: int = 15,
    ) -> list[Item]:
        diagnostic: dict[str, Any] = {
            "query": query, "attempted": True, "request_count": 0,
            "fetched_count": 0, "returned_count": 0, "unique_yield": 0,
            "total_available": None, "exhausted": False, "truncated": False,
            "stop_reason": None, "failure": None,
            "oldest_publication_date": None, "newest_publication_date": None,
            "request_count_includes_retries": False,
            "indexing_delay": None,
        }
        self.last_fetch_diagnostics["queries"].append(diagnostic)
        # Lazy import: pulling ``rank.source_quality`` at module top would trigger
        # ``rank/__init__`` → sentence-transformers on the ingest hot path.
        if upgrade_venue:
            from ..rank.source_quality import recognized_research_venue
        else:
            recognized_research_venue = None  # type: ignore[assignment]

        today = datetime.now(timezone.utc).date()
        window_start = (today - timedelta(days=max(1, days))).isoformat()
        work_types = str(getattr(spec, "openalex_types", "") or "article").strip()
        filter_str = (
            f"from_publication_date:{window_start},"
            f"to_publication_date:{today.isoformat()},type:{work_types}"
        )
        if extra_filter:
            filter_str = f"{filter_str},{extra_filter}"
        diagnostic.update(filter=filter_str, item_cap=cap, page_budget=max_pages)
        base_params: dict[str, str] = {
            "filter": filter_str,
            "per-page": str(min(self.PER_PAGE, cap)),
            "cursor": "*",
            "sort": "publication_date:desc",
        }
        if query:
            base_params["search"] = query

        out: list[Item] = []
        params = dict(base_params)
        pages = 0
        seen_cursors = {"*"}
        while len(out) < cap and pages < max_pages:
            params["per-page"] = str(min(self.PER_PAGE, cap - len(out)))
            diagnostic["request_count"] += 1
            try:
                payload = _get_json(self.BASE, params, headers)
            except Exception as e:
                diagnostic.update(stop_reason="failure", failure=type(e).__name__, truncated=True)
                name = getattr(spec, "name", "OpenAlexSource")
                raise RuntimeError(
                    f"{name} fetch failed: {type(e).__name__}: {str(e)[:200]}"
                ) from e
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                diagnostic.update(stop_reason="failure", failure="invalid_response", truncated=True)
                raise RuntimeError(f"{spec.name} returned an invalid OpenAlex response")
            results: list[dict[str, Any]] = payload["results"]
            diagnostic["fetched_count"] += len(results)
            meta = payload.get("meta") or {}
            if not isinstance(meta, dict) or any(not isinstance(work, dict) for work in results):
                diagnostic.update(stop_reason="failure", failure="invalid_response", truncated=True)
                raise RuntimeError(f"{spec.name} returned an invalid OpenAlex response")
            if isinstance(meta.get("count"), int):
                diagnostic["total_available"] = meta["count"]
            if not results:
                if diagnostic["total_available"] is not None and diagnostic["total_available"] > diagnostic["fetched_count"]:
                    diagnostic.update(stop_reason="failure", failure="inconsistent_empty_page", truncated=True)
                    raise RuntimeError(f"{spec.name} returned an unexpectedly empty OpenAlex page")
                diagnostic.update(exhausted=True, stop_reason="exhausted")
                break

            for work in results:
                if len(out) >= cap:
                    break
                title = (work.get("title") or work.get("display_name") or "").strip()
                if not title:
                    continue
                raw_doi = work.get("doi") or ""
                bare_doi = _canonical_doi(raw_doi)
                title = _repair_acs_title(
                    title,
                    bare_doi,
                    window_start=window_start,
                    window_end=today.isoformat(),
                    polite_email=spec.polite_email or "",
                )
                raw_url = (
                    work.get("doi")
                    or (work.get("primary_location") or {}).get("landing_page_url")
                    or ""
                )
                openalex_id = work.get("id") or ""
                if not raw_url and openalex_id:
                    # Fall back to the canonical OpenAlex page URL so the item
                    # has a working link rather than a bare API identifier.
                    raw_url = openalex_id if openalex_id.startswith("http") else f"https://openalex.org/{openalex_id}"
                url = raw_url
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
                ext = bare_doi or work.get("id") or hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
                # Real publication venue, used to attribute the item to its actual
                # journal (so it earns venue prestige) rather than to the generic
                # aggregator feed name.
                venue = ((work.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
                item_source = spec.name
                if use_venue_source and venue:
                    item_source = venue
                elif upgrade_venue and venue and recognized_research_venue is not None:
                    recognized = recognized_research_venue(venue)
                    if recognized:
                        item_source = recognized
                publication_metadata: dict[str, Any] = {}
                if isinstance(work.get("type"), str) and work["type"].strip():
                    publication_metadata["work_type"] = work["type"].strip()
                if isinstance(work.get("is_retracted"), bool):
                    publication_metadata["is_retracted"] = work["is_retracted"]
                out.append(
                    Item(
                        source=item_source,
                        section=spec.section,
                        external_id=str(ext),
                        url=url,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        published_at=pub,
                        metadata={
                            **{key: value for key, value in (("venue", venue), ("doi", bare_doi)) if value},
                            **publication_metadata,
                        },
                    )
                )

            pages += 1
            next_cursor = meta.get("next_cursor")
            diagnostic["returned_count"] = len(out)
            diagnostic["unique_yield"] = len({item.external_id for item in out})
            if len(out) >= cap:
                # Hitting a cap is incomplete evidence of recall even when the
                # upstream omits its next cursor or reports an estimated total.
                diagnostic.update(truncated=True, stop_reason="item_cap")
                break
            if not next_cursor:
                if diagnostic["total_available"] is not None and diagnostic["total_available"] > diagnostic["fetched_count"]:
                    diagnostic.update(truncated=True, stop_reason="missing_cursor")
                else:
                    diagnostic.update(exhausted=True, stop_reason="exhausted")
                break
            if next_cursor in seen_cursors:
                diagnostic.update(truncated=True, stop_reason="repeated_cursor")
                break
            seen_cursors.add(next_cursor)
            params = dict(base_params)
            params["cursor"] = next_cursor

        if diagnostic["stop_reason"] is None:
            diagnostic.update(truncated=True, stop_reason="request_budget")
        dates = [item.published_at.date().isoformat() for item in out if item.published_at]
        if dates:
            diagnostic.update(oldest_publication_date=min(dates), newest_publication_date=max(dates))
        return out


class OpenAlexVenuesSource:
    """Fetch recent articles directly from specific high-value journals by venue id.

    A retrieval channel for flagship journals whose native RSS is bot-blocked
    (e.g. ACS Nano, Nano Letters, JACS, Chemistry of Materials behind Cloudflare).
    Queries OpenAlex filtered to the configured ``venue_ids`` and tags each item
    with its real journal name (``use_venue_source``) so it earns the correct
    venue prestige. The topic-relevance gate downstream keeps only on-field work,
    so this does not flood the digest with a journal's full daily output.
    """

    MAX_ITEMS = 120

    def __init__(self) -> None:
        self.last_fetch_diagnostics: dict[str, Any] = {"queries": []}

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        self.last_fetch_diagnostics = {"queries": [], "returned_count": 0}
        venue_ids = [v for v in (spec.venue_ids or []) if v]
        if not venue_ids:
            logger.debug("%s: no venue_ids configured", spec.name)
            return []
        works = OpenAlexSource()
        self.last_fetch_diagnostics = works.last_fetch_diagnostics
        headers = works._headers(spec)
        extra_filter = "primary_location.source.id:" + "|".join(venue_ids)
        out = works._fetch_works(
            spec,
            days,
            headers,
            cap=self.MAX_ITEMS,
            extra_filter=extra_filter,
            use_venue_source=True,
        )
        self.last_fetch_diagnostics["returned_count"] = len(out)
        return out


class OpenAlexAuthorsSource:
    """Fetch recent works BY the profile's watched authors, across any venue.

    ``authors_of_interest`` is otherwise only a ranking boost applied to items
    that already arrived via some feed; this makes it a *retrieval* channel, so a
    watched lab's new paper surfaces even from a journal not in the feed list.
    Author names are resolved to OpenAlex author ids (top match), then their
    works in the window are fetched in a single filtered query.
    """

    AUTHORS_URL = "https://api.openalex.org/authors"
    MAX_AUTHORS = 25
    MAX_ITEMS = 120

    def __init__(self) -> None:
        self.last_fetch_diagnostics: dict[str, Any] = {"queries": []}

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        self.last_fetch_diagnostics = {"queries": [], "returned_count": 0}
        names = watched_author_names(self.MAX_AUTHORS)
        if not names:
            return []
        works = OpenAlexSource()
        self.last_fetch_diagnostics = works.last_fetch_diagnostics
        headers = works._headers(spec)
        author_ids: list[str] = []
        for name in names:
            aid = self._resolve_author_id(name, headers)
            if aid:
                author_ids.append(aid)
        failed = any(query["failure"] for query in self.last_fetch_diagnostics.get("author_queries", []))
        self.last_fetch_diagnostics.update(
            author_resolution_complete=not failed, truncated=failed,
        )
        if not author_ids:
            if failed:
                raise RuntimeError(f"{spec.name}: author resolution failed; no author works query possible")
            logger.debug("%s: no watched-author ids resolved", spec.name)
            return []
        extra_filter = "author.id:" + "|".join(author_ids)
        try:
            out = works._fetch_works(
                spec, days, headers, cap=self.MAX_ITEMS, extra_filter=extra_filter
            )
        finally:
            self.last_fetch_diagnostics["truncated"] = failed or any(
                query["truncated"] for query in self.last_fetch_diagnostics["queries"]
            )
        self.last_fetch_diagnostics["returned_count"] = len(out)
        return out

    def _resolve_author_id(self, name: str, headers: dict[str, str]) -> str | None:
        diagnostic: dict[str, Any] = {
            "query": name, "attempted": True, "request_count": 1,
            "request_count_includes_retries": False,
            "failure": None, "stop_reason": None, "resolved_id": None,
        }
        self.last_fetch_diagnostics.setdefault("author_queries", []).append(diagnostic)
        try:
            payload = _get_json(
                self.AUTHORS_URL, {"search": name, "per-page": "1"}, headers
            )
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise RuntimeError("Invalid OpenAlex author response")
            results = payload["results"]
            if not results:
                diagnostic["stop_reason"] = "no_match"
                return None
            raw_id = str(results[0].get("id") or "")
            # id is a URL like https://openalex.org/A5023888391 → keep the bare id.
            author_id = raw_id.rstrip("/").rsplit("/", 1)[-1] or None
            if not author_id:
                raise RuntimeError("OpenAlex author match has no id")
            diagnostic.update(stop_reason="matched", resolved_id=author_id)
            return author_id
        except Exception as e:  # noqa: BLE001
            diagnostic.update(failure=type(e).__name__, stop_reason="failure")
            logger.warning("author resolve failed for %r: %s", name, e)
            return None
