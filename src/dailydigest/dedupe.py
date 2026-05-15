from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from urllib.parse import parse_qsl, quote, unquote, urlencode, urldefrag, urlparse, urlunparse

from .models import Item


def _canonical_doi(raw: str) -> str:
    """Normalize a DOI to bare lowercase form."""
    if not raw:
        return ""
    s = raw.strip()
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^doi:\s*", "", s, flags=re.IGNORECASE)
    return s.lower()


_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_ARXIV_RE = re.compile(
    r"(?:arxiv:)?((?:\d{4}\.\d{4,5})|(?:[a-z-]+(?:\.[A-Z]{2})?/\d{7}))(?:v\d+)?",
    re.IGNORECASE,
)
_PMID_URL_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", re.IGNORECASE)
_TRAILING_DOI_PUNCT = ".),;:"
_TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
_RSS_TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content"}


def canonicalize_url(url: str) -> str:
    """Canonicalize URLs without importing ingest.rss and creating an import cycle."""
    if not url:
        return url
    url, _fragment = urldefrag(url)
    parsed = urlparse(url)
    parsed = parsed._replace(scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower())
    if parsed.query:
        kept = [
            kv
            for kv in parsed.query.split("&")
            if kv and kv.split("=")[0].lower() not in _RSS_TRACKING_PARAMS
        ]
        parsed = parsed._replace(query="&".join(kept))
    return urlunparse(parsed).rstrip("/")


def dedupe_by_url(items: list[Item]) -> list[Item]:
    seen: set[str] = set()
    out: list[Item] = []
    for it in items:
        canon = canonicalize_url(it.url)
        if not canon:
            out.append(it)
            continue
        if canon in seen:
            continue
        seen.add(canon)
        out.append(it)
    return out


def dedupe_ranking_candidates[T](items: list[T]) -> list[T]:
    """Collapse duplicate ranking candidates across stored sources.

    Stored rows are unique only within a source. Ranking works across all sources,
    so use durable cross-source keys first (DOI/PMID/arXiv/URL), then a cautious
    same-day title fallback for feeds that do not expose shared identifiers.
    """
    seen: set[str] = set()
    out: list[T] = []
    for it in items:
        keys = _candidate_keys(it)
        duplicate = any(k in seen for k in keys)
        seen.update(keys)
        if duplicate:
            continue
        out.append(it)
    return out


def _candidate_keys(it: Any) -> list[str]:
    url = str(getattr(it, "url", "") or "")
    external_id = str(getattr(it, "external_id", "") or "")
    source = str(getattr(it, "source", "") or "").lower()
    title = str(getattr(it, "title", "") or "")

    keys: list[str] = []
    doi = _extract_doi(external_id) or _extract_doi(url)
    if doi:
        keys.append(f"doi:{doi}")

    pmid = _extract_pmid(url, external_id, source)
    if pmid:
        keys.append(f"pmid:{pmid}")

    arxiv_id = _extract_arxiv_id(url, external_id, source)
    if arxiv_id:
        keys.append(f"arxiv:{arxiv_id}")

    normalized_url = _normalized_url(url)
    if normalized_url:
        keys.append(f"url:{normalized_url}")

    title_key = _same_day_title_key(title, _effective_date(it))
    if title_key:
        keys.append(f"title_day:{title_key}")
        # Also emit the previous 12h bucket so items near a boundary still match
        parts = title_key.split(":", 2)
        if len(parts) == 3:
            try:
                keys.append(f"title_day:t12h:{int(parts[1]) - 1}:{parts[2]}")
            except ValueError:
                pass

    return keys


def _extract_doi(*values: str) -> str | None:
    for value in values:
        text = unquote(value or "").strip()
        text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^doi:\s*", "", text, flags=re.IGNORECASE)
        match = _DOI_RE.search(text)
        doi = (match.group(0) if match else text if text.lower().startswith("10.") else "")
        doi = doi.strip().rstrip(_TRAILING_DOI_PUNCT).lower()
        if doi.startswith("10.") and "/" in doi:
            return doi
    return None


def _extract_pmid(url: str, external_id: str, source: str) -> str | None:
    match = _PMID_URL_RE.search(url)
    if match:
        return match.group(1)
    if "pubmed" in source and external_id.isdigit():
        return external_id
    return None


def _extract_arxiv_id(url: str, external_id: str, source: str) -> str | None:
    values = [url]
    if "arxiv" in source:
        values.append(external_id)
    for value in values:
        text = unquote(value or "")
        parsed = urlparse(text)
        if parsed.netloc.lower().endswith("arxiv.org") and parsed.path:
            text = parsed.path.rsplit("/", 1)[-1]
        match = _ARXIV_RE.search(text)
        if match:
            return match.group(1).lower()
    return None


def _normalized_url(url: str) -> str:
    canon = canonicalize_url(url)
    if not canon:
        return ""
    parsed = urlparse(canon)
    if not parsed.scheme or not parsed.netloc:
        return canon.rstrip("/")
    query = urlencode(
        sorted(
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS
        ),
        doseq=True,
    )
    path = quote(unquote(parsed.path), safe="/:@")
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower().removeprefix("www."),
            path.rstrip("/") or "/",
            "",
            query,
            "",
        )
    ).rstrip("/")


def _effective_date(it: Any) -> datetime | None:
    published_at = getattr(it, "published_at", None)
    if isinstance(published_at, datetime):
        return published_at
    fetched_at = getattr(it, "fetched_at", None)
    if isinstance(fetched_at, datetime):
        return fetched_at
    return None


def _same_day_title_key(title: str, dt: datetime | None) -> str:
    """Return a dedup key bucketed by 12-hour epoch windows.

    Using 12-hour buckets (vs calendar-day) means items published within
    ~12h of a midnight boundary (e.g. Nature RSS at 23:00 UTC and OpenAlex
    at 02:00 UTC next day) still land in adjacent buckets and can be matched
    via the two-key approach in _candidate_keys.
    """
    if dt is None:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if len(normalized) < 20:
        return ""
    import calendar
    ts = int(dt.timestamp()) if dt.tzinfo is not None else calendar.timegm(dt.timetuple())
    bucket = ts // (12 * 3600)
    return f"t12h:{bucket}:{normalized}"


def filter_english(items: list[Item]) -> list[Item]:
    """Best-effort English filter using langdetect on title+abstract."""
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
    except Exception:
        return items

    out: list[Item] = []
    for it in items:
        sample = (it.title + " " + it.abstract).strip()
        if len(sample) < 20:
            out.append(it)
            continue
        try:
            if detect(sample) == "en":
                out.append(it)
        except Exception:
            out.append(it)
    return out
