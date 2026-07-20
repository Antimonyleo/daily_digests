"""Live citation / impact enrichment via OpenAlex.

Static prestige tiers (``source_quality``) are blind to the individual paper and
biased toward a hand-curated, biomed-heavy venue list. OpenAlex exposes a live
``cited_by_count`` for any DOI, which we convert to a **citation velocity** score
(cites per month, log-squashed). Velocity rewards a fast-rising preprint that the
flat preprint penalty would otherwise bury, and works across every field without
maintaining tier tables.

Off by default (``citation_enrichment``) to keep the daily run local and
reproducible. Network access is injectable (``fetcher``) so the logic is fully
unit-testable offline, and any fetch failure degrades to a no-op.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timedelta, timezone

from ..store import ItemRow

logger = logging.getLogger(__name__)

# DOIs look like 10.1234/suffix. Suffix charset per Crossref guidance.
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)

# Citations/month that should map to a ~1.0 score. A paper accruing ~15
# cites/month is exceptional; most legitimate work sits well below.
_VELOCITY_SCALE = 15.0
CITATION_BOOST = 0.12
# Venue 2yr_mean_citedness (impact-factor-like) that maps to a ~1.0 venue score.
_VENUE_SCALE = 8.0
# Venue-quality score below this marks the item's venue as low-impact, which
# routes it into the low_impact_journal bucket (frequency-capped). 0.4 ~
# 2yr_mean_citedness of ~1.4 — roughly a low-impact-factor journal.
_LOW_VENUE_QUALITY = 0.4
_OPENALEX_URL = "https://api.openalex.org/works"
_OPENALEX_SOURCES_URL = "https://api.openalex.org/sources"

# Reject an OpenAlex response larger than this — a hostile/buggy endpoint could
# otherwise stream a multi-GB body that `resp.json()` buffers entirely, OOMing
# the run. 16 MiB is generous for a 50-record page.
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
# Defensive ceilings on untrusted upstream values.
_MAX_RESULTS_PER_PAGE = 200
_MAX_VENUE_MEAN_CITEDNESS = 1000.0


def _redact(text: str) -> str:
    """Strip a ``mailto=<email>`` query param so the polite-pool email (PII)
    never lands in logs (which may be public CI output)."""
    return re.sub(r"(mailto=)[^&\s]+", r"\1[redacted]", text)


def _get_json_bounded(client, url: str, params: dict) -> dict:
    """GET + JSON-parse with a hard response-size ceiling (CWE-400)."""
    import json

    with client.stream("GET", url, params=params) as resp:
        resp.raise_for_status()
        buf = bytearray()
        for chunk in resp.iter_bytes():
            buf += chunk
            if len(buf) > _MAX_RESPONSE_BYTES:
                raise ValueError("OpenAlex response exceeds size cap")
        return json.loads(buf) if buf else {}


def derive_doi(row: object) -> str | None:
    """Best-effort DOI extraction from an item's url / external_id."""
    for attr in ("url", "external_id"):
        value = getattr(row, attr, None)
        if isinstance(value, str) and value:
            m = _DOI_RE.search(value)
            if m:
                return m.group(0).rstrip(").").lower()
    return None


def citation_score(
    cited_by_count: int | None,
    published_at: datetime | None,
    now: datetime | None = None,
) -> float:
    """Map a citation count to a [0, 1] velocity score (cites per month)."""
    if cited_by_count is None or cited_by_count <= 0:
        return 0.0
    now = now or datetime.now(timezone.utc)
    if isinstance(published_at, datetime):
        ref = published_at if published_at.tzinfo else published_at.replace(tzinfo=timezone.utc)
        age_days = max(1.0, (now - ref).total_seconds() / 86400)
    else:
        age_days = 30.0  # unknown age — assume ~1 month
    age_months = max(age_days / 30.0, 0.5)
    velocity = cited_by_count / age_months
    return min(1.0, math.log1p(velocity) / math.log1p(_VELOCITY_SCALE))


def venue_quality_score(mean_citedness: float | None) -> float | None:
    """Map a venue's 2yr_mean_citedness (impact-factor-like) to [0, 1].

    Returns None when impact is unknown so callers can skip the adjustment
    rather than assume a venue is low quality.
    """
    if mean_citedness is None or mean_citedness < 0:
        return None
    return min(1.0, math.log1p(float(mean_citedness)) / math.log1p(_VENUE_SCALE))


def _normalize_entry(value: object) -> dict:
    """Accept either a bare cited_by_count int or a structured dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, int):
        return {"cited_by_count": value}
    return {}


def fetch_openalex(dois: list[str], email: str = "") -> dict[str, dict]:
    """Return ``{doi: {cited_by_count, venue_impact, venue}}`` from OpenAlex.

    Two best-effort calls: works (citation counts + host venue) then sources
    (per-venue 2yr_mean_citedness). Returns an empty dict on error so callers can
    treat enrichment as optional.
    """
    if not dois:
        return {}
    import httpx

    out: dict[str, dict] = {}
    source_ids: set[str] = set()
    with httpx.Client(timeout=20.0) as client:
        # 1. Works: citation counts + host venue id/name.
        for i in range(0, len(dois), 50):
            batch = dois[i : i + 50]
            params = {
                "filter": "doi:" + "|".join(batch),
                "select": "doi,cited_by_count,primary_location",
                "per-page": "50",
            }
            if email:
                params["mailto"] = email
            try:
                data = _get_json_bounded(client, _OPENALEX_URL, params)
            except Exception as e:  # noqa: BLE001
                logger.warning("OpenAlex works fetch failed: %s", _redact(str(e)))
                continue
            for work in (data.get("results") or [])[:_MAX_RESULTS_PER_PAGE]:
                doi = (work.get("doi") or "").lower().replace("https://doi.org/", "")
                if not doi:
                    continue
                source = ((work.get("primary_location") or {}).get("source") or {})
                source_id = source.get("id")
                if source_id:
                    source_ids.add(source_id)
                out[doi] = {
                    "cited_by_count": work.get("cited_by_count"),
                    "venue": source.get("display_name"),
                    "venue_source_id": source_id,
                    "venue_impact": None,
                }

        # 2. Sources: 2yr_mean_citedness per venue.
        impact_by_source: dict[str, float] = {}
        ids = list(source_ids)
        for i in range(0, len(ids), 50):
            batch = [sid.rsplit("/", 1)[-1] for sid in ids[i : i + 50]]
            params = {
                "filter": "ids.openalex:" + "|".join(batch),
                "select": "id,summary_stats",
                "per-page": "50",
            }
            if email:
                params["mailto"] = email
            try:
                data = _get_json_bounded(client, _OPENALEX_SOURCES_URL, params)
            except Exception as e:  # noqa: BLE001
                logger.warning("OpenAlex sources fetch failed: %s", _redact(str(e)))
                continue
            for src in (data.get("results") or [])[:_MAX_RESULTS_PER_PAGE]:
                sid = src.get("id")
                mc = (src.get("summary_stats") or {}).get("2yr_mean_citedness")
                # Clamp the untrusted upstream value: reject non-finite and bound
                # the magnitude so a hostile/erroneous feed can't drive the
                # venue-impact gate to extremes (CWE-345).
                if sid and isinstance(mc, (int, float)) and math.isfinite(mc):
                    impact_by_source[sid] = min(max(float(mc), 0.0), _MAX_VENUE_MEAN_CITEDNESS)

    for doi, entry in out.items():
        sid = entry.get("venue_source_id")
        if sid and sid in impact_by_source:
            entry["venue_impact"] = impact_by_source[sid]
    return out


# Persisted enrichment older than this is considered stale and re-fetched.
_ENRICHMENT_TTL_DAYS = 14


def load_cached_enrichment(
    item_ids: list[int], max_age_days: int = _ENRICHMENT_TTL_DAYS
) -> dict[int, dict]:
    """Return ``{item_id: {cited_by_count, venue_impact, venue}}`` for fresh rows."""
    if not item_ids:
        return {}
    from sqlalchemy import select

    from ..store import ItemEnrichmentRow, init_db, session_scope

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    init_db()
    out: dict[int, dict] = {}
    with session_scope() as s:
        rows = (
            s.execute(
                select(ItemEnrichmentRow).where(ItemEnrichmentRow.item_id.in_(item_ids))
            )
            .scalars()
            .all()
        )
        for r in rows:
            fetched_at = r.fetched_at
            if fetched_at is not None:
                ref = fetched_at if fetched_at.tzinfo else fetched_at.replace(tzinfo=timezone.utc)
                if ref < cutoff:
                    continue  # stale
            out[int(r.item_id)] = {
                "cited_by_count": r.cited_by_count,
                "venue_impact": r.venue_impact,
                "venue": r.venue,
            }
    return out


def save_enrichment(entries: dict[int, dict]) -> None:
    """Upsert enrichment rows keyed by item id."""
    if not entries:
        return
    from ..store import ItemEnrichmentRow, init_db, session_scope

    init_db()
    now = datetime.now(timezone.utc)
    with session_scope() as s:
        for item_id, entry in entries.items():
            row = s.get(ItemEnrichmentRow, int(item_id))
            if row is None:
                row = ItemEnrichmentRow(item_id=int(item_id))
                s.add(row)
            row.cited_by_count = entry.get("cited_by_count")
            row.venue_impact = entry.get("venue_impact")
            row.venue = entry.get("venue")
            row.fetched_at = now


def enrich_scored(
    scored: list[tuple[ItemRow, float]],
    *,
    settings: object | None = None,
    fetcher=None,
    now: datetime | None = None,
) -> list[tuple[ItemRow, float]]:
    """Apply citation-velocity and venue-impact adjustments when enabled.

    Citation velocity boosts fast-rising work; venue impact rewards high-impact
    journals and penalizes low-impact ones — which is the only quality signal
    available for aggregator (OpenAlex/PubMed) items whose configured source name
    hides the real publication venue.
    """
    if not scored:
        return scored
    if settings is None:
        from ..config import get_settings

        settings = get_settings()
    if not getattr(settings, "citation_enrichment", False):
        return scored

    doi_by_idx: dict[int, str] = {}
    iid_by_idx: dict[int, int] = {}
    for idx, (row, _score) in enumerate(scored):
        doi = derive_doi(row)
        if doi:
            doi_by_idx[idx] = doi
        iid = getattr(row, "id", None)
        if isinstance(iid, int):
            iid_by_idx[idx] = iid
    if not doi_by_idx:
        return scored

    # Reuse persisted enrichment so we don't re-hit OpenAlex for known papers and
    # still have data on a day the fetch fails.
    cached = load_cached_enrichment(list(set(iid_by_idx.values())))
    needed = {
        doi_by_idx[idx]
        for idx in doi_by_idx
        if iid_by_idx.get(idx) not in cached  # uncached (or item has no id)
    }
    fetched: dict[str, dict] = {}
    if needed:
        fetch = fetcher or fetch_openalex
        try:
            raw = fetch(sorted(needed), getattr(settings, "citation_polite_email", "") or "")
            fetched = {doi: _normalize_entry(value) for doi, value in raw.items()}
        except Exception as e:  # noqa: BLE001
            logger.warning("citation enrichment fetch failed (%s); using cache only", e)
        if fetched:
            to_save = {
                iid_by_idx[idx]: fetched[doi_by_idx[idx]]
                for idx in doi_by_idx
                if idx in iid_by_idx and doi_by_idx[idx] in fetched
            }
            if to_save:
                try:
                    save_enrichment(to_save)
                except Exception as e:  # noqa: BLE001
                    logger.warning("could not persist enrichment: %s", e)

    if not cached and not fetched:
        return scored

    def _entry_for(idx: int) -> dict | None:
        iid = iid_by_idx.get(idx)
        if iid is not None and iid in cached:
            return cached[iid]
        doi = doi_by_idx.get(idx)
        return fetched.get(doi) if doi else None

    venue_w = float(getattr(settings, "venue_quality_weight", 0.18))
    boosted: list[tuple[ItemRow, float]] = []
    for idx, (row, score) in enumerate(scored):
        entry = _entry_for(idx)
        if entry:
            score = float(score)
            cs = citation_score(
                entry.get("cited_by_count"), getattr(row, "published_at", None), now=now
            )
            score += CITATION_BOOST * cs
            vq = venue_quality_score(entry.get("venue_impact"))
            if vq is not None:
                # Center at 0.5: high-impact venues gain, low-impact venues lose.
                score += venue_w * (vq - 0.5)
                # Flag genuinely low-impact venues so the selection-stage
                # frequency cap treats them as low_impact_journal even though
                # their configured source (e.g. OpenAlex) hides the real venue.
                if vq < _LOW_VENUE_QUALITY:
                    # Transient (non-persisted) marker read by source_bucket's
                    # low-impact frequency cap. Setting a non-column attribute on
                    # a mapped instance is safe; no swallowing so a real failure
                    # (which would silently disable low-impact gating) surfaces.
                    row.venue_low_impact = True
        boosted.append((row, score))
    boosted.sort(key=lambda t: t[1], reverse=True)
    return boosted
