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

import json
import logging
import math
import re
from datetime import datetime, timedelta, timezone

from ..store import ItemRow
from .source_quality import (
    is_high_quality_journal_source, publication_date_is_known, source_bucket,
)

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
# Venue-quality score at or above this marks the item's venue as a verified
# high-impact journal, exempting it from the low-impact gate even when its name
# is not on the curated venue lists. 0.85 ~ 2yr_mean_citedness of ~5.5, above
# the mega-journal band (PLOS ONE ~3, Scientific Reports ~4, most MDPI/Frontiers
# titles ~4-5) that the curated lists deliberately leave under the cap.
_HIGH_VENUE_QUALITY = 0.85
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


def _results(data):
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("OpenAlex response lacks a results list")
    rows = data["results"]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("OpenAlex results must be objects")
    return rows[:_MAX_RESULTS_PER_PAGE]


def derive_doi(row: object) -> str | None:
    """Best-effort DOI extraction from an item's url / external_id."""
    for attr in ("url", "external_id"):
        value = getattr(row, attr, None)
        if isinstance(value, str) and value:
            m = _DOI_RE.search(value)
            if m:
                return m.group(0).rstrip(").").lower()
    metadata = getattr(row, "metadata", None)
    if not isinstance(metadata, dict):
        raw = getattr(row, "metadata_json", "")
        try:
            metadata = json.loads(raw) if isinstance(raw, str) and raw else {}
        except (TypeError, json.JSONDecodeError):
            metadata = {}
    value = metadata.get("doi") if isinstance(metadata, dict) else None
    if isinstance(value, str):
        match = _DOI_RE.search(value)
        if match:
            return match.group(0).rstrip(").").lower()
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
        if ref > now:
            return 0.0
    else:
        return 0.0  # No publication age means no defensible velocity estimate.
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

    Per-DOI status distinguishes healthy absence from works/source outages.
    Partial observations must not refresh a previously complete cache row.
    """
    if not dois:
        return {}
    import httpx

    out: dict[str, dict] = {}
    source_ids: set[str] = set()
    failed_sources: set[str] = set()
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
                works = _results(_get_json_bounded(client, _OPENALEX_URL, params))
                if any(not isinstance(work.get("doi"), str) for work in works):
                    raise ValueError("OpenAlex work lacks a DOI")
            except Exception as e:  # noqa: BLE001
                logger.warning("OpenAlex works fetch failed: %s", _redact(str(e)))
                out.update({doi: {"status": "works_failed"} for doi in batch})
                continue
            observed_at = datetime.now(timezone.utc).isoformat()
            out.update({doi: {"status": "missing", "observed_at": observed_at} for doi in batch})
            for work in works:
                doi = (work.get("doi") or "").lower().replace("https://doi.org/", "")
                if doi not in batch:
                    continue
                location = work.get("primary_location") or {}
                source = location.get("source") or {} if isinstance(location, dict) else None
                count = work.get("cited_by_count")
                if not isinstance(source, dict) or type(count) is not int or count < 0:
                    out[doi] = {"status": "works_failed"}
                    continue
                source_id = source.get("id")
                if source_id is not None and not isinstance(source_id, str):
                    out[doi] = {"status": "works_failed"}
                    continue
                if source_id:
                    source_ids.add(source_id)
                out[doi] = {
                    "cited_by_count": work.get("cited_by_count"),
                    "venue": source.get("display_name"),
                    "venue_source_id": source_id,
                    "venue_impact": None,
                    "status": "fresh" if source_id else "venue_missing", "observed_at": observed_at,
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
                sources = _results(_get_json_bounded(client, _OPENALEX_SOURCES_URL, params))
                if any(not isinstance(src.get("id"), str)
                       or not isinstance(src.get("summary_stats") or {}, dict) for src in sources):
                    raise ValueError("OpenAlex source has malformed summary statistics")
            except Exception as e:  # noqa: BLE001
                logger.warning("OpenAlex sources fetch failed: %s", _redact(str(e)))
                failed_sources.update(ids[i : i + 50])
                continue
            for src in sources:
                sid = src.get("id")
                mc = (src.get("summary_stats") or {}).get("2yr_mean_citedness")
                # Clamp the untrusted upstream value: reject non-finite and bound
                # the magnitude so a hostile/erroneous feed can't drive the
                # venue-impact gate to extremes (CWE-345).
                if sid and isinstance(mc, (int, float)) and math.isfinite(mc):
                    impact_by_source[sid] = min(max(float(mc), 0.0), _MAX_VENUE_MEAN_CITEDNESS)

    for entry in out.values():
        sid = entry.get("venue_source_id")
        if sid and sid in impact_by_source:
            entry["venue_impact"] = impact_by_source[sid]
        if sid in failed_sources:
            entry["status"] = "sources_failed"
        elif sid and sid not in impact_by_source:
            entry["status"] = "sources_missing"
    return out


# Persisted enrichment older than this is considered stale and re-fetched.
_ENRICHMENT_TTL_DAYS = 14
_ENRICHMENT_STALE_MAX_DAYS = 30


def load_cached_enrichment(
    item_ids: list[int], max_age_days: int = _ENRICHMENT_STALE_MAX_DAYS
) -> dict[int, dict]:
    """Return original observations; only <=max_age_days can be fresh/stale.

    Older rows remain explicit expired evidence and cannot affect serving.
    """
    if not item_ids:
        return {}
    from sqlalchemy import select

    from ..store import ItemEnrichmentRow, init_db, session_scope

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max_age_days)
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
            if fetched_at is None:
                continue  # Unknown age cannot establish freshness or a bounded veto.
            ref = fetched_at if fetched_at.tzinfo else fetched_at.replace(tzinfo=timezone.utc)
            if ref > now:
                continue
            out[int(r.item_id)] = {
                "cited_by_count": r.cited_by_count,
                "venue_impact": r.venue_impact,
                "venue": r.venue,
                "observed_at": ref.isoformat(),
                "status": ("expired" if ref < cutoff else "stale"
                           if ref < now - timedelta(days=_ENRICHMENT_TTL_DAYS) else "fresh"),
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
            if entry.get("status", "fresh") != "fresh":
                continue
            row = s.get(ItemEnrichmentRow, int(item_id))
            if row is None:
                row = ItemEnrichmentRow(item_id=int(item_id))
                s.add(row)
            row.cited_by_count = entry.get("cited_by_count")
            row.venue_impact = entry.get("venue_impact")
            row.venue = entry.get("venue")
            observed_at = entry.get("observed_at")
            row.fetched_at = datetime.fromisoformat(observed_at) if observed_at else now


def _load_enrichment(rows, *, settings, fetcher=None) -> dict[int, dict]:
    """Resolve one batch by row position, reusing fresh cache entries first."""
    doi_by_idx: dict[int, str] = {}
    iid_by_idx: dict[int, int] = {}
    for idx, row in enumerate(rows):
        doi = derive_doi(row)
        if doi:
            doi_by_idx[idx] = doi
        iid = getattr(row, "id", None)
        if isinstance(iid, int):
            iid_by_idx[idx] = iid
    if not doi_by_idx and not iid_by_idx:
        return {}

    # Reuse persisted enrichment so we don't re-hit OpenAlex for known papers and
    # still have data on a day the fetch fails.
    cached = load_cached_enrichment(list(set(iid_by_idx.values())))
    needed = {
        doi_by_idx[idx]
        for idx in doi_by_idx
        if cached.get(iid_by_idx.get(idx), {}).get("status", "fresh") != "fresh"
        or iid_by_idx.get(idx) not in cached
    }
    fetched: dict[str, dict] = {}
    if needed:
        fetch = fetcher or fetch_openalex
        try:
            raw = fetch(sorted(needed), getattr(settings, "citation_polite_email", "") or "")
            fetched = {doi: _normalize_entry(value) for doi, value in raw.items()}
            for doi in needed:
                entry = fetched.setdefault(doi, {"status": "missing"})
                entry.setdefault("status", "fresh")
                if entry["status"] in {"fresh", "missing"}:
                    entry.setdefault("observed_at", datetime.now(timezone.utc).isoformat())
        except Exception as e:  # noqa: BLE001
            logger.warning("citation enrichment fetch failed (%s); using cache only", e)
            fetched = {doi: {"status": "works_failed"} for doi in needed}
        if fetched:
            # Legacy injected fetchers may omit status or fields. They remain
            # supported, but a partial refresh cannot erase known observations.
            for idx, doi in doi_by_idx.items():
                prior = cached.get(iid_by_idx.get(idx))
                update = fetched.get(doi)
                if not prior or not update or update.get("status") != "fresh":
                    continue
                for field in ("cited_by_count", "venue_impact", "venue"):
                    if prior.get(field) not in (None, "") and update.get(field) in (None, ""):
                        update["status"] = "works_failed" if field == "cited_by_count" else "sources_missing"
                        break
            to_save = {
                iid_by_idx[idx]: fetched[doi_by_idx[idx]]
                for idx in doi_by_idx
                if idx in iid_by_idx and doi_by_idx[idx] in fetched
                and fetched[doi_by_idx[idx]].get("status") == "fresh"
            }
            if to_save:
                try:
                    save_enrichment(to_save)
                except Exception as e:  # noqa: BLE001
                    logger.warning("could not persist enrichment: %s", e)

    resolved = {}
    for idx in range(len(rows)):
        prior = cached.get(iid_by_idx.get(idx))
        update = fetched.get(doi_by_idx.get(idx))
        if update and update.get("status") == "fresh":
            resolved[idx] = update
        elif prior:
            resolved[idx] = dict(prior)
            if update:
                resolved[idx]["refresh_status"] = update["status"]
        elif update:
            resolved[idx] = update
    return resolved


def _apply_venue_evidence(row, entry) -> None:
    # These transient attributes are owned by enrichment. Reused row objects
    # must not retain an earlier high-venue admission after its evidence expires.
    row.venue_name = ""
    row.venue_high_impact = False
    row.venue_low_impact = False
    status = entry.get("status", "fresh")
    if status != "fresh":
        # A bounded stale low observation may conservatively veto promotion;
        # stale venue names themselves must not create a curated-journal prior.
        quality = venue_quality_score(entry.get("venue_impact"))
        if status == "stale" and quality is not None and quality < _LOW_VENUE_QUALITY:
            row.venue_low_impact = True
            row.venue_high_impact = False
        return
    venue = str(entry.get("venue") or "").strip()
    if venue:
        row.venue_name = venue
    vq = venue_quality_score(entry.get("venue_impact"))
    if vq is not None:
        row.venue_low_impact = vq < _LOW_VENUE_QUALITY
        row.venue_high_impact = vq >= _HIGH_VENUE_QUALITY


def prepare_enrichment(rows, *, settings=None, fetcher=None) -> None:
    """Attach resolved venue evidence before quality scoring and preference fusion.

    The pipeline bounds this batch using raw topic scores. Citation adjustments
    remain separate and consume this exact evidence after eligibility filtering.
    """
    if settings is None:
        from ..config import get_settings
        settings = get_settings()
    if not rows or not getattr(settings, "citation_enrichment", False):
        return
    entries = _load_enrichment(rows, settings=settings, fetcher=fetcher)
    for idx, entry in entries.items():
        row = rows[idx]
        row.enrichment_evidence = {
            key: entry.get(key) for key in ("cited_by_count", "venue_impact", "venue", "status", "observed_at", "refresh_status")
        }
        row.enrichment_evidence["status"] = entry.get("status", "fresh")
        row.enrichment_evidence["doi"] = derive_doi(row)
        row.enrichment_evidence["quality_resolved_before_scoring"] = True
        _apply_venue_evidence(row, entry)


def enrich_scored(
    scored: list[tuple[ItemRow, float]],
    *,
    settings: object | None = None,
    fetcher=None,
    now: datetime | None = None,
    prepared_only: bool = False,
) -> list[tuple[ItemRow, float]]:
    """Apply citation-velocity and venue-impact adjustments when enabled.

    Citation velocity boosts fast-rising work. Venue impact supplies missing
    journal evidence and penalizes low-impact venues; already-curated journals
    retain their existing source prior without a second positive venue bonus.
    """
    if not scored:
        return scored
    if settings is None:
        from ..config import get_settings

        settings = get_settings()
    if not getattr(settings, "citation_enrichment", False):
        return scored

    rows = [row for row, _score in scored]
    if prepared_only:
        entries = {idx: row.enrichment_evidence for idx, row in enumerate(rows)
                   if isinstance(getattr(row, "enrichment_evidence", None), dict)}
    else:
        entries = _load_enrichment(rows, settings=settings, fetcher=fetcher)
    if not entries:
        return scored

    venue_w = float(getattr(settings, "venue_quality_weight", 0.18))
    boosted: list[tuple[ItemRow, float]] = []
    for idx, (row, score) in enumerate(scored):
        entry = entries.get(idx)
        if entry and entry.get("status", "fresh") != "fresh":
            _apply_venue_evidence(row, entry)
            row.enrichment_evidence = dict(entry) | {
                "doi": derive_doi(row), "citation_boost": 0.0, "venue_adjustment": 0.0,
                "quality_resolved_before_scoring": prepared_only,
            }
            boosted.append((row, score))
            continue
        if entry:
            score = float(score)
            # Check before enrichment reveals a venue or changes impact flags:
            # a newly identified venue may not have received the original prior.
            has_venue_prior = is_high_quality_journal_source(row)
            _apply_venue_evidence(row, entry)
            cs = citation_score(
                entry.get("cited_by_count"),
                getattr(row, "published_at", None) if publication_date_is_known(row) else None,
                now=now
            )
            citation_boost = CITATION_BOOST * cs
            score += citation_boost
            applied_venue_adjustment = 0.0
            vq = venue_quality_score(entry.get("venue_impact"))
            if vq is not None:
                # Metadata availability must not give curated journals a second
                # positive quality bonus. Contrary low-impact evidence still counts.
                venue_adjustment = venue_w * (vq - 0.5)
                low_penalty_already_applied = (
                    prepared_only and source_bucket(row) == "low_impact_journal"
                    and venue_adjustment < 0
                )
                # Prepared low-impact evidence already changed the quality leg;
                # do not penalize the same venue again after preference fusion.
                if not low_penalty_already_applied and (venue_adjustment <= 0 or not has_venue_prior):
                    score += venue_adjustment
                    applied_venue_adjustment = venue_adjustment
            row.enrichment_evidence = {
                key: entry.get(key) for key in ("cited_by_count", "venue_impact", "venue", "observed_at", "refresh_status")
            } | {"doi": derive_doi(row), "citation_boost": citation_boost,
                 "status": entry.get("status", "fresh"),
                 "venue_adjustment": applied_venue_adjustment,
                 "quality_resolved_before_scoring": prepared_only}
            score = max(0.0, min(1.0, score))
        boosted.append((row, score))
    boosted.sort(key=lambda t: t[1], reverse=True)
    return boosted
