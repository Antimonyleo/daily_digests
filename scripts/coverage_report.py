#!/usr/bin/env python
"""READ-ONLY per-interest coverage measurement harness.

This script measures, over the last N *sent* digests, how well the research
section covers each of the reader's core interests and how much off-field
content leaks in. It answers the user's five finite success criteria:

  1. Per-interest coverage  — when a qualified candidate for an interest exists
     in the pool, does a selected item actually cover that interest?
  2. Unattributed selections — how many selected research items per digest lack
     a persisted, qualified canonical-facet attribution?  This is *not*
     semantic precision: attribution comes from the production ranker and must
     not be presented as an independent judgement of topical relevance.
  3. High-profile omissions  — strong, on-interest, high-venue candidates that
     were available but not shown.
  4. Supply responsiveness   — selected count vs qualified-candidate count per
     digest (quiet days should shrink, not pad).
  5. Vote-based precision    — overall pairwise/precision via the existing
     read-only ``evaluate_history`` (skipped cleanly if unavailable).

CRITICAL — this harness is strictly READ-ONLY. It NEVER calls any model
``.fit()``, NEVER writes ``lr_ranker.npz`` / ``calibrator.json``, and NEVER
mutates the database. It only issues SELECTs and reads config. A prior bug had
eval scripts overwriting the production model; this script must not repeat it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

# Make ``src`` importable when run directly (uv run python scripts/coverage_report.py).
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dailydigest.config import get_settings, load_profile  # noqa: E402
from dailydigest.store import (  # noqa: E402
    DigestRow,
    ImpressionRow,
    ItemEnrichmentRow,
    ItemRow,
)
from dailydigest.rank.source_quality import source_bucket  # noqa: E402

RESEARCH_SECTION = "research"

# source_bucket values that indicate a top/high venue (see rank.source_quality).
# A publication database (e.g. PubMed) is not itself evidence of a high-profile
# venue.  An actual top/high/strong journal maps to ``published_journal``.
HIGH_VENUE_BUCKETS = {"published_journal"}


class _VenueProbe:
    """Minimal source-quality input for a persisted source or journal name."""

    section = RESEARCH_SECTION

    def __init__(self, source: str):
        self.source = source


# --------------------------------------------------------------------------- #
# Read-only data access
# --------------------------------------------------------------------------- #
@contextmanager
def session_scope():
    """Yield a SQLite session opened read-only, without schema/pragma writes."""
    db_path = Path(get_settings().db_path)
    if not db_path.exists():
        raise OperationalError("coverage database does not exist", None, None)
    # `mode=ro` prohibits both database creation and journal/WAL changes. Use a
    # local engine rather than the application's write-capable store engine.
    url = f"sqlite+pysqlite:///file:{quote(str(db_path), safe='/')}?mode=ro&uri=true"
    engine = create_engine(url, future=True)
    session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _sent_digest_ids(limit: int, include_unsent: bool = False) -> list[str]:
    """Return up to ``limit`` most-recently SENT digest ids (newest first).

    Read-only SELECT; falls back to created_at ordering only for tie-breaking.
    When ``include_unsent`` is True the ``sent_at`` filter is dropped and digests
    are ordered by ``created_at`` — useful for measuring dry-run/preview brews
    (which never set ``sent_at``) during offline validation.
    """
    try:
        with session_scope() as s:
            q = s.query(DigestRow.id)
            if include_unsent:
                q = q.order_by(DigestRow.created_at.desc())
            else:
                q = q.filter(DigestRow.sent_at.isnot(None)).order_by(
                    DigestRow.sent_at.desc(), DigestRow.created_at.desc()
                )
            rows = q.limit(limit).all()
    except OperationalError:
        # A genuinely fresh database has no schema.  Do not call ``init_db``:
        # this command is a report, not a schema-management operation.
        return []
    return [r[0] for r in rows]


def _latest_run_id(digest_id: str) -> str | None:
    """The most-recent brew run for a digest (a rebrew mints a new run_id)."""
    try:
        with session_scope() as s:
            run_id = (
                s.query(ImpressionRow.run_id)
                .filter(ImpressionRow.digest_id == digest_id)
                .order_by(ImpressionRow.created_at.desc(), ImpressionRow.id.desc())
                .limit(1)
                .scalar()
            )
    except OperationalError:
        # A partially migrated database is still read-only here; report no
        # candidate pool instead of invoking schema setup or crashing.
        return None
    return run_id


def _pool_rows(digest_id: str, run_id: str) -> list[dict]:
    """Return the research candidate pool for one run, enriched with attribution.

    Attribution comes DIRECTLY from the ``impressions`` table (research section,
    both selected AND unselected rows).  Actual source/venue comes from the
    associated item/enrichment records for both populations; feature JSON is
    selected-only and must not be used to measure omissions.
    """
    with session_scope() as s:
        # The report must work against old immutable logs after a newer ORM adds
        # ``primary_facet_score``.  Inspecting SQLite's schema is read-only and
        # avoids selecting a not-yet-migrated column.
        columns = {
            str(row[1])
            for row in s.connection().exec_driver_sql("PRAGMA table_info(impressions)")
        }
        score_column = getattr(ImpressionRow, "primary_facet_score", None)
        has_primary_facet_score = bool(
            score_column is not None and "primary_facet_score" in columns
        )
        selected_columns = [
            ImpressionRow.item_id,
            ImpressionRow.selected,
            ImpressionRow.final_score,
            ImpressionRow.primary_facet,
            ImpressionRow.topic_score,
        ]
        if has_primary_facet_score:
            selected_columns.append(score_column)
        impressions = (
            s.query(*selected_columns, ItemRow.source, ItemEnrichmentRow.venue)
            .outerjoin(ItemRow, ItemRow.id == ImpressionRow.item_id)
            .outerjoin(ItemEnrichmentRow, ItemEnrichmentRow.item_id == ImpressionRow.item_id)
            .filter(
                ImpressionRow.digest_id == digest_id,
                ImpressionRow.run_id == run_id,
                ImpressionRow.section == RESEARCH_SECTION,
            )
            .all()
        )
    pool: list[dict] = []
    for row in impressions:
        item_id, selected, final_score, primary_facet, topic_score = row[:5]
        offset = 5
        primary_facet_score = row[offset] if has_primary_facet_score else None
        if has_primary_facet_score:
            offset += 1
        source, venue = row[offset:offset + 2]
        # ``primary_facet_score`` was added after the first impression schema.
        # Prefer it whenever present; old rows have only the overall topic score.
        score_from_legacy_topic = primary_facet_score is None
        if primary_facet_score is None:
            primary_facet_score = topic_score
        pool.append(
            {
                "item_id": int(item_id),
                "selected": bool(selected),
                "impression_score": float(final_score or 0.0),
                "primary_facet": str(primary_facet or ""),
                "primary_facet_score": float(primary_facet_score or 0.0),
                "facet_score_source": "topic_score" if score_from_legacy_topic else "primary_facet_score",
                "topic_score": float(topic_score or 0.0),
                "final_score": float(final_score or 0.0),
                "source": str(source or ""),
                "venue": str(venue or ""),
            }
        )
    return pool


# --------------------------------------------------------------------------- #
# Report computation
# --------------------------------------------------------------------------- #
def compute_report(n: int = 10, include_unsent: bool = False) -> dict:
    """Compute the coverage/precision report over the last ``n`` sent digests.

    Returns a dict keyed for machine assertion (see module docstring for the
    five criteria). Never writes anything.
    """
    settings = get_settings()
    min_topic_relevance = float(settings.min_topic_relevance)

    try:
        profile = load_profile()
        interests = _canonical_interests(profile)
    except Exception as exc:  # profile missing/invalid — degrade gracefully.
        interests = []
        _profile_error = str(exc)
    else:
        _profile_error = ""

    # Opening SQLite through the application's normal engine would create a
    # missing file (and enable WAL), which is a write. A report over a fresh
    # path therefore returns insufficient data without opening a connection.
    db_exists = Path(settings.db_path).exists()
    digest_ids = _sent_digest_ids(n, include_unsent=include_unsent) if db_exists else []

    report: dict = {
        "n_requested": n,
        "n_digests": len(digest_ids),
        "digest_ids": digest_ids,
        "min_topic_relevance": min_topic_relevance,
        "profile_error": _profile_error,
        # criterion 1
        "per_interest_coverage": {},
        "routinely_absent": [],
        # criterion 2.  This is attribution completeness, not semantic precision.
        "unattributed_selected_per_digest": None,
        "unattributed_selected_target_met": None,
        # criterion 3
        "high_profile_omissions": 0,
        "high_profile_omissions_per_digest": None,
        # criterion 4
        "supply_responsiveness": [],
        # criterion 5
        "vote_eval": None,
        "per_digest": [],
    }

    if not digest_ids:
        report["status"] = "insufficient data: no sent digests found"
        report["watched_interests"] = _watched_status({}, interests)
        report["vote_eval"] = _vote_eval()
        return report

    # Accumulators.
    # coverage[interest] = [eligible_digests, covered_digests]
    coverage: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    unattributed_counts: list[int] = []
    total_omissions = 0
    supply: list[dict] = []
    per_digest: list[dict] = []

    for digest_id in digest_ids:
        run_id = _latest_run_id(digest_id)
        if run_id is None:
            per_digest.append({"digest_id": digest_id, "status": "no impressions"})
            continue
        pool = _pool_rows(digest_id, run_id)
        if not pool:
            per_digest.append({"digest_id": digest_id, "status": "empty research pool"})
            continue

        selected = [r for r in pool if r["selected"]]
        qualified = [
            r
            for r in pool
            if _is_qualified_canonical_attribution(r, interests, min_topic_relevance)
        ]

        # --- criterion 1: per-interest coverage --------------------------- #
        # An interest is "eligible" in this digest if a QUALIFIED candidate for
        # it exists in the pool; "covered" if a SELECTED item has that facet.
        selected_facets = {
            _canonical_facet_name(r, interests)
            for r in selected
            if _is_qualified_canonical_attribution(r, interests, min_topic_relevance)
        }
        qualified_facets = {_canonical_facet_name(r, interests) for r in qualified}
        for interest in interests:
            if interest in qualified_facets:
                coverage[interest][0] += 1
                if interest in selected_facets:
                    coverage[interest][1] += 1

        # --- criterion 2: persisted-attribution completeness -------------- #
        # It would be circular to call a configured primary facet proof that a
        # paper is semantically on-field.  Report only what the log can prove:
        # whether selected items lacked a qualified canonical attribution.
        unattributed = sum(
            1
            for r in selected
            if not _is_qualified_canonical_attribution(r, interests, min_topic_relevance)
        )
        unattributed_counts.append(unattributed)

        # --- criterion 3: high-profile omissions -------------------------- #
        omissions = [
            r
            for r in pool
            if not r["selected"]
            and _is_qualified_canonical_attribution(r, interests, min_topic_relevance)
            and _is_high_venue(r)
        ]
        total_omissions += len(omissions)

        # --- criterion 4: supply responsiveness --------------------------- #
        supply.append(
            {
                "digest_id": digest_id,
                "n_selected": len(selected),
                "n_qualified": len(qualified),
                "pool_size": len(pool),
            }
        )

        per_digest.append(
            {
                "digest_id": digest_id,
                "pool_size": len(pool),
                "n_selected": len(selected),
                "n_qualified": len(qualified),
                "unattributed_selected": unattributed,
                "high_profile_omissions": len(omissions),
                "selected_facets": sorted(f for f in selected_facets if f),
            }
        )

    # Finalize criterion 1.
    per_interest = {}
    routinely_absent = []
    for interest in interests:
        eligible, covered = coverage[interest]
        frac = (covered / eligible) if eligible else None
        per_interest[interest] = {
            "eligible_digests": eligible,
            "covered_digests": covered,
            "coverage_fraction": frac,
        }
        # "Routinely absent": candidates existed but selected in <50% of eligible.
        if eligible > 0 and frac is not None and frac < 0.5:
            routinely_absent.append(interest)
    # Always surface the watched interests' status explicitly.
    watched_status = _watched_status(per_interest, interests)

    n_scored = len(unattributed_counts)
    unattributed_mean = (sum(unattributed_counts) / n_scored) if n_scored else None

    report["per_interest_coverage"] = per_interest
    report["routinely_absent"] = routinely_absent
    report["watched_interests"] = watched_status
    report["unattributed_selected_per_digest"] = unattributed_mean
    report["unattributed_selected_target_met"] = (
        (unattributed_mean < 1.0) if unattributed_mean is not None else None
    )
    report["high_profile_omissions"] = total_omissions
    report["high_profile_omissions_per_digest"] = (
        (total_omissions / n_scored) if n_scored else None
    )
    report["supply_responsiveness"] = supply
    report["per_digest"] = per_digest
    report["vote_eval"] = _vote_eval()
    report["status"] = "ok" if n_scored else "insufficient data: no scorable digests"
    return report


def _watched_status(per_interest: dict, interests: list[str] | None = None) -> dict:
    """Return an explicit coverage status for every configured user interest."""
    watched_status = {}
    for interest in interests or []:
        info = per_interest.get(interest)
        if info is None:
            watched_status[interest] = "no data in window"
        elif info["eligible_digests"] == 0:
            watched_status[interest] = "no qualified candidates in window"
        else:
            watched_status[interest] = (
                f"{info['covered_digests']}/{info['eligible_digests']} eligible digests covered"
            )
    return watched_status


def _canonical_interests(profile) -> list[str]:
    """Return configured canonical facets, with a legacy-keyword fallback."""
    canonical_facets = getattr(profile, "canonical_facets", None)
    if isinstance(canonical_facets, dict) and canonical_facets:
        return list(canonical_facets)
    canonical = getattr(profile, "canonical_interests", None)
    if isinstance(canonical, dict) and canonical:
        return list(canonical)
    if isinstance(canonical, (list, tuple)):
        return [str(v) for v in canonical]
    return list(getattr(profile, "keywords", []) or [])


def _is_qualified_canonical_attribution(
    row: dict, interests: list[str], min_topic_relevance: float
) -> bool:
    """Whether the log proves a qualified attribution to a canonical facet."""
    facet = str(row.get("primary_facet") or "").casefold()
    canonical = {str(interest).casefold() for interest in interests}
    return (
        bool(facet)
        and facet in canonical
        and float(row.get("primary_facet_score") or 0.0) >= min_topic_relevance
    )


def _canonical_facet_name(row: dict, interests: list[str]) -> str:
    """Return the configured spelling of a row's canonical facet, if any."""
    facet = str(row.get("primary_facet") or "").casefold()
    return next(
        (str(interest) for interest in interests if str(interest).casefold() == facet),
        "",
    )


def _is_high_venue(row: dict) -> bool:
    """High-profile only when the candidate's actual source or venue proves it."""
    # Topic-search aggregators often preserve the actual journal only in
    # item_enrichment.  Prefer it; do not substitute a ranking score for venue.
    source = row.get("venue") or row.get("source")
    if not source:
        return False
    return source_bucket(_VenueProbe(str(source))) in HIGH_VENUE_BUCKETS


def _vote_eval() -> dict | None:
    """Explain why the existing evaluator is intentionally not called here."""
    # ``evaluate_history`` currently calls ``init_db``.  Keeping this script
    # SELECT-only means it cannot be called from here until its own schema setup
    # is separated from replay.  State that limitation rather than mutating a DB.
    return {"error": "skipped: evaluate_history initializes schema; this report is SELECT-only"}


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #
def _fmt(x, nd: int = 2) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def print_report(report: dict) -> None:
    print("=" * 70)
    print("DailyDigest coverage report (READ-ONLY)")
    print("=" * 70)
    print(
        f"Window: last {report['n_requested']} sent digests "
        f"(found {report['n_digests']}). "
        f"Topic floor = {report['min_topic_relevance']:.2f}."
    )
    if report.get("profile_error"):
        print(f"  WARNING: profile load failed: {report['profile_error']}")
    if report.get("status", "").startswith("insufficient"):
        print(f"\n{report['status'].upper()}")

    # --- Criterion 1 -----------------------------------------------------
    print("\n[1] PER-INTEREST COVERAGE "
          "(selected among eligible digests where a qualified candidate existed)")
    pic = report.get("per_interest_coverage") or {}
    if not pic:
        print("    insufficient data (no configured interests / no digests)")
    else:
        for interest, info in sorted(pic.items()):
            elig = info["eligible_digests"]
            if elig == 0:
                print(f"    - {interest}: no qualified candidates in window")
                continue
            frac = info["coverage_fraction"]
            flag = "  <-- ROUTINELY ABSENT" if (frac is not None and frac < 0.5) else ""
            print(
                f"    - {interest}: {info['covered_digests']}/{elig} "
                f"({_fmt(frac)}){flag}"
            )
    watched = report.get("watched_interests") or {}
    if watched:
        print("    Configured interest status:")
        for interest, status in watched.items():
            print(f"      * {interest}: {status}")
    if report.get("routinely_absent"):
        print(f"    FAIL flags (routinely absent): {report['routinely_absent']}")

    # --- Criterion 2 -----------------------------------------------------
    print("\n[2] UNATTRIBUTED SELECTED (not semantic off-field precision)")
    print("    mean selected research items without a qualified canonical attribution / digest")
    off = report.get("unattributed_selected_per_digest")
    met = report.get("unattributed_selected_target_met")
    verdict = "PASS" if met else ("FAIL" if met is False else "n/a")
    print(f"    mean = {_fmt(off)} per digest  (target < 1.0)  -> {verdict}")

    # --- Criterion 3 -----------------------------------------------------
    print("\n[3] HIGH-PROFILE OMISSIONS "
          "(on-interest + high-venue candidates available but NOT shown)")
    print(
        f"    total = {report.get('high_profile_omissions', 0)}  "
        f"(mean {_fmt(report.get('high_profile_omissions_per_digest'))}/digest)"
    )

    # --- Criterion 4 -----------------------------------------------------
    print("\n[4] SUPPLY RESPONSIVENESS (selected vs qualified per digest; "
          "quiet days should not be padded)")
    supply = report.get("supply_responsiveness") or []
    if not supply:
        print("    insufficient data")
    else:
        for row in supply:
            print(
                f"    - {row['digest_id']}: selected={row['n_selected']} "
                f"qualified={row['n_qualified']} pool={row['pool_size']}"
            )

    # --- Criterion 5 -----------------------------------------------------
    print("\n[5] VOTE-BASED PRECISION (overall, from evaluate_history — read-only)")
    ve = report.get("vote_eval")
    if not ve or "error" in (ve or {}):
        print(f"    {ve.get('error') if ve else 'insufficient data'}")
    else:
        print(
            f"    digests_scored={ve.get('digests_scored')} "
            f"votes={ve.get('votes_used')} "
            f"pairwise={_fmt(ve.get('pairwise_accuracy'), 3)} "
            f"P@k={_fmt(ve.get('precision_at_k'), 3)} "
            f"MAP={_fmt(ve.get('map'), 3)}"
        )
    print("=" * 70)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n", "--num-digests", type=int, default=10,
        help="Number of most-recent SENT digests to analyze (default 10).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the raw report dict as JSON."
    )
    parser.add_argument(
        "--include-unsent", action="store_true",
        help="Include dry-run/preview digests (no sent_at) — for offline validation.",
    )
    args = parser.parse_args(argv)
    report = compute_report(n=args.num_digests, include_unsent=args.include_unsent)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
