#!/usr/bin/env python
"""READ-ONLY per-interest coverage & precision measurement harness.

This script measures, over the last N *sent* digests, how well the research
section covers each of the reader's core interests and how much off-field
content leaks in. It answers the user's five finite success criteria:

  1. Per-interest coverage  — when a qualified candidate for an interest exists
     in the pool, does a selected item actually cover that interest?
  2. Off-field precision     — how many selected research items per digest are
     off-field (no facet match, or closer to a negative interest)?
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
from pathlib import Path

# Make ``src`` importable when run directly (uv run python scripts/coverage_report.py).
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dailydigest.config import get_settings, load_profile  # noqa: E402
from dailydigest.store import (  # noqa: E402
    DigestItemFeatureRow,
    DigestRow,
    ImpressionRow,
    init_db,
    session_scope,
)

RESEARCH_SECTION = "research"

# source_bucket values that indicate a top/high venue (see rank.source_quality).
HIGH_VENUE_BUCKETS = {"published_journal", "published_database"}

# Interests we specifically flag as "routinely absent" per the success criteria.
WATCHED_INTERESTS = [
    "colloidal self-assembly",
    "DNA nanotechnology",
    "RNA nanotechnology",
]


# --------------------------------------------------------------------------- #
# Read-only data access
# --------------------------------------------------------------------------- #
def _sent_digest_ids(limit: int, include_unsent: bool = False) -> list[str]:
    """Return up to ``limit`` most-recently SENT digest ids (newest first).

    Read-only SELECT; falls back to created_at ordering only for tie-breaking.
    When ``include_unsent`` is True the ``sent_at`` filter is dropped and digests
    are ordered by ``created_at`` — useful for measuring dry-run/preview brews
    (which never set ``sent_at``) during offline validation.
    """
    with session_scope() as s:
        q = s.query(DigestRow.id)
        if include_unsent:
            q = q.order_by(DigestRow.created_at.desc())
        else:
            q = q.filter(DigestRow.sent_at.isnot(None)).order_by(
                DigestRow.sent_at.desc(), DigestRow.created_at.desc()
            )
        rows = q.limit(limit).all()
    return [r[0] for r in rows]


def _latest_run_id(digest_id: str) -> str | None:
    """The most-recent brew run for a digest (a rebrew mints a new run_id)."""
    with session_scope() as s:
        run_id = (
            s.query(ImpressionRow.run_id)
            .filter(ImpressionRow.digest_id == digest_id)
            .order_by(ImpressionRow.created_at.desc(), ImpressionRow.id.desc())
            .limit(1)
            .scalar()
        )
    return run_id


def _pool_rows(digest_id: str, run_id: str) -> list[dict]:
    """Return the research candidate pool for one run, enriched with attribution.

    Each dict carries: item_id, selected, final_score, and the per-candidate
    attribution keys ``primary_facet`` / ``topic_score`` — read DIRECTLY from the
    ``impressions`` table (research section, both selected AND unselected rows).
    This is what lets the harness attribute the UNSELECTED candidate pool:
    ``digest_item_features`` only records the ~selected slate, so unselected
    candidates have no feature JSON. ``source_bucket`` (used only for the
    high-venue heuristic) is still read from the feature JSON when present — it
    exists for selected items and is best-effort for the pool. Read-only.
    """
    with session_scope() as s:
        impressions = (
            s.query(
                ImpressionRow.item_id,
                ImpressionRow.selected,
                ImpressionRow.final_score,
                ImpressionRow.primary_facet,
                ImpressionRow.topic_score,
            )
            .filter(
                ImpressionRow.digest_id == digest_id,
                ImpressionRow.run_id == run_id,
                ImpressionRow.section == RESEARCH_SECTION,
            )
            .all()
        )
        feat_raw = (
            s.query(DigestItemFeatureRow.item_id, DigestItemFeatureRow.features_json)
            .filter(DigestItemFeatureRow.digest_id == digest_id)
            .all()
        )

    # Feature JSON is only present for the SELECTED slate; used solely as a
    # best-effort source for source_bucket (the high-venue heuristic).
    features: dict[int, dict] = {}
    for item_id, raw in feat_raw:
        try:
            payload = json.loads(raw or "{}")
        except (json.JSONDecodeError, TypeError):
            payload = {}
        if isinstance(payload, dict):
            features[int(item_id)] = payload

    pool: list[dict] = []
    for item_id, selected, final_score, primary_facet, topic_score in impressions:
        feat = features.get(int(item_id), {})
        pool.append(
            {
                "item_id": int(item_id),
                "selected": bool(selected),
                "impression_score": float(final_score) if final_score is not None else 0.0,
                # Attribution sourced from the impression row (covers unselected).
                "primary_facet": str(primary_facet or ""),
                "topic_score": float(topic_score) if topic_score is not None else 0.0,
                "final_score": float(
                    final_score if final_score is not None else 0.0
                ),
                "source_bucket": str(feat.get("source_bucket", "") or ""),
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
    init_db()  # read-only: creates schema if a fresh DB, but writes no data rows.
    settings = get_settings()
    min_topic_relevance = float(settings.min_topic_relevance)

    try:
        profile = load_profile()
        interests = list(profile.keywords)
        negative_interests = dict(profile.negative_interests)
    except Exception as exc:  # profile missing/invalid — degrade gracefully.
        interests = []
        negative_interests = {}
        _profile_error = str(exc)
    else:
        _profile_error = ""

    digest_ids = _sent_digest_ids(n, include_unsent=include_unsent)

    report: dict = {
        "n_requested": n,
        "n_digests": len(digest_ids),
        "digest_ids": digest_ids,
        "min_topic_relevance": min_topic_relevance,
        "profile_error": _profile_error,
        # criterion 1
        "per_interest_coverage": {},
        "routinely_absent": [],
        # criterion 2
        "off_field_per_digest": None,
        "off_field_target_met": None,
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
    off_field_counts: list[int] = []
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
            r for r in pool if r["topic_score"] >= min_topic_relevance
        ]

        # --- criterion 1: per-interest coverage --------------------------- #
        # An interest is "eligible" in this digest if a QUALIFIED candidate for
        # it exists in the pool; "covered" if a SELECTED item has that facet.
        selected_facets = {r["primary_facet"] for r in selected if r["primary_facet"]}
        qualified_facets = {
            r["primary_facet"]
            for r in qualified
            if r["primary_facet"]
        }
        for interest in interests:
            if interest in qualified_facets:
                coverage[interest][0] += 1
                if interest in selected_facets:
                    coverage[interest][1] += 1

        # --- criterion 2: off-field precision ----------------------------- #
        off_field = sum(
            1 for r in selected if _is_off_field(r, interests, negative_interests)
        )
        off_field_counts.append(off_field)

        # --- criterion 3: high-profile omissions -------------------------- #
        omissions = [
            r
            for r in pool
            if not r["selected"]
            and r["primary_facet"] != ""
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
                "off_field_selected": off_field,
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

    n_scored = len(off_field_counts)
    off_field_mean = (sum(off_field_counts) / n_scored) if n_scored else None

    report["per_interest_coverage"] = per_interest
    report["routinely_absent"] = routinely_absent
    report["watched_interests"] = watched_status
    report["off_field_per_digest"] = off_field_mean
    report["off_field_target_met"] = (
        (off_field_mean < 1.0) if off_field_mean is not None else None
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
    """Explicit status for the specifically-watched interests (criterion 1)."""
    configured = {i.lower() for i in (interests or [])}
    watched_status = {}
    for interest in WATCHED_INTERESTS:
        info = per_interest.get(interest)
        if info is None:
            # Distinguish a genuinely-unconfigured interest from one that is
            # configured but simply had no data in the window (empty per_interest).
            if configured and interest.lower() not in configured:
                watched_status[interest] = "not a configured core interest"
            else:
                watched_status[interest] = "no data in window"
        elif info["eligible_digests"] == 0:
            watched_status[interest] = "no qualified candidates in window"
        else:
            watched_status[interest] = (
                f"{info['covered_digests']}/{info['eligible_digests']} eligible digests covered"
            )
    return watched_status


def _is_off_field(
    row: dict, interests: list[str], negative_interests: dict[str, float]
) -> bool:
    """Off-field = no clear interest match, or facet is a negative interest.

    We work purely from persisted features (no embedding recompute, keeping the
    harness read-only and cheap): an item is off-field when its ``primary_facet``
    is empty (no core-interest match) OR its facet appears among the reader's
    negative interests. Facet matching is case-insensitive.
    """
    facet = row["primary_facet"]
    if facet == "":
        return True
    facet_lc = facet.lower()
    interest_lc = {i.lower() for i in interests}
    if facet_lc in interest_lc:
        return False
    neg_lc = {k.lower() for k in negative_interests}
    if facet_lc in neg_lc:
        return True
    # A facet that is neither a known core interest nor a known negative is
    # treated as off-field (it did not match the reader's field).
    return True


def _is_high_venue(row: dict) -> bool:
    """High-profile candidate: top/high venue bucket, or a high final score."""
    if row["source_bucket"] in HIGH_VENUE_BUCKETS:
        return True
    # Fallback signal when the source bucket is unknown: a strong final score.
    return row["final_score"] >= 0.85


def _vote_eval() -> dict | None:
    """Read-only overall pairwise/precision via the existing evaluator.

    ``evaluate_history`` only issues SELECTs (it replays persisted orderings and
    reads votes) and never fits or writes — safe to call here. Returns None with
    a note if it cannot run cleanly.
    """
    try:
        from dailydigest.rank.evaluate import evaluate_history

        rep = evaluate_history(k=10)
        d = rep.as_dict()
        return {
            "digests_scored": d.get("digests_scored"),
            "votes_used": d.get("votes_used"),
            "pairwise_accuracy": d.get("pairwise_accuracy"),
            "precision_at_k": d.get("precision_at_k"),
            "map": d.get("map"),
            "ndcg_at_k": d.get("ndcg_at_k"),
        }
    except Exception as exc:  # keep the harness robust; skip criterion 5.
        return {"error": f"skipped (evaluate_history unavailable): {exc}"}


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #
def _fmt(x, nd: int = 2) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def print_report(report: dict) -> None:
    print("=" * 70)
    print("DailyDigest coverage & precision report (READ-ONLY)")
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
        print("    Watched interests:")
        for interest, status in watched.items():
            print(f"      * {interest}: {status}")
    if report.get("routinely_absent"):
        print(f"    FAIL flags (routinely absent): {report['routinely_absent']}")

    # --- Criterion 2 -----------------------------------------------------
    print("\n[2] OFF-FIELD PRECISION (mean off-field selected research items / digest)")
    off = report.get("off_field_per_digest")
    met = report.get("off_field_target_met")
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
