"""Tests for low-impact-journal classification, frequency cap, and relevance floor."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dailydigest.rank.ranker import pick_top_per_section
from dailydigest.rank.source_quality import (
    is_low_impact_research,
    quality_adjusted_score,
    source_bucket,
)


def _row(title: str, source: str, section: str = "research") -> MagicMock:
    row = MagicMock()
    row.title = title
    row.abstract = "primary research with efficacy data"
    row.section = section
    row.source = source
    row.id = None
    return row


# --- classification ----------------------------------------------------------


def test_unknown_venue_is_low_impact_bucket():
    row = _row("A targeted therapeutics study", "Journal of Minor Results")
    assert source_bucket(row) == "low_impact_journal"
    assert is_low_impact_research(row) is True


def test_top_journal_is_not_low_impact():
    row = _row("A targeted therapeutics study", "Nature")
    assert source_bucket(row) == "published_journal"
    assert is_low_impact_research(row) is False


def test_venue_low_impact_flag_overrides_aggregator_bucket():
    # An OpenAlex item would normally bucket as "aggregator"; the enrichment flag
    # reroutes it to low_impact_journal so the frequency cap applies.
    row = _row("A targeted therapeutics study", "OpenAlex (biotech)")
    assert source_bucket(row) == "aggregator"
    row.venue_low_impact = True
    assert source_bucket(row) == "low_impact_journal"
    assert is_low_impact_research(row) is True


def test_low_impact_penalized_vs_top_at_equal_relevance():
    minor = _row("RNA delivery mechanism", "Journal of Minor Results")
    nature = _row("RNA delivery mechanism", "Nature")
    assert quality_adjusted_score(minor, 0.70) < quality_adjusted_score(nature, 0.70)


def test_paywalled_source_is_penalized_in_industry():
    from types import SimpleNamespace

    from dailydigest.rank.source_quality import PAYWALL_PENALTY, quality_adjusted_score

    # Two industry items, identical text/relevance; one from a paywalled venue.
    free = SimpleNamespace(
        title="FDA clears new biotech therapy", abstract="clinical trial results",
        section="industry", source="FierceBiotech", id=1,
    )
    paywalled = SimpleNamespace(
        title="FDA clears new biotech therapy", abstract="clinical trial results",
        section="industry", source="Endpoints News", id=2,
    )
    free_s = quality_adjusted_score(free, 0.60)
    pay_s = quality_adjusted_score(paywalled, 0.60)
    assert pay_s < free_s
    assert free_s - pay_s == pytest.approx(PAYWALL_PENALTY)


def test_quality_adjusted_score_stays_in_unit_interval():
    # The selection thresholds (adaptive_size_bar, low_impact_relevance_floor,
    # the exceptional-preprint cutoff) all assume a [0,1] scale, so the
    # quality-adjusted score must never escape it — even at extreme inputs.
    sources = ["Nature", "Journal of Minor Results", "bioRxiv", "OpenAlex (biotech)", "BBC World"]
    sections = ["research", "industry", "world", "regulatory"]
    bases = [-0.5, 0.0, 0.5, 1.0, 1.5]  # incl. out-of-range inputs
    for section in sections:
        for source in sources:
            for base in bases:
                row = _row("CRISPR efficacy results with method and data", source, section)
                s = quality_adjusted_score(row, base)
                assert 0.0 <= s <= 1.0, f"{section}/{source}/base={base} -> {s}"


# --- frequency cap + relevance floor in selection ----------------------------


def _hq_pool(n: int):
    sources = [
        "Nature Methods",
        "Nature Medicine",
        "Nature Chemistry",
        "Nature Materials",
        "Nature Physics",
        "Nature Catalysis",
    ]
    return [
        (_row(f"High quality paper {i}", sources[i % len(sources)]), 0.70 - i * 0.005)
        for i in range(n)
    ]


def test_low_impact_journals_are_frequency_capped():
    # Low-impact items score *higher* but must still be capped (int(10*0.15)=1).
    scored = _hq_pool(12)
    scored += [
        (_row(f"Minor paper {i}", f"Journal of Minor Results {i}"), 0.90 - i * 0.005)
        for i in range(8)
    ]
    result = pick_top_per_section(scored, {"research": 10})
    low = sum(1 for row, _ in result if source_bucket(row) == "low_impact_journal")
    assert len(result) == 10
    assert low <= 1


def test_low_impact_below_floor_is_excluded():
    # All low-impact items are below the 0.58 relevance floor → none selected when
    # enough high-quality items exist to fill the section.
    scored = _hq_pool(12)
    scored += [
        (_row(f"Minor paper {i}", f"Journal of Minor Results {i}"), 0.40 - i * 0.005)
        for i in range(8)
    ]
    result = pick_top_per_section(scored, {"research": 10})
    low = sum(1 for row, _ in result if source_bucket(row) == "low_impact_journal")
    assert low == 0


def test_low_impact_used_as_last_resort_only_up_to_hard_minimum():
    # Only low-impact items exist (below floor). The last-resort override fills just
    # the small hard minimum (3), not the full cap — a short section of the least-bad
    # items beats padding five weak slots (dynamic cutoff, see min_research/P8).
    scored = [
        (_row(f"Minor paper {i}", f"Journal of Minor Results {i}"), 0.45 - i * 0.01)
        for i in range(6)
    ]
    result = pick_top_per_section(scored, {"research": 5})
    assert len(result) == 3
