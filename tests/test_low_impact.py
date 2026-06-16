"""Tests for low-impact-journal classification, frequency cap, and relevance floor."""

from __future__ import annotations

from unittest.mock import MagicMock

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


def test_low_impact_used_as_last_resort_when_section_would_be_short():
    # Only low-impact items exist (below floor) → last-resort fill still uses them
    # rather than ship an empty section.
    scored = [
        (_row(f"Minor paper {i}", f"Journal of Minor Results {i}"), 0.45 - i * 0.01)
        for i in range(6)
    ]
    result = pick_top_per_section(scored, {"research": 5})
    assert len(result) == 5
