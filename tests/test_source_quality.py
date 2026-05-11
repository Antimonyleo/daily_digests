"""Source-quality heuristics and real source config audits."""

from __future__ import annotations

from pathlib import Path

import pytest

from dailydigest.config import load_sources
from dailydigest.rank.source_quality import (
    _configured_sources_cached,
    infer_source_quality,
    quality_adjusted_score,
)


REAL_SOURCES = Path(__file__).resolve().parents[1] / "config" / "sources.yaml"


def _clear_source_quality_cache() -> None:
    _configured_sources_cached.cache_clear()


def test_trusted_news_tier_uses_trusted_news_prestige_from_config():
    _clear_source_quality_cache()

    quality = infer_source_quality("BBC World", "world")

    assert quality.quality_tier == "trusted-news"
    assert quality.prestige_score == pytest.approx(0.72)


def test_real_research_rss_sources_have_non_unknown_quality():
    _clear_source_quality_cache()

    unknown = []
    for source in load_sources(str(REAL_SOURCES)):
        if source.section != "research" or source.kind != "rss":
            continue
        quality = infer_source_quality(source.name, source.section)
        if quality.quality_tier == "unknown":
            unknown.append(source.name)

    assert unknown == []


def test_science_config_uses_journal_feed_not_news_feed():
    sources = load_sources(str(REAL_SOURCES))
    science = next(source for source in sources if source.section == "research" and source.name == "Science")

    assert "news_current" not in (science.url or "")
    assert science.quality_tier == "top"


def test_science_news_name_does_not_receive_top_research_boost():
    _clear_source_quality_cache()

    science_news = infer_source_quality("Science News", "research")
    science_journal = infer_source_quality("Science", "research")

    assert science_news.quality_tier != "top"
    assert science_news.prestige_score < science_journal.prestige_score


def test_research_promo_language_reduces_quality_adjusted_score():
    class Row:
        source = "Nature"
        section = "research"

        def __init__(self, title: str, abstract: str = "") -> None:
            self.title = title
            self.abstract = abstract

    clean = Row("First-in-class CRISPR delivery study", "Primary research with efficacy data.")
    promo = Row(
        "Sponsored webinar on first-in-class CRISPR delivery",
        "Register now for partner content about an AI discovery platform.",
    )

    assert quality_adjusted_score(promo, 0.70) < quality_adjusted_score(clean, 0.70)
