"""Source-quality heuristics and real source config audits."""

from __future__ import annotations

from pathlib import Path

import pytest

from dailydigest.config import load_sources
from dailydigest.rank.source_quality import (
    _configured_sources_cached,
    infer_source_quality,
    is_arxiv_cs_source,
    quality_adjusted_score,
    score_breakdown,
    source_bucket,
    should_skip_item,
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


def test_openalex_is_downweighted_as_aggregator_source():
    _clear_source_quality_cache()

    quality = infer_source_quality("OpenAlex (biotech)", "research")

    assert quality.quality_tier == "aggregator"
    assert quality.prestige_score == pytest.approx(0.34)


def test_arxiv_cs_is_identified_and_downweighted_below_journal_tie():
    class Row:
        section = "research"
        title = "Machine learning method for biology"
        abstract = "Primary method with benchmark data."

        def __init__(self, source: str) -> None:
            self.source = source

    arxiv = Row("arXiv cs.LG")
    journal = Row("Nature Methods")

    assert is_arxiv_cs_source(arxiv) is True
    assert source_bucket(arxiv) == "arxiv_cs"
    assert quality_adjusted_score(arxiv, 0.72) < quality_adjusted_score(journal, 0.72)


def test_access_friction_language_reduces_quality_adjusted_score():
    class Row:
        source = "Nature"
        section = "research"

        def __init__(self, title: str, abstract: str = "") -> None:
            self.title = title
            self.abstract = abstract

    clean = Row("CRISPR delivery study", "Primary research with efficacy data.")
    gated = Row(
        "CRISPR delivery study",
        "Sign up or log in; subscription required for access.",
    )

    assert quality_adjusted_score(gated, 0.70) < quality_adjusted_score(clean, 0.70)


def test_angew_cover_entries_are_skipped():
    class Row:
        source = "Angew. Chem. Int. Ed."
        section = "research"
        title = "Front Cover: Molecular Catalysts"
        abstract = "Cover picture information."

    assert should_skip_item(Row()) is True


def test_short_editorial_without_new_information_is_skipped():
    class Row:
        source = "Nature"
        section = "research"
        title = "Editorial: The future of biological research"
        abstract = "This editorial discusses broad challenges and opportunities."

    assert should_skip_item(Row()) is True


def test_commentary_with_new_method_signal_is_kept():
    class Row:
        source = "Nature"
        section = "research"
        title = "Commentary on a new single-cell atlas method"
        abstract = "The article reports a new method and dataset for spatial single-cell analysis."

    assert should_skip_item(Row()) is False


def test_topic_fit_can_beat_prestige_when_journal_item_is_weak_match():
    class Row:
        section = "research"
        abstract = "Primary research with efficacy data."

        def __init__(self, source: str, title: str) -> None:
            self.source = source
            self.title = title

    weak_nature = Row("Nature", "Broad cell biology observation")
    strong_minor = Row("Minor Journal", "RNA delivery mechanism for targeted therapeutics")

    assert quality_adjusted_score(strong_minor, 0.70) > quality_adjusted_score(weak_nature, 0.55)


def test_score_breakdown_exposes_user_friendly_reason_tags():
    class Row:
        source = "Nature"
        section = "research"
        title = "First-in-class CRISPR delivery breakthrough"
        abstract = "Primary research with efficacy data."

    breakdown = score_breakdown(Row(), 0.75, learned_score=0.8)

    assert breakdown.topic == pytest.approx(0.75)
    assert "High-quality source" in breakdown.tags
    assert "High novelty" in breakdown.tags
    assert "Matches learned preferences" in breakdown.tags


def test_score_breakdown_exposes_structured_ranking_features():
    class Row:
        source = "Nature"
        section = "research"
        title = "First-in-class CRISPR delivery method"
        abstract = "Sign up or log in for a sponsored webinar about efficacy data."

    breakdown = score_breakdown(Row(), 0.75, learned_score=0.8, reason_penalty=0.07)

    assert breakdown.content_type == "method"
    assert breakdown.promo_penalty > 0
    assert breakdown.access_penalty > 0
    assert breakdown.reason_penalty == pytest.approx(0.07)
    assert "high_quality_source" in breakdown.quality_tags
    assert breakdown.freshness_tags
    assert "Downweighted for promotional language" in breakdown.why_shown
