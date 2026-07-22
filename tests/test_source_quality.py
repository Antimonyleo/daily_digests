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


def test_issue_publication_metadata_is_skipped():
    class Row:
        source = "ACS Chemical Biology"
        section = "research"
        title = "Issue Publication Information"
        abstract = "Volume 21, issue 5 publication metadata."

    assert should_skip_item(Row()) is True


def test_author_intro_metadata_is_skipped():
    class Row:
        source = "ACS Chemical Biology"
        section = "research"
        title = "Introducing Our Authors"
        abstract = "Author profile and issue metadata."

    assert should_skip_item(Row()) is True


def test_topic_fit_beats_prestige_only_when_relevance_gap_is_large():
    # Relevance stays the primary gate: a clearly stronger topic match in a minor
    # venue still beats a weak match in a top venue when the gap is large.
    class Row:
        section = "research"
        abstract = "Primary research with efficacy data."

        def __init__(self, source: str, title: str) -> None:
            self.source = source
            self.title = title

    weak_nature = Row("Nature", "Broad cell biology observation")
    strong_minor = Row("Minor Journal", "RNA delivery mechanism for targeted therapeutics")

    assert quality_adjusted_score(strong_minor, 0.82) > quality_adjusted_score(weak_nature, 0.50)


def test_venue_bonus_is_relevance_gated_not_unconditional():
    # Regression: a relevance-INDEPENDENT venue bonus used to lift an off-niche
    # high-impact paper (low base) above a strongly on-topic item, inverting the
    # topic order. The venue reward must now be gated by relevance so a prestigious
    # but off-topic paper cannot outrank a more on-topic one.
    class Row:
        section = "research"
        abstract = "Primary research with efficacy data."

        def __init__(self, source: str, title: str) -> None:
            self.source = source
            self.title = title

    offtopic_nature = Row("Nature", "Broad tissue gene expression atlas")
    ontopic_preprint = Row("bioRxiv", "De novo protein design method for binders")

    assert quality_adjusted_score(ontopic_preprint, 0.78) > quality_adjusted_score(
        offtopic_nature, 0.69
    )


def test_venue_bonus_scales_with_relevance():
    # The prestige reward must be much larger for an on-topic paper than for the
    # same venue at a near-floor relevance — quality amplifies relevance, never
    # substitutes for it.
    class Row:
        section = "research"
        abstract = "Primary research with efficacy data."

        def __init__(self, source: str, title: str) -> None:
            self.source = source
            self.title = title

    nature = Row("Nature", "RNA delivery mechanism for targeted therapeutics")
    low_delta = quality_adjusted_score(nature, 0.66) - 0.66
    high_delta = quality_adjusted_score(nature, 0.82) - 0.82
    assert high_delta > low_delta + 0.05
    # Near the retrieval floor the venue lift is negligible.
    assert low_delta < 0.05


def test_high_quality_venue_wins_at_similar_relevance():
    # Policy: high quality AND relevant is rewarded. At comparable relevance the
    # top-venue paper should clearly outrank a low-impact-venue paper.
    class Row:
        section = "research"
        abstract = "Primary research with efficacy data."

        def __init__(self, source: str, title: str) -> None:
            self.source = source
            self.title = title

    nature = Row("Nature", "RNA delivery mechanism for targeted therapeutics")
    minor = Row("Minor Journal", "RNA delivery mechanism for targeted therapeutics")

    assert quality_adjusted_score(nature, 0.70) > quality_adjusted_score(minor, 0.70)
    # The gap should be substantial, not a hair-thin tie-breaker.
    assert quality_adjusted_score(nature, 0.70) - quality_adjusted_score(minor, 0.70) > 0.15


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


def test_recognized_research_venue_matches_flagship_journals():
    from dailydigest.rank.source_quality import recognized_research_venue

    # Flagship venues that arrive via aggregators should be recognized so they can
    # be re-attributed to their real (prestigious) identity.
    assert recognized_research_venue("ACS Nano") == "ACS Nano"
    assert recognized_research_venue("Nano Letters") == "Nano Letters"
    assert (
        recognized_research_venue("Journal of the American Chemical Society")
        == "Journal of the American Chemical Society"
    )
    assert recognized_research_venue("Nature Materials") == "Nature Materials"
    # Unknown / low-impact venues keep their aggregator attribution (None).
    assert recognized_research_venue("ACS Omega") is None
    assert recognized_research_venue("Frontiers in Pharmacology") is None
    assert recognized_research_venue("") is None
    assert recognized_research_venue(None) is None


def test_venue_relevance_credit_only_rewards_peer_reviewed_prestige():
    from dailydigest.rank.source_quality import venue_relevance_credit

    class Row:
        def __init__(self, source: str, section: str = "research") -> None:
            self.source = source
            self.section = section
            self.title = ""
            self.abstract = ""

    # High/strong-tier journals get a positive gate credit...
    assert venue_relevance_credit(Row("Nature Materials")) > 0.0
    assert venue_relevance_credit(Row("ACS Nano")) > 0.0
    # ...while preprints, aggregators, and unknown venues get none.
    assert venue_relevance_credit(Row("bioRxiv (recent)")) == 0.0
    assert venue_relevance_credit(Row("OpenAlex (your topics)")) == 0.0
    assert venue_relevance_credit(Row("Some Unknown Journal")) == 0.0
    # News items never get research venue credit.
    assert venue_relevance_credit(Row("Nature", section="industry")) == 0.0
    # Top-tier credit >= high-tier credit >= strong-tier credit (prestige order).
    top = venue_relevance_credit(Row("Nature"))
    strong = venue_relevance_credit(Row("Chemistry of Materials"))  # genuinely strong-tier
    assert top >= strong > 0.0


def test_flagship_acs_journals_are_high_tier():
    """ACS Nano / JACS / Nano Letters rival Nature sub-journals by impact and must
    sit in the high tier so they aren't buried by score compression at the top."""
    for name in ("ACS Nano", "Nano Letters", "Journal of the American Chemical Society"):
        q = infer_source_quality(name, "research")
        assert q.quality_tier == "high", f"{name} should be high tier, got {q.quality_tier}"
        assert q.prestige_score >= 0.90
    # Lower-impact ACS venues stay at strong (not over-promoted).
    assert infer_source_quality("Chemistry of Materials", "research").quality_tier == "strong"
    assert infer_source_quality("JACS Au", "research").quality_tier == "strong"


def test_non_research_frontmatter_is_skipped_from_research():
    from dailydigest.rank.source_quality import is_non_research_content

    class Row:
        section = "research"
        abstract = "Some blurb text."
        def __init__(self, title): self.title = title

    # Front-matter / non-primary content rides journal prestige — must be skipped.
    for t in [
        "Biotech news from around the world",
        "News & Views: Form follows function",
        "Author Correction: A de novo designed protein",
        "Publisher Correction: Nanostructure assembly",
        "Retraction Note: Prior claims",
        "Research Highlights",
        "Editorial: The road ahead",
        "In this issue",
    ]:
        assert is_non_research_content(Row(t)) is True, t
        assert should_skip_item(Row(t)) is True, t

    # Genuine research whose title merely contains a trigger word is NOT skipped.
    assert is_non_research_content(Row("A correction-free error model for DNA origami")) is False
    assert is_non_research_content(Row("De novo design of ligand binding proteins")) is False

    # Career-advice / news-desk content that Nature/Science RSS mixes into the
    # research feed and rides journal prestige — must be skipped.
    for t in [
        "My job interviews for industry positions are going nowhere. How do I stand out?",
        "The science of foresight: how to future-proof your research",
        "How to write a paper that lands a faculty job",
        "Smuggling charges against NIH virologists trigger political uproar",
        "Prominent researcher indicted on fraud charges",
        "Mathematics and mentorship make a recipe for success",
        "Balancing work-life demands as a new PI",
    ]:
        assert is_non_research_content(Row(t)) is True, t

    # Real papers whose titles merely start with "How" or contain "charge" are kept.
    assert is_non_research_content(Row("How the ribosome selects tRNA during elongation")) is False
    assert is_non_research_content(Row("Charge transport in a DNA-templated nanowire")) is False


def test_non_research_filter_only_applies_to_research_section():
    from dailydigest.rank.source_quality import is_non_research_content

    class Row:
        section = "industry"
        abstract = ""
        def __init__(self, title): self.title = title

    # News is expected in the industry section; the filter must not touch it there.
    assert is_non_research_content(Row("Biotech news from around the world")) is False


def test_multi_cosine_downweights_low_weight_context_facets():
    """A context facet (row built at weight < 1) must not let an item win the
    top-1 the way a core facet (weight 1) does."""
    import numpy as np
    from dailydigest.rank.ranker import _multi_cosine

    # Two orthogonal unit directions: e0 = core interest, e1 = context interest.
    core = np.zeros(384, dtype=np.float32); core[0] = 1.0
    context = np.zeros(384, dtype=np.float32); context[1] = 1.0
    # Profile matrix: core at weight 1.0, context at weight 0.45.
    profile = np.vstack([core * 1.0, context * 0.45]).astype(np.float32)

    item_core = core.reshape(1, -1)      # perfectly matches core
    item_context = context.reshape(1, -1)  # perfectly matches context only

    s_core = float(_multi_cosine(item_core, profile)[0])
    s_context = float(_multi_cosine(item_context, profile)[0])
    # Same raw cosine (1.0) to their respective facet, but the context-only match
    # scores strictly lower because its facet weight is clamped-down (0.45).
    assert s_core > s_context
    assert s_context < 0.68  # below a typical relevance floor
