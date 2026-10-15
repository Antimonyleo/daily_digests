"""Local regression checks for field-neutral venue and selection policy."""
import json
from types import SimpleNamespace

import pytest

from dailydigest.rank import source_quality as quality
from dailydigest.rank.ranker import pick_top_per_section


@pytest.fixture(autouse=True)
def unconfigured_sources(monkeypatch):
    monkeypatch.setattr(quality, "_configured_sources_cached", lambda: {})


def row(iid, source, **attributes):
    return SimpleNamespace(id=iid, source=source, section="research",
                           title="A measured research result", abstract="Methods and results.",
                           metadata_json="{}", **attributes)


def settings(fraction=.1):
    return SimpleNamespace(max_preprint_research_frac=.55,
                           max_low_impact_research_frac=fraction,
                           low_impact_relevance_floor=.72, adaptive_relevance_floor=False,
                           research_final_score_floor_frac=0., research_final_score_min_keep=0)


@pytest.mark.parametrize("venue", [
    "Physical Review Letters", "Journal of Machine Learning Research",
    "Advanced materials (Deerfield Beach, Fla.)",
])
def test_cross_field_known_venues_have_exact_priors(venue):
    article = row(1, "PubMed (topics)")
    article.metadata_json = json.dumps({"venue": venue})
    assert quality.recognized_research_venue(venue) == venue
    assert quality.source_bucket(article) == "published_journal"


def test_nlm_angewandte_identity_keeps_its_curated_prior_without_spending_unknown_quota():
    # NLM Catalog 0370543 lists this uniform title and the existing canonical
    # title as the same journal: https://www.ncbi.nlm.nih.gov/nlmcatalog/0370543
    venue = "Angewandte Chemie (International ed. in English)"
    article = row(1, "PubMed (topics)")
    article.metadata_json = json.dumps({"venue": venue})
    canonical = row(2, "Angewandte Chemie International Edition")
    unknown = row(3, "Unlisted nucleic acid journal")
    second_unknown = row(4, "Another unlisted journal")
    assert quality.recognized_research_venue(venue) == venue
    assert quality.infer_source_quality(venue, "research") == quality.infer_source_quality(
        canonical.source, "research")
    assert quality.source_bucket(article) == "published_journal"
    assert quality.is_high_quality_journal_source(article)
    assert quality.source_diversity_key(article) == quality.source_diversity_key(canonical)
    assert quality.quality_adjusted_score(article, .8) == pytest.approx(
        quality.quality_adjusted_score(canonical, .8))
    pool = [(unknown, .99), (article, .98), (second_unknown, .97)]
    selected = pick_top_per_section(pool, {"research": 3}, settings=settings(),
                                   score_features={i: {"topic_score": .8} for i in [1, 3, 4]})
    assert {item.id for item, _ in selected} == {1, 3}


def test_nlm_small_identity_matches_existing_wiley_journal_prior():
    # NLM Catalog 101235338 identifies this uniform title as Small, Wiley-VCH.
    venue = "Small (Weinheim an der Bergstrasse, Germany)"
    article = row(1, "PubMed (topics)")
    article.metadata_json = json.dumps({"venue": venue})
    assert quality.recognized_research_venue(venue) == venue
    assert quality.source_bucket(article) == "published_journal"
    assert quality.infer_source_quality(venue, "research") == quality.infer_source_quality(
        "Small (Wiley)", "research")
    assert quality.source_diversity_key(article) == quality.source_diversity_key(
        row(2, "Small")) == quality.source_diversity_key(row(3, "Small (Wiley)"))


@pytest.mark.parametrize("source,venue,expected", [
    ("Unlisted Computing Proceedings", None, "unknown_journal"),
    ("PubMed (topics)", "Unlisted Mathematics Journal", "unknown_journal"),
    ("PubMed (topics)", None, "unknown_journal"),
    ("OpenAlex (topics)", None, "unknown_journal"),
])
def test_missing_venue_evidence_does_not_assert_low_impact(source, venue, expected):
    article = row(1, source)
    article.metadata_json = json.dumps({"venue": venue})
    assert quality.source_bucket(article) == expected
    assert quality.venue_evidence(article)["status"] == "unknown"
    assert not quality.is_low_impact_research(article)
    assert not quality.is_high_quality_journal_source(article)
    assert quality.venue_relevance_credit(article) == 0


@pytest.mark.parametrize("cap", range(1, 13))
@pytest.mark.parametrize("fraction", [0., .1])
@pytest.mark.parametrize("observed_low", [False, True])
def test_small_digest_uncertain_or_low_allowance_is_bounded_and_zero_is_absolute(cap, fraction, observed_low):
    pool = [(row(i, f"Venue {i}", venue_low_impact=observed_low), .95)
            for i in range(1, cap + 3)]
    features = {article.id: {"topic_score": .9} for article, _ in pool}
    picked = pick_top_per_section(pool, {"research": cap}, settings=settings(fraction),
                                  score_features=features)
    expected = min(cap, max(1, int(cap * fraction))) if fraction > 0 else 0
    assert len(picked) == expected


@pytest.mark.parametrize("base", [.55, .65, .75, .85])
def test_same_preprint_status_and_topic_get_same_penalty_across_fields(base):
    cs = row(1, "arXiv cs.LG")
    physics = row(2, "arXiv cond-mat.soft")
    biology = row(3, "bioRxiv")
    assert quality.quality_adjusted_score(cs, base) == pytest.approx(
        quality.quality_adjusted_score(physics, base))
    assert quality.quality_adjusted_score(cs, base) == pytest.approx(
        quality.quality_adjusted_score(biology, base))


def test_cs_and_other_preprints_share_selection_budget():
    journals = [(row(i, source), .8) for i, source in enumerate(
        ["Nature", "Cell", "Science", "Nature Methods", "Nature Chemistry", "Nature Physics"], 10)]
    cs = [(row(i, f"arXiv cs.{category}"), .94 - i / 100)
          for i, category in enumerate(["LG", "AI", "CV"], 1)]
    physics = [(row(i, f"arXiv {category}"), score) for (article, score), i, category in zip(
        cs, range(1, 4), ["cond-mat.soft", "physics.optics", "quant-ph"], strict=True)]
    cs_result = pick_top_per_section(cs + journals, {"research": 6}, settings=settings())
    physics_result = pick_top_per_section(physics + journals, {"research": 6}, settings=settings())
    assert [item.id for item, _ in cs_result] == [item.id for item, _ in physics_result]


@pytest.mark.parametrize("observed_low", [False, True])
def test_unknown_and_observed_weak_require_raw_topic_fit_even_during_minimum_fill(observed_low):
    article = row(1, "Unlisted Computing Proceedings", venue_low_impact=observed_low)
    selected = pick_top_per_section([(article, .99)], {"research": 3},
                                   score_features={1: {"topic_score": .71}}, settings=settings())
    assert selected == []


def test_unknown_and_observed_weak_have_separate_bounded_budgets():
    unknown = [(row(i, f"Unlisted proceedings {i}"), .95) for i in range(1, 5)]
    weak = [(row(i, f"Measured weak {i}", venue_low_impact=True), .94) for i in range(5, 9)]
    pool = unknown + weak
    features = {article.id: {"topic_score": .9} for article, _ in pool}
    selected = pick_top_per_section(pool, {"research": 6}, score_features=features,
                                   settings=settings())
    assert [quality.source_bucket(article) for article, _ in selected] == [
        "unknown_journal", "low_impact_journal"]


def test_explicit_configured_low_tier_keeps_weak_source_penalty_and_bucket(monkeypatch):
    from dailydigest.models import SourceSpec

    monkeypatch.setattr(quality, "_configured_sources_cached", lambda: {
        "weak journal": SourceSpec(name="Weak journal", kind="rss", section="research",
                                   quality_tier="low"),
    })
    weak = row(1, "Weak journal")
    assert quality.source_bucket(weak) == "low_impact_journal"
    assert quality.quality_adjusted_score(weak, .8) < .8
    assert not quality.is_high_quality_journal_source(weak)
    assert quality.venue_relevance_credit(weak) == 0


@pytest.mark.parametrize("venue", [
    "Physical Review Letters Plus", "Journal of Machine Learning Research Applications",
    "Advanced Materials Research (Deerfield Beach, Fla.)",
    "Angewandte Chemie (International ed. in English) Research",
    "Angewandte Chemie (International ed. in French)",
    "Small Ruminant Research (Weinheim an der Bergstrasse, Germany)",
    "Small (Weinheim an der Bergstrasse, Germany) Reports",
])
def test_alias_fixes_do_not_promote_similar_titles(venue):
    assert quality.recognized_research_venue(venue) is None
    assert quality.source_bucket(row(1, venue)) == "unknown_journal"


@pytest.mark.parametrize("source", ["PubMed (topics)", "OpenAlex (topics)"])
def test_missing_venue_cannot_escape_unknown_raw_floor_or_zero_budget(source):
    article = row(1, source)
    assert not pick_top_per_section([(article, .99)], {"research": 3}, settings=settings(),
                                    score_features={1: {"topic_score": .70}})
    assert not pick_top_per_section([(article, .99)], {"research": 3}, settings=settings(0),
                                    score_features={1: {"topic_score": .99}})


@pytest.mark.parametrize("catch_up", [False, True])
@pytest.mark.parametrize("source", ["bioRxiv", "arXiv cs.LG", "ChemRxiv"])
def test_explicit_zero_preprint_quota_cannot_be_bypassed_by_fill(source, catch_up):
    controls = settings()
    controls.max_preprint_research_frac = 0
    article = row(1, source)
    assert not pick_top_per_section([(article, .99)], {"research": 3}, settings=controls,
                                    catch_up=catch_up)


def test_observed_weak_cannot_keep_high_quality_reporting():
    article = row(1, "Nature", venue_low_impact=True)
    breakdown = quality.score_breakdown(article, .8)
    assert "High-quality source" not in breakdown.tags
    assert "high_quality_source" not in breakdown.quality_tags
    assert not any("high-quality" in reason.casefold() for reason in breakdown.why_shown)
