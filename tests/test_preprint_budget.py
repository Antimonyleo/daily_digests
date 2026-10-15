"""Minimum-size filling preserves mixed-pool publication-status budgets."""
from types import SimpleNamespace

import pytest

from dailydigest.rank import source_quality
from dailydigest.rank.ranker import pick_top_per_section


def row(iid, source):
    return SimpleNamespace(id=iid, source=source, section="research",
                           title="Validated measured result", abstract="Measured methods.",
                           metadata_json="{}")


def controls(fraction):
    return SimpleNamespace(max_preprint_research_frac=fraction,
                           max_low_impact_research_frac=0., low_impact_relevance_floor=.72,
                           adaptive_relevance_floor=False, research_final_score_floor_frac=0.,
                           research_final_score_min_keep=0)


@pytest.fixture(autouse=True)
def isolated_sources(monkeypatch):
    monkeypatch.setattr(source_quality, "_configured_sources_cached", lambda: {})


@pytest.mark.parametrize("cap", [3, 5])
def test_hard_minimum_does_not_exceed_positive_preprint_budget_in_mixed_pool(cap):
    pool = [(row(1, "Nature"), .85), (row(2, "bioRxiv"), .95),
            (row(3, "arXiv quant-ph"), .94), (row(4, "ChemRxiv"), .93)]
    features = {candidate.id: {"topic_score": .9} for candidate, _ in pool}
    selected = pick_top_per_section(pool, {"research": cap}, settings=controls(.1),
                                    score_features=features)
    assert sum(source_quality.is_preprint_source(candidate) for candidate, _ in selected) == 1
    assert {candidate.id for candidate, _ in selected} == {1, 2}


def test_preprint_only_positive_budget_keeps_existing_safety_valve():
    pool = [(row(index, source), .95) for index, source in enumerate(
        ["bioRxiv", "arXiv quant-ph", "ChemRxiv"], 1)]
    selected = pick_top_per_section(pool, {"research": 3}, settings=controls(.1))
    assert len(selected) == 3


def test_single_preprint_server_keeps_existing_minimum_fill_exception():
    pool = [(row(index, "bioRxiv"), .95) for index in range(1, 6)]
    selected = pick_top_per_section(pool, {"research": 5}, settings=controls(.1))
    assert len(selected) == 3


@pytest.mark.parametrize("catch_up", [False, True])
def test_preprint_only_safety_valve_never_waives_explicit_zero(catch_up):
    pool = [(row(1, "bioRxiv"), .95), (row(2, "arXiv quant-ph"), .94)]
    assert pick_top_per_section(pool, {"research": 3}, settings=controls(0.),
                                catch_up=catch_up) == []
