"""Quality constraints cannot disappear in small pools or optional exploration."""
from types import SimpleNamespace

import pytest

from dailydigest.pipeline import _quality_gate
from dailydigest.rank.ranker import apply_exploration, pick_top_per_section


def row(i, source="Nature", title="Programmable DNA origami assemblies", abstract="Primary research with methods and experimental results."):
    return SimpleNamespace(id=i, source=source, section="research", title=title,
                           abstract=abstract, url=f"https://example.com/{i}",
                           external_id=str(i), metadata_json="{}")


@pytest.mark.parametrize("bad", [
    row(1, title="Sponsored: Programmable DNA origami assemblies"),
    row(1, source="Unlisted Journal", abstract=""),
    row(1, title="Editorial: A new method for DNA design"),
])
def test_quality_gate_has_same_decision_in_sparse_and_large_pools(bad):
    alone, alone_drops = _quality_gate([bad])
    crowded, crowded_drops = _quality_gate([bad] + [row(i) for i in range(2, 12)])
    assert bad not in crowded
    assert bad not in alone
    assert alone_drops[0]["reason"] == crowded_drops[0]["reason"]


def settings(**updates):
    return SimpleNamespace(**{
        "max_preprint_research_frac": .55, "max_low_impact_research_frac": 0,
        "low_impact_relevance_floor": .72, "adaptive_relevance_floor": False,
        "research_final_score_floor_frac": .5, "research_final_score_min_keep": 3,
        **updates,
    })


@pytest.mark.parametrize("violation", ["source_cap", "final_score_floor"])
def test_exploration_retains_real_picker_constraints(violation):
    picked = [(row(1), .95), (row(2), .9), (row(3, "Science"), .85),
              (row(4, "Science"), .8), (row(5, "Cell"), .75)]
    candidate = row(6, "Nature" if violation == "source_cap" else "Nature Methods")
    candidate_score = .7 if violation == "source_cap" else .1
    pool = picked + [(candidate, candidate_score)]
    features = {r.id: {"topic_score": .8} for r, _ in pool}
    before = pick_top_per_section(picked, {"research": 5}, score_features=features, settings=settings())
    assert len(before) == 5
    after = apply_exploration(before, pool, {6: 1.0}, slots=1, eligible=lambda r: True,
                              caps={"research": 5}, score_features=features, settings=settings())
    assert [r.id for r, _ in after] == [r.id for r, _ in before]


def test_exploration_can_skip_invalid_candidate_and_use_valid_alternative():
    picked = [(row(1), .95), (row(2), .9), (row(3, "Science"), .85),
              (row(4, "Science"), .8), (row(5, "Cell"), .75)]
    pool = picked + [(row(6), .7), (row(7, "Nature Methods"), .7)]
    features = {r.id: {"topic_score": .8} for r, _ in pool}
    after = apply_exploration(picked, pool, {6: 1.0, 7: .9}, slots=1, eligible=lambda r: True,
                              caps={"research": 5}, score_features=features, settings=settings())
    assert {r.id for r, _ in after} == {1, 2, 3, 4, 7}
