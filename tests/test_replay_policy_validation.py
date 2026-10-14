"""Offline quality metrics cannot certify an arbitrary policy-bypassing slate."""
from copy import deepcopy

from dailydigest.rank.replay import evaluate_replay, frozen_selection, paired_run_metrics


def _snapshot():
    return {
        "schema_version": 1, "run_id": "policy-run", "digest_id": "2026-10-01",
        "decision_at": "2026-10-01T12:00:00Z", "reading_mode": "full",
        "complete_candidate_pool": True, "section_budgets": {"research": 2},
        "selection_settings": {
            "max_preprint_research_frac": 0.55, "max_low_impact_research_frac": 0.5,
            "low_impact_relevance_floor": 0.7, "research_final_score_floor_frac": 0,
            "research_final_score_min_keep": 0, "adaptive_relevance_floor": False,
        },
        "decision_local_date": "2026-10-01", "catch_up": False, "exploration_slots": 0,
        "candidates": [{
            "item_id": iid, "family_id": str(iid), "section": "research", "eligible": True,
            "selected": iid == 2, "position": 0 if iid == 2 else None,
            "input_position": iid - 1, "ranking_score": 1 - iid / 10,
            "features": {"topic_score": 0.01 if iid == 1 else 0.9},
            "metadata": {"source": f"Journal {iid}", "title": f"Distinct paper {iid}",
                         "abstract": "", "metadata_json": "{}"},
            "selection_facts": {
                "source_bucket": "low_impact_journal" if iid == 1 else "published_journal",
                "is_arxiv_cs": False, "is_preprint": False, "is_published": True,
                "is_high_quality": iid == 2, "is_low_impact": iid == 1,
            },
        } for iid in (1, 2)],
    }


def _evaluate(snapshot, slate=None):
    return evaluate_replay(snapshot, [
        {"id": 1, "item_id": 1, "value": 1, "grade": 100,
         "recorded_at": "2026-10-01T13:00:00Z"},
    ], slate=slate, outcome_as_of="2026-10-02T00:00:00Z")


def test_below_floor_challenger_is_diagnostic_not_policy_verified():
    snapshot = _snapshot()
    assert frozen_selection(snapshot) == {"research": [2]}
    report = _evaluate(snapshot, {"research": [1]})
    assert report["sections"]["research"]["positive_count"] == 1
    assert report["policy_validation"]["status"] == "unverified"
    assert "frozen" in report["policy_validation"]["reason"]


def test_recorded_and_explicit_real_selection_are_independently_verified():
    snapshot = _snapshot()
    assert _evaluate(snapshot)["policy_validation"]["status"] == "verified"
    assert _evaluate(snapshot, frozen_selection(snapshot))["policy_validation"]["status"] == "verified"


def test_recorded_flags_cannot_self_certify_a_bad_slate():
    snapshot = _snapshot()
    snapshot["candidates"][0].update(selected=True, position=0)
    snapshot["candidates"][1].update(selected=False, position=None)
    assert _evaluate(snapshot)["policy_validation"]["status"] == "unverified"


def test_missing_frozen_facts_remain_unverified_and_pair_ineligible():
    snapshot = _snapshot()
    del snapshot["candidates"][1]["selection_facts"]
    report = _evaluate(snapshot)
    assert report["policy_validation"]["status"] == "unverified"
    paired = paired_run_metrics([report], [deepcopy(report)])
    assert paired["policy_comparison_eligible"] is False
    assert paired["unverified_policy_runs"] == ["policy-run"]


def test_policy_bypassing_gain_does_not_qualify_as_policy_preserving_comparison():
    snapshot = _snapshot()
    baseline = _evaluate(snapshot)
    challenger = _evaluate(snapshot, {"research": [1]})
    paired = paired_run_metrics([baseline], [challenger])
    assert paired["mean_positive_count_delta"] == 1
    assert paired["policy_comparison_eligible"] is False
    assert paired["unverified_policy_runs"] == ["policy-run"]
    matched = paired_run_metrics([baseline], [deepcopy(baseline)])
    assert matched["policy_comparison_eligible"] is True
    assert matched["unverified_policy_runs"] == []
