"""Scope interventions retain the real frozen serving policy and its limits."""
from copy import deepcopy

import pytest

from dailydigest.rank.replay import IncompleteReplay, frozen_selection
from dailydigest.rank.scope_replay import compare_scope_slates


def _snapshot():
    candidates = []
    for iid in range(1, 10):
        research = iid < 9
        candidates.append({
            "item_id": iid, "family_id": str(iid),
            "section": "research" if research else "industry", "eligible": True,
            "selected": iid <= 4 or not research,
            "position": iid - 1 if iid <= 4 else (0 if not research else None),
            "input_position": iid - 1, "ranking_score": 1 - iid / 100,
            "features": {"topic_score": 0.9, "scoring_mode": "hybrid_knn", "learned_score": 0.8},
            "metadata": {"source": f"Journal {iid}", "title": f"Paper {iid}",
                         "abstract": "", "metadata_json": "{}"},
            "selection_facts": {"source_bucket": "published_journal", "is_arxiv_cs": False,
                                "is_preprint": False, "is_published": True,
                                "is_high_quality": True, "is_low_impact": False},
        })
    return {
        "schema_version": 1, "run_id": "scope-run", "digest_id": "2026-10-01",
        "decision_at": "2026-10-01T12:00:00Z", "reading_mode": "full",
        "complete_candidate_pool": True, "section_budgets": {"research": 4, "industry": 1},
        "selection_settings": {
            "max_preprint_research_frac": 0.55, "max_low_impact_research_frac": 0.25,
            "low_impact_relevance_floor": 0.7, "research_final_score_floor_frac": 0,
            "research_final_score_min_keep": 0, "adaptive_relevance_floor": False,
        },
        "decision_local_date": "2026-10-01", "catch_up": False, "exploration_slots": 0,
        "candidates": candidates,
    }


def _decisions(snapshot=None):
    snapshot = _snapshot() if snapshot is None else snapshot
    return {c["item_id"]: {"label": "core", "facet": "molecular design", "reason": "design evidence"}
            for c in snapshot["candidates"] if c["eligible"] and c["section"] == "research"}


def _compare(snapshot=None, decisions=None, policy=None):
    snapshot = _snapshot() if snapshot is None else snapshot
    return compare_scope_slates(snapshot, _decisions(snapshot) if decisions is None else decisions,
                               policy={"version": "test-v1"} if policy is None else policy)


def test_filters_then_refills_from_overflow_without_mutating_inputs():
    snapshot = _snapshot()
    decisions = _decisions(snapshot)
    decisions[1].update(label="off_interest", facet="", reason="diagnostic assay")
    decisions[3].update(label="context", facet="", reason="adjacent descriptive biology")
    original, original_decisions = deepcopy(snapshot), deepcopy(decisions)
    report = _compare(snapshot, decisions)
    assert report["baseline_slate"] == {"research": [1, 2, 3, 4], "industry": [9]}
    assert report["challenger_slate"] == {"research": [2, 4, 5, 6], "industry": [9]}
    assert report["removed_research_ids"] == [1, 3]
    assert report["added_research_ids"] == [5, 6]
    assert report["decision_counts"] == {"core": 6, "context": 1, "off_interest": 1, "uncertain": 0}
    assert report["baseline_recorded_parity"]["status"] == "matched"
    assert snapshot == original
    assert decisions == original_decisions
    explanations = {e["item_id"]: e for e in report["research_explanations"]}
    assert explanations[1]["selection_status"] == "excluded_by_scope"
    assert explanations[5]["selection_status"] == "selected"
    assert explanations[8]["selection_status"] == "not_selected_by_frozen_policy"


def test_keeps_raw_topic_floor_and_low_impact_cap_when_refilling():
    snapshot = _snapshot()
    for c in snapshot["candidates"][:5]:
        c["selection_facts"].update(is_low_impact=True, is_high_quality=False,
                                    source_bucket="low_impact_journal")
    snapshot["candidates"][0]["features"]["topic_score"] = 0.01
    decisions = _decisions(snapshot)
    decisions[6]["label"] = "off_interest"
    report = _compare(snapshot, decisions)
    selected = report["challenger_slate"]["research"]
    assert 1 not in selected
    assert len(set(selected).intersection(range(1, 6))) == 1
    assert 6 not in selected
    assert report["coverage"]["challenger_research_count"] == 3
    assert report["coverage"]["shorter_than_baseline"]


def test_original_ineligible_candidates_cannot_be_resurrected():
    snapshot = _snapshot()
    snapshot["candidates"][0]["eligible"] = False
    report = _compare(snapshot)
    assert 1 not in report["challenger_slate"]["research"]
    entry = next(e for e in report["research_explanations"] if e["item_id"] == 1)
    assert entry["scope_decision"] is None
    assert entry["selection_status"] == "originally_ineligible"
    with pytest.raises(ValueError, match="exactly"):
        _compare(snapshot, _decisions())


@pytest.mark.parametrize("change", [
    lambda d: d.pop(1),
    lambda d: d.update({99: d[1]}),
    lambda d: d.update({"1": d.pop(1)}),
    lambda d: d.update({True: d.pop(1)}),
    lambda d: d[1].update(label="maybe"),
    lambda d: d[1].update(facet=None),
    lambda d: d[1].update(facet=""),
    lambda d: d[1].update(reason="  "),
    lambda d: d[1].update(scores={"core": float("nan")}),
    lambda d: d[1].update(scores={"core": float("inf")}),
])
def test_incomplete_or_invalid_decisions_fail_closed(change):
    decisions = _decisions()
    change(decisions)
    with pytest.raises(ValueError):
        _compare(decisions=decisions)


@pytest.mark.parametrize("policy", [{}, {"version": ""}, {"version": float("nan")}, []])
def test_missing_or_invalid_provenance_fails_closed(policy):
    with pytest.raises(ValueError, match="policy"):
        _compare(policy=policy)


def test_empty_challenger_reports_coverage_loss_not_improvement():
    decisions = _decisions()
    for decision in decisions.values():
        decision.update(label="uncertain", facet="", reason="insufficient abstract evidence")
    report = _compare(decisions=decisions)
    assert report["challenger_slate"] == {"research": [], "industry": [9]}
    assert report["coverage"]["empty_research_slate"]
    assert report["coverage"]["uncertain_fraction"] == 1.0
    assert report["publication_allowed"] is False
    assert report["relevance_validation"] == "not_evaluated"


def test_incomplete_frozen_selection_evidence_is_not_certified():
    snapshot = _snapshot()
    del snapshot["candidates"][4]["selection_facts"]
    with pytest.raises(IncompleteReplay):
        _compare(snapshot)


def test_recorded_slate_mismatch_is_visible_and_not_used_as_baseline():
    snapshot = _snapshot()
    snapshot["candidates"][0]["selected"] = False
    report = _compare(snapshot)
    assert report["baseline_slate"] == frozen_selection(snapshot)
    assert report["baseline_recorded_parity"]["status"] == "mismatch"
    assert report["baseline_recorded_parity"]["recorded"]["research"] == [2, 3, 4]


@pytest.mark.parametrize("mode, budget, limit", [("full", 4, 4), ("minimal", 8, 5)])
def test_uses_actual_exploration_and_reading_mode(mode, budget, limit):
    snapshot = _snapshot()
    snapshot["exploration_slots"] = 1
    snapshot["reading_mode"] = mode
    snapshot["section_budgets"]["research"] = budget
    snapshot["candidates"][7]["features"]["learned_score"] = 0.5
    decisions = _decisions(snapshot)
    decisions[1]["label"] = "off_interest"
    expected = deepcopy(snapshot)
    expected["candidates"][0]["eligible"] = False
    report = _compare(snapshot, decisions)
    assert report["baseline_slate"] == frozen_selection(snapshot)
    assert report["challenger_slate"] == frozen_selection(expected)
    assert len(report["challenger_slate"]["research"]) == limit
    if mode == "full":
        assert 8 in report["challenger_slate"]["research"]


def test_provenance_and_decisions_are_frozen_in_report():
    policy = {"version": "test-v1", "thresholds": {"core": 0.7}}
    decisions = _decisions()
    report = _compare(decisions=decisions, policy=policy)
    first_signature = report["decision_signature"]
    policy["thresholds"]["core"] = 0.1
    decisions[1]["reason"] = "changed later"
    assert report["policy_provenance"]["thresholds"]["core"] == 0.7
    assert report["research_explanations"][0]["scope_decision"]["reason"] == "design evidence"
    assert _compare(decisions=decisions)["decision_signature"] != first_signature
