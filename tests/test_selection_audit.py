from copy import deepcopy
import json

from dailydigest.rank.selection_audit import audit_selection_snapshot


def candidate(i, source, eligible, selected, score, topic, facet="Engineering"):
    return {"item_id": i, "section": "research", "eligible": eligible, "selected": selected,
            "ranking_score": score, "features": {"topic_score": topic, "primary_facet": facet},
            "metadata": {"source": source}, "selection_facts": {"source_bucket": "published_journal"},
            "first_rejected_stage": None if eligible else "after_topic_gate"}


def test_report_distinguishes_source_volume_from_selection_rate_and_omission():
    snapshot = {"run_id": "frozen", "reading_mode": "usual", "section_budgets": {"research": 3},
                "candidates": [candidate(1, "A", True, True, .9, .8),
                               candidate(2, "A", True, False, .85, .9),
                               candidate(3, "A", False, False, .6, .4),
                               candidate(4, "B", True, True, .7, .75, "Context")]}
    before = deepcopy(snapshot)
    report = audit_selection_snapshot(snapshot)
    a = report["groups"]["source"]["A"]
    assert (a["pool_count"], a["eligible_count"], a["selected_count"], a["selection_rate"]) == (3, 2, 1, .5)
    assert report["higher_rank_score_omissions"] == [{"item_id": 2, "rank_score": .85,
                                                    "source": "A", "primary_facet": "Engineering"}]
    assert report["rejection_stages"] == {"none": 3, "after_topic_gate": 1}
    assert snapshot == before
    json.dumps(report, allow_nan=False)


def test_missing_topic_evidence_and_empty_groups_do_not_gain_zero_scores():
    snapshot = {"section_budgets": {"research": 0}, "candidates": []}
    assert audit_selection_snapshot(snapshot)["higher_rank_score_omissions"] == []
    row = candidate(1, "A", True, False, .7, None)
    snapshot["candidates"] = [row]
    values = audit_selection_snapshot(snapshot)["groups"]["source"]["A"]["eligible_scores"]["topic_score"]
    assert values == {"observed": 0, "missing": 1, "p05": None, "median": None, "p95": None}
