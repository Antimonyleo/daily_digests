"""Useful-paper outcomes remain separate from scope and missing judgments."""
from copy import deepcopy

import pytest

from dailydigest.rank.digest_evaluation import (
    build_judgment_packet, evaluate_digest_variants, expand_packet_judgments,
)


CUTOFF = "2026-10-05T00:00:00Z"


def snapshot(day=1):
    ids = list(range(1, 7)) if day == 1 else [1, 8, 9, 10, 11, 12]
    candidates = []
    for position, iid in enumerate(ids):
        candidates.append({
            "item_id": iid, "family_id": f"family-{iid}", "section": "research",
            "eligible": True, "selected": position < 3, "position": position if position < 3 else None,
            "input_position": position, "ranking_score": .95 - position * .05,
            "features": {"topic_score": .9, "primary_facet": "RNA design" if iid % 2 else "Protein design"},
            "metadata": {"title": f"Paper {iid}", "abstract": f"Supplied abstract for paper {iid}.",
                         "source": f"Journal {iid}", "metadata_json": "{}"},
            "selection_facts": {"source_bucket": "published_journal", "is_arxiv_cs": False,
                                "is_preprint": False, "is_published": True, "is_high_quality": True,
                                "is_low_impact": False},
        })
    return {"schema_version": 1, "run_id": f"run-{day}", "digest_id": f"2026-10-0{day}",
            "decision_at": f"2026-10-0{day}T12:00:00Z", "decision_local_date": f"2026-10-0{day}",
            "reading_mode": "full", "catch_up": False, "exploration_slots": 0,
            "complete_candidate_pool": True, "section_budgets": {"research": 3},
            "selection_settings": {"max_preprint_research_frac": .5, "max_low_impact_research_frac": .1,
                                   "low_impact_relevance_floor": .7, "research_final_score_floor_frac": 0,
                                   "research_final_score_min_keep": 3, "adaptive_relevance_floor": False},
            "candidates": candidates}


def judgment(iid, **fields):
    return {"run_id": "run-1", "item_id": iid, "recorded_at": "2026-10-03T12:00:00Z",
            "rater": "reader", **fields}


def compare(judgments=(), **kwargs):
    return evaluate_digest_variants([snapshot()], judgments, outcome_as_of=CUTOFF, **kwargs)


def test_scope_annotations_never_become_usefulness_and_unknown_bounds_remain():
    report = compare([judgment(1, useful=True, important=True),
                      judgment(2, useful=False, grade=10),
                      judgment(3, rater="analyst", scope="off_interest", rejection_reason="diagnostic association")])
    baseline = report["variants"]["baseline"]
    metrics = baseline["aggregate"]["research"]
    assert metrics["useful_count"] == 1
    assert metrics["judged"] == 2
    assert metrics["unknown"] == 1
    assert metrics["useful_fraction_bounds"] == pytest.approx([1 / 3, 2 / 3])
    assert metrics["at_5"]["useful_count"] == 1
    scope = baseline["per_day"][0]["diagnostics"]["research"]["scope"]
    assert scope["counts"] == {"off_interest": 1}
    assert scope["unknown"] == 2
    assert scope["rater_counts"] == {"analyst": 1}
    assert baseline["whole_slate"]["unknown"] == 1
    assert not report["promotion_eligible"]


def test_missed_important_papers_use_captured_pool_and_explicit_reader_labels():
    labels = [judgment(1, useful=True, important=True), judgment(4, useful=True, important=True)]
    challenger = {"candidate": {"run-1": {"research": [4, 1, 3]}}}
    report = compare(labels, challengers=challenger)
    base = report["variants"]["baseline"]["per_day"][0]
    other = report["variants"]["candidate"]["per_day"][0]
    assert base["diagnostics"]["research"]["observed_important_missed_ids"] == [4]
    assert other["diagnostics"]["research"]["observed_important_missed_ids"] == []
    paired = report["comparisons"]["candidate"]["research"]
    assert paired["per_run"][0]["must_read_recovered_delta"] == 1
    assert paired["per_run"][0]["must_read_available"] == 2
    assert not paired["policy_comparison_eligible"]  # Undeclared score changes cannot claim parity.


def test_declared_counterfactual_scores_verify_policy_without_changing_utility_pool():
    original = snapshot()
    changed = deepcopy(original)
    ordering = [4, 1, 3, 2, 5, 6]
    for candidate in changed["candidates"]:
        candidate["input_position"] = ordering.index(candidate["item_id"])
    descriptor = {"snapshot": changed, "slate": {"research": [1, 3, 4]}}
    before = deepcopy((original, changed))
    report = evaluate_digest_variants([original], [judgment(4, useful=True, important=True)],
                                      challengers={"contribution": {"run-1": descriptor}}, outcome_as_of=CUTOFF)
    assert (original, changed) == before
    variant = report["variants"]["contribution"]
    assert variant["per_day"][0]["policy_validation"]["status"] == "verified"
    assert report["comparisons"]["contribution"]["research"]["policy_comparison_eligible"]
    assert variant["replay_reports"][0]["pool_signature"] == report["variants"]["baseline"]["replay_reports"][0]["pool_signature"]
    assert not report["promotion_eligible"]


@pytest.mark.parametrize("mutation", ["text", "source", "family", "budget", "source_policy", "missing", "reenable"])
def test_counterfactual_cannot_rewrite_evidence_or_expand_eligibility(mutation):
    original = snapshot()
    changed = deepcopy(original)
    if mutation == "text":
        changed["candidates"][0]["metadata"]["abstract"] = "Different paper"
    elif mutation == "source":
        changed["candidates"][0]["selection_facts"]["is_high_quality"] = False
    elif mutation == "family":
        changed["candidates"][0]["family_id"] = "new-family"
    elif mutation == "budget":
        changed["section_budgets"]["research"] = 9
    elif mutation == "source_policy":
        changed["selection_settings"]["max_low_impact_research_frac"] = 1
    elif mutation == "missing":
        changed["candidates"].pop()
    else:
        original["candidates"][-1]["eligible"] = False
    with pytest.raises(ValueError, match="Counterfactual"):
        evaluate_digest_variants([original], [], outcome_as_of=CUTOFF,
                                 challengers={"bad": {"run-1": {"snapshot": changed, "slate": {"research": [1, 2, 3]}}}})


def test_multiday_repetition_uses_earlier_observed_slates_only():
    first, second = snapshot(1), snapshot(2)
    # The frozen family alias changes, but stable item identity still joins it.
    second["candidates"][0]["family_id"] = "changed-alias"
    report = evaluate_digest_variants([second, first], [], outcome_as_of=CUTOFF)
    rows = report["variants"]["baseline"]["per_day"]
    assert rows[0]["diagnostics"]["research"]["recent_repeated_ids"] == []
    assert rows[1]["diagnostics"]["research"]["recent_repeated_ids"] == [1]
    assert report["days"] == 2
    assert report["variants"]["baseline"]["aggregate"]["research"]["unknown"] == 6
    assert report["variants"]["baseline"]["diagnostic_totals"]["research"]["recent_repeated_count"] == 1
    isolated = evaluate_digest_variants([second], [], outcome_as_of=CUTOFF)
    assert isolated["variants"]["baseline"]["per_day"][0]["diagnostics"]["research"]["recent_repeated_ids"] == []


def test_duplicate_families_are_reported_invalid_without_sanitizing_slate():
    snap = snapshot()
    snap["candidates"][1]["family_id"] = snap["candidates"][0]["family_id"]
    report = evaluate_digest_variants([snap], [], outcome_as_of=CUTOFF)
    variant = report["variants"]["baseline"]
    assert variant["invalid_runs"] == 1
    assert variant["per_day"][0]["reader_metrics"] is None
    assert variant["per_day"][0]["diagnostics"]["research"]["duplicate_family_count"] == 1
    assert variant["per_day"][0]["slate"]["research"] == [1, 2, 3]


def test_future_labels_are_ignored_and_explicit_unknown_clears_usefulness():
    labels = [judgment(1, useful=True), judgment(1, useful=None, recorded_at="2026-10-04T00:00:00Z"),
              judgment(2, useful=True, recorded_at=CUTOFF)]
    report = compare(labels)
    assert report["judgments_used"] == 2
    assert report["variants"]["baseline"]["aggregate"]["research"]["unknown"] == 3


@pytest.mark.parametrize("fields", [
    {"rater": "analyst", "useful": True}, {"rater": "model", "scope": "core"},
    {"useful": False, "important": True}, {"useful": True, "grade": 10},
    {"useful": True, "grade": float("nan")}, {"scope": "context", "facets": ["RNA"]},
    {"recorded_at": "2026-10-03"}, {"scope": "maybe"}, {"family_id": "unrelated"},
])
def test_invalid_or_conflated_judgments_fail_loudly(fields):
    with pytest.raises(ValueError):
        compare([judgment(1, **fields)])


def test_blind_packet_is_deterministic_and_keeps_arm_membership_in_manifest():
    challenger = {"candidate": {"run-1": {"research": [4, 1, 3]}}}
    packet = build_judgment_packet([snapshot()], challengers=challenger, rejected_per_run=1, seed=7)
    assert packet == build_judgment_packet([snapshot()], challengers=challenger, rejected_per_run=1, seed=7)
    assert len(packet["examples"]) == 5
    for row in packet["examples"]:
        assert not set(row) & {"source", "score", "rank", "origins", "run_id", "item_id"}
        assert row["useful"] is None and row["scope"] is None
    refs = [ref for group in packet["manifest"]["references"].values() for ref in group]
    assert {1, 2, 3, 4} <= {ref["item_id"] for ref in refs}
    assert sum("rejected_sample" in ref["origins"] for ref in refs) == 1


def test_packet_families_and_inspected_status_never_cross_splits():
    first, second = snapshot(1), snapshot(2)
    second["candidates"][0]["metadata"]["abstract"] = "A different supplied version of the same paper."
    packet = build_judgment_packet([first, second], rejected_per_run=0, inspected_family_ids=["family-1"])
    by_family = {}
    for row in packet["examples"]:
        by_family.setdefault(row["family_id"], set()).add(row["split"])
        if row["title"] == "Paper 1":
            assert row["split"] == "inspected"
            assert row["inspection_status"] == "inspected"
        else:
            assert row["inspection_status"] == "unknown"
    assert all(len(splits) == 1 for splits in by_family.values())


def test_completed_packet_binds_judgments_to_exact_snapshot_and_text():
    snap = snapshot()
    packet = build_judgment_packet([snap], rejected_per_run=0)
    completed = deepcopy(packet["examples"][:1])
    completed[0].update(rater="reader", recorded_at="2026-10-03T00:00:00Z", useful=True)
    rows = expand_packet_judgments(completed, packet["manifest"], [snap])
    report = evaluate_digest_variants([snap], rows, outcome_as_of=CUTOFF)
    assert report["variants"]["baseline"]["aggregate"]["research"]["useful_count"] == 1
    completed[0]["abstract"] = "Substituted evidence"
    with pytest.raises(ValueError, match="text or family"):
        expand_packet_judgments(completed, packet["manifest"], [snap])
    snap["candidates"][0]["ranking_score"] = .1
    with pytest.raises(ValueError, match="different snapshots"):
        expand_packet_judgments([], packet["manifest"], [snap])
