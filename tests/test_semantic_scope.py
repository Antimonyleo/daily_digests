"""Scope decisions require contrastive evidence, independently of prestige."""
import numpy as np
import pytest

from dailydigest.rank.scope import (
    ScopeThresholds, anchor_records, classify_scope, score_scope, calibrate_scope,
    scope_metrics, validate_benchmark,
)


ABSTRACT = "We design and validate programmable molecular structures with controlled geometry and specific interactions."
DEFINITIONS = {"core": {"DNA design": ["Designed DNA assemblies"],
                         "Protein design": ["Designed protein assemblies"]},
               "context": ["Natural molecular mechanisms"],
               "off_interest": ["Disease diagnosis"]}


def evidence(scores=(.85, .82, .6, .5)):
    records = anchor_records(DEFINITIONS)
    # One orthogonal dimension per anchor lets scores be controlled exactly.
    return score_scope(np.array([scores]), np.eye(4), records)[0]


def test_multiple_core_matches_do_not_compete_with_each_other():
    ev = evidence((.85, .85, .6, .5))
    result = classify_scope(ev, ABSTRACT, ScopeThresholds(.4, .03))
    assert result["label"] == "core"
    assert result["facets"] == ["DNA design", "Protein design"]


@pytest.mark.parametrize("scores,label", [((.6, .5, .9, .3), "context"),
                                           ((.6, .5, .3, .9), "off_interest")])
def test_contrastive_winner_does_not_receive_a_core_label(scores, label):
    result = classify_scope(evidence(scores), ABSTRACT, ScopeThresholds(.4, .03))
    assert result["label"] == label
    assert result["facet"] == ""
    assert result["facets"] == []


@pytest.mark.parametrize("abstract", ["", "Promising new molecular design."])
def test_strong_title_similarity_cannot_replace_abstract_evidence(abstract):
    result = classify_scope(evidence(), abstract, ScopeThresholds(.4, .03))
    assert result["label"] == "uncertain"
    assert result["reason"] == "insufficient_abstract"


def test_cross_class_tie_and_low_similarity_abstain():
    assert classify_scope(evidence((.8, .5, .8, .3)), ABSTRACT,
                          ScopeThresholds(.4, 0))["label"] == "uncertain"
    assert classify_scope(evidence(), ABSTRACT,
                          ScopeThresholds(.99, .01))["label"] == "uncertain"


def test_zero_document_vector_abstains():
    ev = score_scope(np.zeros((1, 4)), np.eye(4), anchor_records(DEFINITIONS))[0]
    assert classify_scope(ev, ABSTRACT, ScopeThresholds(0, 0))["label"] == "uncertain"


@pytest.mark.parametrize("items,anchors", [
    (np.array([[np.nan, 1, 0, 0]]), np.eye(4)),
    (np.ones((1, 3)), np.eye(4)),
    (np.ones((1, 4)), np.zeros((4, 4))),
])
def test_invalid_vectors_fail_loudly(items, anchors):
    with pytest.raises(ValueError):
        score_scope(items, anchors, anchor_records(DEFINITIONS))


def test_row_scaling_cannot_change_cosine_decisions():
    records = anchor_records(DEFINITIONS)
    a = score_scope(np.array([[.8, .5, .3, .2]]), np.eye(4), records)
    b = score_scope(np.array([[8, 5, 3, 2]]), 10 * np.eye(4), records)
    assert b[0]["scores"] == pytest.approx(a[0]["scores"])


def test_metrics_expose_abstention_and_missed_core():
    examples = [{"label": "core", "facet": "DNA design"}, {"label": "context", "facet": ""}]
    metrics = scope_metrics(examples, [{"label": "uncertain"}, {"label": "uncertain"}])
    assert metrics["coverage"] == 0
    assert metrics["core_recall"] == 0
    assert metrics["core_precision"] is None
    assert metrics["counts"]["uncertain"] == 2


def test_uncertain_annotation_is_not_reported_as_proven_off_scope():
    examples = [{"label": label, "facet": "DNA design" if label == "core" else ""}
                for label in ("core", "context", "uncertain")]
    metrics = scope_metrics(examples, [{"label": "core"}] * 3)
    assert metrics["known_noncore_as_core"] == 1
    assert metrics["uncertain_as_core"] == 1
    assert metrics["core_precision_known_labels"] == .5
    assert metrics["core_precision_bounds"] == pytest.approx([1 / 3, 2 / 3])
    assert metrics["core_precision"] == 1 / 3  # frozen calibration convention


def test_calibration_refuses_evaluation_rows_and_repeated_families():
    example = {"id": "a", "family_id": "a", "split": "evaluation", "label": "core",
               "facet": "DNA design", "abstract": ABSTRACT}
    with pytest.raises(ValueError, match="calibration"):
        calibrate_scope([evidence()], [example])
    example["split"] = "calibration"
    with pytest.raises(ValueError, match="famil"):
        calibrate_scope([evidence(), evidence()], [example, {**example, "id": "b"}])


def test_calibration_marks_degenerate_classifier_unviable():
    examples = [{"id": "a", "family_id": "a", "split": "calibration", "label": "core",
                 "facet": "DNA design", "abstract": ABSTRACT}]
    calibrated = calibrate_scope([evidence((0, 0, 0, 0))], examples)
    assert calibrated["coverage_viable"] is False
    assert calibrated["promotion_eligible"] is False


def test_benchmark_cannot_leak_a_paper_family_across_splits():
    row = {"id": "a", "family_id": "same", "split": "calibration", "title": "A title",
           "abstract": ABSTRACT, "label": "core", "facet": "DNA design", "rationale": "Design"}
    data = {"schema_version": 1, "provenance": {"kind": "synthetic"}, "definitions": DEFINITIONS,
            "examples": [row, {**row, "id": "b", "split": "evaluation"}]}
    with pytest.raises(ValueError, match="famil"):
        validate_benchmark(data)


def test_different_dates_and_abstract_versions_cannot_hide_cross_split_title_alias():
    row = {"id": "a", "family_id": "old-date", "split": "calibration",
           "title": "Programmable DNA assemblies & molecular control across different length scales",
           "abstract": ABSTRACT, "label": "core", "facet": "DNA design", "rationale": "Design"}
    alias = {**row, "id": "b", "family_id": "new-date", "split": "evaluation",
             "title": "PROGRAMMABLE DNA ASSEMBLIES &amp; MOLECULAR CONTROL ACROSS DIFFERENT LENGTH SCALES.",
             "abstract": ABSTRACT + " A longer abstract version from another source."}
    data = {"schema_version": 1, "provenance": {"kind": "synthetic"}, "definitions": DEFINITIONS,
            "examples": [row, alias]}
    with pytest.raises(ValueError, match="title"):
        validate_benchmark(data)
