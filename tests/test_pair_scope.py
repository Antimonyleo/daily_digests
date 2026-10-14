"""Raw pair relevance logits must not masquerade as cosine or confidence."""
import numpy as np
import pytest

from dailydigest.rank.pair_scope import (
    PairScopeThresholds, pair_queries, pair_evidence, classify_pair_scope, calibrate_pair_scope,
)


DEFINITIONS = {"core": {"DNA design": ["Designed DNA structures", "broad alias"],
                         "Protein design": ["Designed proteins"]},
               "context": ["Natural mechanisms"], "off_interest": ["Diagnostic associations"]}
ABSTRACT = "This work designs and experimentally evaluates a programmable molecular structure with controlled function."


def evidence(scores):
    return pair_evidence(np.asarray([scores]), pair_queries(DEFINITIONS))[0]


def test_specific_first_core_anchor_and_competitors_are_frozen():
    queries = pair_queries(DEFINITIONS)
    assert [q["text"] for q in queries] == ["Designed DNA structures", "Designed proteins", "Natural mechanisms", "Diagnostic associations"]


def test_raw_logits_and_multiple_core_facets_are_supported():
    decision = classify_pair_scope(evidence([6, 5.8, -2, -4]), ABSTRACT, PairScopeThresholds(2, 1))
    assert decision["label"] == "core"
    assert decision["facets"] == ["DNA design", "Protein design"]
    assert decision["scores"]["core"] == 6
    assert "not probabilities" in decision["score_meaning"]


@pytest.mark.parametrize("scores", [[6, 5, 6, -4], [-9, -10, -10, -11]])
def test_ambiguous_or_weak_raw_logits_abstain(scores):
    assert classify_pair_scope(evidence(scores), ABSTRACT, PairScopeThresholds(-4, 0))["label"] == "uncertain"


def test_pair_score_does_not_replace_missing_article_evidence():
    result = classify_pair_scope(evidence([100, 2, 0, 0]), "", PairScopeThresholds(2, 1))
    assert result["label"] == "uncertain"
    assert result["reason"] == "insufficient_abstract"


@pytest.mark.parametrize("scores", [np.ones((2, 3)), [[1, np.nan, 3, 4]]])
def test_invalid_logit_matrix_fails(scores):
    with pytest.raises(ValueError):
        pair_evidence(scores, pair_queries(DEFINITIONS))


def test_calibration_requires_precision_as_well_as_retention():
    # Uninformative classifier predicts core for positives and negatives alike.
    examples = [{"id": str(i), "family_id": str(i), "split": "calibration",
                 "label": "core" if i == 0 else "context", "facet": "DNA design" if i == 0 else "",
                 "abstract": ABSTRACT} for i in range(5)]
    report = calibrate_pair_scope([evidence([5, 4, -2, -3])] * 5, examples)
    assert not report["development_constraints_met"]
    assert not report["promotion_eligible"]


def test_calibration_never_consumes_evaluation_labels():
    row = {"id": "a", "family_id": "a", "split": "evaluation", "label": "core",
           "facet": "DNA design", "abstract": ABSTRACT}
    with pytest.raises(ValueError, match="calibration"):
        calibrate_pair_scope([evidence([5, 4, -2, -3])], [row])
