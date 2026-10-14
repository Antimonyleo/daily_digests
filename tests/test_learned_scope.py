"""Guard the supervised experiment against label and family leakage."""
from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from dailydigest.rank.learned_scope import fit_learned_scope
from dailydigest.rank.pair_scope import pair_evidence
from dailydigest.rank.scope import fingerprint


ABSTRACT = "This supplied abstract describes a substantive contribution and enough evidence to permit an assessment of research scope."


@pytest.fixture
def sample():
    queries = [
        {"label": label, "facet": f"facet-{i}" if label == "core" else "",
         "text": f"Contribution anchor {label} {i}"}
        for label in ("core", "context", "off_interest") for i in range(8)
    ]
    examples, values = [], []
    rng = np.random.default_rng(4)
    for class_index, label in enumerate(("core", "context", "off_interest")):
        for group in range(6):
            for version in range(2):
                family = f"{label}-{group}"
                examples.append({
                    "id": f"{family}-{version}", "family_id": family, "split": "calibration",
                    "label": label, "facet": "facet-0" if label == "core" else "",
                    "abstract": ABSTRACT,
                })
                row = rng.normal(0, .2, 24)
                row[class_index * 8:(class_index + 1) * 8] += 4
                values.append(row)
    return np.asarray(values), queries, examples


def test_uncertain_rows_never_train_model_scaler_or_thresholds(sample):
    values, queries, examples = sample
    baseline = fit_learned_scope(values, queries, examples)
    uncertain = {"id": "ambiguous", "family_id": "ambiguous", "split": "calibration",
                 "label": "uncertain", "facet": "", "abstract": ""}
    fitted = fit_learned_scope(np.vstack([values, np.full(24, 1e6)]), queries,
                               examples + [uncertain])
    provenance = fitted.provenance
    assert provenance["excluded_uncertain_ids"] == ["ambiguous"]
    assert "ambiguous" not in provenance["training_ids"]
    assert "ambiguous" not in provenance["calibration_report"]["calibration_ids"]
    assert provenance["model_fingerprint"] == baseline.provenance["model_fingerprint"]
    assert provenance["thresholds"] == baseline.provenance["thresholds"]
    np.testing.assert_allclose(fitted.scaler.mean_, values.mean(axis=0))
    assert not provenance["promotion_eligible"]


def test_oof_holds_entire_families_out_and_calibrates_only_representatives(sample):
    values, queries, examples = sample
    fitted = fit_learned_scope(values, queries, examples)
    provenance = fitted.provenance
    by_id = {example["id"]: example for example in examples}
    held_out = []
    for fold in provenance["oof_folds"]:
        assert set(fold["train_families"]).isdisjoint(fold["validation_families"])
        assert set(fold["train_ids"]).isdisjoint(fold["validation_ids"])
        assert set(fold["train_ids"]) | set(fold["validation_ids"]) == set(by_id)
        for ident in fold["validation_ids"]:
            assert by_id[ident]["family_id"] in fold["validation_families"]
        held_out.extend(fold["validation_ids"])
    assert Counter(held_out) == Counter(by_id.keys())
    ids = provenance["calibration_report"]["calibration_ids"]
    assert len(ids) == len({example["family_id"] for example in examples})
    assert all(ident.endswith("-0") for ident in ids)


def test_prediction_preserves_row_order_and_relative_facet_query_gaps(sample):
    values, queries, examples = sample
    fitted = fit_learned_scope(values, queries, examples)
    prediction_rows = values[[27, 1, 19]].copy()
    prediction_rows[1, 0] = 9
    prediction_rows[1, 1] = 7.25
    evidence = fitted.predict_evidence(prediction_rows)
    model_scores = fitted.classifier.decision_function(fitted.scaler.transform(prediction_rows))
    raw_evidence = pair_evidence(prediction_rows, queries)
    for row, scores, raw in zip(evidence, model_scores, raw_evidence, strict=True):
        assert row["valid_vector"] is True
        assert row["matched_anchors"] == raw["matched_anchors"]
        for label, score in zip(fitted.classifier.classes_, scores, strict=True):
            assert row["scores"][label] == pytest.approx(score)
        assert max(row["facet_scores"].values()) == pytest.approx(row["scores"]["core"])
        for facet, raw_score in raw["facet_scores"].items():
            assert row["scores"]["core"] - row["facet_scores"][facet] == pytest.approx(
                max(raw["facet_scores"].values()) - raw_score
            )
    assert evidence[1]["facet_scores"]["facet-0"] - evidence[1]["facet_scores"]["facet-1"] == pytest.approx(1.75)
    assert fitted.predict_evidence(np.empty((0, 24))) == []


@pytest.mark.parametrize("bad", [np.ones(24), np.ones((2, 23)), np.full((1, 24), np.nan),
                                 np.full((1, 24), np.inf)])
def test_prediction_rejects_misaligned_or_nonfinite_features(sample, bad):
    fitted = fit_learned_scope(*sample)
    with pytest.raises(ValueError, match="finite document-by-query"):
        fitted.predict_evidence(bad)


@pytest.mark.parametrize("split", ["evaluation", "test"])
def test_noncalibration_labels_rejected_even_when_uncertain(sample, split):
    values, queries, examples = sample
    examples[0].update(split=split, label="uncertain", facet="")
    with pytest.raises(ValueError, match="Only calibration"):
        fit_learned_scope(values, queries, examples)


def test_missing_class_or_too_few_independent_families_fails(sample):
    values, queries, examples = sample
    indices = [i for i, ex in enumerate(examples)
               if ex["label"] != "off_interest" or int(ex["family_id"].rsplit("-", 1)[1]) < 3]
    with pytest.raises(ValueError, match="at least 4 calibration families"):
        fit_learned_scope(values[indices], queries, [examples[i] for i in indices])
    indices = [i for i, ex in enumerate(examples) if ex["label"] != "off_interest"]
    with pytest.raises(ValueError, match="at least 4 calibration families"):
        fit_learned_scope(values[indices], queries, [examples[i] for i in indices])


def test_unknown_class_and_conflicting_family_labels_fail(sample):
    values, queries, examples = sample
    invalid = deepcopy(examples)
    invalid[0]["label"] = "interesting"
    with pytest.raises(ValueError, match="Unsupported scope class"):
        fit_learned_scope(values, queries, invalid)
    invalid = deepcopy(examples)
    invalid[1].update(label="context", facet="")
    with pytest.raises(ValueError, match="conflicting definitive"):
        fit_learned_scope(values, queries, invalid)


def test_provenance_captures_feature_order_and_fixed_fitted_parameters(sample):
    values, queries, examples = sample
    fitted = fit_learned_scope(values, queries, examples)
    provenance = fitted.provenance
    assert provenance["feature_ordering"] == queries
    assert provenance["query_hash"] == fingerprint(queries)
    state = provenance["model_state"]
    assert state["model_parameters"]["C"] == 1
    assert state["model_parameters"]["class_weight"] == "balanced"
    assert state["model_parameters"]["max_iter"] == 2000
    assert state["model_parameters"]["random_state"] == 0
    assert provenance["model_fingerprint"] == fingerprint(state)
    assert set(state["classes"]) == {"core", "context", "off_interest"}
    assert np.asarray(state["coefficients"]).shape == (3, 24)
    assert np.asarray(state["intercepts"]).shape == (3,)
    queries[0]["text"] = "Caller mutated the input after fitting"
    assert fitted.queries[0]["text"] != queries[0]["text"]
    assert provenance["feature_ordering"] == fitted.queries


@pytest.mark.parametrize("kind", ["width", "nan", "count", "query_label", "query_facet"])
def test_fit_rejects_invalid_input(sample, kind):
    values, queries, examples = sample
    if kind == "width":
        values = values[:, :-1]
    elif kind == "nan":
        values[0, 0] = np.nan
    elif kind == "count":
        examples = examples[:-1]
    elif kind == "query_label":
        queries[0]["label"] = "uncertain"
    else:
        queries[0]["facet"] = ""
    with pytest.raises(ValueError):
        fit_learned_scope(values, queries, examples)


def test_each_oof_prediction_comes_from_a_model_that_never_saw_its_family(sample):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    values, queries, examples = sample
    # Outlier families make leakage through a global scaler observable.
    values[0:2] *= 8
    fitted = fit_learned_scope(values, queries, examples)
    by_id = {example["id"]: index for index, example in enumerate(examples)}
    saved = {row["id"]: row for row in fitted.provenance["oof_evidence"]}
    for fold in fitted.provenance["oof_folds"]:
        train = [by_id[ident] for ident in fold["train_ids"]]
        validation = [by_id[ident] for ident in fold["validation_ids"]]
        scaler = StandardScaler().fit(values[train])
        classifier = LogisticRegression(C=1, class_weight="balanced", max_iter=2000, random_state=0)
        classifier.fit(scaler.transform(values[train]), [examples[index]["label"] for index in train])
        expected = classifier.decision_function(scaler.transform(values[validation]))
        for ident, row in zip(fold["validation_ids"], expected, strict=True):
            assert saved[ident]["scores"] == pytest.approx(dict(zip(classifier.classes_, row, strict=True)))
