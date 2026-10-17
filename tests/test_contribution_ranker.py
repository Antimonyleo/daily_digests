"""Pairwise contribution learning must remain calibration-only and reproducible."""
from copy import deepcopy
import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from dailydigest.rank.contribution_ranker import (
    MODEL_PARAMETERS, ContributionRanker, fit_contribution_ranker,
)
from dailydigest.rank.scope import fingerprint


@pytest.fixture
def sample():
    queries = [
        {"label": label, "facet": f"facet-{i}" if label == "core" else "",
         "text": f"Frozen anchor {label} {i}"}
        for label in ("core", "context", "off_interest") for i in range(8)
    ]
    examples, values = [], []
    rng = np.random.default_rng(4)
    for tier, label in enumerate(("core", "context", "off_interest")):
        for family in range(6):
            examples.append({"id": f"{label}-{family}", "family_id": f"{label}-{family}",
                             "split": "calibration", "label": label})
            row = rng.normal(0, .2, 24)
            row[tier * 8:(tier + 1) * 8] += 4
            values.append(row)
    return np.asarray(values), queries, examples


def test_scores_rerank_tiers_and_preserve_document_order(sample):
    values, queries, examples = sample
    model = fit_contribution_ranker(values, queries, examples)
    scores = model.predict_scores(values, queries=queries)
    assert scores.shape == (18,)
    assert np.all((scores >= 0) & (scores <= 1))
    assert min(scores[:6]) > max(scores[6:12]) > max(scores[12:])
    np.testing.assert_allclose(model.predict_scores(values[[14, 0, 8]]), scores[[14, 0, 8]])
    assert len(set(np.round(scores[:6], 6))) > 1  # Continuous ranking, not a class gate.
    assert model.predict_scores(np.empty((0, 24))).shape == (0,)
    assert model.provenance["promotion_eligible"] is False
    assert "not probabilities" in model.provenance["score_meaning"]


def test_uncertain_and_family_versions_cannot_change_training(sample):
    values, queries, examples = sample
    baseline = fit_contribution_ranker(values, queries, examples)
    extra = [dict(examples[0], id="another-version"),
             {"id": "unknown", "family_id": "unknown", "split": "calibration", "label": "uncertain"}]
    model = fit_contribution_ranker(np.vstack([values, np.full((2, 24), 1e6)]), queries,
                                    examples + extra)
    np.testing.assert_allclose(model.predict_scores(values), baseline.predict_scores(values))
    assert model.provenance["excluded_uncertain_ids"] == ["unknown"]
    assert model.provenance["excluded_family_version_ids"] == ["another-version"]
    np.testing.assert_allclose(model.mean, values.mean(axis=0))
    assert model.provenance["training_ids"] == [example["id"] for example in examples]


def test_uncertain_first_version_does_not_mask_definitive_family(sample):
    values, queries, examples = sample
    uncertain = dict(examples[0], id="uncertain-first", label="uncertain")
    model = fit_contribution_ranker(np.vstack([np.full(24, 1e6), values]), queries,
                                    [uncertain] + examples)
    assert examples[0]["id"] in model.provenance["training_ids"]
    np.testing.assert_allclose(model.mean, values.mean(axis=0))


def test_oof_families_disjoint_and_scale_is_held_out_only(sample):
    values, queries, examples = sample
    model = fit_contribution_ranker(values, queries, examples)
    provenance = model.provenance
    by_id = {example["id"]: i for i, example in enumerate(examples)}
    oof = {}
    assert len(provenance["oof_folds"]) == 4
    for fold in provenance["oof_folds"]:
        assert set(fold["train_families"]).isdisjoint(fold["validation_families"])
        train_rows = [by_id[ident] for ident in fold["train_ids"]]
        np.testing.assert_allclose(fold["state"]["mean"], values[train_rows].mean(axis=0))
        for ident, projection in zip(fold["validation_ids"], fold["validation_projections"], strict=True):
            assert ident not in oof
            oof[ident] = projection
            expected = ((values[by_id[ident]] - fold["state"]["mean"]) /
                        fold["state"]["scale"]) @ fold["state"]["coefficients"]
            assert projection == pytest.approx(expected)
    assert set(oof) == set(by_id)
    assert model.oof_mean == pytest.approx(np.mean(list(oof.values())))
    assert model.oof_std == pytest.approx(np.std(list(oof.values())))
    final_projection = (values - model.mean) / model.scale @ model.coefficients
    assert model.oof_std != pytest.approx(final_projection.std())
    assert provenance["oof_metrics"]["balanced_pair_accuracy"] == 1


def test_pair_weights_equalize_uneven_tier_pairs(sample):
    values, queries, examples = sample
    indices = list(range(18)) + [0, 1]
    examples = examples + [dict(examples[i], id=f"extra-{i}", family_id=f"extra-{i}") for i in (0, 1)]
    model = fit_contribution_ranker(values[indices], queries, examples)
    pairs = model.provenance["training_pairs"]
    assert pairs["unordered_pair_counts"] == {"core_context": 48, "core_off_interest": 48,
                                               "context_off_interest": 36}
    assert set(pairs["total_weight_by_tier_pair"].values()) == {88.0}
    assert pairs["mirrored_rows"] == 264
    assert "core_core" not in pairs["unordered_pair_counts"]


def test_actual_training_uses_mirrored_unequal_tier_pairs_and_fixed_parameters(sample, monkeypatch):
    values, queries, examples = sample
    calls = []
    original = LogisticRegression.fit

    def inspect_fit(self, features, targets, *, sample_weight):
        calls.append((self.get_params(), features.copy(), targets.copy(), sample_weight.copy()))
        return original(self, features, targets, sample_weight=sample_weight)

    monkeypatch.setattr(LogisticRegression, "fit", inspect_fit)
    model = fit_contribution_ranker(values, queries, examples)
    assert len(calls) == 5
    parameters, features, targets, weights = calls[-1]
    assert all(parameters[key] == value for key, value in MODEL_PARAMETERS.items())
    scaled = (values - model.mean) / model.scale
    expected = np.asarray([scaled[i] - scaled[j]
                           for left, right in ((range(6), range(6, 12)),
                                               (range(6), range(12, 18)),
                                               (range(6, 12), range(12, 18)))
                           for i in left for j in right])
    np.testing.assert_allclose(features[:108], expected)
    np.testing.assert_allclose(features[108:], -expected)
    np.testing.assert_array_equal(targets[:108], np.ones(108))
    np.testing.assert_array_equal(targets[108:], np.zeros(108))
    np.testing.assert_array_equal(weights, np.ones(216))


@pytest.mark.parametrize("field,value", [("split", "evaluation"), ("label", "unsupported"),
                                         ("id", ""), ("family_id", "")])
def test_reject_invalid_or_noncalibration_examples(sample, field, value):
    values, queries, examples = sample
    examples = deepcopy(examples)
    examples[-1][field] = value
    with pytest.raises(ValueError):
        fit_contribution_ranker(values, queries, examples)


def test_conflicting_family_and_duplicate_ids_rejected(sample):
    values, queries, examples = sample
    conflicting = dict(examples[0], id="conflicting-version", label="context")
    with pytest.raises(ValueError, match="conflicting"):
        fit_contribution_ranker(np.vstack([values, values[0]]), queries, examples + [conflicting])
    with pytest.raises(ValueError, match="unique"):
        fit_contribution_ranker(np.vstack([values, values[0]]), queries, examples + [examples[0]])


def test_insufficient_independent_families_rejected(sample):
    values, queries, examples = sample
    with pytest.raises(ValueError, match="4"):
        fit_contribution_ranker(values[3:], queries, examples[3:])


@pytest.mark.parametrize("bad", [np.ones(24), np.ones((18, 23)), np.full((18, 24), np.nan),
                                 np.full((18, 24), np.inf)])
def test_invalid_features_rejected_at_fit_and_predict(sample, bad):
    values, queries, examples = sample
    with pytest.raises(ValueError):
        fit_contribution_ranker(bad, queries, examples)
    model = fit_contribution_ranker(values, queries, examples)
    with pytest.raises(ValueError):
        model.predict_scores(bad)


def test_query_provenance_order_and_input_copying(sample):
    values, queries, examples = sample
    model = fit_contribution_ranker(values, queries, examples)
    reordered = list(reversed(queries))
    with pytest.raises(ValueError, match="queries"):
        model.predict_scores(values, queries=reordered)
    original_hash = model.provenance["query_hash"]
    queries[0]["text"] = "changed"
    assert model.provenance["query_hash"] == original_hash
    assert fingerprint(model.queries) == original_hash


def test_constant_features_have_finite_neutral_ranking(sample):
    values, queries, examples = sample
    model = fit_contribution_ranker(np.zeros_like(values), queries, examples)
    assert model.oof_std == 1.0
    assert model.provenance["degenerate_oof_scale"] is True
    np.testing.assert_array_equal(model.predict_scores(values), np.full(18, .5))


def test_json_roundtrip_and_existing_output_preserved(sample, tmp_path):
    values, queries, examples = sample
    model = fit_contribution_ranker(values, queries, examples)
    target = tmp_path / "ranker.json"
    model.save(target)
    before = target.read_bytes()
    loaded = ContributionRanker.load(target)
    np.testing.assert_array_equal(loaded.predict_scores(values), model.predict_scores(values))
    assert loaded.to_artifact() == json.loads(before)
    with pytest.raises(FileExistsError):
        model.save(target)
    assert target.read_bytes() == before


@pytest.mark.parametrize("mutation", ["hash", "version", "width", "scale", "query_order", "nonfinite"])
def test_corrupt_or_incompatible_artifacts_rejected(sample, mutation):
    values, queries, examples = sample
    artifact = fit_contribution_ranker(values, queries, examples).to_artifact()
    if mutation == "hash":
        artifact["state"]["coefficients"][0] += 1
    else:
        if mutation == "version":
            artifact["version"] = "future"
        elif mutation == "width":
            artifact["state"]["coefficients"].pop()
        elif mutation == "scale":
            artifact["state"]["scale"][0] = 0
        elif mutation == "query_order":
            artifact["queries"].reverse()
        elif mutation == "nonfinite":
            artifact["state"]["coefficients"][0] = float("nan")
        if mutation != "nonfinite":
            artifact["fingerprint"] = fingerprint({k: v for k, v in artifact.items() if k != "fingerprint"})
    with pytest.raises(ValueError):
        ContributionRanker.from_artifact(artifact)
