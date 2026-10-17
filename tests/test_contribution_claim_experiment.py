"""Declared author-claim feature caches must match the serving transformation."""

from copy import deepcopy
import hashlib

import pytest

from dailydigest.rank.contribution_claims import FEATURE_PROTOCOL, contribution_text
from scripts import experiment_contribution_ranker as experiment
from tests.test_contribution_ranker_experiment import data as data


def _declare(data):
    benchmark, rows, manifest, snapshot = deepcopy(data)
    manifest["feature_protocol"] = deepcopy(FEATURE_PROTOCOL)
    for row in rows:
        row["input_text"] = contribution_text(row["title"], row["abstract"])
        row["input_text_sha256"] = hashlib.sha256(row["input_text"].encode()).hexdigest()
    return benchmark, rows, manifest, snapshot


@pytest.mark.parametrize("bad", ["unsupported", "null_protocol", "missing_text", "hash", "transform", "unused_row", "original_text", "model_hash"])
def test_invalid_declared_feature_cache_fails_before_fit(data, monkeypatch, bad):
    benchmark, rows, manifest, snapshot = _declare(data)
    if bad == "unsupported":
        manifest["feature_protocol"]["version"] = "unknown-transform"
    elif bad == "null_protocol":
        manifest["feature_protocol"] = None
    elif bad == "original_text":
        rows[0]["text_sha256"] = "a" * 64
    elif bad == "model_hash":
        manifest["model"]["threads"] += 1
    elif bad == "missing_text":
        rows[0].pop("input_text")
    elif bad == "hash":
        rows[0]["input_text_sha256"] = "a" * 64
    else:
        row = rows[0] if bad == "transform" else deepcopy(rows[0])
        row["input_text"] = "Different text with a self-consistent hash"
        row["input_text_sha256"] = hashlib.sha256(row["input_text"].encode()).hexdigest()
        if bad == "unused_row":
            row["id"] = "unused-extra-row"
            rows.append(row)

    def no_fit(*args, **kwargs):
        pytest.fail("Invalid feature protocol reached fitting")

    monkeypatch.setattr(experiment, "fit_contribution_ranker", no_fit)
    with pytest.raises(ValueError, match="feature|input|protocol|fingerprint"):
        experiment.run_experiment(benchmark, rows, manifest, snapshot=snapshot)


def test_declared_protocol_is_bound_into_artifact_and_legacy_is_unchanged(data):
    legacy = experiment.run_experiment(*data[:3], snapshot=data[3])
    declared = _declare(data)
    report = experiment.run_experiment(*declared[:3], snapshot=declared[3])
    assert "feature_protocol" not in legacy["artifact"]
    assert report["artifact"]["feature_protocol"] == FEATURE_PROTOCOL
    assert report["artifact"]["ranker"] == legacy["artifact"]["ranker"]
    assert report["challenger_slate"] == legacy["challenger_slate"]
    assert report["artifact"]["fingerprint"] != legacy["artifact"]["fingerprint"]


def test_snapshot_evaluation_compares_actual_final_scores_and_reports_exclusions(data):
    benchmark, rows, manifest, snapshot = _declare(data)
    for iid, label in ((2, "off_interest"), (6, "context"), (12, "core")):
        candidate = snapshot["candidates"][iid - 1]
        benchmark["examples"].append({
            "id": str(iid), "family_id": candidate["family_id"], "split": "evaluation",
            "title": candidate["metadata"]["title"], "abstract": candidate["metadata"]["abstract"],
            "label": label, "facet": next(iter(benchmark["definitions"]["core"])) if label == "core" else "",
            "rationale": "Frozen held-out analyst example", "evaluation_group": "present_holdout",
        })
    report = experiment.run_experiment(benchmark, rows, manifest, snapshot=snapshot)
    metrics = report["evaluation_final_score_metrics"]
    assert metrics["compared_example_count"] == 3
    assert metrics["excluded_missing_original_scores_count"] > 0
    assert metrics["baseline"]["balanced_pair_accuracy"] == 0.
    assert metrics["challenger"]["balanced_pair_accuracy"] == 1.
    assert metrics["groups"]["present_holdout"]["compared_example_count"] == 3
    assert metrics["groups"]["present_holdout"]["excluded_missing_original_scores_count"] == 0
    training = report["artifact"]["ranker"]["provenance"]["training_ids"]
    assert not {"2", "6", "12"}.intersection(training)
    changed = deepcopy(benchmark)
    for example in changed["examples"]:
        if example["split"] == "evaluation":
            example.update(label="off_interest", facet="")
    second = experiment.run_experiment(changed, rows, manifest, snapshot=snapshot)
    assert second["artifact"] == report["artifact"]
    assert second["evaluation_final_score_metrics"] != metrics


def test_legacy_historical_evaluation_does_not_invent_original_final_scores(data):
    report = experiment.run_experiment(*data[:3], snapshot=data[3])
    metrics = report["evaluation_final_score_metrics"]
    assert metrics["compared_example_count"] == 0
    assert metrics["excluded_missing_original_scores_count"] == metrics["evaluation_example_count"]
    assert metrics["baseline"]["balanced_pair_accuracy"] is None
    assert metrics["challenger"]["balanced_pair_accuracy"] is None
