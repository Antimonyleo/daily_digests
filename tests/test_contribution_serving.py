"""Explicit local serving artifacts must match training columns and reader scope."""

from copy import deepcopy
import importlib
import json
from datetime import date, datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from dailydigest.rank.contribution_ranker import fit_contribution_ranker
from dailydigest.rank.pair_scope import pair_queries
from dailydigest.rank.scope import fingerprint
from dailydigest.rank.semantic_rerank import LocalContributionReranker, interest_signature


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    # Initialize import-time aliases before installing a per-test settings guard.
    importlib.import_module('dailydigest.pipeline')
    definitions = {"core": {f"facet-{i}": [f"Core interest {i}"] for i in range(8)},
                   "context": [f"Context interest {i}" for i in range(8)],
                   "off_interest": [f"Excluded interest {i}" for i in range(8)]}
    queries = pair_queries(definitions)
    examples, values = [], []
    rng = np.random.default_rng(12)
    for tier, label in enumerate(("core", "context", "off_interest")):
        for family in range(6):
            examples.append({"id": f"{label}-{family}", "family_id": f"{label}-{family}",
                             "split": "calibration", "label": label})
            row = rng.normal(0, .15, 24)
            row[tier * 8:(tier + 1) * 8] += 3
            values.append(row)
    model = fit_contribution_ranker(np.asarray(values), queries, examples)
    profile = {"bio": "Engineering", "keywords": ["protein design"],
               "canonical_facets": {}, "negative_interests": {"clinical trials": 1.},
               "context_keywords": ["natural structures"]}
    provenance = {"max_length": 512, "batch_size": 16, "threads": 1,
                  "assets_sha256": {"model.onnx": "frozen-assets"}, "fingerprint": "encoder"}
    artifact = {"schema_version": 1, "kind": "local_contribution_reranker",
                "ranker": model.to_artifact(), "queries": queries, "definitions": definitions,
                "encoder_provenance": provenance,
                "profile_interest_signature": interest_signature(profile)}
    artifact["fingerprint"] = fingerprint(artifact)
    path = tmp_path / "ranker.json"
    path.write_text(json.dumps(artifact))
    calls = []
    logits = {"Core paper": np.asarray(values[0]), "Context paper": np.asarray(values[6])}

    class Encoder:
        def __init__(self, model_dir, **settings):
            calls.append((model_dir, settings))
            self.provenance = deepcopy(provenance)

        def score_pairs(self, pairs):
            calls.append(pairs)
            assert [query for query, _text in pairs] == [q["text"] for q in queries]
            return logits[pairs[0][1].split(". ", 1)[0]].copy()

    monkeypatch.setattr("dailydigest.rank.cross_encoder.LocalCrossEncoder", Encoder)

    def forbidden(*args, **kwargs):
        pytest.fail("Local contribution serving must not access network or default providers")

    monkeypatch.setattr("httpx.Client", forbidden)
    monkeypatch.setattr("dailydigest.config.get_settings", forbidden)
    return SimpleNamespace(path=path, artifact=artifact, profile=profile, model=model,
                           queries=queries, values=values, calls=calls, logits=logits)


def _write(bundle, artifact):
    artifact = deepcopy(artifact)
    artifact["fingerprint"] = fingerprint({k: v for k, v in artifact.items() if k != "fingerprint"})
    bundle.path.write_text(json.dumps(artifact))


def _row(iid, title="Core paper", section="research", abstract=None):
    return SimpleNamespace(id=iid, title=title, section=section, abstract=(
        "This abstract provides sufficient text to describe the investigated contribution."
        if abstract is None else abstract))


def test_serving_scores_match_actual_fitted_model_and_do_not_call_network(bundle):
    serving = LocalContributionReranker(bundle.path, "/explicit/local/model", bundle.profile)
    core, context = _row(1), _row(2, "Context paper")
    features = {1: {"topic_score": .85, "quality_score": .9},
                2: {"topic_score": .85, "quality_score": .9}}
    predictions = bundle.model.predict_scores(np.asarray([bundle.values[0], bundle.values[6]]),
                                              queries=bundle.queries)
    result = serving.rerank([(context, .9), (core, .8)], features)
    scores = {row.id: score for row, score in result}
    assert scores[1] == pytest.approx(.7 * predictions[0] + .3 * .8)
    assert scores[2] == pytest.approx(.7 * predictions[1] + .3 * .9)
    assert [row.id for row, _score in result] == [1, 2]
    assert bundle.calls[0] == ("/explicit/local/model", {"max_length": 512, "batch_size": 16, "threads": 1})
    for iid, score in scores.items():
        assert features[iid]["final_score"] == score
        assert features[iid]["quality_score"] == .9
        assert features[iid]["topic_score"] == .85
        assert features[iid]["contribution_rerank"]["model_fingerprint"] == bundle.artifact["fingerprint"]


@pytest.mark.parametrize("field,value", [("negative_interests", {"diagnostics": 1.}),
                                         ("context_keywords", ["medicine"]),
                                         ("bio", "Clinical medicine")])
def test_actual_reader_interest_changes_reject_artifact_before_encoder_load(bundle, field, value):
    profile = {**bundle.profile, field: value}
    with pytest.raises(ValueError, match="reader interest"):
        LocalContributionReranker(bundle.path, "/local", profile)
    assert not bundle.calls


def test_outer_artifact_corruption_is_rejected_before_encoder_load(bundle):
    artifact = deepcopy(bundle.artifact)
    artifact["ranker"]["state"]["coefficients"][0] += 1
    bundle.path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="identity or schema"):
        LocalContributionReranker(bundle.path, "/local", bundle.profile)
    assert not bundle.calls


def test_rehashed_envelope_cannot_hide_corrupted_numeric_ranker(bundle):
    artifact = deepcopy(bundle.artifact)
    artifact["ranker"]["state"]["scale"][0] = 0
    ranker = artifact["ranker"]
    ranker["fingerprint"] = fingerprint({k: v for k, v in ranker.items() if k != "fingerprint"})
    _write(bundle, artifact)
    with pytest.raises(ValueError, match="scales"):
        LocalContributionReranker(bundle.path, "/local", bundle.profile)
    assert not bundle.calls


def test_query_definitions_cannot_change_columns_without_retraining(bundle):
    artifact = deepcopy(bundle.artifact)
    artifact["definitions"]["context"].reverse()
    artifact["queries"] = pair_queries(artifact["definitions"])
    _write(bundle, artifact)
    with pytest.raises(ValueError, match="frozen ordered queries"):
        LocalContributionReranker(bundle.path, "/local", bundle.profile)
    assert not bundle.calls


def test_declared_queries_must_equal_definition_queries(bundle):
    artifact = deepcopy(bundle.artifact)
    artifact["queries"][0]["text"] = "Different query"
    _write(bundle, artifact)
    with pytest.raises(ValueError, match="query definitions"):
        LocalContributionReranker(bundle.path, "/local", bundle.profile)
    assert not bundle.calls


def test_encoder_asset_mismatch_is_rejected(bundle):
    artifact = deepcopy(bundle.artifact)
    artifact["encoder_provenance"]["assets_sha256"]["model.onnx"] = "different-assets"
    _write(bundle, artifact)
    with pytest.raises(ValueError, match="assets/runtime"):
        LocalContributionReranker(bundle.path, "/local", bundle.profile)


def test_missing_abstract_and_nonresearch_rows_do_not_call_encoder_or_change_features(bundle):
    serving = LocalContributionReranker(bundle.path, "/local", bundle.profile)
    rows = [_row(1, abstract="Short."), _row(2, section="world")]
    features = {r.id: {"final_score": .8} for r in rows}
    before = deepcopy(features)
    result = serving.rerank([(r, .8) for r in rows], features)
    assert result == [(r, .8) for r in rows]
    assert len(bundle.calls) == 1
    assert features == before


def test_later_encoder_failure_cannot_partially_mutate_features(bundle):
    serving = LocalContributionReranker(bundle.path, "/local", bundle.profile)
    rows = [_row(1), _row(2, "Context paper")]
    features = {r.id: {"final_score": .8} for r in rows}
    before = deepcopy(features)
    original = serving.encoder.score_pairs
    calls = 0

    def fail_second(pairs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Local encoder failed")
        return original(pairs)

    serving.encoder.score_pairs = fail_second
    with pytest.raises(RuntimeError, match="Local encoder failed"):
        serving.rerank([(r, .8) for r in rows], features)
    assert features == before


def test_nonfinite_encoder_scores_cannot_partially_mutate_features(bundle):
    serving = LocalContributionReranker(bundle.path, "/local", bundle.profile)
    serving.encoder.score_pairs = lambda pairs: np.full(len(pairs), np.nan)
    features = {1: {"final_score": .8}}
    with pytest.raises(ValueError, match="finite"):
        serving.rerank([(_row(1), .8)], features)
    assert features == {1: {"final_score": .8}}


@pytest.mark.parametrize("paths,weight", [(("", ""), .7), (("broken.json", "model"), 0.)])
def test_pipeline_disabled_paths_or_weight_never_load_a_model(bundle, monkeypatch, paths, weight):
    from dailydigest import pipeline

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled contribution reranking must not construct a model")

    monkeypatch.setattr("dailydigest.rank.semantic_rerank.LocalContributionReranker", forbidden)
    settings = SimpleNamespace(contribution_reranker_path=paths[0], contribution_encoder_dir=paths[1],
                               contribution_reranker_weight=weight)
    scored, features = [(_row(1), .8)], {1: {"final_score": .8}}
    assert pipeline._apply_semantic_reranking(scored, features, bundle.profile, settings) is scored
    assert features == {1: {"final_score": .8}}
    assert not bundle.calls


def test_pipeline_requires_both_explicit_local_paths(bundle):
    from dailydigest import pipeline

    settings = SimpleNamespace(contribution_reranker_path=str(bundle.path), contribution_encoder_dir="")
    with pytest.raises(ValueError, match="both local paths"):
        pipeline._apply_semantic_reranking([(_row(1), .8)], {}, bundle.profile, settings)
    assert not bundle.calls


def test_settings_defaults_leave_learned_reranking_inactive():
    from dailydigest.config import Settings

    assert Settings.model_fields["contribution_reranker_path"].default == ""
    assert Settings.model_fields["contribution_encoder_dir"].default == ""


def test_adjusted_top_candidate_survives_actual_dedupe_and_snapshot_scores_agree(bundle, monkeypatch):
    from dailydigest import pipeline
    from dailydigest.config import Settings
    from dailydigest.dedupe import cap_near_duplicates
    from dailydigest.models import Profile
    from dailydigest.rank import snapshot

    profile = Profile.model_validate(bundle.profile)
    rows = [_row(1), _row(2, "Context paper")]
    for row in rows:
        row.source, row.external_id = "Nature", str(row.id)
        row.url, row.metadata_json = f"https://example.org/{row.id}", "{}"
        row.authors = "A. Researcher"
        row.published_at = datetime(2026, 10, 1, tzinfo=timezone.utc)
        row.fetched_at = datetime(2026, 10, 2, tzinfo=timezone.utc)
    settings = Settings(_env_file=None, contribution_reranker_path=str(bundle.path),
                        contribution_encoder_dir="/explicit/local/model")
    features = {row.id: {"topic_score": .9, "quality_score": .9} for row in rows}
    reranked = pipeline._apply_semantic_reranking([(rows[1], .9), (rows[0], .8)],
                                                 features, profile, settings)
    kept = cap_near_duplicates([row for row, _score in reranked], np.ones((2, 3)), .86)
    picked = [reranked[index] for index in kept]
    assert [row.id for row, _score in picked] == [1]
    monkeypatch.setattr(snapshot, "active_embedding_signature", lambda **kwargs: "frozen")
    monkeypatch.setattr(snapshot, "active_embedding_provenance", lambda: {
        "model": settings.embed_model, "actual_backend": "fixture", "status": "resolved",
    })
    facts = {row.id: {"source_bucket": "published_journal", "is_preprint": False,
                     "is_arxiv_cs": False, "is_published": True,
                     "is_high_quality": True, "is_low_impact": False} for row in rows}
    frozen = snapshot.build_ranking_snapshot(
        run_id="semantic-test", digest_id="2026-10-02",
        decision_at=datetime(2026, 10, 2, tzinfo=timezone.utc),
        decision_local_date=date(2026, 10, 2), reading_mode="full", candidates=rows,
        scored=[(rows[1], .9), (rows[0], .8)], eligible=reranked,
        labeled=[(row, score, "R1") for row, score in picked], features=features,
        section_budgets={"research": 1}, settings=settings, profile=profile,
        profile_matrix=np.array([[1., 0.]]), catch_up=False, ingest_stats=[],
        feedback_before={}, feedback_after={}, stages={}, selection_facts=facts,
    )
    selected = next(row for row in frozen["candidates"] if row["selected"])
    assert frozen["embedding_provenance"]["actual_backend"] == "fixture"
    assert selected["item_id"] == 1
    assert selected["ranking_score"] == picked[0][1]
    assert selected["features"]["final_score"] == picked[0][1]
    assert selected["features"]["contribution_rerank"]["model_fingerprint"] == bundle.artifact["fingerprint"]
