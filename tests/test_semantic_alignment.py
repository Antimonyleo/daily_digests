"""Bounded joint topic matching, with no network or model inference."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from dailydigest.models import Profile


def row(iid, *, abstract=None, section="research"):
    return SimpleNamespace(id=iid, title="Measured result", section=section,
                           abstract=abstract if abstract is not None else "Measured evidence. " * 10)


def install(monkeypatch, results=None):
    from dailydigest.rank import semantic_alignment as module

    records = []

    class Backend:
        def score_records(self, batch, *, check_cancel=None):
            records.extend(batch)
            if check_cancel:
                check_cancel()
            return results if results is not None else [
                {"logit": 6., "windows": 1, "complete": True} for _ in batch]

    monkeypatch.setattr(module, "_build_backend", lambda model_dir=None: Backend())
    return module, records


def test_explicit_intent_verifies_without_replacing_quality_scores(monkeypatch):
    module, records = install(monkeypatch)
    profile = Profile(bio="Private biography must not become a query", keywords=["catalysis"],
                      canonical_facets={"catalysis": {"description": "Design selective catalysts",
                                                       "aliases": ["organocatalysis"]}})
    features = {1: {"topic_score": .72, "final_score": .6}}
    result = module.apply_topic_alignment([(row(1), .6)], features, profile)
    assert records[0]["query"].startswith("Design selective catalysts")
    assert "catalysis" in records[0]["query"] and "organocatalysis" in records[0]["query"]
    assert "Private" not in records[0]["query"]
    assert features[1]["topic_score"] == .72
    alignment = features[1]["topic_alignment"]
    assert alignment["kind"] == "local_joint_match_v1"
    assert alignment["model_revision"] == module.SEMANTIC_REVISION
    assert alignment["window_protocol"] == module.WINDOW_PROTOCOL
    assert alignment["topic"] == "catalysis" and alignment["logit"] == 6
    assert result[0][1] == .6
    assert features[1]["final_score"] == .6
    assert alignment["mode"] == "verification_only"


def test_balanced_shortlist_limits_pairs_and_skips_unassigned_multitopic(monkeypatch):
    module, records = install(monkeypatch)
    profile = Profile(bio="", keywords=["topic A", "topic B"])
    rows = [row(i) for i in range(1, 7)]
    features = {i: {"primary_facet": "topic A" if i < 5 else "topic B",
                    "topic_score": 1 - i / 10} for i in range(1, 7)}
    features[6]["primary_facet"] = ""
    module.apply_topic_alignment([(r, 1 - r.id / 10) for r in rows], features, profile, limit=2)
    assert len(records) == 2
    assert {record["query"] for record in records} == {"topic A", "topic B"}
    assert "topic_alignment" not in features[6]


def test_short_missing_and_nonresearch_abstracts_keep_legacy_scores(monkeypatch):
    module, records = install(monkeypatch)
    scored = [(row(1, abstract=""), .7), (row(2, abstract="Short."), .6),
              (row(3, section="world"), .5)]
    features = {}
    assert module.apply_topic_alignment(scored, features, Profile(bio="", keywords=["topic"])) == scored
    assert features == {}
    assert records == []


@pytest.mark.parametrize("results", [[], [{"logit": float("nan"), "windows": 1, "complete": True}],
                                      [{"logit": 2, "windows": 4, "complete": True}]])
def test_invalid_batch_leaves_features_untouched(monkeypatch, results):
    module, _ = install(monkeypatch, results)
    features = {1: {"topic_score": .7}}
    before = deepcopy(features)
    with pytest.raises(ValueError):
        module.apply_topic_alignment([(row(1), .7)], features, Profile(bio="", keywords=["topic"]))
    assert features == before


@pytest.mark.parametrize("complete,rejected", [(True, True), (False, False)])
def test_negative_match_is_only_marked_with_complete_coverage(monkeypatch, complete, rejected):
    module, _ = install(monkeypatch, [{"logit": -10., "windows": 3, "complete": complete}])
    features = {}
    result = module.apply_topic_alignment([(row(1), .7)], features, Profile(bio="", keywords=["topic"]))
    assert len(result) == 1  # The serving gate belongs to the pipeline.
    assert features[1]["topic_alignment"]["uncertain"] is rejected


def test_cancellation_and_backend_failure_propagate(monkeypatch):
    module, _ = install(monkeypatch)

    def cancelled():
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        module.apply_topic_alignment([(row(1), .7)], {}, Profile(bio="", keywords=["topic"]),
                                     check_cancel=cancelled)
    monkeypatch.setattr(module, "_build_backend", lambda *_: cancelled())
    with pytest.raises(RuntimeError, match="cancelled"):
        module.apply_topic_alignment([(row(1), .7)], {}, Profile(bio="", keywords=["topic"]))


def test_explicit_model_requires_verified_pinned_files(tmp_path):
    from dailydigest.rank import semantic_alignment as module

    module._build_backend.cache_clear()
    with pytest.raises((FileNotFoundError, ValueError)):
        module._build_backend(str(tmp_path))


def test_default_download_is_pinned_verified_and_cached(monkeypatch, tmp_path):
    import hashlib
    import huggingface_hub
    from dailydigest.rank import semantic_alignment as module
    from dailydigest.rank import cross_encoder

    module._build_backend.cache_clear()
    hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in module.SEMANTIC_FILES}
    monkeypatch.setattr(module, "SEMANTIC_FILES", hashes)
    monkeypatch.setattr(module, "_BROWSER_MODEL_DIR", tmp_path / "absent")
    calls = []

    def download(*, repo_id, filename, revision):
        calls.append((repo_id, filename, revision))
        path = tmp_path / "snapshot" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(filename)
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(cross_encoder, "LocalCrossEncoder", lambda root: Path(root))
    try:
        backend = module._build_backend()
        assert backend == tmp_path / "snapshot"
        assert module._build_backend() == backend
        assert len(calls) == 5
        assert all(repo == module.SEMANTIC_REPOSITORY and revision == module.SEMANTIC_REVISION
                   for repo, _, revision in calls)
    finally:
        module._build_backend.cache_clear()


def test_pair_cache_reuses_only_identical_model_query_and_text(monkeypatch):
    module, records = install(monkeypatch)
    backend = module._build_backend()
    monkeypatch.setattr(module, "_build_backend", lambda *_: backend)
    profile = Profile(bio="", keywords=["topic"])
    scored = [(row(1), .7)]
    for _ in range(2):
        module.apply_topic_alignment(scored, {}, profile)
    assert len(records) == 1
    module.apply_topic_alignment(scored, {}, Profile(bio="", keywords=["new topic"]))
    assert len(records) == 2
    scored[0][0].abstract += " Additional measured result."
    module.apply_topic_alignment(scored, {}, profile)
    assert len(records) == 3


def test_original_nonfinite_score_is_rejected_before_inference(monkeypatch):
    module, records = install(monkeypatch)
    with pytest.raises(ValueError, match="finite"):
        module.apply_topic_alignment([(row(1), float("nan"))], {}, Profile(bio="", keywords=["topic"]))
    assert records == []


def test_shortlist_checks_current_selection_leaders_with_weaker_raw_topic(monkeypatch):
    module, records = install(monkeypatch)
    profile = Profile(bio="", keywords=["topic"])
    rows = [row(index) for index in range(100)]
    for candidate in rows:
        candidate.title = f"Article {candidate.id}"
    features = {candidate.id: {"topic_score": 1 - candidate.id / 101} for candidate in rows}
    scored = [(candidate, .99 if candidate.id == 99 else .5) for candidate in rows]
    module.apply_topic_alignment(scored, features, profile)
    assert len(records) == 48
    assert "Article 99" in {record["title"] for record in records}
    assert sum(feature['topic_alignment_status'] == 'matched' for feature in features.values()) == 48
    assert sum(feature['topic_alignment_status'] == 'outside_shortlist' for feature in features.values()) == 52


def test_unmatched_research_status_distinguishes_missing_text_and_interest(monkeypatch):
    module, _ = install(monkeypatch)
    features = {1: {'primary_facet': 'topic A'}}
    scored = [(row(1), .8), (row(2, abstract='short'), .7), (row(3), .6)]
    module.apply_topic_alignment(scored, features, Profile(bio='', keywords=['topic A', 'topic B']))
    assert features[1]['topic_alignment_status'] == 'matched'
    assert features[2]['topic_alignment_status'] == 'insufficient_text'
    assert features[3]['topic_alignment_status'] == 'unassigned_interest'


def test_partial_query_coverage_survives_cache_and_cannot_mark_rejection(monkeypatch):
    module, records = install(monkeypatch, [
        {'logit': -11., 'windows': 1, 'complete': False, 'query_truncated': True},
    ])
    backend = module._build_backend()
    monkeypatch.setattr(module, '_build_backend', lambda *_: backend)
    profile = Profile(bio='', keywords=['topic'])
    for _ in range(2):
        features = {}
        module.apply_topic_alignment([(row(1), .7)], features, profile)
        assert features[1]['topic_alignment']['query_truncated']
        assert not features[1]['topic_alignment']['uncertain']
    assert records[0]['topic'] == 'topic'
    assert len(records) == 1


def test_complete_batch_validation_precedes_any_feature_update(monkeypatch):
    module, _ = install(monkeypatch, [
        {"logit": 6., "windows": 1, "complete": True},
        {"logit": float("inf"), "windows": 1, "complete": True},
    ])
    features = {1: {"topic_score": .7}, 2: {"topic_score": .6}}
    before = deepcopy(features)
    with pytest.raises(ValueError):
        module.apply_topic_alignment([(row(1), .7), (row(2), .6)], features,
                                     Profile(bio="", keywords=["topic"]))
    assert features == before


def test_explicit_model_rejects_wrong_checksums(tmp_path):
    from dailydigest.rank import semantic_alignment as module

    module._build_backend.cache_clear()
    for name in module.SEMANTIC_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("wrong asset")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module._build_backend(str(tmp_path))
