"""Old feedback policies cannot silently influence current semantic ranking."""
from pathlib import Path

import numpy as np
import pytest

from dailydigest import config, store, votes
from dailydigest.models import Profile
from dailydigest.rank import profile, ranker


@pytest.mark.parametrize("version", [None, "reason-aware-v1", "older-policy", "future-policy"])
def test_incompatible_rocchio_is_preserved_and_static_profile_used(monkeypatch, version):
    path = Path(config.get_settings().db_path).parent / "learned_profile.npz"
    values = {"profile": np.array([0., 1.]), "vote_count": np.array([60])}
    if version is not None:
        values["feedback_semantics_version"] = np.array([version])
    np.savez(path, **values)
    before = path.read_bytes(), path.stat().st_mtime_ns
    static = np.eye(2, dtype=np.float32)
    monkeypatch.setattr(profile, "build_profile_matrix", lambda _p: static.copy())
    result = profile.build_profile_matrix_with_rocchio(Profile(bio="Reader"), 60)
    np.testing.assert_array_equal(result, static)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_new_vote_builds_versioned_rocchio_accepted_by_reader(monkeypatch):
    static = np.eye(2, dtype=np.float32)
    monkeypatch.setattr(profile, "build_profile_matrix", lambda _p: static.copy())
    monkeypatch.setattr(votes, "embed_item_rows", lambda rows: np.tile([1., 0.], (len(rows), 1)))
    store.init_db()
    with store.session_scope() as session:
        rows = [store.ItemRow(source="Nature", section="research", external_id=str(i),
                              url=f"https://example.test/{i}", title=f"Methods {i}") for i in range(2)]
        session.add_all(rows)
        session.flush()
        ids = [row.id for row in rows]
    for iid in ids:
        assert votes.record_vote_by_id(iid, 1)
    path = Path(config.get_settings().db_path).parent / "learned_profile.npz"
    with np.load(path, allow_pickle=False) as data:
        assert str(data["feedback_semantics_version"][0]) == votes.FEEDBACK_SEMANTICS_VERSION
        assert str(data["feedback_reasons_signature"][0]) == votes.feedback_reasons_signature()
        assert str(data["feedback_votes_signature"][0]) == votes.feedback_votes_signature()
    result = profile.build_profile_matrix_with_rocchio(Profile(bio="Reader"), 2)
    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result[:2], static)
    assert result[2, 0] > 0


@pytest.mark.parametrize("version", [None, "reason-aware-v1", "older-policy", "future-policy"])
def test_incompatible_lr_is_preserved_and_rejected_even_with_current_features(version):
    values = {
        "coef": np.ones((1, votes.LR_FEATURE_DIM), dtype=np.float32),
        "intercept": np.array([0.]), "classes": np.array([-1, 1]),
        "feature_mean": np.zeros(votes.LR_FEATURE_DIM),
        "feature_scale": np.ones(votes.LR_FEATURE_DIM),
        "feature_dim": np.array([votes.LR_FEATURE_DIM]),
        "feature_schema_version": np.array([votes.LR_FEATURE_SCHEMA_VERSION]),
    }
    if version is not None:
        values["feedback_semantics_version"] = np.array([version])
    path = ranker._lr_weights_path()
    np.savez(path, **values)
    before = path.read_bytes(), path.stat().st_mtime_ns
    assert ranker.LRRanker().load() is False
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_weights_trained_with_cross_author_matching_are_preserved_but_not_reused():
    path = ranker._lr_weights_path()
    np.savez(path, coef=np.ones((1, votes.LR_FEATURE_DIM), dtype=np.float32),
             intercept=np.array([0.]), classes=np.array([-1, 1]),
             feature_mean=np.zeros(votes.LR_FEATURE_DIM), feature_scale=np.ones(votes.LR_FEATURE_DIM),
             feature_dim=np.array([votes.LR_FEATURE_DIM]),
             feature_schema_version=np.array(['lr_ranker_preference_affinity_v6']),
             feedback_semantics_version=np.array([votes.FEEDBACK_SEMANTICS_VERSION]),
             feedback_reasons_signature=np.array([votes.feedback_reasons_signature()]))
    before = path.read_bytes(), path.stat().st_mtime_ns
    assert ranker.LRRanker().load() is False
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_practical_ratings_do_not_inflate_strength_of_two_informative_votes(monkeypatch):
    path = Path(config.get_settings().db_path).parent / "learned_profile.npz"
    np.savez(path, profile=np.array([1., 0.]), vote_count=np.array([2]),
             feedback_semantics_version=np.array([votes.FEEDBACK_SEMANTICS_VERSION]),
             feedback_reasons_signature=np.array([votes.feedback_reasons_signature()]),
             feedback_votes_signature=np.array([votes.feedback_votes_signature()]))
    static = np.eye(2, dtype=np.float32)
    monkeypatch.setattr(profile, "build_profile_matrix", lambda _p: static.copy())
    # 98 practical ratings + 2 informative: the persisted informative count is
    # authoritative for composition even though the raw signed count is 100.
    result = profile.build_profile_matrix_with_rocchio(Profile(bio="Reader"), 100)
    expected = profile.append_learned_profile(static, np.array([1., 0.]), 2)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("change", ["add", "remove"])
def test_reason_changes_invalidate_cached_and_persisted_lr_without_rewriting(change, monkeypatch):
    monkeypatch.setattr(votes, "_update_rocchio", lambda *_a, **_k: None)
    store.init_db()
    with store.session_scope() as session:
        row = store.ItemRow(source="Nature", section="research", external_id="negative",
                            url="https://example.test/negative", title="Negative")
        session.add(row)
        session.flush()
        iid = row.id
    votes.record_vote_by_id(iid, -1)
    if change == "remove":
        votes.record_vote_reason(iid, "already_known")
    matrix = np.zeros((4, votes.LR_FEATURE_DIM), dtype=np.float32)
    matrix[:, 0] = [1., .8, -.8, -1.]
    fitted = ranker.LRRanker()
    fitted.fit(matrix, np.array([1, 1, -1, -1]))
    assert ranker.get_lr_ranker() is not None
    path = ranker._lr_weights_path()
    before = path.read_bytes(), path.stat().st_mtime_ns
    if change == "remove":
        votes.remove_vote_reason(iid, "already_known")
    else:
        votes.record_vote_reason(iid, "already_known")
    assert ranker.get_lr_ranker() is None
    assert ranker.LRRanker().load() is False
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_failed_rocchio_rebuild_after_practical_reason_uses_static_and_preserves_artifact(monkeypatch):
    static = np.eye(2, dtype=np.float32)
    monkeypatch.setattr(profile, "build_profile_matrix", lambda _p: static.copy())
    store.init_db()
    with store.session_scope() as session:
        rows = [store.ItemRow(source="Nature", section="research", external_id=str(i),
                              url=f"https://example.test/{i}", title=f"Methods {i}") for i in range(2)]
        session.add_all(rows)
        session.flush()
        positive, negative = [row.id for row in rows]
    monkeypatch.setattr(votes, "embed_item_rows", lambda rows: np.array(
        [[1., 0.] if row.id == positive else [0., 1.] for row in rows]))
    votes.record_vote_by_id(positive, 1)
    votes.record_vote_by_id(negative, -1)
    path = Path(config.get_settings().db_path).parent / "learned_profile.npz"
    before = path.read_bytes(), path.stat().st_mtime_ns

    def unavailable(_rows):
        raise RuntimeError("Local encoder unavailable")
    monkeypatch.setattr(votes, "embed_item_rows", unavailable)
    assert votes.record_vote_reason(negative, "already_known")
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    # The old file's negative topic pull is incompatible with current reasons.
    result = profile.build_profile_matrix_with_rocchio(Profile(bio="Reader"), 2)
    np.testing.assert_array_equal(result, static)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


@pytest.mark.parametrize("change", [(-1, 10), (1, 70), (0, None)])
@pytest.mark.parametrize("failure", ["encoder", "write"])
def test_changed_votes_cannot_reuse_old_rocchio_when_rebuild_fails(monkeypatch, change, failure):
    static = np.eye(2, dtype=np.float32)
    monkeypatch.setattr(profile, "build_profile_matrix", lambda _p: static.copy())
    monkeypatch.setattr(votes, "embed_item_rows", lambda rows: np.tile([1., 0.], (len(rows), 1)))
    store.init_db()
    with store.session_scope() as session:
        rows = [store.ItemRow(source="Nature", section="research", external_id=str(i),
                             url=f"https://example.test/{i}", title=f"Methods {i}") for i in range(2)]
        session.add_all(rows)
        session.flush()
        ids = [row.id for row in rows]
    for iid in ids:
        assert votes.record_vote_by_id(iid, 1, 100)
    path = Path(config.get_settings().db_path).parent / "learned_profile.npz"
    before = path.read_bytes(), path.stat().st_mtime_ns
    assert profile.build_profile_matrix_with_rocchio(Profile(bio="Reader"), 2).shape == (3, 2)

    def unavailable(*_args, **_kwargs):
        raise OSError("Fixture backend unavailable")

    with monkeypatch.context() as fail:
        if failure == "encoder":
            fail.setattr(votes, "embed_item_rows", unavailable)
        else:
            fail.setattr(np, "savez", unavailable)
        assert votes.record_vote_by_id(ids[0], *change)
    assert votes.get_vote_value(ids[0]) == change[0]
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    np.testing.assert_array_equal(profile.build_profile_matrix_with_rocchio(Profile(bio="Reader"), 2), static)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    # A later successful rebuild binds the artifact to the new canonical state.
    votes._rebuild_rocchio_profile()
    with np.load(path, allow_pickle=False) as data:
        assert str(data["feedback_votes_signature"][0]) == votes.feedback_votes_signature()
