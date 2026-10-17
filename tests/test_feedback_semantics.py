"""Practical rejection reasons cannot become inferred topic aversion."""
from types import SimpleNamespace

import numpy as np
import pytest

from dailydigest import store, votes


def setup_feedback(tmp_path, monkeypatch):
    vectors = {}
    monkeypatch.setattr(votes, "embed_item_rows", lambda rows: np.array(
        [vectors[int(row.id)] for row in rows], dtype=np.float32))
    learned_path = tmp_path / "learned_profile.npz"
    monkeypatch.setattr(votes, "_learned_profile_path", lambda: learned_path)
    store.init_db()

    def add(key, vector, value):
        with store.session_scope() as session:
            row = store.ItemRow(source="Nature", section="research", external_id=key,
                                url=f"https://example.test/{key}", title=key,
                                abstract="A substantive research methods abstract.")
            session.add(row)
            session.flush()
            iid = row.id
        vectors[iid] = vector
        votes.record_vote_by_id(iid, value)
        return iid
    return add, vectors, learned_path


@pytest.mark.parametrize("reason", ["already_known", "not_urgent", "duplicate", "access_friction",
                                   "low_quality", "low_impact", "promotional"])
def test_practical_reasons_do_not_teach_affinity_knn_or_rocchio_aversion(tmp_path, monkeypatch, reason):
    add, vectors, learned_path = setup_feedback(tmp_path, monkeypatch)
    positive = add("liked-engineering", [1., 0.], 1)
    negative = add("useful-but-practical-problem", [0., 1.], -1)
    assert votes.record_vote_reason(negative, reason)
    pos, neg = votes._load_vote_exemplars()
    assert pos[0].tolist() == [positive]
    assert neg[0].tolist() == []
    vectors[999] = [0., 1.]
    assert votes.knn_preference_scores([SimpleNamespace(id=999)], k=1)[0] == 0
    learned = np.load(learned_path)["profile"]
    assert learned[0] > 0
    assert learned[1] == 0
    # Preserve the actual rating and precise explanation for evaluation/UI.
    assert votes.get_vote_value(negative) == -1
    assert votes.signed_vote_count() == 2
    assert votes.get_vote_reasons(negative) == [reason]


@pytest.mark.parametrize("semantic_reason", ["off_topic", "too_contextual"])
def test_mixed_topic_reason_restores_semantic_aversion_and_removal_rebuilds(tmp_path, monkeypatch, semantic_reason):
    add, vectors, path = setup_feedback(tmp_path, monkeypatch)
    positive = add("liked", [1., 0.], 1)
    negative = add("known-and-off-topic", [0., 1.], -1)
    votes.record_vote_reason(negative, "already_known")
    assert np.load(path)["profile"][1] == 0
    votes.record_vote_reason(negative, semantic_reason)
    assert votes._load_vote_exemplars()[1][0].tolist() == [negative]
    assert np.load(path)["profile"][1] < 0
    votes.remove_vote_reason(negative, semantic_reason)
    assert votes._load_vote_exemplars()[1][0].size == 0
    assert np.load(path)["profile"][1] == 0
    # An explicitly relevant re-vote keeps positive influence and clears chips.
    votes.record_vote_by_id(negative, 1)
    assert votes.get_vote_reasons(negative) == []
    assert set(votes._load_vote_exemplars()[0][0]) == {positive, negative}
    with np.load(path, allow_pickle=False) as data:
        assert str(data["feedback_reasons_signature"][0]) == votes.feedback_reasons_signature()
    from dailydigest.models import Profile
    from dailydigest.rank import profile
    monkeypatch.setattr(profile, "build_profile_matrix", lambda _p: np.eye(2, dtype=np.float32))
    assert profile.build_profile_matrix_with_rocchio(Profile(bio="Reader"), 2).shape == (3, 2)


def test_last_semantic_vote_becoming_practical_removes_stale_rocchio_profile(tmp_path, monkeypatch):
    add, _vectors, path = setup_feedback(tmp_path, monkeypatch)
    iid = add("duplicate", [1., 0.], -1)
    assert path.exists()
    votes.record_vote_reason(iid, "duplicate")
    assert not path.exists()
    votes.remove_vote_reason(iid, "duplicate")
    assert np.load(path)["profile"][0] < 0


def test_lr_dataset_excludes_practical_negative_while_retaining_raw_votes(tmp_path, monkeypatch):
    add, _vectors, _path = setup_feedback(tmp_path, monkeypatch)
    positive = add("liked", [1., 0.], 1)
    practical = add("already-known", [0., 1.], -1)
    disliked = add("off-interest", [0., 1.], -1)
    votes.record_vote_reason(practical, "already_known")
    monkeypatch.setattr(votes, "MIN_VOTES_FOR_LR", 2)
    from dailydigest.rank import profile
    monkeypatch.setattr(profile, "build_profile_matrix_with_rocchio", lambda *_a, **_k: np.array([[1., 0.]]))
    built = []

    def features(rows, _profile):
        built.extend(row.id for row in rows)
        return np.array([[1.] if row.id == positive else [0.] for row in rows], dtype=np.float32)
    monkeypatch.setattr(votes, "_build_item_features", features)
    dataset = votes.vote_dataset()
    assert dataset is not None
    assert set(built) == {positive, disliked}
    assert practical not in built
    assert votes.vote_counts()["signed"] == 3


@pytest.mark.parametrize("reason", ["too_contextual", "low_quality"])
def test_article_specific_reason_does_not_penalize_other_papers_from_source(tmp_path, monkeypatch, reason):
    add, _vectors, _path = setup_feedback(tmp_path, monkeypatch)
    negative = add("rejected", [1., 0.], -1)
    future = add("different-paper", [0., 1.], 0)
    assert votes.record_vote_reason(negative, reason)
    with store.session_scope() as session:
        row = session.get(store.ItemRow, future)
        penalties = votes.reason_penalty_map([row])
    assert penalties[str(negative)] > 0
    assert penalties.get(str(future), 0) == 0


def test_clearing_rating_removes_only_its_topic_learning(tmp_path, monkeypatch):
    add, _vectors, path = setup_feedback(tmp_path, monkeypatch)
    positive = add("keep-this-preference", [1., 0.], 1)
    cleared = add("clear-this-preference", [0., 1.], -1)
    votes.record_vote_reason(cleared, "too_contextual")
    assert np.load(path)["profile"][1] < 0
    assert votes.clear_rating_by_id(cleared)
    assert votes.get_vote_value(cleared) is None
    assert votes.get_vote_reasons(cleared) == []
    pos, neg = votes._load_vote_exemplars()
    assert pos[0].tolist() == [positive]
    assert neg[0].size == 0
    assert np.load(path)["profile"][1] == 0
    assert votes.signed_vote_count() == 1
    assert votes.clear_rating_by_id(positive)
    assert not path.exists()


def test_clearing_positive_invalidates_stale_models_even_if_rebuild_fails(tmp_path, monkeypatch):
    from dailydigest.rank import ranker

    add, _vectors, path = setup_feedback(tmp_path, monkeypatch)
    cleared = add("clear-positive", [1., 0.], 1)
    add("keep-negative", [0., 1.], -1)
    weights = ranker._lr_weights_path()
    weights.write_bytes(b"obsolete learned weights")

    def unavailable(_rows):
        raise RuntimeError("Local encoder unavailable")

    monkeypatch.setattr(votes, "embed_item_rows", unavailable)
    assert votes.clear_rating_by_id(cleared)
    assert not path.exists()
    assert not weights.exists()
    assert votes.get_vote_value(cleared) is None
