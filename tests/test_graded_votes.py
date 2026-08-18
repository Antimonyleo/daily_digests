"""Tests for 4-level graded feedback (must-read/relevant/hmmm/not-for-me)."""

from __future__ import annotations

from dailydigest import votes as v


def test_grade_to_value_sign():
    assert v.grade_to_value(100) == 1
    assert v.grade_to_value(70) == 1
    assert v.grade_to_value(40) == -1
    assert v.grade_to_value(10) == -1


def test_value_to_grade_legacy_mapping():
    assert v.value_to_grade(1) == 70
    assert v.value_to_grade(0) == 40
    assert v.value_to_grade(-1) == 10


def test_grade_to_weight_ordering_and_bounds():
    w = {g: v.grade_to_weight(g) for g in (100, 70, 40, 10)}
    # strictly decreasing preference strength
    assert w[100] > w[70] > w[40] > w[10]
    assert w[100] == 1.0 and w[10] == -0.8
    assert -1.0 <= w[10] and w[100] <= 1.0


def test_vote_levels_constant():
    assert v.VOTE_LEVELS == {"must_read": 100, "relevant": 70, "hmmm": 40, "not_for_me": 10}


def test_record_vote_stores_grade(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    from dailydigest import config as cfg, store as st
    cfg.reload_settings(); st.SETTINGS = cfg.SETTINGS
    st._ENGINE = None; st._SessionLocal = None; st._INITIALIZED = False
    st.init_db()
    with st.session_scope() as s:
        row = st.ItemRow(source="Test", section="research", external_id="g1",
                         url="https://e.com/g1", title="T", abstract="a")
        s.add(row); s.flush(); item_id = int(row.id)
    # avoid the embedding/Rocchio side effect in this unit test
    monkeypatch.setattr(v, "_update_rocchio", lambda *a, **k: None)
    assert v.record_vote_by_id(item_id, 1, 100) is True
    from sqlalchemy import select
    with st.session_scope() as s:
        vr = s.execute(select(st.VoteRow).where(st.VoteRow.item_id == item_id)).scalars().first()
        assert vr.value == 1 and vr.grade == 100


def _stub_exemplars(monkeypatch, pos, neg):
    """Install synthetic (ids, unit_vecs, weights) exemplar sets."""
    import numpy as np

    def _pack(entries):
        if not entries:
            return (
                np.zeros(0, dtype=np.int64),
                np.zeros((0, 3), np.float32),
                np.zeros(0, np.float32),
            )
        ids = np.array([e[0] for e in entries], dtype=np.int64)
        vecs = np.array([e[1] for e in entries], dtype=np.float32)
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        ws = np.array([e[2] for e in entries], dtype=np.float32)
        return ids, vecs, ws

    monkeypatch.setattr(v, "_load_vote_exemplars", lambda: (_pack(pos), _pack(neg)))


def test_knn_preference_scores_uses_graded_memory(monkeypatch):
    import numpy as np
    from types import SimpleNamespace

    # Exemplars on three orthogonal axes: loved x, hmmm y (weight 0.2),
    # not-for-me z (weight 0.8).
    _stub_exemplars(
        monkeypatch,
        pos=[(1, [1.0, 0.0, 0.0], 0.4)],
        neg=[(2, [0.0, 1.0, 0.0], 0.2), (3, [0.0, 0.0, 1.0], 0.8)],
    )
    cands = [
        SimpleNamespace(id=None, title="like the loved one"),
        SimpleNamespace(id=None, title="like the hmmm one"),
        SimpleNamespace(id=None, title="like the not-for-me one"),
    ]
    monkeypatch.setattr(
        v,
        "embed_item_rows",
        lambda rows: np.eye(3, dtype=np.float32)[: len(rows)],
    )
    scores = v.knn_preference_scores(cands, k=1)
    # Sign and magnitude follow the grades: relevant > hmmm > not-for-me.
    assert scores[0] > 0 > scores[1] > scores[2]
    assert abs(scores[1]) < abs(scores[2])


def test_knn_preference_scores_leave_one_out_and_empty(monkeypatch):
    import numpy as np
    from types import SimpleNamespace

    _stub_exemplars(
        monkeypatch,
        pos=[(7, [1.0, 0.0, 0.0], 1.0)],
        neg=[(8, [0.9, 0.1, 0.0], 0.8)],
    )
    monkeypatch.setattr(
        v,
        "embed_item_rows",
        lambda rows: np.array([[1.0, 0.0, 0.0]] * len(rows), dtype=np.float32),
    )
    # A voted candidate must not match its own vote: with itself excluded, item 7
    # sees only the negative neighbour and scores negative.
    voted = [SimpleNamespace(id=7, title="already voted")]
    assert v.knn_preference_scores(voted, k=1)[0] < 0

    # No votes at all -> neutral zeros, not an error.
    _stub_exemplars(monkeypatch, pos=[], neg=[])
    fresh = [SimpleNamespace(id=None, title="anything")]
    assert v.knn_preference_scores(fresh)[0] == 0.0
    assert v.knn_preference_scores([]).shape == (0,)
