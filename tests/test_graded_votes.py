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
