"""Tests for usage-gap catch-up: wider window + scaled research ceiling."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dailydigest import store as st
from dailydigest.config import get_settings
from dailydigest.pipeline import _catch_up_window, _research_ceiling_for_window


def _reset(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    from dailydigest import config as cfg
    cfg.reload_settings()
    st.SETTINGS = cfg.SETTINGS
    st._ENGINE = None
    st._SessionLocal = None
    st._INITIALIZED = False
    st.init_db()


def test_window_is_two_with_no_prior_digest(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    assert _catch_up_window("2026-07-02", None) == 2


def test_window_widens_to_gap_plus_one(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    old = datetime.now(timezone.utc) - timedelta(days=7)
    with st.session_scope() as s:
        s.add(st.DigestRow(id="2026-06-25", item_count=5, created_at=old))
    assert _catch_up_window("2026-07-02", None) == 8  # 7 + 1


def test_window_capped_at_max_backfill(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    monkeypatch.setattr(get_settings(), "max_backfill_days", 21)
    old = datetime.now(timezone.utc) - timedelta(days=60)
    with st.session_scope() as s:
        s.add(st.DigestRow(id="2026-05-02", item_count=5, created_at=old))
    assert _catch_up_window("2026-07-02", None) == 21


def test_explicit_backfill_overrides(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    assert _catch_up_window("2026-07-02", 14) == 14


def test_research_ceiling_scales_with_window(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "top_research", 15)
    monkeypatch.setattr(s, "max_research_backlog", 40)
    assert _research_ceiling_for_window(2) == 15          # normal day: unchanged
    assert _research_ceiling_for_window(8) > 15           # week gap: grows
    assert _research_ceiling_for_window(8) <= 40          # bounded by backlog cap
    assert _research_ceiling_for_window(60) == 40         # far gap: hits the cap


def test_research_ceiling_no_growth_when_cap_equals_base(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "top_research", 15)
    monkeypatch.setattr(s, "max_research_backlog", 15)
    assert _research_ceiling_for_window(30) == 15
