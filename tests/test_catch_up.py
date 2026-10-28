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


# --- R0.3: the gap is counted in the reader's calendar days, not truncated 24 h --- #

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from dailydigest import pipeline as pipeline_mod  # noqa: E402


def _set_tz(monkeypatch, tz: str) -> None:
    monkeypatch.setattr(get_settings(), "user_tz", tz)


def _add_digest(created_at: datetime, digest_id: str = "previous") -> None:
    with st.session_scope() as s:
        s.add(st.DigestRow(id=digest_id, item_count=5, created_at=created_at))


def test_gap_of_three_days_twenty_three_hours_is_four_calendar_days(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    _set_tz(monkeypatch, "UTC")
    now = datetime(2026, 10, 5, 23, 30, tzinfo=timezone.utc)
    _add_digest(now - timedelta(days=3, hours=23))  # 2026-10-02 00:30
    assert st.days_since_last_digest(now=now) == 3  # 10-02 -> 10-05
    later = datetime(2026, 10, 6, 0, 10, tzinfo=timezone.utc)
    # 3 d 23 h 40 min elapsed: old truncation said 3, the calendar says 4.
    assert st.days_since_last_digest(now=later) == 4
    monkeypatch.setattr(st, "datetime", _FrozenDatetime.at(later))
    assert _catch_up_window("today", None) == 5


def test_late_evening_then_just_after_midnight_is_one_day(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    _set_tz(monkeypatch, "America/Los_Angeles")
    # 23:50 PDT on 10-04 and 00:10 PDT on 10-05: 20 minutes, one calendar day.
    last = datetime(2026, 10, 5, 6, 50, tzinfo=timezone.utc)
    now = datetime(2026, 10, 5, 7, 10, tzinfo=timezone.utc)
    _add_digest(last)
    assert st.days_since_last_digest(now=now) == 1
    # The same instants are the same UTC day; the reader's zone decides.
    _set_tz(monkeypatch, "UTC")
    assert st.days_since_last_digest(now=now) == 0


def test_dst_spring_forward_23_hour_day_counts_one_day(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    _set_tz(monkeypatch, "America/New_York")
    # 2026-03-07 23:30 EST -> 2026-03-08 23:00 EDT: 22.5 h elapsed on a 23 h day.
    last = datetime(2026, 3, 8, 4, 30, tzinfo=timezone.utc)
    now = datetime(2026, 3, 9, 3, 0, tzinfo=timezone.utc)
    assert (now - last).days == 0  # what truncation used to report
    _add_digest(last)
    assert st.days_since_last_digest(now=now) == 1


def test_dst_fall_back_25_hour_day_is_still_same_day(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    _set_tz(monkeypatch, "America/New_York")
    # 2026-11-01 00:10 EDT -> 2026-11-01 23:50 EST: 25 h 40 min on one 25 h day.
    last = datetime(2026, 11, 1, 4, 10, tzinfo=timezone.utc)
    now = datetime(2026, 11, 2, 4, 50, tzinfo=timezone.utc)
    assert (now - last).days == 1  # ceil/duration would also say 1+
    _add_digest(last)
    assert st.days_since_last_digest(now=now) == 0


def test_naive_stored_timestamp_is_treated_as_utc(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    _set_tz(monkeypatch, "Asia/Tokyo")
    _add_digest(datetime(2026, 10, 4, 14, 30))  # naive UTC == 10-04 23:30 JST
    now = datetime(2026, 10, 4, 15, 30, tzinfo=timezone.utc)  # 10-05 00:30 JST
    assert st.days_since_last_digest(now=now) == 1


def test_no_history_is_minus_one_and_window_two(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    assert st.days_since_last_digest() == -1
    assert _catch_up_window("today", None) == 2


def test_excluded_current_digest_is_not_its_own_baseline(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    now = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)
    _add_digest(now - timedelta(days=3), "older")
    _add_digest(now - timedelta(minutes=5), "today")
    assert st.days_since_last_digest(exclude_digest_id="today", now=now) == 3
    assert st.days_since_last_digest(now=now) == 0


def test_future_timestamp_clamps_to_zero(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    now = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)
    _add_digest(now + timedelta(days=2))
    assert st.days_since_last_digest(now=now) == 0


class _FrozenDatetime(datetime):
    _now: datetime

    @classmethod
    def at(cls, moment: datetime):
        return type("Frozen", (cls,), {"_now": moment})

    @classmethod
    def now(cls, tz=None):
        return cls._now if tz is None else cls._now.astimezone(tz)


@pytest.mark.parametrize(("gap", "cap", "expected"), [(0, 21, 2), (1, 21, 2), (4, 21, 5), (20, 21, 21), (30, 21, 21), (9, 7, 7)])
def test_window_is_gap_plus_one_floored_and_capped(monkeypatch, gap, cap, expected):
    monkeypatch.setattr(get_settings(), "max_backfill_days", cap)
    monkeypatch.setattr(pipeline_mod, "days_since_last_digest", lambda exclude_digest_id=None: gap)
    assert _catch_up_window("today", None) == expected
    # An explicit backfill always wins, even beyond the automatic cap.
    assert _catch_up_window("today", 30) == 30


def _capture_pool_days(monkeypatch, gap: int) -> list[int]:
    captured: list[int] = []
    monkeypatch.setattr(pipeline_mod, "days_since_last_digest", lambda exclude_digest_id=None: gap)
    monkeypatch.setattr(pipeline_mod, "_digest_id", lambda: "2026-10-05")
    monkeypatch.setattr(pipeline_mod, "ingest_all", lambda **kwargs: captured.append(kwargs["days"]) or 0)
    monkeypatch.setattr(pipeline_mod, "load_profile", lambda: SimpleNamespace(bio="", keywords=[], downweight=[]))
    monkeypatch.setattr(pipeline_mod, "_build_profile_with_rocchio", None)
    monkeypatch.setattr(pipeline_mod, "build_profile_matrix", lambda _p: __import__("numpy").zeros((1, 3)))
    monkeypatch.setattr(pipeline_mod, "recent_items", lambda days=2: captured.append(days) or [])
    monkeypatch.setattr(pipeline_mod, "score_items", lambda *a, **k: [])
    monkeypatch.setattr(pipeline_mod, "pick_top_per_section", lambda *a, **k: [])
    monkeypatch.setattr(pipeline_mod, "summarize_items", lambda *a, **k: {})
    monkeypatch.setattr(pipeline_mod, "send_digest", lambda *a, **k: False)
    return captured


@pytest.mark.parametrize(("gap", "min_window", "backfill", "expected_pool"), [
    (1, 4, None, 4),    # daily use: pool still reaches min_window_days
    (4, 4, None, 5),    # 4 calendar days away: window 5 exceeds the floor
    (10, 4, None, 11),
    (1, 6, None, 6),
    (10, 4, 3, 3),      # explicit backfill is used as-is, below min_window_days
])
def test_pool_days_combines_window_and_min_window(tmp_path, monkeypatch, gap, min_window, backfill, expected_pool):
    _reset(tmp_path, monkeypatch)
    monkeypatch.setattr(get_settings(), "min_window_days", min_window)
    captured = _capture_pool_days(monkeypatch, gap)
    pipeline_mod.run_all(dry_run=True, backfill_days=backfill)
    assert captured[:2] == [expected_pool, expected_pool]
