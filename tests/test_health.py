"""Tests for dailydigest.health.weekly_summary and should_show.

health.py has two pure aggregation functions (weekly_summary, should_show)
that are fully testable without any DB or network I/O. The function reads
a JSON file from SETTINGS.db_path parent directory; we redirect it via
a monkeypatched SETTINGS.db_path so it reads from tmp_path.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dailydigest.health import latest_snapshot, should_show, weekly_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_health(path: Path, history: list[dict]) -> None:
    """Write a health.json file with the given history entries."""
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "latest": [],
        "history": history,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_health_payload(path: Path, latest: list[dict], history: list[dict] | None = None) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "latest": latest,
        "history": history or [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _recent_ts(days_ago: float = 0.5) -> str:
    """ISO timestamp within the last 7 days."""
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.isoformat()


def _old_ts() -> str:
    """ISO timestamp older than 7 days."""
    dt = datetime.now(timezone.utc) - timedelta(days=8)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# weekly_summary
# ---------------------------------------------------------------------------

class TestWeeklySummary:
    def test_no_health_file_returns_empty(self, tmp_path, monkeypatch):
        fake_settings = type("S", (), {"db_path": str(tmp_path / "digest.db")})()
        monkeypatch.setattr("dailydigest.health.get_settings", lambda: fake_settings)
        result = weekly_summary()
        assert result == []

    def test_aggregates_items_across_runs(self, tmp_path, monkeypatch):
        db_path = tmp_path / "digest.db"
        fake_settings = type("S", (), {"db_path": str(db_path)})()
        monkeypatch.setattr("dailydigest.health.get_settings", lambda: fake_settings)
        health_file = tmp_path / "health.json"

        history = [
            {
                "recorded_at": _recent_ts(1),
                "stats": [{"source": "Nature", "items": 10, "ok": True}],
            },
            {
                "recorded_at": _recent_ts(2),
                "stats": [{"source": "Nature", "items": 8, "ok": True}],
            },
        ]
        _write_health(health_file, history)

        result = weekly_summary()
        assert len(result) == 1
        assert result[0]["source"] == "Nature"
        assert result[0]["items_7d"] == 18
        assert result[0]["failures_7d"] == 0

    def test_counts_failures(self, tmp_path, monkeypatch):
        db_path = tmp_path / "digest.db"
        fake_settings = type("S", (), {"db_path": str(db_path)})()
        monkeypatch.setattr("dailydigest.health.get_settings", lambda: fake_settings)
        health_file = tmp_path / "health.json"

        history = [
            {
                "recorded_at": _recent_ts(1),
                "stats": [{"source": "BadFeed", "items": 0, "ok": False, "error": "timeout"}],
            },
            {
                "recorded_at": _recent_ts(2),
                "stats": [{"source": "BadFeed", "items": 5, "ok": True}],
            },
        ]
        _write_health(health_file, history)

        result = weekly_summary()
        bad = next(r for r in result if r["source"] == "BadFeed")
        assert bad["failures_7d"] == 1
        assert bad["last_error"] == "timeout"
        assert bad["items_7d"] == 5

    def test_entries_older_than_7_days_excluded(self, tmp_path, monkeypatch):
        db_path = tmp_path / "digest.db"
        fake_settings = type("S", (), {"db_path": str(db_path)})()
        monkeypatch.setattr("dailydigest.health.get_settings", lambda: fake_settings)
        health_file = tmp_path / "health.json"

        history = [
            {
                "recorded_at": _old_ts(),
                "stats": [{"source": "OldSource", "items": 999, "ok": True}],
            },
            {
                "recorded_at": _recent_ts(1),
                "stats": [{"source": "NewSource", "items": 5, "ok": True}],
            },
        ]
        _write_health(health_file, history)

        result = weekly_summary()
        sources = [r["source"] for r in result]
        assert "OldSource" not in sources
        assert "NewSource" in sources

    def test_result_sorted_alphabetically(self, tmp_path, monkeypatch):
        db_path = tmp_path / "digest.db"
        fake_settings = type("S", (), {"db_path": str(db_path)})()
        monkeypatch.setattr("dailydigest.health.get_settings", lambda: fake_settings)
        health_file = tmp_path / "health.json"

        history = [
            {
                "recorded_at": _recent_ts(1),
                "stats": [
                    {"source": "Zebra", "items": 1, "ok": True},
                    {"source": "Alpha", "items": 2, "ok": True},
                    {"source": "Middle", "items": 3, "ok": True},
                ],
            }
        ]
        _write_health(health_file, history)

        result = weekly_summary()
        names = [r["source"] for r in result]
        assert names == sorted(names, key=str.lower)


def test_latest_snapshot_returns_most_recent_source_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "digest.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from dailydigest import config as config_mod
    config_mod.reload_settings()
    health_file = tmp_path / "health.json"
    latest = [
        {"source": "Nature", "items": 12, "ok": True},
        {"source": "Broken", "items": 0, "ok": False, "error": "timeout"},
    ]
    _write_health_payload(health_file, latest)

    assert latest_snapshot() == latest


# ---------------------------------------------------------------------------
# should_show
# ---------------------------------------------------------------------------

class TestShouldShow:
    def test_true_when_any_failure(self):
        summary = [
            {"source": "A", "failures_7d": 0},
            {"source": "B", "failures_7d": 1},
        ]
        assert should_show(summary) is True

    def test_false_when_no_failures(self):
        summary = [
            {"source": "A", "failures_7d": 0},
            {"source": "B", "failures_7d": 0},
        ]
        assert should_show(summary) is False

    def test_false_for_empty_summary(self):
        assert should_show([]) is False

    def test_true_when_multiple_failures(self):
        summary = [
            {"source": "A", "failures_7d": 3},
            {"source": "B", "failures_7d": 2},
        ]
        assert should_show(summary) is True


def test_coverage_distinguishes_empty_partial_failed_and_unknown_sources(tmp_path, monkeypatch):
    from dailydigest.health import coverage_summary

    monkeypatch.setattr('dailydigest.health.get_settings', lambda: type('S', (), {'db_path': str(tmp_path/'digest.db')})())
    payload = {'updated_at': '2026-10-14T12:00:00+03:00', 'latest': [
        {'source': 'Direct', 'ok': True, 'items': 3},
        {'source': 'Empty', 'ok': True, 'items': 0},
        {'source': 'Partial', 'ok': True, 'items': 2, 'diagnostics': {'queries': [{'failure': 'private-key'}]}},
        {'source': 'Public', 'ok': True, 'items': 4, 'diagnostics': {'transport': 'public_snapshot', 'snapshot_fetched_at': '2026-10-14T10:00:00+03:00'}},
        {'source': 'Failed', 'ok': False, 'items': 0, 'error': 'private-token'},
        {'source': 'Legacy', 'items': 1},
    ]}
    (tmp_path/'health.json').write_text(json.dumps(payload))
    result = coverage_summary()
    assert (result['total'], result['responded'], result['unavailable'], result['limited'], result['unknown']) == (6, 4, 1, 1, 1)
    assert result['rows'][1]['status'] == 'No items returned'
    assert result['rows'][2]['status'] == 'Limited results'
    assert result['public_oldest_label'] == 'Oct 14, 2026 07:00 UTC'
    assert result['checked_label'] == 'Oct 14, 2026 09:00 UTC'
    assert 'private-' not in json.dumps(result)


def test_coverage_does_not_invent_feed_freshness_and_reports_old_check(tmp_path, monkeypatch):
    from dailydigest.health import coverage_summary

    monkeypatch.setattr('dailydigest.health.get_settings', lambda: type('S', (), {'db_path': str(tmp_path/'digest.db')})())
    payload = {'updated_at': (datetime.now(timezone.utc)-timedelta(days=2)).isoformat(), 'latest': [
        {'source': 'Public', 'ok': True, 'items': 1, 'diagnostics': {'transport': 'public_snapshot', 'snapshot_fetched_at': 'unknown'}},
    ]}
    (tmp_path/'health.json').write_text(json.dumps(payload))
    result = coverage_summary()
    assert result['checked_stale'] is True
    assert result['public_oldest_label'] == ''
    assert result['rows'][0]['freshness'] == 'Public feed time not recorded'


def test_coverage_missing_or_damaged_optional_metadata_does_not_block_reading(tmp_path, monkeypatch):
    from dailydigest.health import coverage_summary

    monkeypatch.setattr('dailydigest.health.get_settings', lambda: type('S', (), {'db_path': str(tmp_path/'digest.db')})())
    assert coverage_summary()['total'] == 0
    (tmp_path/'health.json').write_text('[]')
    assert coverage_summary()['total'] == 0
    (tmp_path/'health.json').write_text(json.dumps({'latest': [None]}))
    assert coverage_summary()['unknown'] == 1
