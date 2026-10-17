import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from dailydigest import health


def test_rebrews_do_not_erase_failures_in_last_seven_days(tmp_path, monkeypatch):
    path = tmp_path / 'health.json'
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    path.write_text(json.dumps({'history': [{'recorded_at': yesterday.isoformat(),
        'stats': [{'source': 'feed', 'items': 0, 'ok': False, 'error': 'timeout'}]}]}))
    monkeypatch.setattr(health, '_health_path', lambda: path)
    monkeypatch.setattr(health, 'ensure_data_dir', lambda: None)
    monkeypatch.setattr(health, 'get_settings', lambda: SimpleNamespace(db_path=str(tmp_path / 'test.db')))
    for _ in range(10):
        health.record([health.IngestStats(source='feed', items=2)])
    assert health.weekly_summary() == [dict(source='feed', items_7d=20, failures_7d=1, last_error='timeout')]


def test_history_retains_seven_calendar_days_without_losing_same_day_checks(tmp_path, monkeypatch):
    path = tmp_path / 'health.json'
    old = datetime.now(timezone.utc) - timedelta(days=8)
    path.write_text(json.dumps({'history': [{'recorded_at': old.isoformat(),
        'stats': [{'source': 'feed', 'items': 900, 'ok': False}]}]}))
    monkeypatch.setattr(health, '_health_path', lambda: path)
    monkeypatch.setattr(health, 'ensure_data_dir', lambda: None)
    health.record([health.IngestStats(source='feed', ok=False, error='first failure')])
    health.record([health.IngestStats(source='feed', items=3)])
    assert len(json.loads(path.read_text())['history']) == 2
    assert health.weekly_summary() == [dict(source='feed', items_7d=3, failures_7d=1, last_error='first failure')]
