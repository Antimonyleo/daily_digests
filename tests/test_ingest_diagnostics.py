"""Query-level health records preserve new diagnostics and legacy snapshots."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from dailydigest import health


@pytest.fixture
def health_path(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "get_settings", lambda: SimpleNamespace(db_path=tmp_path / "digest.db"))
    monkeypatch.setattr(health, "ensure_data_dir", lambda: None)
    return tmp_path / "health.json"


def test_legacy_ingest_stats_default_to_independent_empty_diagnostics():
    first = health.IngestStats.model_validate({"source": "old", "items": 5})
    second = health.IngestStats(source="other")
    assert first.diagnostics == {}
    first.diagnostics["queries"] = []
    assert second.diagnostics == {}


@pytest.mark.parametrize("ok", [True, False])
def test_query_diagnostics_persist_in_latest_and_history(health_path, ok):
    diagnostics = {
        "item_budget": 180,
        "queries": [{"query": "RNA", "attempted": True, "fetched_count": 20,
                     "unique_yield": 18, "truncated": True, "exhausted": False,
                     "stop_reason": "item_cap" if ok else "failure",
                     "total_available": 200, "indexing_delay": None}],
    }
    stat = health.IngestStats(source="OpenAlex", items=18 if ok else 0, ok=ok,
                              error=None if ok else "timeout", diagnostics=diagnostics)
    health.record([stat])
    saved = json.loads(health_path.read_text())
    assert saved["latest"][0]["diagnostics"] == diagnostics
    assert saved["history"][0]["stats"][0]["diagnostics"] == diagnostics
    assert health.latest_snapshot()[0]["diagnostics"] == diagnostics
    assert health.weekly_summary()[0]["failures_7d"] == int(not ok)


def test_new_record_preserves_legacy_history_without_inventing_diagnostics(health_path):
    old = {"source": "OpenAlex", "items": 5, "ok": True}
    health_path.write_text(json.dumps({
        "latest": [old], "history": [{
            "recorded_at": datetime.now(timezone.utc).isoformat(), "stats": [old],
        }],
    }))
    assert health.latest_snapshot() == [old]
    health.record([health.IngestStats(source="OpenAlex", items=2, diagnostics={"queries": []})])
    history = json.loads(health_path.read_text())["history"]
    assert history[0]["stats"] == [old]
    assert history[1]["stats"][0]["diagnostics"] == {"queries": []}
    assert health.weekly_summary()[0]["items_7d"] == 7
