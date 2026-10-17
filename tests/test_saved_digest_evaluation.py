"""Daily audit reads immutable decisions and events without initializing a DB."""
import json
import sqlite3
import zlib

import pytest

from dailydigest.rank.digest_evaluation import evaluate_digest_variants
from tests.test_digest_evaluation import snapshot


def database(tmp_path, snapshots=(), events=()):
    path = tmp_path / "private" / "digest.db"
    path.parent.mkdir(exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE ranking_snapshots (run_id TEXT, created_at TEXT, payload_zlib BLOB)")
        db.execute("CREATE TABLE feedback_events (id INTEGER, item_id INTEGER, value INTEGER, grade INTEGER, recorded_at TEXT)")
        for snap in snapshots:
            db.execute("INSERT INTO ranking_snapshots VALUES (?, ?, ?)",
                       (snap["run_id"], snap["decision_at"], zlib.compress(json.dumps(snap).encode())))
        db.executemany("INSERT INTO feedback_events VALUES (?, ?, ?, ?, ?)", events)
    return path


def load(path, **kwargs):
    from dailydigest.rank.saved_evaluation import load_saved_evaluation
    return load_saved_evaluation(path, outcome_as_of="2026-10-05T00:00:00Z", **kwargs)


def test_latest_run_per_observed_day_and_cutoff_are_frozen(tmp_path):
    first, upgraded, second, future = snapshot(), snapshot(), snapshot(2), snapshot(3)
    upgraded.update(run_id="run-upgraded", decision_at="2026-10-01T18:00:00Z")
    future.update(run_id="future", decision_at="2026-10-05T00:00:00Z")
    path = database(tmp_path, [first, upgraded, second, future])
    before = path.read_bytes()
    result = load(path, days=2)
    assert [snap["run_id"] for snap in result["snapshots"]] == ["run-upgraded", "run-2"]
    assert path.read_bytes() == before
    assert len(load(path, days=1)["snapshots"]) == 1


def test_signed_feedback_clear_and_important_omission_are_evaluated(tmp_path):
    path = database(tmp_path, [snapshot()], [
        (1, 1, 1, 70, "2026-10-02T00:00:00Z"),
        (2, 1, 0, None, "2026-10-03T00:00:00Z"),
        (3, 2, -1, 40, "2026-10-03T00:00:00Z"),
        (4, 3, 1, None, "2026-10-03T00:00:00Z"),
        (5, 4, 1, 100, "2026-10-03T00:00:00Z"),
        (6, 2, 1, 70, "2026-10-05T00:00:00Z"),
        (7, 999, 1, 100, "2026-10-03T00:00:00Z"),
    ])
    result = load(path)
    report = evaluate_digest_variants(result["snapshots"], result["judgments"], outcome_as_of="2026-10-05T00:00:00Z")
    metrics = report["variants"]["baseline"]["aggregate"]["research"]
    assert metrics["useful_count"] == 1 and metrics["judged"] == 2 and metrics["unknown"] == 1
    assert report["variants"]["baseline"]["per_day"][0]["diagnostics"]["research"]["observed_important_missed_ids"] == [4]
    assert result["coverage"]["important_outside_supplied_pools_item_ids"] == [999]
    assert result["coverage"]["signed_items_in_supplied_pools"] == 3


def test_simultaneous_events_use_append_order_and_neutral_does_not_become_dislike(tmp_path):
    path = database(tmp_path, [snapshot()], [(1, 1, 1, 70, "2026-10-03"), (2, 1, 0, 50, "2026-10-03")])
    row = load(path)["judgments"][0]
    assert row["useful"] is None and row["grade"] is None and row["important"] is None


def test_absent_database_is_not_created(tmp_path):
    path = tmp_path / "missing.db"
    with pytest.raises(ValueError, match="read"):
        load(path)
    assert not path.exists()


def test_empty_database_is_reported_without_creating_schema(tmp_path):
    path = tmp_path / "empty.db"
    with sqlite3.connect(path):
        pass
    with pytest.raises(ValueError, match="snapshot"):
        load(path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT name FROM sqlite_master").fetchall() == []


def test_no_snapshots_before_cutoff_is_explicit(tmp_path):
    path = database(tmp_path)
    assert load(path)["snapshots"] == []


def test_manifest_pins_original_run_after_newer_same_day_brew(tmp_path):
    old, newer = snapshot(), snapshot()
    newer.update(run_id="newer", decision_at="2026-10-01T18:00:00Z")
    path = database(tmp_path, [old, newer])
    assert load(path, run_ids={"run-1"})["snapshots"] == [old]
    with pytest.raises(ValueError, match="Pinned snapshots"):
        load(path, run_ids={"absent"})


def test_missing_feedback_history_does_not_backfill_current_votes(tmp_path):
    path = database(tmp_path, [snapshot()])
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE feedback_events")
        db.execute("CREATE TABLE votes (item_id INTEGER, value INTEGER, grade INTEGER)")
        db.execute("INSERT INTO votes VALUES (1, 1, 100)")
    result = load(path)
    assert result["judgments"] == []
    assert result["coverage"]["feedback_history"] == "missing"


def test_important_ineligible_paper_exposes_the_recorded_rejection_stage(tmp_path):
    snap = snapshot()
    snap["candidates"][3].update(eligible=False, first_rejected_stage="after_topic_gate")
    path = database(tmp_path, [snap], [(1, 4, 1, 100, "2026-10-03T00:00:00Z")])
    assert load(path)["coverage"]["important_ineligible_candidates"] == [
        {"run_id": "run-1", "item_id": 4, "first_rejected_stage": "after_topic_gate"}]


@pytest.mark.parametrize("days", [0, -1, True, 1.5])
def test_invalid_day_count_is_rejected(tmp_path, days):
    with pytest.raises(ValueError, match="positive integer"):
        load(tmp_path / "missing.db", days=days)
