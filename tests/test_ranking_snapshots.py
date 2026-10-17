"""Immutable measurement records must survive rebrews and failed publishes."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.exc import IntegrityError


def _item():
    from dailydigest import store

    store.init_db()
    with store.session_scope() as session:
        row = store.ItemRow(
            source="Nature", section="research", external_id="snapshot-paper",
            url="https://example.com/snapshot-paper", title="A research paper",
            abstract="Frozen evidence", published_at=datetime.now(timezone.utc),
        )
        session.add(row)
        session.flush()
        return int(row.id)


def _snapshot(item_id, run_id="run-1"):
    return {
        "schema_version": 1, "run_id": run_id, "digest_id": "2026-10-02",
        "decision_at": "2026-10-02T12:00:00+00:00",
        "complete_candidate_pool": True, "reading_mode": "full",
        "section_budgets": {"research": 1},
        "candidates": [{
            "item_id": item_id, "family_id": "paper", "section": "research",
            "eligible": True, "ranking_score": 0.8, "quality_score": 0.7,
            "selected": True, "position": 0,
            "metadata": {"title": "Original title", "abstract": "Frozen evidence"},
        }],
    }


def test_rebrew_keeps_original_snapshot_and_metadata():
    from dailydigest import store

    item_id = _item()
    for run_id, title in [("run-1", "Original title"), ("run-2", "Revised title")]:
        snapshot = _snapshot(item_id, run_id)
        snapshot["candidates"][0]["metadata"]["title"] = title
        store.publish_digest(
            "2026-10-02", [("R1", item_id, 0.8)], [], [],
            run_id=run_id, ranking_snapshot=snapshot,
        )
    with store.session_scope() as session:
        session.get(store.ItemRow, item_id).title = "Changed later"
    first = store.load_ranking_snapshot("run-1")
    second = store.load_ranking_snapshot("run-2")
    assert first["candidates"][0]["metadata"]["title"] == "Original title"
    assert second["candidates"][0]["metadata"]["title"] == "Revised title"
    assert len(store.list_ranking_snapshots()) == 2


def test_run_listing_is_bounded_and_cli_does_not_expose_payload():
    import json
    from typer.testing import CliRunner
    from dailydigest import store
    from dailydigest.cli import app

    item_id = _item()
    for run_id in ("older", "newer"):
        store.publish_digest(
            "2026-10-02", [("R1", item_id, 0.8)], [], [], run_id=run_id,
            ranking_snapshot=_snapshot(item_id, run_id),
        )
    result = CliRunner().invoke(app, ["ranking-runs", "--limit", "1"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["count"] == 1
    assert payload["runs"][0]["run_id"] == "newer"
    assert "candidates" not in payload["runs"][0]
    assert "Original title" not in result.output
    assert CliRunner().invoke(app, ["ranking-runs", "--limit", "0"]).exit_code == 2


def test_cli_absent_run_reports_incomplete_without_inventing_a_replay():
    import json
    from typer.testing import CliRunner
    from dailydigest.cli import app

    result = CliRunner().invoke(app, ["replay-run", "never-captured"])
    assert result.exit_code == 2
    assert json.loads(result.output)["status"] == "incomplete"


def test_duplicate_run_id_rolls_back_whole_publication():
    from dailydigest import store

    item_id = _item()
    store.publish_digest(
        "2026-10-02", [("R1", item_id, 0.8)], [], [],
        run_id="run-1", ranking_snapshot=_snapshot(item_id),
    )
    with pytest.raises(IntegrityError):
        store.publish_digest(
            "2026-10-02", [("R2", item_id, 0.2)], [], [],
            run_id="run-1", ranking_snapshot=_snapshot(item_id),
        )
    with store.session_scope() as session:
        slate = session.query(store.DigestItemRow).one()
        assert slate.item_label == "R1"
        assert slate.score == 0.8


def test_feedback_changes_append_events_and_rollback_atomically():
    from dailydigest import store, votes

    item_id = _item()
    with store.session_scope() as session:
        votes._upsert_vote(session, item_id, 1, 100)
    with store.session_scope() as session:
        votes._upsert_vote(session, item_id, -1, 40)
    with pytest.raises(RuntimeError, match="interrupted"):
        with store.session_scope() as session:
            votes._upsert_vote(session, item_id, -1, 10)
            raise RuntimeError("interrupted")
    events = store.load_feedback_events()
    assert [(event["value"], event["grade"]) for event in events] == [(1, 100), (-1, 40)]
    with store.session_scope() as session:
        latest = session.query(store.VoteRow).one()
        assert (latest.value, latest.grade) == (-1, 40)
    assert store.load_feedback_events(before=datetime(2000, 1, 1, tzinfo=timezone.utc)) == []


def test_pruning_expired_snapshot_keeps_feedback_and_voted_item():
    from dailydigest import store, votes

    item_id = _item()
    store.publish_digest(
        "2026-10-02", [("R1", item_id, 0.8)], [], [],
        run_id="run-1", ranking_snapshot=_snapshot(item_id),
    )
    old = datetime.now(timezone.utc) - timedelta(days=60)
    with store.session_scope() as session:
        session.get(store.DigestRow, "2026-10-02").created_at = old
        session.get(store.ItemRow, item_id).fetched_at = old
        votes._upsert_vote(session, item_id, 1, 70)
    assert store.prune_old_digests(30) == 1
    assert store.load_ranking_snapshot("run-1") is None
    assert store.prune(30) == 0
    assert len(store.load_feedback_events()) == 1


@pytest.mark.parametrize("field,value", [
    ("run_id", "wrong-run"), ("digest_id", "wrong-day"),
    ("schema_version", 0), ("schema_version", True),
    ("quality_score", float("nan")),
])
def test_invalid_snapshot_does_not_modify_existing_slate(field, value):
    from dailydigest import store

    item_id = _item()
    store.publish_digest("2026-10-02", [("R1", item_id, 0.8)], [], [])
    snapshot = _snapshot(item_id)
    snapshot[field] = value
    with pytest.raises(ValueError):
        store.publish_digest(
            "2026-10-02", [("R2", item_id, 0.2)], [], [],
            run_id="run-1", ranking_snapshot=snapshot,
        )
    assert store.list_ranking_snapshots() == []
    with store.session_scope() as session:
        slate = session.query(store.DigestItemRow).one()
        assert (slate.item_label, slate.score) == ("R1", 0.8)


def test_snapshot_metadata_filter_and_payload_are_independent():
    import json
    import zlib
    from dailydigest import store

    item_id = _item()
    snapshot = _snapshot(item_id)
    store.publish_digest(
        "2026-10-02", [("R1", item_id, 0.8)], [], [],
        run_id="run-1", ranking_snapshot=snapshot,
    )
    snapshot["candidates"].clear()
    loaded = store.load_ranking_snapshot("run-1")
    assert len(loaded["candidates"]) == 1
    assert store.load_ranking_snapshot("unknown") is None
    assert store.list_ranking_snapshots(digest_id="other-day") == []
    assert store.list_ranking_snapshots(digest_id="2026-10-02")[0]["schema_version"] == 1
    with store.session_scope() as session:
        stored = session.get(store.RankingSnapshotRow, "run-1")
        assert json.loads(zlib.decompress(stored.payload_zlib)) == loaded


def test_feedback_state_preserves_legacy_baseline_without_backfill():
    from dailydigest import store, votes

    store.init_db()
    assert store.feedback_state() == {"max_event_id": 0, "votes": []}
    item_id = _item()
    with store.session_scope() as session:
        session.add(store.VoteRow(item_id=item_id, value=1, grade=70))
    state = store.feedback_state()
    assert state["max_event_id"] == 0
    assert state["votes"][0]["grade"] == 70
    assert store.load_feedback_events() == []
    with store.session_scope() as session:
        votes._upsert_vote(session, item_id, -1, 10)
    state = store.feedback_state()
    assert state["max_event_id"] == 1
    assert state["votes"][0]["grade"] == 10
    event = store.load_feedback_events()[0]
    recorded_at = datetime.fromisoformat(event["recorded_at"])
    assert recorded_at.tzinfo is not None
    assert store.load_feedback_events(before=recorded_at) == []
    assert len(store.load_feedback_events(before=recorded_at + timedelta(microseconds=1))) == 1


def test_feedback_events_protect_history_when_current_vote_is_removed():
    from sqlalchemy import delete
    from dailydigest import store, votes

    item_id = _item()
    with store.session_scope() as session:
        votes._upsert_vote(session, item_id, 1, 70)
        session.get(store.ItemRow, item_id).fetched_at = datetime.now(timezone.utc) - timedelta(days=60)
    with store.session_scope() as session:
        session.execute(delete(store.VoteRow).where(store.VoteRow.item_id == item_id))
    assert store.prune(30) == 0
    assert len(store.load_feedback_events()) == 1
