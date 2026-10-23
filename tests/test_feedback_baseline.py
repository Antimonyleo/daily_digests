"""Read-only evidence separates current opinions, immutable events, and saves."""
import json
import sqlite3
import zlib

import pytest

from dailydigest.rank.feedback_baseline import (
    build_feedback_baseline, build_feedback_cases, forward_holdout_protocol, main,
)
from tests.test_digest_evaluation import snapshot


CUTOFF = "2026-10-05T00:00:00Z"


def database(tmp_path, *, snapshots=(), events=(), votes=(), saves=(), displays=()):
    path = tmp_path / "private" / "digest.db"
    path.parent.mkdir(exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE ranking_snapshots (run_id TEXT, created_at TEXT, payload_zlib BLOB)")
        db.execute("CREATE TABLE feedback_events (id INTEGER, item_id INTEGER, value INTEGER, grade INTEGER, recorded_at TEXT)")
        db.execute("CREATE TABLE votes (item_id INTEGER, value INTEGER, grade INTEGER)")
        db.execute("CREATE TABLE bookmarks (item_id INTEGER)")
        db.execute("CREATE TABLE digest_items (digest_id TEXT, item_id INTEGER, item_label TEXT)")
        for snap in snapshots:
            db.execute("INSERT INTO ranking_snapshots VALUES (?, ?, ?)",
                       (snap["run_id"], snap["decision_at"], zlib.compress(json.dumps(snap).encode())))
        db.executemany("INSERT INTO feedback_events VALUES (?, ?, ?, ?, ?)", events)
        db.executemany("INSERT INTO votes VALUES (?, ?, ?)", votes)
        db.executemany("INSERT INTO bookmarks VALUES (?)", [(iid,) for iid in saves])
        db.executemany("INSERT INTO digest_items VALUES (?, ?, ?)", displays)
    return path


def report(path):
    return build_feedback_baseline(path, outcome_as_of=CUTOFF)


def test_snapshot_read_preserves_database_and_exports_no_private_text(tmp_path):
    snap = snapshot()
    snap["profile"] = {"name": "PRIVATE NAME", "interests": ["SECRET TOPIC"]}
    snap["profile_signature"] = "PRIVATE SIGNATURE"
    snap["candidates"][0]["metadata"]["title"] = "PRIVATE PAPER TITLE"
    path = database(tmp_path, snapshots=[snap])
    before, mtime = path.read_bytes(), path.stat().st_mtime_ns
    result = report(path)
    assert path.read_bytes() == before and path.stat().st_mtime_ns == mtime
    serialized = json.dumps(result)
    assert all(secret not in serialized for secret in ("PRIVATE", "SECRET TOPIC", "run-1", "family-1"))
    assert not result["first_shot_quality_certified"]


def test_current_votes_and_saves_never_backfill_historical_events(tmp_path):
    path = database(tmp_path, snapshots=[snapshot()], votes=[(1, 1, 100), (2, -1, 10)],
                    saves=[1, 2, 3], displays=[("day", 1, "R1"), ("day", 2, "R2"), ("day", 3, "R3")])
    result = report(path)
    history = result["timestamped_development_only"]
    assert history["selected"]["judged"] == 0
    assert history["top_five"]["useful_fraction_bounds"] == [0, 1]
    current = result["current_state_development_only"]
    assert current["weak_saves"] == {"items": 3, "with_explicit_positive": 1,
                                      "with_explicit_negative": 1, "without_signed_rating": 1,
                                      "used_as_positive_labels": False}
    assert current["retained_research"]["top_five"]["useful_fraction_bounds"] == [1 / 3, 2 / 3]


def test_event_cutoff_is_exclusive_neutral_clear_wins_and_future_is_ignored(tmp_path):
    path = database(tmp_path, snapshots=[snapshot()], events=[
        (1, 1, 1, 100, "2026-10-02"), (2, 1, 0, 50, "2026-10-02"),
        (3, 2, -1, 10, "2026-10-03"), (4, 3, 1, 70, CUTOFF),
        (5, 2, 1, 100, "2026-10-06"), (6, 4, 1, 100, "2026-10-03"),
    ])
    evidence = report(path)["timestamped_development_only"]
    assert evidence["timestamped_events_before_cutoff"] == 4
    assert evidence["selected"]["positive"] == 0
    assert evidence["selected"]["negative"] == 1
    assert evidence["selected"]["unknown"] == 2
    assert evidence["eligible_omitted"]["positive"] == 1


def test_latest_run_per_day_does_not_double_count_and_excludes_cutoff_run(tmp_path):
    first, second, future = snapshot(), snapshot(), snapshot(2)
    second.update(run_id="updated", decision_at="2026-10-01T18:00:00Z")
    second["candidates"][0]["selected"] = False
    second["candidates"][0]["position"] = None
    for candidate in second["candidates"][1:3]:
        candidate["position"] -= 1
    future["decision_at"] = CUTOFF
    path = database(tmp_path, snapshots=[first, second, future])
    result = report(path)["timestamped_development_only"]
    assert result["retained_runs_before_cutoff"] == 2
    assert result["observed_days"] == 1
    assert result["selected"]["occurrences"] == 2


def test_short_slate_is_not_counted_as_fully_judged_five(tmp_path):
    path = database(tmp_path, snapshots=[snapshot()], events=[
        (iid, iid, 1, 100, "2026-10-03") for iid in range(1, 4)
    ])
    result = report(path)["timestamped_development_only"]
    assert result["top_five"]["useful_fraction_bounds"] == [1, 1]
    assert result["days_with_fewer_than_five"] == 1
    assert result["days_with_five_fully_judged"] == 0
    assert result["days_with_first_three_fully_judged"] == 1


def test_eligible_omissions_ineligible_positives_and_outside_pool_are_separate(tmp_path):
    snap = snapshot()
    snap["candidates"][4]["eligible"] = False
    path = database(tmp_path, snapshots=[snap], events=[
        (iid, iid, 1, 100, "2026-10-03") for iid in (4, 5, 999)
    ])
    result = report(path)["timestamped_development_only"]
    assert result["eligible_omitted"]["positive"] == 1
    assert result["ineligible"]["positive"] == 1
    assert result["signed_items_outside_captured_research_pools"] == 1
    assert "999" not in json.dumps(result)


def test_profile_drift_and_prior_feedback_are_counted_without_identity_export(tmp_path):
    first, second = snapshot(), snapshot(2)
    first.update(profile_signature="alpha", scorer_version="version-a")
    second.update(profile_signature="beta", scorer_version="version-b")
    path = database(tmp_path, snapshots=[first, second], events=[(1, 1, 1, 100, "2026-10-01T14:00:00Z")])
    result = report(path)["timestamped_development_only"]
    assert result["distinct_recorded_profile_signatures"] == 2
    assert result["distinct_recorded_scorer_versions"] == 2
    assert result["runs_with_observed_predecision_signed_feedback"] == 1
    assert result["runs_without_observed_predecision_signed_feedback"] == 1


def test_omitted_on_a_later_day_is_not_counted_as_never_shown(tmp_path):
    first, second = snapshot(), snapshot(2)
    second["candidates"][0].update(selected=False, position=None)
    for candidate in second["candidates"][1:3]:
        candidate["position"] -= 1
    path = database(tmp_path, snapshots=[first, second], events=[(1, 1, 1, 100, "2026-10-03")])
    result = report(path)["timestamped_development_only"]["unique_items"]
    assert result["eligible_omitted"]["positive"] == 1
    assert result["selected"]["positive"] == 1
    assert result["eligible_omitted_never_selected_in_supplied_runs"]["positive"] == 0


def test_missing_database_or_schema_is_never_created(tmp_path):
    path = tmp_path / "absent.db"
    with pytest.raises(ValueError, match="read"):
        report(path)
    assert not path.exists()
    with sqlite3.connect(path):
        pass
    with pytest.raises(ValueError, match="No feedback"):
        report(path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT name FROM sqlite_master").fetchall() == []


def test_missing_history_tables_are_explicit_and_current_votes_remain_separate(tmp_path):
    path = database(tmp_path, votes=[(1, 1, 100)])
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE feedback_events")
        db.execute("DROP TABLE ranking_snapshots")
    result = report(path)
    assert "feedback_events" not in result["available_tables"]
    assert result["timestamped_development_only"]["signed_items_at_cutoff"] == 0
    assert result["current_state_development_only"]["votes"]["positive"] == 1


@pytest.mark.parametrize("mutation", ["duplicate_vote", "invalid_event", "selected_ineligible", "bad_position"])
def test_ambiguous_or_inconsistent_evidence_fails_closed(tmp_path, mutation):
    snap = snapshot()
    if mutation == "selected_ineligible":
        snap["candidates"][0]["eligible"] = False
    if mutation == "bad_position":
        snap["candidates"][0]["position"] = 8
    path = database(tmp_path, snapshots=[snap],
                    votes=[(1, 1, 70), (1, -1, 10)] if mutation == "duplicate_vote" else [],
                    events=[(1, 1, 2, 100, "2026-10-02")] if mutation == "invalid_event" else [])
    with pytest.raises(ValueError):
        report(path)


def test_forward_protocol_never_rebrands_historical_data_as_holdout():
    protocol = forward_holdout_protocol(frozen_at=CUTOFF, evidence_cutoff=CUTOFF)
    assert protocol["completed_human_evaluations"] == 0
    assert not protocol["promotion_gate"]["release_approved"]
    assert "after this freeze" in protocol["forward_holdout"]["enrollment"]
    assert "no same family" in protocol["forward_holdout"]["separation"]
    with pytest.raises(ValueError, match="precede"):
        forward_holdout_protocol(frozen_at="2026-10-01", evidence_cutoff=CUTOFF)


def test_cli_never_overwrites_an_existing_file_or_database(tmp_path):
    path = database(tmp_path, snapshots=[snapshot()])
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        main(["--database", str(path), "--outcome-as-of", CUTOFF, "--output", str(path)])
    assert path.read_bytes() == before
    output = tmp_path / "reports" / "aggregate.json"
    main(["--database", str(path), "--outcome-as-of", CUTOFF, "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["forward_holdout_protocol"]["status"] == "protocol_only_not_enrolled_or_validated"


def test_development_export_pseudonyms_are_consistent_within_export_and_not_identifying(tmp_path):
    first, second = snapshot(), snapshot(2)
    for snap in (first, second):
        snap["profile_signature"] = "PRIVATE GOAL SIGNATURE"
        snap["profile"] = {"name": "PRIVATE READER", "bio": "PRIVATE GOAL TEXT"}
        snap["stage_results"] = {"after_quality_gate": [1], "PRIVATE STAGE": [1]}
        snap["candidates"][0]["family_id"] = "PRIVATE FAMILY"
        snap["candidates"][0]["features"].update(primary_facet="PRIVATE FACET", topic_score=.8)
        snap["candidates"][0]["metadata"]["title"] = "PRIVATE TITLE"
    first["candidates"][1]["family_id"] = "PRIVATE FAMILY"
    path = database(tmp_path, snapshots=[first, second], events=[(1, 1, 1, 100, "2026-10-03")])
    before = path.read_bytes()
    result = build_feedback_cases(path, outcome_as_of=CUTOFF)
    rows = result["cases"]
    assert rows[0]["paper_id"] == rows[6]["paper_id"]
    assert rows[0]["goal_id"] == rows[6]["goal_id"]
    assert rows[0]["decision_id"] != rows[6]["decision_id"]
    assert rows[0]["family_id"] == rows[1]["family_id"] == rows[6]["family_id"]
    assert rows[0]["scores"]["topic_score"] == .8
    assert rows[0]["stages"] == {"after_quality_gate": True}
    assert rows[0]["outcome_signed"] == 1
    assert rows[0]["outcome_goal_alignment"] == "unknown"
    encoded = json.dumps(result)
    assert all(secret not in encoded for secret in ("PRIVATE", "Paper 1", "family-", "run-1", "example.org", "RNA design", "item_id"))
    assert all(row["split"] == "development" for row in rows)
    assert path.read_bytes() == before
    another = build_feedback_cases(path, outcome_as_of=CUTOFF)
    assert another["cases"][0]["paper_id"] != rows[0]["paper_id"]


def test_development_export_neutral_clear_and_absent_events_never_use_current_votes(tmp_path):
    path = database(tmp_path, snapshots=[snapshot()], votes=[(1, 1, 100)],
                    events=[(1, 1, 1, 100, "2026-10-02"), (2, 1, 0, 50, "2026-10-03")])
    result = build_feedback_cases(path, outcome_as_of=CUTOFF)
    assert all(row["outcome_signed"] is None for row in result["cases"])
    assert all(row["outcome_grade"] is None for row in result["cases"])
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE feedback_events")
    result = build_feedback_cases(path, outcome_as_of=CUTOFF)
    assert all(row["outcome_signed"] is None for row in result["cases"])
    assert all(not row["item_had_predecision_signed_feedback"] for row in result["cases"])


def test_development_export_marks_absent_goal_family_and_sanitizes_unknown_fields(tmp_path):
    snap = snapshot()
    candidate = snap["candidates"][0]
    candidate.pop("family_id")
    candidate["input_position"] = "PRIVATE POSITION"
    candidate["first_rejected_stage"] = "PRIVATE STAGE"
    candidate["features"].update(topic_score=float('nan'), final_score="PRIVATE SCORE")
    path = database(tmp_path, snapshots=[snap])
    result = build_feedback_cases(path, outcome_as_of=CUTOFF)
    row = result["cases"][0]
    assert row["family_id"] is None and row["family_basis"] == "unavailable"
    assert row["goal_id"] is None and row["goal_basis"] == "unknown"
    assert row["input_position_zero_based"] is None
    assert row["first_rejected_stage"] == "other_recorded_stage"
    assert row["scores"]["topic_score"] is None and row["scores"]["final_score"] is None
    assert "PRIVATE" not in json.dumps(result, allow_nan=False)


def test_development_cli_requires_ignored_output_and_exclusive_write(tmp_path, monkeypatch):
    from dailydigest.rank import feedback_baseline
    path = database(tmp_path, snapshots=[snapshot()])
    root = tmp_path / "ignored"
    monkeypatch.setattr(feedback_baseline, "_CASE_OUTPUT_ROOT", root)
    output = tmp_path / "aggregate.json"
    args = ["--database", str(path), "--outcome-as-of", CUTOFF, "--output", str(output)]
    with pytest.raises(ValueError, match="must stay"):
        main([*args, "--cases-output", str(tmp_path / "public.json")])
    assert not output.exists()
    cases = root / "development.json"
    main([*args, "--cases-output", str(cases)])
    assert json.loads(cases.read_text())["case_count"] == 6
    before = cases.read_bytes()
    with pytest.raises(FileExistsError):
        main([*args, "--cases-output", str(cases)])
    assert cases.read_bytes() == before


def test_aggregate_and_cases_cli_share_one_read_snapshot(tmp_path, monkeypatch):
    from dailydigest.rank import feedback_baseline
    path = database(tmp_path, snapshots=[snapshot()])
    monkeypatch.setattr(feedback_baseline, "_CASE_OUTPUT_ROOT", tmp_path)
    original = feedback_baseline._read
    reads = []

    def read_once(database):
        reads.append(database)
        return original(database)

    monkeypatch.setattr(feedback_baseline, "_read", read_once)
    main(["--database", str(path), "--outcome-as-of", CUTOFF,
          "--output", str(tmp_path / "aggregate.json"), "--cases-output", str(tmp_path / "development.json")])
    assert reads == [path]
