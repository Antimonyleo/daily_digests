"""Frozen offline evaluation: no leakage, unknown labels stay unknown, explicit ship rule."""
from __future__ import annotations

import hashlib
import json
import socket
import sqlite3
import zlib
from copy import deepcopy

import numpy as np
import pytest

from dailydigest.rank import offline_eval as oe

DECISION = "2026-10-10T12:00:00+00:00"
SIGNATURE = "test-model:abc"
DIM = 4


def _unit(values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _text(title, abstract):
    return f"{title}. {abstract}" if abstract else title


def _candidate_vector(iid: int) -> np.ndarray:
    angle = iid / 3.0
    return _unit([1.0, 0.4 * np.cos(angle), 0.4 * np.sin(angle), 0.1 * (iid % 3)])


def _exemplar_vector(iid: int) -> np.ndarray:
    # Even exemplars sit near the first axis (liked), odd ones near the third.
    return _unit([1.0, 0.2, 0.05, 0.0]) if iid % 2 == 0 else _unit([0.2, 0.0, 1.0, 0.3])


def _snapshot(*, votes=None, scorer_version=None):
    from dailydigest.rank.source_quality import RANKER_VERSION

    candidates = []
    for iid in range(1, 13):
        metadata = {"source": f"Frozen Journal {iid % 4}", "title": f"paper {iid}",
                    "abstract": f"abstract {iid}", "url": f"https://example.org/{iid}",
                    "external_id": f"ext-{iid}", "authors": "", "metadata_json": "{}",
                    "published_at": "2026-10-09T00:00:00", "fetched_at": "2026-10-09T01:00:00"}
        candidates.append({
            "item_id": iid, "family_id": f"fam-{iid}", "section": "research",
            "identity_keys": [f"doi:10.1/{iid}"], "eligible": True, "input_position": iid - 1,
            "ranking_score": 1 - iid / 100, "selected": iid <= 3, "position": iid - 1 if iid <= 3 else None,
            "label": None, "metadata": metadata, "first_rejected_stage": None,
            "features": {"topic_score": 0.9, "preference_signal": 0.0, "confidence_score": 0.9,
                         "scoring_mode": "hybrid_knn", "reason_penalty": 0.0,
                         "primary_facet": "alpha" if iid % 2 else "beta", "primary_facet_score": 0.9,
                         "topic_priority": 0.0, "topic_priority_bonus": 0.0},
            "selection_facts": {"source_bucket": "published_journal", "is_arxiv_cs": False,
                                "is_preprint": False, "is_published": True, "is_high_quality": True,
                                "is_low_impact": False, "source_diversity_key": f"journal {iid}"},
        })
    if votes is None:
        votes = [{"item_id": 100 + i, "value": 1 if i % 2 == 0 else -1, "grade": 70 if i % 2 == 0 else 10,
                  "created_at": "2026-10-01T00:00:00+00:00"} for i in range(32)]
    ids = [c["item_id"] for c in candidates]
    return {
        "schema_version": 1, "run_id": "run-1", "digest_id": "2026-10-10",
        "decision_at": DECISION, "decision_local_date": "2026-10-10", "reading_mode": "full",
        "catch_up": False, "exploration_slots": 0, "complete_candidate_pool": True,
        "reading_order_policy": "selection_priority", "section_budgets": {"research": 3},
        "selection_settings": {"max_preprint_research_frac": 0.55, "max_low_impact_research_frac": 0.2,
                               "low_impact_relevance_floor": 0.0, "research_final_score_floor_frac": 0.0,
                               "research_final_score_min_keep": 3, "adaptive_relevance_floor": False},
        "ranking_settings": {"min_topic_relevance": 0.0, "research_final_score_floor_frac": 0.0,
                             "low_impact_relevance_floor": 0.0, "citation_enrichment": False,
                             "within_day_dedupe": False, "topic_coverage_bonus_scale": 0.0,
                             "embed_model": "test-model"},
        "profile": {"bio": "", "keywords": ["alpha", "beta"]},
        "profile_matrix": [_unit([1, 0, 0, 0]).tolist(), _unit([0.8, 0.6, 0, 0]).tolist(),
                           (2.0 * _unit([1, 0.1, 0, 0])).tolist()],
        "scorer_version": scorer_version or RANKER_VERSION, "embedding_signature": SIGNATURE,
        "feedback_at_start": {"max_event_id": 0, "votes": votes},
        "stage_results": {name: ids for name in (
            "after_reviewed", "after_shown_and_known", "after_identity_dedupe",
            "after_cross_day_dedupe", "after_quality_gate", "after_actionable_gate", "after_topic_gate")},
        "candidates": candidates,
    }


def _database(path, snapshot, *, label_votes=()):
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE items (id INTEGER PRIMARY KEY, title TEXT, abstract TEXT);
        CREATE TABLE votes (id INTEGER PRIMARY KEY, item_id INTEGER, value INTEGER, grade INTEGER,
                            created_at DATETIME);
        CREATE TABLE item_embeddings (id INTEGER PRIMARY KEY, item_id INTEGER, model TEXT,
                                      text_hash TEXT, dim INTEGER, vector BLOB);
        CREATE TABLE ranking_snapshots (run_id TEXT PRIMARY KEY, digest_id TEXT, schema_version INTEGER,
                                        payload_zlib BLOB, created_at DATETIME);
    """)

    def add_item(iid, title, abstract, vector):
        connection.execute("INSERT INTO items VALUES (?, ?, ?)", (iid, title, abstract))
        text_hash = hashlib.sha256(_text(title, abstract).encode()).hexdigest()
        connection.execute("INSERT INTO item_embeddings (item_id, model, text_hash, dim, vector) "
                           "VALUES (?, ?, ?, ?, ?)", (iid, SIGNATURE, text_hash, DIM, vector.tobytes()))

    for candidate in snapshot["candidates"]:
        meta = candidate["metadata"]
        add_item(candidate["item_id"], meta["title"], meta["abstract"], _candidate_vector(candidate["item_id"]))
    for iid in range(100, 140):
        add_item(iid, f"voted {iid}", "", _exemplar_vector(iid))
    for vote in snapshot["feedback_at_start"]["votes"]:
        connection.execute("INSERT INTO votes (item_id, value, grade, created_at) VALUES (?, ?, ?, ?)",
                           (vote["item_id"], vote["value"], vote["grade"], vote["created_at"][:19]))
    for vote in label_votes:
        connection.execute("INSERT INTO votes (item_id, value, grade, created_at) VALUES (?, ?, ?, ?)", vote)
    connection.execute("INSERT INTO ranking_snapshots VALUES (?, ?, 1, ?, ?)",
                       (snapshot["run_id"], snapshot["digest_id"],
                        zlib.compress(json.dumps(snapshot).encode()), snapshot["decision_at"][:19]))
    connection.commit()
    connection.close()
    return path


CONFIGS = [{"name": "current", "preference_weight": 0.5}, {"name": "knn_065", "preference_weight": 0.65}]
MARGINS = {"precision": 0.0, "auc": 0.0}


def _rescore(path, snapshot):
    connection = oe._connect(path)
    try:
        frozen = oe.freeze_run(snapshot, stage="veteran")
        embeddings = oe.FrozenEmbeddings(connection)
        result = oe.rescore_run(frozen, oe.EvalConfig.parse(CONFIGS[0]), embeddings=embeddings,
                                reasons={}, item_rows=oe._exemplar_rows(snapshot, connection))
    finally:
        connection.close()
    return {iid: (row["topic_score"], row["preference_signal"], row["ranking_score"])
            for iid, row in result["research"].items()}, result["slate"]


def test_future_votes_and_labels_never_change_past_scores(tmp_path):
    base = _snapshot()
    scores, slate = _rescore(_database(tmp_path / "a.db", base), base)
    assert {row[1] for row in scores.values()} != {0.0}  # the frozen kNN leg is active

    leaked = deepcopy(base)
    # A vote recorded after the decision and a contaminating future label.
    leaked["feedback_at_start"]["votes"].append(
        {"item_id": 138, "value": -1, "grade": 10, "created_at": "2026-10-11T00:00:00+00:00"})
    path = _database(tmp_path / "b.db", leaked, label_votes=[(1, -1, 10, "2026-10-12 00:00:00")])
    leaked_scores, leaked_slate = _rescore(path, leaked)
    assert leaked_scores == scores and leaked_slate == slate

    control = deepcopy(base)
    control["feedback_at_start"]["votes"].append(
        {"item_id": 138, "value": -1, "grade": 10, "created_at": "2026-10-09T00:00:00+00:00"})
    control_scores, _ = _rescore(_database(tmp_path / "c.db", control), control)
    assert control_scores != scores  # pre-decision state does reach the scorer


def _recorded_from_frozen_scorer(tmp_path, snapshot):
    """Fixture history whose recorded components equal what the frozen scorer gives."""
    connection = oe._connect(_database(tmp_path / "seed.db", snapshot))
    try:
        result = oe.rescore_run(oe.freeze_run(snapshot, stage="veteran"), oe.EvalConfig.parse(CONFIGS[0]),
                                embeddings=oe.FrozenEmbeddings(connection), reasons={},
                                item_rows=oe._exemplar_rows(snapshot, connection))
    finally:
        connection.close()
    for candidate in snapshot["candidates"]:
        row = result["research"][candidate["item_id"]]
        for key in ("topic_score", "preference_signal", "confidence_score"):
            candidate["features"][key] = row[key]
    return snapshot


def test_evaluate_reports_separate_modes_parity_and_development_evidence(tmp_path):
    snapshot = _recorded_from_frozen_scorer(tmp_path, _snapshot())
    path = _database(tmp_path / "d.db", snapshot, label_votes=[
        (1, 1, 70, "2026-10-11 00:00:00"), (2, -1, 10, "2026-10-11 00:00:00"),
        (3, -1, 40, "2026-10-11 00:00:00"), (101, 1, 100, "2026-10-11 00:00:00")])
    report = oe.evaluate(path, CONFIGS, baseline="current", margins=MARGINS, n_boot=50,
                         min_runs=1, min_strict_pairs=1)
    run = report["runs"][0]
    assert report["evidence_class"] == "development" and report["defaults_changed"] is False
    assert run["selector_replay"]["scope"].endswith("not rescoring")
    assert run["parity"]["current"]["component_parity"] == "ok"
    assert run["retrospective"] is True
    assert "vote_reasons_current_state" in run["freeze"]["veteran"]["retrospective_inputs"]
    # Item 101 was rated before the decision: training state, never an outcome.
    classes = oe._labels_for_run(oe.read_labels(oe._connect(path)), {101}, oe._time(DECISION))
    assert classes[101] == "pre_decision" and classes[2] == "strict_negative" and classes[3] == "weak_negative"
    stage = report["analyses"]["all"]["veteran"]
    assert set(stage["configs"]["current"]["contrasts"]) == set(oe.CONTRASTS)
    single = stage["configs"]["current"]["contrasts"][oe.PRIMARY_CONTRAST]["pairwise_accuracy"]
    delta = stage["comparisons_vs_baseline"]["knn_065"]["contrasts"][oe.PRIMARY_CONTRAST]["delta"]
    assert single["reference"] == 0.5 and delta["reference"] == 0.0
    funnel = {row["stage"]: row for row in stage["configs"]["current"]["funnel"]}
    assert funnel["selected"]["candidates"] == 3 and funnel["captured_pool"]["unique_works"] == 12
    assert report["analyses"]["all"]["overall_decisions"]["knn_065"] in oe.DECISIONS

    drifted = deepcopy(snapshot)
    drifted["candidates"][0]["features"]["topic_score"] += 0.05  # reconstruction mismatch
    report = oe.evaluate(_database(tmp_path / "drift.db", drifted), CONFIGS, baseline="current",
                         margins=MARGINS, n_boot=20, stages=("veteran",), min_runs=1, min_strict_pairs=1)
    assert report["runs"][0]["parity"]["current"]["component_parity"] == "failed"
    decision = report["analyses"]["all"]["veteran"]["decisions"]["knn_065"]
    assert decision["decision"] == "insufficient_evidence"
    assert "reconstruction parity failed on some runs" in decision["reasons"]


def test_first_shot_proxy_drops_learned_row_and_feedback(tmp_path):
    snapshot = _snapshot()
    frozen = oe.freeze_run(snapshot, stage="first_shot_proxy")
    assert frozen.profile_matrix.shape == (2, DIM) and frozen.votes == []
    assert frozen.freeze_notes["static_profile"] == "static_rows_verified_by_profile_weights"
    path = _database(tmp_path / "e.db", snapshot)
    report = oe.evaluate(path, CONFIGS, baseline="current", margins=MARGINS, n_boot=20,
                         min_runs=1, min_strict_pairs=1)
    slates = report["runs"][0]["slates"]["first_shot_proxy"]
    assert slates["current"] == slates["knn_065"]  # no votes: the kNN weight cannot act


def test_unknown_labels_bound_precision_instead_of_counting_as_negative():
    counts = oe._slate_counts([1, 2, 3, 4], {1: "relevant", 2: "strict_negative"})
    assert counts["unknown"] == 2 and counts["precision_bounds"] == [0.25, 0.75]
    assert counts["utility_per_slot_bounds"] == [0.25, (1 + 2 * oe.MAX_GAIN) / 4]
    design = oe._Design(["r"], {("r", 1): 0, ("r", 2): 1, ("r", 3): 2}, 3)
    run_w, fam_w = np.ones((1, 1)), np.ones((1, 3))
    positive = lambda c: 1.0 if c in oe.POSITIVE else 0.0  # noqa: E731
    lo, hi = oe._delta_bounds(design, run_w, fam_w, {"r": [1, 2]}, {"r": [1, 3]},
                              {"r": {1: "relevant"}}, positive, 1.0)
    assert (lo[0], hi[0]) == (-0.5, 0.5)  # swap of two unknowns: sign unidentified
    lo, hi = oe._delta_bounds(design, run_w, fam_w, {"r": [1, 2]}, {"r": [1, 3]},
                              {"r": {1: "relevant", 3: "relevant"}}, positive, 1.0)
    assert (lo[0], hi[0]) == (0.0, 0.5)
    lo, hi = oe._delta_bounds(design, run_w, fam_w, {"r": [1, 2]}, {"r": [1, 2]},
                              {"r": {}}, positive, 1.0)
    assert (lo[0], hi[0]) == (0.0, 0.0)  # shared unknowns cancel exactly


def _comparison(prec, delta_ci, pairs=50, runs=7, parity=0):
    return {"n_runs": runs, "parity_failures": parity, "precision_delta_interval": prec,
            "contrasts": {oe.PRIMARY_CONTRAST: {"pairs": pairs, "delta": {"ci95": delta_ci}}}}


@pytest.mark.parametrize(("prec", "auc", "margins", "expected"), [
    ([0.01, 0.2], [0.01, 0.1], MARGINS, "better"),
    ([0.0, 0.0], [0.0, 0.0], MARGINS, "non_inferior"),
    ([-0.02, 0.1], [0.0, 0.05], MARGINS, "insufficient_evidence"),
    ([-0.02, 0.1], [0.0, 0.05], {"precision": 0.05, "auc": 0.0}, "non_inferior"),
    ([-0.3, -0.1], [0.0, 0.1], MARGINS, "worse"),
    ([0.0, 0.1], [-0.2, -0.05], MARGINS, "worse"),
])
def test_decision_uses_explicit_margins(prec, auc, margins, expected):
    result = oe.decide(_comparison(prec, auc), margins=margins, min_runs=3, min_strict_pairs=10)
    assert result["decision"] == expected and result["rule"]["margins"]["precision"] == margins["precision"]


def test_decision_requires_margins_and_evidence():
    with pytest.raises(TypeError):
        oe.decide(_comparison([0, 0], [0, 0]), min_runs=1, min_strict_pairs=1)
    with pytest.raises(oe.OfflineEvalError):
        oe.decide(_comparison([0, 0], [0, 0]), margins={"precision": 0.0}, min_runs=1, min_strict_pairs=1)
    for comparison in (_comparison([0.1, 0.2], [0.1, 0.2], runs=2),
                       _comparison([0.1, 0.2], [0.1, 0.2], pairs=3),
                       _comparison([0.1, 0.2], [0.1, 0.2], parity=1),
                       _comparison([None, None], [0.1, 0.2])):
        assert oe.decide(comparison, margins=MARGINS, min_runs=3,
                         min_strict_pairs=10)["decision"] == "insufficient_evidence"
    assert oe.combine_decisions({"veteran": {"decision": "better"},
                                 "first_shot_proxy": {"decision": "insufficient_evidence"}}) \
        == "insufficient_evidence"


def test_config_rejects_missing_settings_knob_and_env_cannot_leak(monkeypatch):
    with pytest.raises(oe.OfflineEvalError, match="no existing settings knob"):
        oe.EvalConfig.parse({"name": "venue_cap", "settings": {"venue_ordering_cap": 0.05}})
    monkeypatch.setenv("MIN_TOPIC_RELEVANCE", "0.99")
    settings = oe._frozen_settings(_snapshot(), oe.EvalConfig.parse(CONFIGS[0]), "hybrid_knn")
    assert settings.min_topic_relevance == 0.0


def test_boundary_blocks_network_and_current_state_then_restores(tmp_path):
    from dailydigest import config, store
    from dailydigest import votes as votes_mod

    snapshot = _snapshot()
    settings = oe._frozen_settings(snapshot, oe.EvalConfig.parse(CONFIGS[0]), "hybrid_knn")
    connection = oe._connect(_database(tmp_path / "f.db", snapshot))
    embeddings = oe.FrozenEmbeddings(connection)
    original = votes_mod.session_scope
    with oe.serving_boundary(settings=settings, clock=oe._time(DECISION), embeddings=embeddings,
                             exemplars=None, signed_votes=0, viewed_history={}):
        assert config.get_settings() is settings
        with pytest.raises(oe.IncompleteReplay):
            socket.create_connection(("example.org", 443))
        with pytest.raises(oe.IncompleteReplay):
            votes_mod.session_scope()
        with pytest.raises(oe.IncompleteReplay):
            store.init_db()
    assert votes_mod.session_scope is original
    connection.close()


def test_missing_frozen_vector_is_incomplete_not_silently_replaced(tmp_path):
    snapshot = _snapshot()
    path = _database(tmp_path / "g.db", snapshot)
    connection = sqlite3.connect(path)
    connection.execute("DELETE FROM item_embeddings WHERE item_id = 5")
    connection.commit()
    connection.close()
    report = oe.evaluate(path, CONFIGS, baseline="current", margins=MARGINS, n_boot=10,
                         stages=("veteran",), min_runs=1, min_strict_pairs=1)
    assert "absent from the frozen cache" in report["runs"][0]["slates"]["veteran"]["current"]
    assert report["analyses"]["all"]["veteran"]["status"] == "no_rescorable_runs"
    assert report["analyses"]["all"]["overall_decisions"]["knn_065"] == "insufficient_evidence"


def test_cli_refuses_data_directory_and_existing_output(tmp_path, tmp_path_factory):
    from pathlib import Path

    snapshot = _snapshot()
    path = _database(tmp_path / "h.db", snapshot)
    configs = tmp_path / "configs.json"
    configs.write_text(json.dumps(CONFIGS))
    repo_data = Path(oe.__file__).resolve().parents[3] / "data" / "offline-eval.json"
    with pytest.raises(oe.OfflineEvalError):
        oe.run_cli(database=path, configs_path=configs, baseline="current", margin_precision=0.0,
                   margin_auc=0.0, output=repo_data, n_boot=10, min_runs=1, min_strict_pairs=1)
    with pytest.raises(oe.OfflineEvalError):  # never beside the evaluated database
        oe.run_cli(database=path, configs_path=configs, baseline="current", margin_precision=0.0,
                   margin_auc=0.0, output=tmp_path / "beside.json", n_boot=10, min_runs=1,
                   min_strict_pairs=1)
    output = tmp_path_factory.mktemp("reports") / "out" / "report.json"
    oe.run_cli(database=path, configs_path=configs, baseline="current", margin_precision=0.0,
               margin_auc=0.0, output=output, n_boot=10, min_runs=1, min_strict_pairs=1)
    assert json.loads(output.read_text())["margins"] == MARGINS
    with pytest.raises(FileExistsError):
        oe.run_cli(database=path, configs_path=configs, baseline="current", margin_precision=0.0,
                   margin_auc=0.0, output=output, n_boot=10, min_runs=1, min_strict_pairs=1)


def test_dd_eval_offline_subcommand_forwards_arguments(tmp_path, tmp_path_factory):
    from typer.testing import CliRunner

    from dailydigest.cli import app

    path = _database(tmp_path / "i.db", _snapshot())
    configs = tmp_path / "configs.json"
    configs.write_text(json.dumps(CONFIGS))
    output = tmp_path_factory.mktemp("cli") / "report.json"
    result = CliRunner().invoke(app, [
        "eval-offline", "--database", str(path), "--configs", str(configs), "--baseline", "current",
        "--margin-precision", "0", "--margin-auc", "0", "--output", str(output), "--bootstrap", "10",
        "--min-runs", "1", "--min-strict-pairs", "1"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output.strip().splitlines()[-1])["knn_065"] in oe.DECISIONS
    assert json.loads(output.read_text())["defaults_changed"] is False


def _unverifiable(snapshot):
    """Static row weights no longer match the current profile rules (2 static + 1 learned)."""
    snapshot["profile_matrix"][0] = (0.7 * _unit([1, 0, 0, 0])).tolist()
    return snapshot


def test_first_shot_with_unverifiable_static_rows_is_marked_not_dropped():
    frozen = oe.freeze_run(_unverifiable(_snapshot()), stage="first_shot_proxy")
    assert frozen.freeze_notes["static_profile"] == "static_rows_unverified"
    assert oe.FIRST_SHOT_UNVERIFIED in frozen.retrospective_inputs
    assert frozen.profile_matrix.shape == (3, DIM)  # learned row may still be present


def test_unverified_first_shot_runs_never_enter_a_decision(tmp_path):
    path = _database(tmp_path / "u.db", _unverifiable(_snapshot()))
    report = oe.evaluate(path, CONFIGS, baseline="current", margins=MARGINS, n_boot=20,
                         min_runs=1, min_strict_pairs=1)
    first = report["analyses"]["all"]["first_shot_proxy"]
    assert first["status"] == "static_profile_unverified"
    assert first["unverified_runs"] == ["run-1"]
    assert first["decisions"]["knn_065"]["decision"] == "insufficient_evidence"
    assert "first_shot_proxy" in report["runs"][0]["slates"]  # kept as a diagnostic


def test_recorded_static_row_count_verifies_first_shot():
    snapshot = _unverifiable(_snapshot())
    snapshot["profile_static_rows"] = 2
    frozen = oe.freeze_run(snapshot, stage="first_shot_proxy")
    assert frozen.freeze_notes["static_profile"] == "static_rows_recorded"
    assert frozen.profile_matrix.shape == (2, DIM)
    assert oe.FIRST_SHOT_UNVERIFIED not in frozen.retrospective_inputs


@pytest.mark.parametrize("bad", [0, 4, True, "2", None])
def test_invalid_recorded_static_row_count_falls_back_to_weights(bad):
    snapshot = _snapshot()
    snapshot["profile_static_rows"] = bad
    frozen = oe.freeze_run(snapshot, stage="first_shot_proxy")
    assert frozen.freeze_notes["static_profile"] == "static_rows_verified_by_profile_weights"
