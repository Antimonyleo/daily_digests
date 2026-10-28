"""R0.1: a stale learned profile is rejected with a reason and recovered visibly.

The learned profile is derived from votes. Reading it never repairs it; a
user-started brew rebuilds a repairable file once, keeps one ``.bak``, and
records what the ranking actually used in the ``learned_profile`` audit.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dailydigest import pipeline, store, votes
from dailydigest.rank import profile as profile_mod

STATIC = np.eye(3, dtype=np.float32)[:2]


def _path() -> Path:
    return votes._learned_profile_path()


def _items(n: int = 2) -> list[int]:
    store.init_db()
    with store.session_scope() as session:
        rows = [store.ItemRow(source="Nature", section="research", external_id=f"r{i}",
                              url=f"https://example.test/{i}", title=f"Methods {i}",
                              abstract="A substantive research methods abstract.")
                for i in range(n)]
        session.add_all(rows)
        session.flush()
        return [row.id for row in rows]


@pytest.fixture
def voted(monkeypatch):
    """Two positive votes with a current learned profile on disk."""
    monkeypatch.setattr(votes, "embed_item_rows", lambda rows: np.tile([1., 0., 0.], (len(rows), 1)))
    ids = _items()
    for iid in ids:
        assert votes.record_vote_by_id(iid, 1)
    assert profile_mod.inspect_learned_profile().usable
    return ids


def _write(path: Path, **overrides) -> None:
    values = {
        "profile": np.array([0., 1., 0.], dtype=np.float32),
        "vote_count": np.array([2], dtype=np.int32),
        "feedback_semantics_version": np.array([votes.FEEDBACK_SEMANTICS_VERSION]),
        "feedback_reasons_signature": np.array([votes.feedback_reasons_signature()]),
        "feedback_votes_signature": np.array([votes.feedback_votes_signature()]),
    }
    for key, value in overrides.items():
        if value is None:
            values.pop(key, None)
        else:
            values[key] = value
    np.savez(path, **values)


def _write_v1_like_production(path: Path) -> None:
    """The shape of the real stale file: v1 stamp, no votes signature."""
    _write(path, feedback_semantics_version=np.array(["reason-aware-v1"]), feedback_votes_signature=None)


def _no_temp_files(directory: Path) -> bool:
    return not [p for p in directory.iterdir() if ".tmp" in p.name]


# --- Reasons ---------------------------------------------------------------- #


@pytest.mark.parametrize(("mutate", "reason"), [
    (lambda p: p.unlink(), "missing"),
    (_write_v1_like_production, "stale_semantics_version"),
    (lambda p: _write(p, feedback_semantics_version=None), "stale_semantics_version"),
    (lambda p: _write(p, feedback_reasons_signature=np.array(["0" * 64])), "stale_reasons_signature"),
    (lambda p: _write(p, feedback_votes_signature=None), "stale_votes_signature"),
    (lambda p: p.write_bytes(b"not a zip archive"), "corrupt"),
    (lambda p: _write(p, profile=None), "corrupt"),
    (lambda p: _write(p, profile=np.array([np.nan, 0., 0.])), "corrupt"),
    (lambda p: _write(p, embedding_model=np.array(["other/model"])), "incompatible_embedding"),
    (lambda p: _write(p, vote_count=np.array([0])), "no_topic_votes"),
])
def test_each_rejection_has_an_explicit_reason_and_is_read_only(voted, monkeypatch, mutate, reason):
    path = _path()
    mutate(path)
    before = path.read_bytes() if path.exists() else None
    state = profile_mod.inspect_learned_profile()
    assert state.reason == reason
    assert not state.usable and state.vector is None
    assert (path.read_bytes() if path.exists() else None) == before
    # The reader falls back to the static matrix and reports why.
    used: dict = {}
    monkeypatch.setattr(profile_mod, "build_profile_matrix", lambda _p: STATIC.copy())
    result = profile_mod.build_profile_matrix_with_rocchio(SimpleNamespace(), 2, state_out=used)
    np.testing.assert_array_equal(result, STATIC)
    assert used["reason"] == reason
    assert used["static_rows"] == len(STATIC)  # recorded for first-use offline replay
    assert (path.read_bytes() if path.exists() else None) == before


def test_dimension_mismatch_is_incompatible_embedding(voted):
    _write(_path(), profile=np.array([1., 0.], dtype=np.float32))
    assert profile_mod.inspect_learned_profile().usable  # no expected dimension known
    assert profile_mod.inspect_learned_profile(expected_dim=3).reason == "incompatible_embedding"


def test_current_file_is_usable_and_records_the_embedding_model(voted):
    state = profile_mod.inspect_learned_profile(expected_dim=3)
    assert state.usable and state.vote_count == 2
    assert state.embedding_model == profile_mod.current_embedding_model()


def test_unreadable_feedback_is_not_misreported_as_a_corrupt_profile(voted):
    votes._vote_reasons_path().write_text("{not json")
    assert profile_mod.inspect_learned_profile().reason == "feedback_unreadable"


# --- Rebuild result ------------------------------------------------------------ #


def test_rebuild_reports_verified_write_and_keeps_one_backup(voted):
    path = _path()
    _write_v1_like_production(path)
    stale = path.read_bytes()
    result = votes._rebuild_rocchio_profile(keep_backup=True)
    assert result.written and result.failure == ""
    assert result.topic_votes == 2 and result.vectors_total == 2
    assert result.vectors_missing == 2  # fixture encoder bypasses the SQLite cache
    assert result.duration_ms >= 0
    assert Path(result.backup_path).read_bytes() == stale
    assert profile_mod.inspect_learned_profile(expected_dim=3).usable
    # A second recovery replaces, never accumulates, the single backup.
    _write_v1_like_production(path)
    assert votes._rebuild_rocchio_profile(keep_backup=True).written
    backups = [p.name for p in path.parent.iterdir() if p.name.startswith("learned_profile.npz.")]
    assert backups == ["learned_profile.npz.bak"]
    assert _no_temp_files(path.parent)


def test_vote_change_rebuild_does_not_create_backups(voted):
    assert votes._rebuild_rocchio_profile().written
    assert not votes._learned_profile_backup_path(_path()).exists()


@pytest.mark.parametrize("failure", ["encoder", "write", "interrupt", "short"])
def test_failed_or_interrupted_rebuild_leaves_old_file_and_no_partial(voted, monkeypatch, failure):
    path = _path()
    _write_v1_like_production(path)
    before = path.read_bytes(), path.stat().st_mtime_ns

    def broken(*_a, **_k):
        if failure == "interrupt":
            raise KeyboardInterrupt
        raise OSError("fixture failure")

    if failure == "encoder":
        monkeypatch.setattr(votes, "embed_item_rows", broken)
    elif failure == "short":
        monkeypatch.setattr(votes, "embed_item_rows", lambda rows: np.ones((len(rows) - 1, 3)))
    else:
        monkeypatch.setattr(np, "savez", broken)
    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            votes._rebuild_rocchio_profile(keep_backup=True)
    else:
        result = votes._rebuild_rocchio_profile(keep_backup=True)
        assert result.status == "failed"
        assert result.failure == {"encoder": "encoder_unavailable", "write": "write_failed",
                                  "short": "missing_vectors"}[failure]
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert not votes._learned_profile_backup_path(path).exists()
    assert _no_temp_files(path.parent)


def test_verification_failure_is_not_a_success(voted, monkeypatch):
    path = _path()
    _write_v1_like_production(path)
    before = path.read_bytes()
    real = votes.feedback_votes_signature
    calls = {"n": 0}

    def drifting(records=None):
        # The rebuild signs its rows; verification then sees a different state,
        # as if a vote landed mid-rebuild.
        if records is None:
            calls["n"] += 1
            return "changed-concurrently"
        return real(records)

    monkeypatch.setattr(votes, "feedback_votes_signature", drifting)
    result = votes._rebuild_rocchio_profile(keep_backup=True)
    assert result.status == "failed" and result.failure == "verification_failed"
    assert path.read_bytes() == before
    assert _no_temp_files(path.parent)


def test_only_practical_ratings_is_static_and_retires_stale_file_to_backup(monkeypatch):
    monkeypatch.setattr(votes, "embed_item_rows", lambda rows: np.tile([1., 0., 0.], (len(rows), 1)))
    monkeypatch.setattr(votes, "_update_rocchio", lambda *a, **k: None)
    (iid,) = _items(1)
    votes.record_vote_by_id(iid, -1)
    votes.record_vote_reason(iid, "already_known")
    _write_v1_like_production(_path())
    result = votes._rebuild_rocchio_profile(keep_backup=True)
    assert result.status == "no_topic_votes"
    assert not _path().exists()
    assert Path(result.backup_path).exists()


# --- Brew-time recovery ------------------------------------------------------- #


@pytest.fixture
def brew(monkeypatch):
    digest_id = "2026-10-26"
    monkeypatch.setattr(pipeline, "_digest_id", lambda: digest_id)
    monkeypatch.setattr(pipeline, "ingest_all", lambda **kwargs: 0)
    monkeypatch.setattr(pipeline, "load_profile", lambda: SimpleNamespace(bio="Reader", keywords=[], downweight=[]))
    monkeypatch.setattr(profile_mod, "build_profile_matrix", lambda _p: np.eye(3, dtype=np.float32)[:2].copy())
    monkeypatch.setattr(pipeline, "build_profile_matrix", lambda _p: np.eye(3, dtype=np.float32)[:2].copy())
    monkeypatch.setattr(pipeline, "_build_neg_centroid", None)
    monkeypatch.setattr(pipeline, "recent_items", lambda **kwargs: [])
    monkeypatch.setattr(pipeline, "score_items", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline, "pick_top_per_section", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline, "summarize_items", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline, "render_digest", lambda *args, **kwargs: "<html>ready</html>")
    monkeypatch.setattr(pipeline, "send_digest", lambda *args, **kwargs: False)
    from dailydigest.rank import calibrate
    monkeypatch.setattr(calibrate, "fit_calibrator", lambda: None)
    monkeypatch.setattr(profile_mod, "build_attribution_context", lambda profile: None)
    real_builder = pipeline._build_profile_with_rocchio
    matrices: list[np.ndarray] = []

    def spy(*args, **kwargs):
        out = real_builder(*args, **kwargs)
        matrices.append(out)
        return out

    monkeypatch.setattr(pipeline, "_build_profile_with_rocchio", spy)
    rebuilds: list[object] = []
    real_rebuild = votes._rebuild_rocchio_profile

    def counting(**kwargs):
        result = real_rebuild(**kwargs)
        rebuilds.append(result)
        return result

    monkeypatch.setattr(votes, "_rebuild_rocchio_profile", counting)
    events: list[tuple[str, dict]] = []

    def run():
        matrices.clear()
        pipeline.run_all(dry_run=True, progress_callback=lambda s, p: events.append((s, p)))
        (audit,) = store.load_digest_audit(digest_id, "learned_profile")
        return audit, matrices[-1]

    return SimpleNamespace(run=run, rebuilds=rebuilds, events=events, digest_id=digest_id)


def test_brew_rebuilds_stale_profile_once_then_reuses_it(voted, brew):
    path = _path()
    _write_v1_like_production(path)
    stale = path.read_bytes()
    feedback = store.feedback_state()
    audit, matrix = brew.run()
    assert audit["status"] == "rebuilt"
    assert audit["reason"] == "stale_semantics_version"
    assert "rebuilt" in audit["message"]
    assert audit["topic_votes"] == 2 and audit["vectors_total"] == 2
    assert {"duration_ms", "vectors_missing"} <= audit.keys()
    assert len(brew.rebuilds) == 1
    assert votes._learned_profile_backup_path(path).read_bytes() == stale
    # The scorer receives the rebuilt learned row, not the static matrix.
    assert matrix.shape == (3, 3) and matrix[2, 0] > 0
    assert ("stage_start", {"name": "learned_profile"}) in brew.events
    # Second brew: current file, no rebuild, nothing rewritten, votes untouched.
    mtime = path.stat().st_mtime_ns
    audit, matrix = brew.run()
    assert audit["status"] == "ok" and audit["reason"] == "current"
    assert len(brew.rebuilds) == 1
    assert path.stat().st_mtime_ns == mtime
    assert matrix.shape == (3, 3)
    assert store.feedback_state() == feedback


@pytest.mark.parametrize(("mutate", "reason"), [
    (lambda p: p.unlink(), "missing"),
    (lambda p: _write(p, feedback_votes_signature=None), "stale_votes_signature"),
    (lambda p: _write(p, feedback_reasons_signature=np.array(["0" * 64])), "stale_reasons_signature"),
    (lambda p: p.write_bytes(b"garbage"), "corrupt"),
    (lambda p: _write(p, embedding_model=np.array(["other/model"])), "incompatible_embedding"),
])
def test_brew_recovers_each_repairable_reason(voted, brew, mutate, reason):
    mutate(_path())
    audit, matrix = brew.run()
    assert (audit["status"], audit["reason"]) == ("rebuilt", reason)
    assert len(brew.rebuilds) == 1
    assert matrix.shape == (3, 3)


def test_brew_with_current_profile_never_rebuilds_or_backs_up(voted, brew):
    audit, _matrix = brew.run()
    assert (audit["status"], audit["reason"]) == ("ok", "current")
    assert brew.rebuilds == []
    assert not votes._learned_profile_backup_path(_path()).exists()
    assert not any(p.get("name") == "learned_profile" for _s, p in brew.events)


def test_brew_without_votes_is_static_not_a_failure(brew):
    audit, matrix = brew.run()
    assert (audit["status"], audit["reason"]) == ("static", "no_topic_votes")
    assert brew.rebuilds == []
    np.testing.assert_array_equal(matrix, STATIC)


def test_failed_rebuild_still_brews_and_says_unavailable(voted, brew, monkeypatch):
    path = _path()
    _write_v1_like_production(path)
    before = path.read_bytes()

    def broken(_rows):
        raise RuntimeError("Local encoder unavailable")

    monkeypatch.setattr(votes, "embed_item_rows", broken)
    audit, matrix = brew.run()
    assert audit["status"] == "unavailable"
    assert audit["reason"] == "stale_semantics_version"
    assert audit["failure"] == "encoder_unavailable"
    assert "static topics only" in audit["message"]
    np.testing.assert_array_equal(matrix, STATIC)
    assert path.read_bytes() == before
    assert len(brew.rebuilds) == 1
    # Each brew attempts once; there is no retry loop within a brew.
    brew.run()
    assert len(brew.rebuilds) == 2


def test_claimed_success_without_a_loadable_file_is_not_reported_rebuilt(voted, brew, monkeypatch):
    _write_v1_like_production(_path())
    monkeypatch.setattr(votes, "_rebuild_rocchio_profile",
                        lambda **_k: votes.RocchioRebuildResult("written", topic_votes=2))
    audit, matrix = brew.run()
    assert audit["status"] == "unavailable"
    assert audit["reason"] == "stale_semantics_version"
    np.testing.assert_array_equal(matrix, STATIC)


def test_dimension_gap_found_at_composition_rebuilds_once(voted, brew, caplog):
    # An older current-policy file without a model stamp, from a 2-d model.
    _write(_path(), profile=np.array([1., 0.], dtype=np.float32), embedding_model=None)
    caplog.set_level("INFO", logger="dailydigest.pipeline")
    audit, loader_matrix = brew.run()
    assert (audit["status"], audit["reason"]) == ("rebuilt", "incompatible_embedding")
    assert len(brew.rebuilds) == 1
    # The loader saw the gap and returned static; the brew appended the verified row.
    np.testing.assert_array_equal(loader_matrix, STATIC)
    assert "ranker: profile_rows=3 votes=2" in caplog.text


def test_unreadable_feedback_is_unavailable_without_rebuild(voted, brew):
    # Scoring itself refuses unreadable reasons later; recovery must not mask it.
    votes._vote_reasons_path().write_text("{not json")
    recovery = pipeline._LearnedProfileRecovery(None, {})
    recovery.prepare(2)
    assert (recovery.audit["status"], recovery.audit["reason"]) == ("unavailable", "feedback_unreadable")
    assert brew.rebuilds == [] and not recovery.attempted


def test_brew_with_only_practical_ratings_stays_static_not_unavailable(brew, monkeypatch):
    # Signed votes exist, but every one carries a practical reason, so nothing
    # teaches topic preference: the rebuild writes no file and that is correct.
    monkeypatch.setattr(votes, "embed_item_rows", lambda rows: np.tile([1., 0., 0.], (len(rows), 1)))
    monkeypatch.setattr(votes, "_update_rocchio", lambda *a, **k: None)
    (iid,) = _items(1)
    votes.record_vote_by_id(iid, -1)
    votes.record_vote_reason(iid, "already_known")
    audit, matrix = brew.run()
    assert (audit["status"], audit["reason"]) == ("static", "no_topic_votes")
    assert "unavailable" not in audit["message"].lower()
    np.testing.assert_array_equal(matrix, STATIC)
