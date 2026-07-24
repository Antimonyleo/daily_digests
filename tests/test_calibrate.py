"""Tests for Platt score calibration and the adaptive relevance floor."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from dailydigest import store as store_mod
from dailydigest import votes as votes_mod
from dailydigest.rank import calibrate as calib_mod
from dailydigest.rank.calibrate import (
    adaptive_relevance_floor,
    calibrated_probability,
    fit_calibrator,
    load_calibrator,
)
from dailydigest.rank.source_quality import RANKER_VERSION


def _insert_item(title: str) -> int:
    store_mod.init_db()
    with store_mod.session_scope() as s:
        row = store_mod.ItemRow(
            source="Test",
            section="research",
            external_id=title,
            url=f"https://example.com/{title}",
            title=title,
            abstract="abstract",
            published_at=datetime.now(timezone.utc),
        )
        s.add(row)
        s.flush()
        return int(row.id)


def _seed_votes_correlated_with_score(n_each: int = 10) -> None:
    """High-scored items upvoted, low-scored items downvoted.

    Feature rows are stamped with the current RANKER_VERSION so they pass the
    same-policy filter in ``_calibration_dataset`` and the calibrator can fit.
    """
    feats = []
    actions = []
    for i in range(n_each):
        up = _insert_item(f"up{i}")
        feats.append((f"R{i}", up, 0.80, {"ranker_version": RANKER_VERSION}))
        actions.append((up, 1))
        down = _insert_item(f"down{i}")
        feats.append((f"D{i}", down, 0.30, {"ranker_version": RANKER_VERSION}))
        actions.append((down, -1))
    store_mod.write_digest_features("2026-01-01", feats)
    for item_id, value in actions:
        votes_mod.record_vote_by_id(item_id, value)


def test_no_calibrator_returns_default_floor():
    assert load_calibrator() is None
    assert adaptive_relevance_floor(0.58) == 0.58
    assert calibrated_probability(0.7) is None


def test_fit_requires_enough_votes():
    # Only 4 votes → below MIN_VOTES_FOR_CALIBRATION.
    feats = []
    for i in range(2):
        up = _insert_item(f"u{i}")
        down = _insert_item(f"d{i}")
        feats.append((f"R{i}", up, 0.8, {}))
        feats.append((f"D{i}", down, 0.3, {}))
    store_mod.write_digest_features("2026-01-02", feats)
    assert fit_calibrator() is None


def test_fit_and_apply_calibrator():
    _seed_votes_correlated_with_score(10)
    params = fit_calibrator()
    assert params is not None
    assert params["n"] == 20
    assert params["a"] > 0  # higher score → higher P(relevant)

    # Probability is monotonic in score.
    p_hi = calibrated_probability(0.80)
    p_lo = calibrated_probability(0.30)
    assert p_hi is not None and p_lo is not None
    assert p_hi > 0.5 > p_lo

    # Adaptive floor lands between the up/down score bands and is clamped near
    # the configured default.
    floor = adaptive_relevance_floor(0.58)
    assert 0.48 <= floor <= 0.78


def test_load_round_trip():
    _seed_votes_correlated_with_score(10)
    fit_calibrator()
    loaded = load_calibrator()
    assert loaded is not None and "a" in loaded and "b" in loaded


def test_inverted_calibrator_falls_back_to_default():
    # a <= 0 means the score is non-informative/inverted → keep the default.
    assert adaptive_relevance_floor(0.58, calib={"a": -1.0, "b": 0.0}) == 0.58
    assert adaptive_relevance_floor(0.58, calib={"a": 0.0, "b": 0.0}) == 0.58


def test_fit_stamps_current_schema_and_load_accepts_it():
    _seed_votes_correlated_with_score(10)
    params = fit_calibrator()
    assert params is not None
    assert params["schema"] == votes_mod.LR_FEATURE_SCHEMA_VERSION
    # The policy identity is stamped alongside the schema.
    assert params["policy"] == RANKER_VERSION
    # A same-schema, same-policy fit round-trips through the loader.
    loaded = load_calibrator()
    assert loaded is not None
    assert loaded["schema"] == votes_mod.LR_FEATURE_SCHEMA_VERSION
    assert loaded["policy"] == RANKER_VERSION


def test_load_invalidates_stale_or_missing_policy():
    """A calibrator whose stored policy != current RANKER_VERSION (or is missing)
    is treated as absent, even when its feature schema still matches — scores
    depend on the ranking policy too."""
    import json

    _seed_votes_correlated_with_score(10)
    assert fit_calibrator() is not None
    path = calib_mod._calibrator_path()

    # Correct schema but WRONG policy → loader returns None.
    data = json.loads(path.read_text())
    assert data["schema"] == votes_mod.LR_FEATURE_SCHEMA_VERSION  # schema is fine
    data["policy"] = "2025-01-01-old-policy-v0"
    path.write_text(json.dumps(data))
    assert load_calibrator() is None
    # Downstream consumers (which load internally) also degrade safely.
    assert calibrated_probability(0.7) is None
    assert adaptive_relevance_floor(0.58) == 0.58

    # Missing policy (pre-policy-versioning calibrator) → also invalidated.
    data.pop("policy", None)
    path.write_text(json.dumps(data))
    assert load_calibrator() is None

    # Re-fitting under the current policy restores a usable calibrator.
    assert fit_calibrator() is not None
    assert load_calibrator() is not None


def test_load_invalidates_stale_or_missing_schema():
    """A calibrator whose stored schema != current (or is missing) is treated as
    absent so it refits rather than applying contaminated params."""
    import json

    _seed_votes_correlated_with_score(10)
    assert fit_calibrator() is not None
    path = calib_mod._calibrator_path()

    # Stale schema → loader returns None.
    data = json.loads(path.read_text())
    data["schema"] = "some_old_schema_v0"
    path.write_text(json.dumps(data))
    assert load_calibrator() is None
    # calibrated_probability (which loads internally) also degrades to None.
    assert calibrated_probability(0.7) is None
    # adaptive floor falls back to the default when the calibrator is invalidated.
    assert adaptive_relevance_floor(0.58) == 0.58

    # Missing schema (pre-versioning calibrator) → also invalidated.
    data.pop("schema", None)
    path.write_text(json.dumps(data))
    assert load_calibrator() is None

    # Re-fitting under the current schema restores a usable calibrator.
    assert fit_calibrator() is not None
    assert load_calibrator() is not None


def test_multi_digest_score_selection_is_deterministic_latest_digest():
    """When an item appears in several digests, the calibration set uses the
    LATEST digest's score (by created_at), deterministically."""
    from datetime import timedelta

    item_id = _insert_item("multi")
    other = _insert_item("other")

    # Two digests for the same item with different final_scores; the later digest
    # carries the newer (correct) score. Write the NEWER row first to prove the
    # selection is by created_at, not by insertion/write order.
    base = datetime.now(timezone.utc)
    store_mod.init_db()
    with store_mod.session_scope() as s:
        # Parent digests (FK target for digest_item_features.digest_id).
        s.add(store_mod.DigestRow(id="d-old", item_count=1, created_at=base))
        s.add(store_mod.DigestRow(
            id="d-new", item_count=2, created_at=base + timedelta(hours=1)
        ))
        s.flush()
        s.add(store_mod.DigestItemFeatureRow(
            digest_id="d-new", item_id=item_id, item_label="R1",
            final_score=0.90, features_json='{"ranker_version": "%s"}' % RANKER_VERSION,
            created_at=base + timedelta(hours=1),
        ))
        s.add(store_mod.DigestItemFeatureRow(
            digest_id="d-old", item_id=item_id, item_label="R1",
            final_score=0.10, features_json='{"ranker_version": "%s"}' % RANKER_VERSION,
            created_at=base,
        ))
        s.add(store_mod.DigestItemFeatureRow(
            digest_id="d-new", item_id=other, item_label="R2",
            final_score=0.20, features_json='{"ranker_version": "%s"}' % RANKER_VERSION,
            created_at=base + timedelta(hours=1),
        ))

    # Sign votes so both items enter the calibration dataset (needs +/- labels).
    votes_mod.record_vote_by_id(item_id, 1)
    votes_mod.record_vote_by_id(other, -1)

    scores, labels = calib_mod._calibration_dataset()
    by_label = {int(lab): float(sc) for sc, lab in zip(scores, labels)}
    # The upvoted multi item (label 1) must carry the latest digest's score 0.90,
    # not the older 0.10 — proving deterministic latest-digest selection.
    assert by_label[1] == pytest.approx(0.90, abs=1e-5)
    assert by_label[0] == pytest.approx(0.20, abs=1e-5)


def test_calibration_dataset_excludes_other_policy_rows():
    """Only feature rows stamped with the current RANKER_VERSION enter the
    calibration set; rows from an older ranking policy are excluded."""
    cur_up = _insert_item("cur_up")
    cur_down = _insert_item("cur_down")
    old_up = _insert_item("old_up")
    old_down = _insert_item("old_down")

    feats = [
        ("R1", cur_up, 0.85, {"ranker_version": RANKER_VERSION}),
        ("R2", cur_down, 0.25, {"ranker_version": RANKER_VERSION}),
        # Old-policy rows carry a different ranker_version and must be dropped.
        ("R3", old_up, 0.85, {"ranker_version": "2025-01-01-old-policy-v0"}),
        ("R4", old_down, 0.25, {"ranker_version": "2025-01-01-old-policy-v0"}),
    ]
    store_mod.write_digest_features("2026-01-03", feats)
    for item_id in (cur_up, old_up):
        votes_mod.record_vote_by_id(item_id, 1)
    for item_id in (cur_down, old_down):
        votes_mod.record_vote_by_id(item_id, -1)

    scores, labels = calib_mod._calibration_dataset()
    # Exactly the two current-policy rows survive; the two old-policy rows do not.
    assert len(scores) == 2
    assert sorted(float(s) for s in scores) == pytest.approx([0.25, 0.85])


def test_fit_skips_when_only_a_few_current_policy_rows():
    """With a MIX of policies, the calibrator fits ONLY from current-policy rows.
    When too few of those exist (below MIN_VOTES_FOR_CALIBRATION), it returns
    not-enough-data so the loader falls back to the safe default instead of
    fitting on cross-policy-contaminated snapshots."""
    feats = []
    # Only 2 current-policy voted rows — below MIN_VOTES_FOR_CALIBRATION (12).
    cur_up = _insert_item("only_cur_up")
    cur_down = _insert_item("only_cur_down")
    feats.append(("R1", cur_up, 0.85, {"ranker_version": RANKER_VERSION}))
    feats.append(("R2", cur_down, 0.25, {"ranker_version": RANKER_VERSION}))
    # Plenty of old-policy rows that would clear the threshold IF they counted —
    # they must be excluded, so the fit is still refused.
    old_actions = []
    for i in range(10):
        up = _insert_item(f"old_up{i}")
        down = _insert_item(f"old_down{i}")
        feats.append((f"OU{i}", up, 0.85, {"ranker_version": "2025-01-01-old-policy-v0"}))
        feats.append((f"OD{i}", down, 0.25, {"ranker_version": "2025-01-01-old-policy-v0"}))
        old_actions.append((up, 1))
        old_actions.append((down, -1))
    store_mod.write_digest_features("2026-01-04", feats)
    votes_mod.record_vote_by_id(cur_up, 1)
    votes_mod.record_vote_by_id(cur_down, -1)
    for item_id, value in old_actions:
        votes_mod.record_vote_by_id(item_id, value)

    # 22 total voted rows, but only 2 same-policy → below threshold → no fit.
    assert fit_calibrator() is None
    assert load_calibrator() is None
    assert adaptive_relevance_floor(0.58) == 0.58
