"""Tests for Platt score calibration and the adaptive relevance floor."""

from __future__ import annotations

from datetime import datetime, timezone

from dailydigest import store as store_mod
from dailydigest import votes as votes_mod
from dailydigest.rank import calibrate as calib_mod
from dailydigest.rank.calibrate import (
    adaptive_relevance_floor,
    calibrated_probability,
    fit_calibrator,
    load_calibrator,
)


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
    """High-scored items upvoted, low-scored items downvoted."""
    feats = []
    actions = []
    for i in range(n_each):
        up = _insert_item(f"up{i}")
        feats.append((f"R{i}", up, 0.80, {}))
        actions.append((up, 1))
        down = _insert_item(f"down{i}")
        feats.append((f"D{i}", down, 0.30, {}))
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
