"""Tests for unbiased pairwise training-pair sampling."""

from __future__ import annotations

from dailydigest.votes import _sample_training_pairs


def test_returns_all_pairs_when_under_cap():
    pairs = _sample_training_pairs([0, 1], [2, 3], max_pairs=300)
    assert sorted(pairs) == [(0, 2), (0, 3), (1, 2), (1, 3)]


def test_caps_pair_count():
    ups = list(range(40))
    downs = list(range(40, 80))
    pairs = _sample_training_pairs(ups, downs, max_pairs=300)
    assert len(pairs) == 300


def test_sampling_is_not_biased_to_first_positives():
    # 40 ups × 40 downs = 1600 pairs, capped at 300. The OLD fixed-order code
    # would only ever use ups[0:8]; uniform sampling must cover far more.
    ups = list(range(40))
    downs = list(range(40, 80))
    pairs = _sample_training_pairs(ups, downs, max_pairs=300)
    used_ups = {u for u, _ in pairs}
    # Expect broad coverage of the up-voted items, not just the first handful.
    assert len(used_ups) >= 30
    assert max(used_ups) >= 30  # late-voted positives participate


def test_sampling_is_deterministic():
    ups = list(range(40))
    downs = list(range(40, 80))
    assert _sample_training_pairs(ups, downs) == _sample_training_pairs(ups, downs)
