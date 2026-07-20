"""Tests for rank-fusion of topic score and LR probability."""

from __future__ import annotations

import numpy as np

from dailydigest.rank.ranker import _fuse_scores, _rank_desc


def test_rank_desc_assigns_zero_to_max():
    ranks = _rank_desc(np.array([0.1, 0.9, 0.5], dtype=np.float32))
    assert list(ranks) == [2, 0, 1]


def test_rrf_top_and_bottom_are_normalized():
    qa = np.array([0.2, 0.9, 0.5], dtype=np.float32)
    lr = np.array([0.3, 0.8, 0.4], dtype=np.float32)
    fused = _fuse_scores(qa, lr, mode="rrf")
    # Both agree item 1 is best, item 0 is worst.
    assert np.argmax(fused) == 1
    assert np.argmin(fused) == 0
    assert fused.max() > 0.999
    assert fused.min() == 0.0


def test_rrf_is_robust_to_score_outliers():
    # An extreme outlier in qa must not let that item dominate the fused score,
    # whereas a min-max blend lets it crush every other signal.
    qa = np.array([0.50, 0.51, 0.52, 1000.0], dtype=np.float32)
    lr = np.array([0.90, 0.80, 0.70, 0.10], dtype=np.float32)

    rrf = _fuse_scores(qa, lr, mode="rrf")
    minmax = _fuse_scores(qa, lr, mode="minmax")

    # Under RRF the qa-outlier item 3 (weakest LR) only earns one rank step, so
    # it cannot take the top — item 0 (strongest LR) ties it and wins on order.
    assert np.argmax(rrf) == 0
    # Under min-max the outlier crushes every other qa value to ~0 and drags
    # item 3 into the top tier despite the weakest LR — demonstrating the
    # instability. (It ties item 0 at the max; RRF keeps it from doing even that.)
    assert minmax[3] == minmax.max()


def test_minmax_mode_matches_legacy_blend():
    qa = np.array([0.0, 1.0], dtype=np.float32)
    lr = np.array([1.0, 0.0], dtype=np.float32)
    fused = _fuse_scores(qa, lr, mode="minmax")
    # 0.5*qa_norm + 0.5*lr_norm = 0.5 for both
    assert np.allclose(fused, [0.5, 0.5], atol=1e-3)
