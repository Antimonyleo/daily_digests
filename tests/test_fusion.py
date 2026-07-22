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
    fused = _fuse_scores(qa, lr)
    # Both agree item 1 is best, item 0 is worst.
    assert np.argmax(fused) == 1
    assert np.argmin(fused) == 0
    assert fused.max() > 0.999
    assert fused.min() == 0.0


def test_rrf_is_robust_to_score_outliers():
    # RRF fuses rankings, not raw values, so an extreme outlier in qa earns only
    # one rank step and cannot dominate — item 3 (the outlier) has the weakest LR
    # and must not take the top; item 0 (strongest LR) wins.
    qa = np.array([0.50, 0.51, 0.52, 1000.0], dtype=np.float32)
    lr = np.array([0.90, 0.80, 0.70, 0.10], dtype=np.float32)

    fused = _fuse_scores(qa, lr)
    # The outlier cannot run away with the top: the best it can do is TIE item 0
    # on fused rank (each is best on one signal, worst on the other), and argmax
    # resolves to item 0. A raw-value blend would instead let it crush everything.
    assert np.argmax(fused) == 0
    assert fused[0] == fused.max()
    assert np.isclose(fused[3], fused[0])
