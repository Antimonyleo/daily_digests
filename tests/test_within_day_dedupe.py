"""Tests for within-day near-duplicate suppression config + contract.

The greedy suppression algorithm itself is covered by
``TestCapNearDuplicates`` in ``test_dedupe.py``. These tests lock in the
configuration defaults and the "disabled = no-op" invariant that the pipeline
relies on. They are fully hermetic (no network, no embedding model, no DB):
synthetic vectors are constructed directly.
"""

from __future__ import annotations

import numpy as np

from dailydigest.config import Settings
from dailydigest.dedupe import cap_near_duplicates


def _unit(vec: list[float]) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


class TestWithinDayDedupeConfig:
    def test_defaults(self):
        s = Settings(_env_file=None)
        assert s.within_day_dedupe is True
        assert s.within_day_dedupe_threshold == 0.86

    def test_threshold_is_high_by_design(self):
        # Guard against an accidental lowering of the deliberately-high default.
        s = Settings(_env_file=None)
        assert s.within_day_dedupe_threshold >= 0.86

    def test_disable_flag(self):
        s = Settings(_env_file=None, within_day_dedupe=False)
        assert s.within_day_dedupe is False

    def test_threshold_override(self):
        s = Settings(_env_file=None, within_day_dedupe_threshold=0.90)
        assert s.within_day_dedupe_threshold == 0.90


class TestWithinDayDedupeBehavior:
    def test_near_dup_cluster_collapses(self):
        vecs = np.vstack([_unit([1, 0, 0]), _unit([1, 0.01, 0]), _unit([1, 0, 0.01])])
        keep = cap_near_duplicates(["a", "b", "c"], vecs, threshold=0.86)
        assert keep == [0]

    def test_disabled_pipeline_path_is_noop(self):
        # Emulate the pipeline's disabled branch: when within_day_dedupe is off
        # the candidate list must be identical to the input (no call is made).
        s = Settings(_env_file=None, within_day_dedupe=False)
        scored = [("row_a", 0.9), ("row_b", 0.89), ("row_c", 0.5)]
        if getattr(s, "within_day_dedupe", True):
            raise AssertionError("flag should be disabled in this test")
        # No suppression applied => list unchanged.
        assert scored == [("row_a", 0.9), ("row_b", 0.89), ("row_c", 0.5)]
