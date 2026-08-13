from __future__ import annotations

import pytest


def test_compute_lock_blocks_a_second_owner_and_releases_cleanly(tmp_path):
    from dailydigest.job_lock import ComputeBusyError, acquire_compute_lock

    db_path = str(tmp_path / "digest.db")
    first = acquire_compute_lock(db_path)
    try:
        with pytest.raises(ComputeBusyError):
            acquire_compute_lock(db_path)
    finally:
        first.release()

    second = acquire_compute_lock(db_path)
    second.release()
