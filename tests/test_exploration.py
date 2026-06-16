"""Tests for the active-learning exploration slot."""

from __future__ import annotations

from dailydigest.rank.ranker import apply_exploration


class _Row:
    def __init__(self, rid: int, eligible: bool, section: str = "research") -> None:
        self.id = rid
        self.eligible = eligible
        self.section = section
        self.source = "Test"


def _eligible(r) -> bool:
    return getattr(r, "eligible", False)


def _setup():
    picked = [(_Row(1, True), 0.9), (_Row(2, True), 0.8), (_Row(3, True), 0.5)]
    extra = [(_Row(4, True), 0.6), (_Row(5, True), 0.55), (_Row(6, False), 0.7)]
    candidates = picked + extra
    # r6 is the most uncertain but NOT eligible (low quality); r4 next.
    uncertainty = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.9, 5: 0.8, 6: 0.99}
    return picked, candidates, uncertainty


def test_swaps_lowest_pick_for_most_uncertain_eligible():
    picked, candidates, uncertainty = _setup()
    out = apply_exploration(
        picked, candidates, uncertainty, slots=1, eligible=_eligible
    )
    ids = [row.id for row, _ in out]
    assert 4 in ids       # most-uncertain eligible swapped in
    assert 3 not in ids    # lowest-scored pick swapped out
    assert 6 not in ids    # low-quality item never explored despite top uncertainty
    assert len(out) == len(picked)


def test_never_selects_ineligible_items():
    picked, candidates, uncertainty = _setup()
    # Make the only unselected candidates ineligible.
    candidates = picked + [(_Row(6, False), 0.7), (_Row(7, False), 0.7)]
    uncertainty = {1: 0.1, 2: 0.1, 3: 0.1, 6: 0.99, 7: 0.95}
    out = apply_exploration(
        picked, candidates, uncertainty, slots=2, eligible=_eligible
    )
    assert {row.id for row, _ in out} == {1, 2, 3}  # unchanged


def test_zero_slots_is_noop():
    picked, candidates, uncertainty = _setup()
    out = apply_exploration(picked, candidates, uncertainty, slots=0, eligible=_eligible)
    assert out == picked


def test_two_slots_swap_two_lowest():
    picked, candidates, uncertainty = _setup()
    out = apply_exploration(
        picked, candidates, uncertainty, slots=2, eligible=_eligible
    )
    ids = {row.id for row, _ in out}
    # Two lowest picks (3, 2) replaced by the two most-uncertain eligible (4, 5).
    assert {4, 5} <= ids
    assert 1 in ids  # top pick retained
