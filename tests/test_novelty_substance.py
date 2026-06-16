"""Hype-novelty should require structured substance to count fully."""

from __future__ import annotations

from dailydigest.rank.source_quality import novelty_score


class _Row:
    section = "research"
    source = "Nature"

    def __init__(self, title: str, abstract: str = "") -> None:
        self.title = title
        self.abstract = abstract


def test_substance_free_hype_is_dampened():
    hype_only = _Row(
        "A landmark breakthrough, practice-changing and first-in-class",
        "An exciting curative announcement.",
    )
    hype_with_substance = _Row(
        "A landmark breakthrough, practice-changing and first-in-class",
        "We report a new method with efficacy and survival data.",
    )
    assert novelty_score(hype_with_substance) > novelty_score(hype_only)


def test_numbers_count_as_substance():
    with_numbers = _Row(
        "A landmark first-in-class result",
        "Reported a 42% improvement in the cohort.",
    )
    without = _Row(
        "A landmark first-in-class result",
        "An exciting announcement with no specifics.",
    )
    assert novelty_score(with_numbers) > novelty_score(without)


def test_legitimate_result_still_high_novelty():
    # A real result with hype + substance should remain clearly high-novelty.
    row = _Row(
        "First-in-class CRISPR delivery breakthrough",
        "Primary research with efficacy data.",
    )
    assert novelty_score(row) >= 0.55
