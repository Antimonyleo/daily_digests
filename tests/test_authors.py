"""Tests for author/lab watchlist matching."""

from __future__ import annotations

from dailydigest.models import Profile
from dailydigest.rank.authors import author_match_score, load_watchlist


def test_full_name_matches_reordered_byline():
    wl = ["Jennifer Doudna"]
    assert author_match_score("Doudna, Jennifer A.; Smith, John", wl) == 1.0


def test_no_match_returns_zero():
    assert author_match_score("Alice Brown, Bob Green", ["Jennifer Doudna"]) == 0.0


def test_partial_token_subset_does_not_match():
    # "Doudna Zhang" requires BOTH tokens present; only Doudna here -> no match.
    assert author_match_score("Doudna, Jennifer", ["Doudna Zhang"]) == 0.0


def test_single_token_matches_any_surname():
    assert author_match_score("Feng Zhang, et al.", ["Zhang"]) == 1.0


def test_empty_inputs():
    assert author_match_score("", ["Doudna"]) == 0.0
    assert author_match_score("Doudna", []) == 0.0


def test_load_watchlist_strips_and_filters():
    p = Profile(bio="x", authors_of_interest=["  Jennifer Doudna ", "", "  "])
    assert load_watchlist(p) == ["Jennifer Doudna"]


def test_load_watchlist_default_empty():
    assert load_watchlist(Profile(bio="x")) == []
