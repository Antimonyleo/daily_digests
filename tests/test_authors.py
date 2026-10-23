"""Tests for author/lab watchlist matching."""

from __future__ import annotations

import pytest

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


def test_watchlist_does_not_assemble_a_name_from_different_authors():
    for byline in ["John Doe, Jane Smith", "John Doe; Jane Smith", "John Doe\nJane Smith",
                   "John Doe, Jane Smith; Other Person", "John Doe and Jane Smith", "John Doe & Jane Smith"]:
        assert author_match_score(byline, ["John Smith"]) == 0.0
        assert author_match_score(byline, ["Jane Smith"]) == 1.0
    assert author_match_score("Doudna, Jennifer A.; Smith, John", ["Jennifer Smith"]) == 0.0


@pytest.mark.parametrize("byline, watched", [
    ("Van Nguyen, John Smith", "John Nguyen"),
    ("Van Nguyen, J. Smith", "J. Nguyen"),
    ("Van Nguyen, John Smith; Other Person", "John Nguyen"),
    ("De Vries, Jane Smith", "Jane Vries"),
])
def test_surname_particle_cannot_join_two_complete_forward_authors(byline, watched):
    assert author_match_score(byline, [watched]) == 0.0
    assert author_match_score(byline, ["Van Nguyen" if byline.startswith("Van") else "De Vries"]) == 1.0


@pytest.mark.parametrize("byline, watched", [
    ("van Gogh, Vincent", "Vincent van Gogh"),
    ("van der Waals, Johannes D.", "Johannes van der Waals"),
    ("van der Waals, J. D.", "J. van der Waals"),
    ("de Gennes, Pierre-Gilles", "Pierre-Gilles de Gennes"),
    ("Doudna, Jennifer A.", "Jennifer Doudna"),
    ("Doudna, Jennifer Anne", "Jennifer Doudna"),
])
def test_credible_inverted_names_keep_given_names_and_initials(byline, watched):
    assert author_match_score(byline, [watched]) == 1.0


def test_inverted_names_and_institution_byline_remain_usable():
    assert author_match_score("Doudna, Jennifer A.", ["Jennifer Doudna"]) == 1.0
    assert author_match_score("van der Waals, Johannes; Smith, John", ["Johannes van der Waals"]) == 1.0
    assert author_match_score("Alice Brown, Broad Institute", ["Broad Institute"]) == 1.0
    assert author_match_score("Alice Brown, Institute of Science and Technology", ["Institute of Science and Technology"]) == 1.0


def test_author_names_preserve_unicode_letters_instead_of_ascii_fragments():
    assert author_match_score("Иван Петров, Jane Smith", ["Иван Петров"]) == 1.0
    assert author_match_score("José García", ["José García"]) == 1.0
    assert author_match_score("Jos Garc A", ["José García"]) == 0.0


def test_single_token_entry_is_too_generic_to_match():
    """A bare surname must NOT match.

    Matching is token-subset, so "Zhang" alone would hand the watched-author
    boost to every paper with any Zhang on it — a large share of each day's
    output. That both distorts the ranking and empties the signal of meaning,
    so an entry needs at least two name tokens to count.
    """
    from dailydigest.rank.authors import unusable_watchlist_entries

    assert author_match_score("Feng Zhang, et al.", ["Zhang"]) == 0.0
    assert author_match_score("Feng Zhang, et al.", ["Feng Zhang"]) == 1.0
    # The generic entry is reported rather than silently ignored.
    assert unusable_watchlist_entries(["Zhang", "Feng Zhang"]) == ["Zhang"]


def test_empty_inputs():
    assert author_match_score("", ["Doudna"]) == 0.0
    assert author_match_score("Doudna", []) == 0.0


def test_load_watchlist_strips_and_filters():
    p = Profile(bio="x", authors_of_interest=["  Jennifer Doudna ", "", "  "])
    assert load_watchlist(p) == ["Jennifer Doudna"]


def test_load_watchlist_default_empty():
    assert load_watchlist(Profile(bio="x")) == []
