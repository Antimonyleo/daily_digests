"""Author / lab watchlist matching.

A researcher filters on *who* wrote a paper as much as *what* it is about, but
the captured ``authors`` string was previously unused by ranking. This module
matches an item's author list against a profile watchlist
(``authors_of_interest``) so matched work can be boosted and exposed as a
learnable feature.

Matching is token-subset based: a watchlist entry matches when every token of
the entry appears in the item's author string (case- and punctuation-
insensitive). So ``"Jennifer Doudna"`` matches ``"Doudna, Jennifer A."`` and a
lab/institution like ``"Broad Institute"`` matches its byline, while avoiding
naive substring false positives. Single-token entries (e.g. a bare surname)
match broadly by design — prefer full names for precision.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def load_watchlist(profile: object) -> list[str]:
    """Return the cleaned ``authors_of_interest`` list from a profile."""
    raw = getattr(profile, "authors_of_interest", None) or []
    return [str(entry).strip() for entry in raw if str(entry).strip()]


def author_match_score(authors: str, watchlist: list[str]) -> float:
    """Return 1.0 if any watchlist entry matches the author string, else 0.0.

    A watchlist entry must carry at least two tokens (e.g. "Peng Yin", or an
    initial plus surname). A bare surname is not specific enough: matching is
    token-subset, so a single common name like "Wang" or "Li" would match a
    large share of every day's papers and hand them all the same boost, which
    both distorts the ranking and destroys the signal's meaning. Single-token
    entries are ignored here and reported by :func:`unusable_watchlist_entries`
    so the reader can fix them rather than wonder why nothing matched.
    """
    if not authors or not watchlist:
        return 0.0
    author_tokens = _tokens(authors)
    if not author_tokens:
        return 0.0
    for entry in watchlist:
        entry_tokens = _tokens(entry)
        if len(entry_tokens) >= 2 and entry_tokens <= author_tokens:
            return 1.0
    return 0.0


def unusable_watchlist_entries(watchlist: list[str]) -> list[str]:
    """Watchlist entries too generic to match on (fewer than two name tokens)."""
    return [entry for entry in watchlist if len(_tokens(entry)) < 2]
