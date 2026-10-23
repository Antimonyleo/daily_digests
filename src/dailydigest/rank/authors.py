"""Author / lab watchlist matching.

A researcher filters on *who* wrote a paper as much as *what* it is about, but
the captured ``authors`` string was previously unused by ranking. This module
matches an item's author list against a profile watchlist
(``authors_of_interest``) so matched work can be boosted and exposed as a
learnable feature.

Matching is token-subset based within an individual byline entry: every token of
the watchlist entry must appear in the same author's name (case- and punctuation-
insensitive). So ``"Jennifer Doudna"`` matches ``"Doudna, Jennifer A."`` and a
lab/institution like ``"Broad Institute"`` matches its byline, while avoiding
naive substring false positives. Single-token entries (e.g. a bare surname)
are ignored — use full names for precision.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.casefold()))


def _byline_tokens(authors: str) -> list[set[str]]:
    # Semicolons/newlines explicitly delimit names and allow "Surname, Given".
    # Feed adapters generally join forward names with commas. For a single
    # inverted name, accept a one-token surname before the comma. A surname
    # particle needs more evidence: "Van Nguyen, John Smith" is two authors.
    names = []
    for group in re.split(r"[;\n]+|\s+(?:and|&)\s+", authors, flags=re.I):
        parts = [name.strip() for name in group.split(",") if name.strip()]
        tokens = [_tokens(name) for name in parts]
        names.extend(tokens)
        surname_prefix = bool(parts and re.match(r"(?:van|von|de|del|da|di|dos|du|le|la)\s", parts[0], re.I))
        if len(tokens) == 2:
            # A particle can also be a given name (Van Nguyen). The right side
            # must not look like another complete author. Keep hyphenated given
            # names intact, and permit only initials after the first word.
            given_suffix = _tokens(" ".join(parts[1].split()[1:]))
            if len(tokens[0]) == 1 or (surname_prefix and all(len(token) == 1 for token in given_suffix)):
                names.append(tokens[0] | tokens[1])
    return names


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
    authors_tokens = _byline_tokens(authors)
    # A full literal institution name may itself contain "and". Exact token
    # equality preserves it without accepting subsets spanning two people.
    literal_entries = [_tokens(name) for name in re.split(r"[;,\n]+", authors)]
    for entry in watchlist:
        entry_tokens = _tokens(entry)
        if len(entry_tokens) >= 2 and (
            any(entry_tokens <= tokens for tokens in authors_tokens)
            or entry_tokens in literal_entries
        ):
            return 1.0
    return 0.0


def unusable_watchlist_entries(watchlist: list[str]) -> list[str]:
    """Watchlist entries too generic to match on (fewer than two name tokens)."""
    return [entry for entry in watchlist if len(_tokens(entry)) < 2]
