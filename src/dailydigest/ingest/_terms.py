"""Profile-derived search terms for active retrieval adapters.

Turns the user's profile keywords / watched authors into concrete queries so
OpenAlex and PubMed actively *search all venues* for the user's interests,
rather than only polling the curated RSS feed list. Bounded so the extra API
volume (and the embedding of new items) stays reasonable.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def profile_search_terms(max_terms: int = 12) -> list[str]:
    """Return up to ``max_terms`` distinct keyword phrases from the profile."""
    try:
        from ..config import load_profile

        prof = load_profile()
    except Exception as e:  # noqa: BLE001
        logger.warning("profile_search_terms: could not load profile: %s", e)
        return []
    seen: set[str] = set()
    out: list[str] = []
    # Core keywords first, then context ("keep-me-informed") terms: context terms
    # are down-weighted for *research relevance* but must still drive retrieval so
    # the industry/regulatory sections keep getting clinical/FDA/etc. items.
    core = list(getattr(prof, "keywords", []) or [])
    context = list(getattr(prof, "context_keywords", []) or [])
    for kw in core + context:
        term = str(kw).strip()
        low = term.lower()
        if term and low not in seen:
            seen.add(low)
            out.append(term)
        if len(out) >= max_terms:
            break
    return out


def watched_author_names(max_authors: int = 10) -> list[str]:
    """Return up to ``max_authors`` watched author names from the profile."""
    try:
        from ..config import load_profile

        prof = load_profile()
    except Exception as e:  # noqa: BLE001
        logger.warning("watched_author_names: could not load profile: %s", e)
        return []
    out: list[str] = []
    for a in getattr(prof, "authors_of_interest", []) or []:
        name = str(a).strip()
        if name:
            out.append(name)
        if len(out) >= max_authors:
            break
    return out
