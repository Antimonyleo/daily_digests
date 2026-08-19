"""Profile-derived search terms for active retrieval adapters.

Turns the user's profile keywords / watched authors into concrete queries so
OpenAlex and PubMed actively *search all venues* for the user's interests,
rather than only polling the curated RSS feed list. Bounded so the extra API
volume (and the embedding of new items) stays reasonable.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def profile_search_terms(max_terms: int = 32) -> list[str]:
    """Return distinct keyword phrases from the profile to drive active retrieval.

    ALL core keywords are always included (they define the user's field — dropping
    any silently blinds retrieval to that interest; previously a 12-term cap hid
    the 13th+ core interests). Context ("keep-me-informed") terms then fill up to
    ``max_terms``. The cap only bounds the context tail and total API volume.
    """
    try:
        from ..config import load_profile

        prof = load_profile()
    except Exception as e:  # noqa: BLE001
        logger.warning("profile_search_terms: could not load profile: %s", e)
        return []
    seen: set[str] = set()
    out: list[str] = []
    core = list(getattr(prof, "keywords", []) or [])
    context = list(getattr(prof, "context_keywords", []) or [])
    for i, kw in enumerate(core + context):
        term = str(kw).strip()
        low = term.lower()
        if term and low not in seen:
            seen.add(low)
            out.append(term)
        # Never truncate a core keyword; only cap once we are into the context tail.
        if len(out) >= max_terms and i >= len(core):
            break
    return out


def watched_author_names(max_authors: int = 25) -> list[str]:
    """Return up to ``max_authors`` watched author names from the profile.

    Truncation is logged: a silently-dropped tail is why a long
    ``authors_of_interest`` list can look like it does nothing.
    """
    try:
        from ..config import load_profile

        prof = load_profile()
    except Exception as e:  # noqa: BLE001
        logger.warning("watched_author_names: could not load profile: %s", e)
        return []
    configured = [
        str(a).strip() for a in getattr(prof, "authors_of_interest", []) or []
    ]
    configured = [name for name in configured if name]
    out = configured[:max_authors]
    if len(configured) > len(out):
        logger.warning(
            "watched authors: using %d of %d configured names; the rest are not "
            "queried as a retrieval channel",
            len(out),
            len(configured),
        )
    return out
