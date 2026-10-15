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
    core, context = profile_search_term_groups(max_terms)
    return core + context


def profile_search_term_groups(
    max_terms: int = 32, *, canonical_order: bool = False, include_aliases: bool = True
) -> tuple[list[str], list[str]]:
    """Keep core and context separate so adapters can reserve core query budget.

    Canonical order makes bounded retrieval independent of profile list order.
    All core terms survive the soft term limit; only context terms are capped.
    """
    try:
        from ..config import load_profile

        prof = load_profile()
    except Exception as e:  # noqa: BLE001
        logger.warning("profile_search_terms: could not load profile: %s", e)
        return [], []
    # Search synonyms only when explicitly supplied by the reader. Canonical
    # names remain included even for profiles with different legacy keywords.
    canonical = getattr(prof, "canonical_facets", None) or {}
    keywords = list(getattr(prof, "keywords", []) or [])
    extras = []
    for label, spec in canonical.items():
        if label.casefold() not in {str(term).casefold() for term in keywords}:
            keywords.append(label)
        extras.extend(getattr(spec, "aliases", [])[:3])
    seen: set[str] = set()
    groups: list[list[str]] = []
    for field in ("keywords", "context_keywords"):
        terms = [str(kw).strip() for kw in (keywords + (extras if include_aliases else []) if field == "keywords" else getattr(prof, field, []) or [])]
        if canonical_order:
            terms.sort(key=lambda term: (term.casefold(), term))
        distinct = []
        for term in terms:
            low = term.casefold()
            if term and low not in seen:
                seen.add(low)
                distinct.append(term)
        groups.append(distinct)
    core, context = groups
    return core, context[:max(0, max_terms - len(core))]


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
