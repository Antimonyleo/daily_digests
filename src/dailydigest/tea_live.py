"""Tea-break cards computed fresh for each brew, with no model in the loop.

The curated banks in :mod:`dailydigest.tea_break` are finite, so however large
they grow they eventually come round again. These generators add cards whose
details vary with each brew or day:

* :func:`brew_observations` reports on the run that just happened, using the
  ``candidate_funnel`` audit the pipeline already writes;
* :func:`corpus_observations` measures the window's literature in aggregate;
* :func:`generated_jokes` fills a grammar from lab vocabulary.

The first two can still repeat when their measured inputs are unchanged. The
third is *combinatorial*, which is not the same thing: readers recognise a
template long before they exhaust its fillings, so its real novelty is closer
to the number of templates than to the product of the slot sizes. It is
therefore a minority of the deck, not the backbone.

Sentence-level extraction from abstracts was measured and deliberately rejected:
over 234 papers that actually reached the reader's slate, a strict filter kept
4 sentences, and those still read as fragments ("The net improvement over the
2-parameter size-only prediction is of 2.5-fold."). Abstracts report findings
that depend on their setup; a tea-break fact has to survive leaving that setup
behind.

Every generator returns [] rather than raising: Pip must never break the page.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256

logger = logging.getLogger(__name__)

BREW_PREFIX = "From your brew — "
FEED_PREFIX = "From the feed — "
JOKE_PREFIX = "Lab joke — "


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    return singular if n == 1 else (plural or f"{singular}s")


# --------------------------------------------------------------------------- #
# 1. Observations about the brew that just ran
# --------------------------------------------------------------------------- #


def brew_observations(digest_id: str) -> list[str]:
    """Describe the run behind today's digest, from data it already recorded."""
    try:
        return _brew_observations(digest_id)
    except Exception as exc:  # noqa: BLE001 - a tea card is never worth an error
        logger.debug("brew observations unavailable: %s", exc)
        return []


def _brew_observations(digest_id: str) -> list[str]:
    from .store import DigestItemRow, ItemRow, VoteRow, load_digest_audit, session_scope

    out: list[str] = []
    funnel_rows = load_digest_audit(digest_id, "candidate_funnel")
    funnel = funnel_rows[0] if funnel_rows else {}

    with session_scope() as s:
        slate = (
            s.query(DigestItemRow)
            .filter(DigestItemRow.digest_id == digest_id)
            .all()
        )
        research_ids = [r.item_id for r in slate if str(r.item_label or "").startswith("R")]
        rows = [s.get(ItemRow, i) for i in research_ids]
        rows = [r for r in rows if r is not None]
        sources = [str(getattr(r, "source", "") or "") for r in rows]
        published = [r.published_at for r in rows if getattr(r, "published_at", None)]
        vote_total = s.query(VoteRow).count()

    # Votes are global history, not evidence that this digest was brewed. On a
    # morning before the first run, a non-empty vote table must not produce a
    # misleading "From your brew" card.
    if not funnel_rows and not slate:
        return []

    shown = len(rows)
    # ``recent_items`` spans every enabled section, while ``shown`` below is
    # research-only. New audits persist the matching denominator; retain the
    # old field as a compatibility fallback for decks brewed before that change.
    considered = int(funnel.get("recent_research_items") or funnel.get("recent_items") or 0)
    window = int(funnel.get("window_days") or 0)

    if considered and shown:
        out.append(
            f"{BREW_PREFIX}Pip considered {considered:,} research papers from the last "
            f"{window} {_plural(window, 'day')} and kept {shown}. That is about one in "
            f"{max(1, round(considered / shown))}."
        )

    near_dups = funnel.get("cross_day_near_dup_drops") or []
    near_dup_count = max(
        0,
        int(funnel.get("after_cross_source_dedupe") or 0)
        - int(funnel.get("after_cross_day_near_dup") or 0),
    ) or len(near_dups)
    if near_dup_count:
        best = max(
            (float(d.get("max_similarity") or 0) for d in near_dups),
            default=0.0,
        )
        detail = f" The closest matched at {best * 100:.0f}%." if best > 0 else ""
        out.append(
            f"{BREW_PREFIX}{near_dup_count} {_plural(near_dup_count, 'item')} in the "
            f"candidate pool turned out to be a near-copy of something you have already "
            f"seen.{detail}"
        )

    dropped = funnel.get("quality_gate_drops") or []
    dropped_count = max(
        0,
        int(funnel.get("after_cross_day_near_dup") or 0)
        - int(funnel.get("after_quality_gate") or 0),
    ) or len(dropped)
    if dropped_count:
        out.append(
            f"{BREW_PREFIX}{dropped_count} items were shown the door before ranking even started."
        )

    misses = load_digest_audit(digest_id, "missed_top_journals")
    if misses:
        top = max(misses, key=lambda m: float(m.get("score") or 0))
        source = str(top.get("source") or "a journal")
        out.append(
            f"{BREW_PREFIX}The nearest miss was {source}, scoring "
            f"{float(top.get('score') or 0):.3f} and finishing just outside the cut."
        )

    if len(sources) >= 2:
        distinct = len(set(sources))
        out.append(
            f"{BREW_PREFIX}Today's {shown} papers came from {distinct} different "
            f"{_plural(distinct, 'source')}."
        )
        busiest, count = Counter(sources).most_common(1)[0]
        if count >= 2:
            out.append(f"{BREW_PREFIX}{busiest} supplied {count} of them.")

    if published:
        oldest = min(published)
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=UTC)
        age = (datetime.now(UTC) - oldest).days
        if age >= 1:
            out.append(
                f"{BREW_PREFIX}The oldest paper on today's slate went online "
                f"{age} {_plural(age, 'day')} ago."
            )

    if vote_total:
        out.append(
            f"{BREW_PREFIX}You have graded {vote_total:,} items so far. Pip is keeping score."
        )

    return out


# --------------------------------------------------------------------------- #
# 2. Aggregate facts about the window's literature
# --------------------------------------------------------------------------- #

def _normalize_topic_text(value: str) -> str:
    """Normalize punctuation while retaining whole-word phrase boundaries."""
    return " ".join(re.sub(r"[\W_]+", " ", value.casefold()).split())


def _watched_topics() -> list[tuple[str, tuple[str, ...]]]:
    """Return up to ten topic labels and aliases from the reader's profile."""
    from .config import load_profile

    try:
        profile = load_profile()
    except (FileNotFoundError, ValueError):
        return []
    topics: list[tuple[str, tuple[str, ...]]] = []
    canonical = getattr(profile, "canonical_facets", None) or {}
    if canonical:
        for label, facet in list(canonical.items())[:10]:
            phrases = [label, *(getattr(facet, "aliases", None) or [])]
            normalized = tuple(
                dict.fromkeys(
                    term for phrase in phrases if (term := _normalize_topic_text(str(phrase)))
                )
            )
            if normalized:
                topics.append((str(label), normalized))
        return topics

    for keyword in (getattr(profile, "keywords", None) or [])[:10]:
        label = str(keyword).strip()
        normalized = _normalize_topic_text(label)
        if normalized:
            topics.append((label, (normalized,)))
    return topics


def _topic_counts(titles: list[str], topics: list[tuple[str, tuple[str, ...]]]) -> dict[str, int]:
    normalized_titles = [f" {_normalize_topic_text(title)} " for title in titles]
    return {
        label: sum(any(f" {phrase} " in title for phrase in phrases) for title in normalized_titles)
        for label, phrases in topics
    }


def corpus_observations(window_days: int = 4, limit: int = 6000) -> list[str]:
    """Measure the window's literature. Computed, so it cannot lose its context."""
    try:
        return _corpus_observations(window_days, limit)
    except Exception as exc:  # noqa: BLE001
        logger.debug("corpus observations unavailable: %s", exc)
        return []


def _corpus_observations(window_days: int, limit: int) -> list[str]:
    from sqlalchemy import select

    from .store import ItemRow, session_scope

    window_days = max(1, int(window_days))
    limit = int(limit)
    if limit <= 0:
        return []
    now = datetime.now(UTC)
    cutoff = now - timedelta(days=window_days)
    previous_cutoff = cutoff - timedelta(days=window_days)

    with session_scope() as s:
        # Query the two windows independently. A single unordered LIMIT across
        # both could contain only old rows on SQLite, yielding no current facts;
        # it also made trend claims compare unequal, arbitrary samples.
        current_rows = s.execute(
            select(ItemRow.source, ItemRow.title, ItemRow.fetched_at)
            .where(
                ItemRow.section == "research",
                ItemRow.fetched_at >= cutoff,
                ItemRow.fetched_at <= now,
            )
            .order_by(ItemRow.fetched_at.desc())
            .limit(limit + 1)
        ).all()
        earlier_rows = s.execute(
            select(ItemRow.source, ItemRow.title, ItemRow.fetched_at)
            .where(
                ItemRow.section == "research",
                ItemRow.fetched_at >= previous_cutoff,
                ItemRow.fetched_at < cutoff,
            )
            .order_by(ItemRow.fetched_at.desc())
            .limit(limit + 1)
        ).all()

    current_truncated = len(current_rows) > limit
    earlier_truncated = len(earlier_rows) > limit
    current = list(current_rows[:limit])
    earlier = list(earlier_rows[:limit])
    if not current:
        return []

    out: list[str] = []
    topics = _watched_topics()
    current_titles = [str(title or "") for _src, title, _fetched in current]
    prior_titles = [str(title or "") for _src, title, _fetched in earlier]

    counts = _topic_counts(current_titles, topics)
    ranked = [(term, n) for term, n in counts.items() if n]
    ranked.sort(key=lambda kv: -kv[1])

    if len(ranked) >= 2:
        (top_term, top_n), (second_term, second_n) = ranked[0], ranked[1]
        scope = (
            f"In a sample of the latest {len(current):,} research papers"
            if current_truncated
            else f"Across {len(current):,} newly fetched research papers"
        )
        out.append(
            f"{FEED_PREFIX}{scope}, "
            f"{top_term} shows up {top_n} {_plural(top_n, 'time')} and "
            f"{second_term} {second_n}."
        )

    # A term that moved sharply against the previous window of the same length.
    if prior_titles and not current_truncated and not earlier_truncated:
        prior_counts = _topic_counts(prior_titles, topics)
        for term, count_now in ranked[:6]:
            before = prior_counts.get(term, 0)
            if before >= 3 and count_now >= 3:
                change = (count_now - before) / before
                if abs(change) >= 0.5:
                    direction = "up" if change > 0 else "down"
                    out.append(
                        f"{FEED_PREFIX}Mentions of {term} are {direction} "
                        f"{abs(change) * 100:.0f}% against the previous {window_days} days "
                        f"({before} to {count_now})."
                    )
                    break

    source_counts = Counter(str(src or "") for src, _t, _f in current)
    if source_counts:
        busiest, n = source_counts.most_common(1)[0]
        if current_truncated:
            out.append(
                f"{FEED_PREFIX}{busiest} supplied {n:,} papers in that "
                f"{len(current):,}-title sample."
            )
        else:
            out.append(
                f"{FEED_PREFIX}{busiest} alone posted {n:,} papers in the last "
                f"{window_days} days. You will see a handful."
            )
        quiet = [name for name, c in source_counts.items() if c == 1]
        if quiet:
            scope = (
                "appeared exactly once in that sample"
                if current_truncated
                else "contributed exactly one paper this window"
            )
            out.append(f"{FEED_PREFIX}{len(quiet)} {_plural(len(quiet), 'source')} {scope}.")

    longest = max(current, key=lambda r: len(str(r[1] or "")))
    words = len(str(longest[1] or "").split())
    if words >= 20:
        scope = "sample" if current_truncated else "window"
        out.append(
            f"{FEED_PREFIX}The longest title in the {scope} runs to {words} words. "
            f"Pip counted them so you do not have to."
        )

    return out


# --------------------------------------------------------------------------- #
# 3. Combinatorial jokes
# --------------------------------------------------------------------------- #

_SLOTS: dict[str, tuple[str, ...]] = {
    "instrument": (
        "the mass spec", "the plate reader", "the confocal", "the centrifuge",
        "the HPLC", "the NMR", "the flow cytometer", "the thermocycler",
        "the sonicator", "the freeze dryer", "the microscope", "the autoclave",
    ),
    "reagent": (
        "the buffer", "the antibody", "the enzyme stock", "the primer set",
        "the master mix", "the competent cells", "the crosslinker", "the gel stain",
    ),
    "artefact": (
        "the manuscript", "the rebuttal", "the grant", "the poster",
        "the slide deck", "the protocol", "the thesis chapter", "the figure legend",
    ),
    "code": (
        "the analysis script", "the pipeline", "the notebook", "the config file",
        "the container image", "the plotting code",
    ),
    "occasion": (
        "overnight", "on a Friday", "during the site visit", "mid-demo",
        "two days before the deadline", "while nobody was watching",
        "the week it was finally cited",
    ),
}

# Labels make the generated cards sound like a rotating lab cast rather than
# one anonymous narrator.  Each label is unique across families, and every
# family has at least as many viewpoints as it has templates below.
_SLOT_PERSPECTIVES: dict[str, tuple[str, ...]] = {
    "instrument": (
        "Instrument's view",
        "Facility manager's view",
        "Core technician's view",
        "Service engineer's view",
        "Booking calendar's view",
        "Night-shift researcher's view",
        "Maintenance log's view",
    ),
    "reagent": (
        "Reagent's view",
        "Bench scientist's view",
        "Lab manager's view",
        "Purchasing office's view",
        "Freezer's view",
    ),
    "artefact": (
        "PI's view",
        "Reviewer's view",
        "Editor's view",
        "Co-author's view",
        "Grant panel's view",
    ),
    "code": (
        "Dataset's view",
        "Analyst's view",
        "Computer's view",
        "Future maintainer's view",
        "Repository's view",
        "Model's view",
    ),
}

# Each template uses one slot family, so no combination can read oddly.
_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("instrument", "{x} is working perfectly. It has simply chosen not to demonstrate that today."),
    ("instrument", "{x} broke {occasion}, as is traditional."),
    ("instrument", "Nobody has touched {x} since it started making that noise, and nobody intends to."),
    ("instrument", "The booking sheet says {x} is free. The booking sheet is an optimist."),
    ("instrument", "{x} was serviced last week and is now broken in an entirely new way."),
    ("instrument", "{x} produces beautiful data on samples we do not have."),
    ("instrument", "Someone has recalibrated {x} and left no note, which is its own kind of message."),
    ("reagent", "{x} is fresh, in the sense that somebody made it once."),
    ("reagent", "{x} has been relabelled twice and trusted once."),
    ("reagent", "{x} works perfectly in the hands of exactly one person, who has left."),
    ("reagent", "We are out of {x}, and the order was placed with great optimism in March."),
    ("reagent", "{x} expired last month, which we will be treating as a suggestion."),
    ("artefact", "{x} is nearly finished, and has been nearly finished for five weeks."),
    ("artefact", "{x} was due yesterday, which is why it now has an appendix."),
    ("artefact", "I have read {x} so many times that I can no longer tell whether it is English."),
    ("artefact", "Everyone has approved {x} except the person who has not read it."),
    ("artefact", "{x} improved enormously once I deleted the paragraph I could not defend."),
    ("code", "{x} runs. What {x} does is a separate research question."),
    ("code", "I fixed {x} and broke {x} in the same commit."),
    ("code", "{x} has one comment, and it says TODO."),
    ("code", "{x} works on my machine, which is now formally a piece of scientific equipment."),
    ("code", "{x} was written to be temporary and is now three years into its tenure."),
    ("code", "Nobody understands {x}, including the version of me that wrote it."),
)


def generated_jokes(count: int, day: date, exclude: set[str] | None = None) -> list[str]:
    """Fill the grammar deterministically for *day*, skipping *exclude*.

    Combinatorial variety is shallow -- a reader recognises the template long
    before the fillings run out -- so callers should take only a couple of these
    per deck and leave the curated bank to carry the rest.
    """
    try:
        return _generated_jokes(count, day, exclude or set())
    except Exception as exc:  # noqa: BLE001
        logger.debug("generated jokes unavailable: %s", exc)
        return []


def _generated_jokes(count: int, day: date, exclude: set[str]) -> list[str]:
    if count <= 0:
        return []
    seed = int.from_bytes(sha256(day.isoformat().encode()).digest()[:8], "big")
    out: list[str] = []
    used_templates: set[int] = set()
    used_perspectives: set[str] = set()
    # Walk the template list on a day-dependent stride so consecutive days do
    # not open with the same shape.
    stride = 1 + (seed % (len(_TEMPLATES) - 1))
    index = seed % len(_TEMPLATES)
    for step in range(len(_TEMPLATES) * 3):
        if len(out) >= count:
            break
        template_index = index % len(_TEMPLATES)
        index += stride
        if template_index in used_templates:
            continue
        slot, template = _TEMPLATES[template_index]
        perspectives = _SLOT_PERSPECTIVES[slot]
        perspective_start = (seed // (template_index + 1) + step) % len(perspectives)
        perspective = next(
            (
                perspectives[(perspective_start + offset) % len(perspectives)]
                for offset in range(len(perspectives))
                if perspectives[(perspective_start + offset) % len(perspectives)]
                not in used_perspectives
            ),
            None,
        )
        if perspective is None:
            continue
        values = _SLOTS[slot]
        value = values[(seed // (template_index + 1) + step) % len(values)]
        body = template.format(
            x=value,
            occasion=_SLOTS["occasion"][(seed // 7 + step) % len(_SLOTS["occasion"])],
        )
        body = body[0].upper() + body[1:]
        text = f"{JOKE_PREFIX}{perspective} — {body}"
        if text in exclude or text in out:
            continue
        used_templates.add(template_index)
        used_perspectives.add(perspective)
        out.append(text)
    return out
