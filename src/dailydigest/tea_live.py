"""Tea-break cards computed fresh for each brew, with no model in the loop.

The curated banks in :mod:`dailydigest.tea_break` are finite, so however large
they grow they eventually come round again. These three generators do not:

* :func:`brew_observations` reports on the run that just happened, using the
  ``candidate_funnel`` audit the pipeline already writes;
* :func:`corpus_observations` measures the window's literature in aggregate;
* :func:`generated_jokes` fills a grammar from lab vocabulary.

The first two are genuinely unrepeatable because their inputs change every day.
The third is *combinatorial*, which is not the same thing: readers recognise a
template long before they exhaust its fillings, so its real novelty is closer to
the number of templates than to the product of the slot sizes. It is therefore a
minority of the deck, not the backbone.

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

    shown = len(rows)
    considered = int(funnel.get("recent_items") or 0)
    window = int(funnel.get("window_days") or 0)

    if considered and shown:
        out.append(
            f"{BREW_PREFIX}Pip read {considered:,} papers from the last {window} "
            f"{_plural(window, 'day')} and kept {shown}. That is one in {considered // shown}."
        )

    near_dups = funnel.get("cross_day_near_dup_drops") or []
    if near_dups:
        best = max(float(d.get("max_similarity") or 0) for d in near_dups)
        out.append(
            f"{BREW_PREFIX}{len(near_dups)} {_plural(len(near_dups), 'paper')} today "
            f"turned out to be a near-copy of something you have already seen. "
            f"The closest matched at {best * 100:.0f}%."
        )

    dropped = funnel.get("quality_gate_drops") or []
    if dropped:
        out.append(
            f"{BREW_PREFIX}{len(dropped)} items were shown the door before ranking "
            f"even started."
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
            f"{BREW_PREFIX}You have graded {vote_total:,} items so far. "
            f"Pip is keeping score."
        )

    return out


# --------------------------------------------------------------------------- #
# 2. Aggregate facts about the window's literature
# --------------------------------------------------------------------------- #

# Counted over titles only: cheap, and a title states the subject plainly.
_WATCHED_TERMS = (
    "DNA origami",
    "self-assembly",
    "CRISPR",
    "machine learning",
    "protein design",
    "nanoparticle",
    "mRNA",
    "cryo-EM",
    "foundation model",
    "phase separation",
)


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

    cutoff = datetime.now(UTC) - timedelta(days=max(1, window_days))
    previous_cutoff = cutoff - timedelta(days=max(1, window_days))

    with session_scope() as s:
        rows = (
            s.execute(
                select(ItemRow.source, ItemRow.title, ItemRow.fetched_at)
                .where(ItemRow.section == "research", ItemRow.fetched_at >= previous_cutoff)
                .limit(limit)
            )
            .all()
        )

    current = [r for r in rows if r[2] is not None and _aware(r[2]) >= cutoff]
    earlier = [r for r in rows if r[2] is not None and _aware(r[2]) < cutoff]
    if not current:
        return []

    out: list[str] = []
    titles = " \n ".join(str(t or "") for _src, t, _f in current).casefold()
    prior_titles = " \n ".join(str(t or "") for _src, t, _f in earlier).casefold()

    counts = {
        term: len(re.findall(re.escape(term.casefold()), titles)) for term in _WATCHED_TERMS
    }
    ranked = [(term, n) for term, n in counts.items() if n]
    ranked.sort(key=lambda kv: -kv[1])

    if len(ranked) >= 2:
        (top_term, top_n), (second_term, second_n) = ranked[0], ranked[1]
        out.append(
            f"{FEED_PREFIX}Across {len(current):,} new papers, "
            f"{top_term} shows up {top_n} {_plural(top_n, 'time')} and "
            f"{second_term} {second_n}."
        )

    # A term that moved sharply against the previous window of the same length.
    if prior_titles:
        for term, now in ranked[:6]:
            before = len(re.findall(re.escape(term.casefold()), prior_titles))
            if before >= 3 and now >= 3:
                change = (now - before) / before
                if abs(change) >= 0.5:
                    direction = "up" if change > 0 else "down"
                    out.append(
                        f"{FEED_PREFIX}Mentions of {term} are {direction} "
                        f"{abs(change) * 100:.0f}% on the previous {window_days} days "
                        f"({before} to {now})."
                    )
                    break

    source_counts = Counter(str(src or "") for src, _t, _f in current)
    if source_counts:
        busiest, n = source_counts.most_common(1)[0]
        out.append(
            f"{FEED_PREFIX}{busiest} alone posted {n:,} papers in the last "
            f"{window_days} days. You will see a handful."
        )
        quiet = [name for name, c in source_counts.items() if c == 1]
        if quiet:
            out.append(
                f"{FEED_PREFIX}{len(quiet)} {_plural(len(quiet), 'source')} "
                f"contributed exactly one paper this window."
            )

    longest = max(current, key=lambda r: len(str(r[1] or "")))
    words = len(str(longest[1] or "").split())
    if words >= 20:
        out.append(
            f"{FEED_PREFIX}The longest title in the window runs to {words} words. "
            f"Pip counted them so you do not have to."
        )

    return out


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


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
        values = _SLOTS[slot]
        value = values[(seed // (template_index + 1) + step) % len(values)]
        text = JOKE_PREFIX + template.format(
            x=value,
            occasion=_SLOTS["occasion"][(seed // 7 + step) % len(_SLOTS["occasion"])],
        )
        # Capitalise after the prefix when the slot starts the sentence.
        head = len(JOKE_PREFIX)
        text = text[:head] + text[head].upper() + text[head + 1 :]
        if text in exclude or text in out:
            continue
        used_templates.add(template_index)
        out.append(text)
    return out
