"""Adapter for official event feeds with explicit event and deadline dates."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

import feedparser
from dateutil import parser as date_parser

from ..models import Item, SourceSpec
from .rss import (
    _external_id,
    _extract_abstract,
    _http_get_bytes,
    _parsed_dt,
    _strip_html,
    canonicalize_url,
)


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _field(text: str, name: str, following: tuple[str, ...]) -> str:
    stops = "|".join(re.escape(label) for label in following)
    match = re.search(
        rf"\b{re.escape(name)}\s*:\s*(.+?)(?=\s+(?:{stops})\s*:|$)",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip(" .;|") if match else ""


def _parse_date(value: str) -> date | None:
    try:
        return date_parser.parse(value, fuzzy=True).date()
    except (ValueError, TypeError, OverflowError):
        return None


def _event_range(value: str) -> tuple[date | None, date | None]:
    text = re.sub(r"\s+", " ", value).strip()
    if not text:
        return None, None
    # Common EMBL form: "17 - 25 Aug 2026".
    same_month = re.fullmatch(
        r"(\d{1,2})\s*[-–]\s*(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})",
        text,
    )
    if same_month:
        first, last, month, year = same_month.groups()
        return _parse_date(f"{first} {month} {year}"), _parse_date(f"{last} {month} {year}")
    parts = re.split(r"\s+(?:to|until)\s+|\s+[–-]\s+", text, maxsplit=1)
    if len(parts) == 2:
        end = _parse_date(parts[1])
        start_text = parts[0]
        if end and not re.search(r"\b\d{4}\b", start_text):
            start_text = f"{start_text} {end.year}"
        start = _parse_date(start_text)
        return start, end
    parsed = _parse_date(text)
    return parsed, parsed


_DATE_PATTERN = (
    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|"
    r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\b"
)


def _deadline(value: str) -> tuple[date | None, bool]:
    """Return the LATEST still-open deadline, and whether every route is closed.

    Official event pages list several routes to attend, e.g.

        Abstract submission: Closed
        Registration (On-site): Closed
        Registration (Virtual): 25 Aug 2026

    Reading only the first "Closed" declared the whole event shut, which
    discarded conferences the reader could still register for -- measured on the
    live EMBL feed, 8 of 10 events were dropped while on-site or virtual
    registration was still open, leaving the events section permanently empty.
    An event is closed only when NO route remains: no future date anywhere.
    """
    if not value.strip():
        return None, False
    dates = [
        parsed
        for parsed in (_parse_date(m.group(0)) for m in re.finditer(_DATE_PATTERN, value))
        if parsed is not None
    ]
    if dates:
        # The most distant deadline is the last chance to act on the event.
        return max(dates), False
    # No date anywhere: only now does an explicit "closed" settle it.
    if re.search(r"\bclosed\b", value, flags=re.IGNORECASE):
        return None, True
    return _parse_date(value), False


def _event_type(title: str, description: str = "") -> str:
    """Classify an event from its title AND its description.

    Returns "" when the kind cannot be determined. That matters: the
    opportunity gate passes an EMPTY type through as "unknown, needs review"
    but drops a type that fails to match the reader's preferences, so labelling
    every unclassifiable entry with a generic "event" silently discarded the
    whole section -- measured on the live EMBL feed, all 6 open conferences
    were typed "event" and dropped as "type is outside preferences" because
    their titles ("Chemical biology 2026", "The complex life of RNA") never
    say what kind of gathering they are.
    """
    haystack = f"{title}\n{description}".casefold()
    for kind in ("conference", "workshop", "course", "symposium", "webinar", "lecture", "seminar"):
        if kind in haystack:
            return kind
    return ""


class EventsRSSSource:
    """Read official feeds, rejecting closed, past, or undated entries."""

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        del days
        if not spec.url:
            return []
        content = _http_get_bytes(spec.url)
        if not content:
            raise RuntimeError(f"{spec.name} event feed returned an empty response")
        feed = feedparser.parse(content)
        if feed.bozo and not feed.entries:
            raise RuntimeError(f"{spec.name} event feed could not be parsed")

        today = _today()
        horizon = today + timedelta(days=spec.lookahead_days)
        out: list[Item] = []
        for entry in feed.entries[:200]:
            url = canonicalize_url(entry.get("link", ""))
            title = _strip_html(entry.get("title", "")).strip()
            if not url or not title:
                continue
            feed_abstract = _extract_abstract(entry)
            try:
                detail_html = _http_get_bytes(url).decode("utf-8", errors="replace")
                detail_text = _strip_html(detail_html)
            except Exception:
                detail_text = ""
            # The official EMBL feed deliberately omits its date/location
            # sidebar from content:encoded; verify those fields on the linked
            # official page. Other official feeds may already carry them.
            facts_text = detail_text or feed_abstract
            date_text = _field(facts_text, "Date", ("Location", "Deadline(s)", "Deadline"))
            location = _field(facts_text, "Location", ("Deadline(s)", "Deadline", "Date"))
            deadline_text = _field(
                facts_text, "Deadline(s)", ("Date", "Location", "Organisers")
            ) or _field(facts_text, "Deadline", ("Date", "Location", "Organisers"))
            start, end = _event_range(date_text)
            deadline, closed = _deadline(deadline_text)
            if closed or start is None or start < today or start > horizon:
                continue
            if deadline is not None and deadline < today:
                continue
            fmt = (
                "online"
                if "online" in location.casefold()
                else "hybrid"
                if "hybrid" in location.casefold()
                else "in_person"
            )
            metadata = {
                "record_type": "event",
                "event_type": _event_type(title, facts_text),
                "status": "open",
                "official": True,
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "event_start": start.isoformat(),
                "event_end": (end or start).isoformat(),
                "deadline": deadline.isoformat() if deadline else None,
                "deadline_timezone": "",
                "deadlines": (
                    [{"type": "application", "date": deadline.isoformat()}] if deadline else []
                ),
                "location": location,
                "format": fmt,
                "organizer": spec.name,
                "official_id": entry.get("id") or entry.get("guid") or url,
            }
            out.append(
                Item(
                    source=spec.name,
                    section=spec.section,
                    external_id=_external_id(entry, url),
                    url=url,
                    title=title,
                    abstract=(feed_abstract or detail_text)[:8000],
                    published_at=_parsed_dt(entry),
                    metadata=metadata,
                )
            )
        return out
