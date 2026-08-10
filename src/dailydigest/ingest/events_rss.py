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


def _deadline(value: str) -> tuple[date | None, bool]:
    if re.search(r"\bclosed\b", value, flags=re.IGNORECASE):
        return None, True
    explicit = re.search(
        r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|"
        r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\b",
        value,
    )
    return _parse_date(explicit.group(0) if explicit else value), False


def _event_type(title: str) -> str:
    lower = title.casefold()
    for kind in ("workshop", "conference", "course", "symposium", "webinar"):
        if kind in lower:
            return kind
    return "event"


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
                "event_type": _event_type(title),
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
