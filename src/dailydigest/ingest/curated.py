"""Hand-kept fellowships, awards and meetings from official pages.

Some funders publish schemes only as prose, and some organisers (Gordon
Research Conferences, MRS, ACS, RSC) block automated access. Their entries
live in ``dailydigest/curated/*.yaml`` with the official URL and the date the
facts were checked. A stated deadline is used only when the official page
gave it; a deadline seen only in a search result is dropped. A meeting dated
from a search result keeps its dates but is labelled "Verify source".
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from importlib.resources import files
from typing import Any

import yaml

from ..config import user_local_date
from ..models import Item, SourceSpec
from ..opportunities import CALL_TYPES
from .events_api import _event_item
from .events_rss import _event_type
from .rss import canonicalize_url



def _today() -> date:
    return user_local_date()


def _load(name: str) -> list[dict[str, Any]]:
    text = files("dailydigest").joinpath("curated", name).read_text(encoding="utf-8")
    rows = yaml.safe_load(text) or []
    if not isinstance(rows, list):
        raise RuntimeError(f"Curated list {name} must be a YAML list")
    return [row for row in rows if isinstance(row, dict)]


def _date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


def _checked_at(row: dict[str, Any]) -> str | None:
    """When the hand-kept facts were last checked, never the brew time."""
    checked = _date(row.get("verified_on"))
    return datetime.combine(checked, time.min, tzinfo=timezone.utc).isoformat() if checked else None


def _codes(value: object) -> list[str]:
    return [str(code).strip().casefold() for code in value or [] if str(code).strip()]


class CuratedCallsSource:
    """Recurring fellowships and awards; the reader's country picks which apply."""

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        del days
        today = _today()
        horizon = today + timedelta(days=spec.lookahead_days)
        items = []
        for row in _load(spec.endpoint or "calls.yaml"):
            identifier, title = str(row.get("id") or "").strip(), str(row.get("title") or "").strip()
            url = canonicalize_url(str(row.get("url") or ""))
            if not identifier or not title or not url:
                continue
            deadline = _date(row.get("next_deadline"))
            if str(row.get("deadline_evidence") or "").startswith("search_snippet"):
                deadline = None  # Not read on the official page itself.
            if deadline is not None and not today <= deadline <= horizon:
                deadline = None  # A past round; the scheme itself recurs.
            kind = str(row.get("type") or "fellowship").casefold()
            stages = ", ".join(str(stage) for stage in row.get("career_stage") or [])
            text = " · ".join(part for part in (
                str(row.get("summary") or "").strip(),
                f"Fields: {row['fields']}" if row.get("fields") else "",
                f"Career stage: {stages}" if stages else "",
            ) if part)
            checked = _checked_at(row)
            items.append(Item(
                source=spec.name,
                section=spec.section,
                external_id=f"curated-{identifier}",
                url=url,
                title=title,
                abstract=text,
                authors=str(row.get("funder") or "").strip(),
                published_at=None,
                metadata={
                    "record_type": "call",
                    "opportunity_type": kind if kind in CALL_TYPES else "fellowship",
                    "status": "open" if deadline else "recurring",
                    # An entry with no check date is not presented as verified.
                    "official": checked is not None,
                    "verified_at": checked,
                    "funder": str(row.get("funder") or "").strip(),
                    "deadline": deadline.isoformat() if deadline else None,
                    "deadline_timezone": "",
                    "deadlines": (
                        [{"type": "application", "date": deadline.isoformat(), "timezone": ""}] if deadline else []
                    ),
                    "eligibility": "",
                    "eligibility_tags": [],
                    "eligibility_description": str(row.get("eligibility_note") or "").strip(),
                    "applicant_regions": _codes(row.get("applicant_regions")),
                    "host_countries": _codes(row.get("host_countries")),
                    "cycle": str(row.get("cycle") or "").strip(),
                    "official_id": identifier,
                },
            ))
        return items


class CuratedEventsSource:
    """Flagship meetings from organisers whose sites block automated access."""

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        del days
        today = _today()
        horizon = today + timedelta(days=spec.lookahead_days)
        items = []
        for row in _load(spec.endpoint or "meetings.yaml"):
            start, end = _date(row.get("start")), _date(row.get("end"))
            identifier, title = str(row.get("id") or "").strip(), str(row.get("title") or "").strip()
            url = canonicalize_url(str(row.get("url") or ""))
            if not identifier or not title or not url or start is None or not today <= start <= horizon:
                continue
            from_page = str(row.get("evidence_kind") or "") == "page"
            deadline = _date(row.get("deadline")) if from_page else None
            kind = str(row.get("event_type") or "").casefold()
            fmt = str(row.get("format") or "in_person").casefold()
            item = _event_item(
                spec, external_id=f"curated-{identifier}", url=url, title=title,
                abstract=str(row.get("topics") or "").strip(), start=start, end=end,
                deadlines=[(str(row.get("deadline_type") or "deadline"), deadline)] if deadline else [],
                location=str(row.get("location") or "").strip(),
                fmt=fmt if fmt in {"in_person", "online", "hybrid"} else "in_person",
                event_type=_event_type(title, kind), official_id=identifier,
                organizer=str(row.get("organizer") or "").strip(),
            )
            checked = _checked_at(row)
            item.metadata["official"] = from_page and checked is not None
            item.metadata["verified_at"] = checked
            items.append(item)
        return items
