"""Structured event listings from EMBL and EMBO.

Both organisers publish every event fact as data, so nothing is read from
pages. EMBL's REST listing replaces its RSS feed, which carried only the ten
newest posts and needed one page fetch per event for dates.
"""

from __future__ import annotations

import hashlib
import html
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ..config import user_local_date
from ..models import Item, SourceSpec
from ._retry import transient_http_error
from .events_rss import _event_type
from .rss import canonicalize_url

EMBL_URL = "https://www.embl.org/about/info/course-and-conference-office/wp-json/wp/v2/vf_events"
EMBL_FIELDS = ",".join([
    "id", "guid", "link", "title", "vf_event_start_date", "vf_event_end_date",
    "vf_event_submission_closing", "vf_event_registration_closing",
    "vf_event_registration_closing_on-site", "vf_event_summary", "vf_event_canceled",
    "vf_event_venue", "vf_event_event_type", "vf_event_location", "vf_event_displayed",
])
EMBO_URL = "https://www.embo.org/wp-json/embo/v1/events/new"
_TAG_RE = re.compile(r"<[^>]+>")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(transient_http_error),
    reraise=True,
)
def _get_json(url: str, params: dict[str, str] | None = None) -> Any:
    with httpx.Client(timeout=30.0, follow_redirects=True,
                      headers={"Accept": "application/json", "User-Agent": "dailydigest/0.1"}) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.json()


def _today() -> date:
    return user_local_date()


def _plain(value: object) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG_RE.sub(" ", str(value or "")))).strip()


def _ymd(value: object) -> date | None:
    text = re.sub(r"[^0-9]", "", str(value or ""))[:8]
    if len(text) != 8:
        return None
    try:
        return date(int(text[:4]), int(text[4:6]), int(text[6:]))
    except ValueError:
        return None


def _field(row: dict[str, Any], name: str) -> Any:
    value = row.get(name)
    return value.get("value") if isinstance(value, dict) else value


def _event_item(spec: SourceSpec, *, external_id: str, url: str, title: str, abstract: str,
                start: date, end: date | None, deadlines: list[tuple[str, date]], location: str,
                fmt: str, event_type: str, official_id: str, organizer: str) -> Item:
    today = _today()
    upcoming = sorted((when, kind) for kind, when in deadlines if when >= today)
    # The last open route (often virtual registration) is the last chance to act.
    deadline = upcoming[-1][0] if upcoming else None
    return Item(
        source=spec.name,
        section=spec.section,
        external_id=external_id,
        url=url,
        title=title,
        abstract=abstract[:8000],
        published_at=None,
        metadata={
            "record_type": "event",
            "event_type": event_type,
            "status": "open",
            "official": True,
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "event_start": start.isoformat(),
            "event_end": (end or start).isoformat(),
            "deadline": deadline.isoformat() if deadline else None,
            "deadline_timezone": "",
            "deadlines": [{"type": kind, "date": when.isoformat()} for when, kind in upcoming],
            "location": location,
            "format": fmt,
            "organizer": organizer,
            "official_id": official_id,
        },
    )


class EMBLEventsSource:
    """EMBL conferences and courses, read from the organiser's REST listing."""

    MAX_PAGES = 5

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        del days
        today = _today()
        horizon = today + timedelta(days=spec.lookahead_days)
        items: list[Item] = []
        for page in range(1, self.MAX_PAGES + 1):
            rows = _get_json(spec.url or EMBL_URL, {
                "per_page": "100", "page": str(page), "orderby": "date", "order": "desc",
                "_fields": EMBL_FIELDS,
            })
            if not isinstance(rows, list):
                raise RuntimeError(f"{spec.name} returned an invalid event listing")
            upcoming_on_page = 0
            for row in rows:
                item = self._item(spec, row, today, horizon)
                end = _ymd(_field(row, "vf_event_end_date")) or _ymd(_field(row, "vf_event_start_date"))
                if end is not None and end >= today:
                    upcoming_on_page += 1
                if item is not None:
                    items.append(item)
            # Posts are ordered by publication, not by event date. A full page
            # with no upcoming event means the rest are older, finished events.
            if len(rows) < 100 or upcoming_on_page == 0:
                break
        return items

    def _item(self, spec: SourceSpec, row: dict[str, Any], today: date, horizon: date) -> Item | None:
        if str(_field(row, "vf_event_canceled") or "").casefold() == "yes":
            return None
        start = _ymd(_field(row, "vf_event_start_date"))
        end = _ymd(_field(row, "vf_event_end_date"))
        title = _plain((row.get("title") or {}).get("rendered"))
        link = canonicalize_url(str(row.get("link") or ""))
        if start is None or not title or not link or start < today or start > horizon:
            return None
        deadlines = [
            (kind, when) for kind, name in (
                ("abstract", "vf_event_submission_closing"),
                ("registration", "vf_event_registration_closing"),
                ("on-site registration", "vf_event_registration_closing_on-site"),
            ) if (when := _ymd(_field(row, name))) is not None
        ]
        if deadlines and all(when < today for _, when in deadlines):
            return None  # Every route to attend has closed.
        location_value = row.get("vf_event_location") or {}
        places = location_value.get("formatted") if isinstance(location_value, dict) else []
        places = [str(place) for place in places or [] if str(place).strip()]
        physical = [place for place in places if place.casefold() != "virtual"]
        fmt = "hybrid" if physical and len(physical) < len(places) else "online" if places and not physical else "in_person"
        venue = _plain(_field(row, "vf_event_venue"))
        location = ", ".join(dict.fromkeys([*physical, *([venue] if venue else [])])) or "Virtual"
        if "heidelberg" in location.casefold() and "germany" not in location.casefold():
            location += ", Germany"
        kind = str(_field(row, "vf_event_event_type") or "").casefold()
        label = _plain(_field(row, "vf_event_displayed"))
        summary = _plain(_field(row, "vf_event_summary"))
        guid = html.unescape(str((row.get("guid") or {}).get("rendered") or link))
        return _event_item(
            spec,
            # The RSS adapter hashed the same guid, so saved and rated events keep their identity.
            external_id=hashlib.sha1(guid.encode("utf-8")).hexdigest()[:16],
            url=link, title=title, abstract=" ".join(part for part in (label, summary) if part),
            start=start, end=end, deadlines=deadlines, location=location, fmt=fmt,
            event_type=_event_type(title, f"{kind} {label} {summary}"),
            official_id=guid, organizer=label or "EMBL",
        )


class EMBOEventsSource:
    """EMBO workshops, courses, lectures and symposia held worldwide."""

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        del days
        payload = _get_json(spec.url or EMBO_URL)
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError(f"{spec.name} returned an invalid event listing")
        today = _today()
        horizon = today + timedelta(days=spec.lookahead_days)
        items = []
        for row in rows:
            item = self._item(spec, row, today, horizon)
            if item is not None:
                items.append(item)
        return items

    def _item(self, spec: SourceSpec, row: dict[str, Any], today: date, horizon: date) -> Item | None:
        # The listing also names a contact person; only event facts are kept.
        start = _ymd(row.get("start_date"))
        end = _ymd(row.get("end_date"))
        title = _plain(row.get("title"))
        url = canonicalize_url(str(row.get("website") or ""))
        event_id = str(row.get("event_id") or row.get("mosform_internal_id") or "").strip()
        if start is None or not title or not url or not event_id or start < today or start > horizon:
            return None
        if str(row.get("registration_status") or "").casefold() == "closed":
            return None
        deadline = _ymd(row.get("deadline_date"))
        deadlines = [("deadline", deadline)] if deadline else []
        if deadline is not None and deadline < today:
            return None
        medium = str(row.get("event_medium") or "").casefold()
        fmt = "online" if medium == "virtual" else "hybrid" if medium == "hybrid" else "in_person"
        place = ", ".join(part for part in (_plain(row.get("city")), _plain(row.get("country"))) if part)
        kind = _plain(row.get("event_type"))
        topics = " ".join(part for part in (_plain(row.get("topic_areas")), _plain(row.get("event_keywords"))) if part)
        return _event_item(
            spec, external_id=f"embo-{event_id}", url=url, title=title,
            abstract=" · ".join(part for part in (kind, topics) if part),
            start=start, end=end, deadlines=deadlines,
            location=place or ("Virtual" if fmt == "online" else ""), fmt=fmt,
            # An unknown kind stays empty so it is reviewed, not filtered out.
            event_type=_event_type(title, kind.replace("Training", "course")),
            official_id=event_id, organizer=kind or "EMBO",
        )
