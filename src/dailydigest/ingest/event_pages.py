"""Official event listings that are published only as web pages.

ISSCR exposes its Squarespace collection as JSON; the other organisers publish
server-rendered HTML with stable labels. Each adapter reads the organiser's own
listing (and, where the listing lacks the venue, deadlines or topic text, a
bounded number of the organiser's event pages). Rows that cannot be read are skipped; a
listing that yields no recognisable rows raises, so a redesigned page shows as
a failed source instead of an empty success.
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse
from zoneinfo import ZoneInfo

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ..config import user_local_date
from ..models import Item, SourceSpec
from ._retry import transient_http_error
from .events_api import _event_item, _plain
from .events_rss import _event_type
from .rss import canonicalize_url

logger = logging.getLogger(__name__)

# A plain descriptive agent: ISSCR's robots.txt refuses AI-crawler agents.
USER_AGENT = "dailydigest/0.1 (+https://github.com/Antimonyleo/daily_digests)"

ISSCR_URL = "https://www.isscr.org/upcoming-events"
KEYSTONE_URL = "https://www.keystonesymposia.org/conferences/conference-listing?time_range=today-forward"
CSHL_URL = "https://meetings.cshl.edu/meetings"
NATURE_URL = "https://conferences.nature.com/"
WELLCOME_CS_URL = "https://coursesandconferences.wellcomeconnectingscience.org/our-events/conferences/"
CRS_URL = "https://www.controlledreleasesociety.org/events"
FEBS_URL = "https://www.febs.org/listings/events-calendar/"
ECI_URL = "https://engconf.org/conferences/"
BPS_URL = "https://www.biophysics.org/events-calendar"
FASEB_SRC_URL = "https://www.faseb.org/meetings-and-events/science-research-conferences"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(transient_http_error),
    reraise=True,
)
def _get_text(url: str, params: dict[str, str] | None = None) -> str:
    with httpx.Client(timeout=30.0, follow_redirects=True,
                      headers={"Accept": "text/html,application/xhtml+xml", "User-Agent": USER_AGENT}) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.text


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(transient_http_error),
    reraise=True,
)
def _get_json(url: str, params: dict[str, str] | None = None) -> Any:
    with httpx.Client(timeout=30.0, follow_redirects=True,
                      headers={"Accept": "application/json", "User-Agent": USER_AGENT}) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.json()


def _today() -> date:
    return user_local_date()


# -- shared parsing helpers ---------------------------------------------------

_MONTH_RE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?(?![a-z])", re.IGNORECASE)
_MONTHS = {name: number for number, name in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), start=1)}
_DAY_RE = re.compile(r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?(?!\d)")
_YEAR_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def _date_parts(text: str) -> tuple[int | None, int | None, int | None]:
    month = _MONTH_RE.search(text)
    year = _YEAR_RE.search(text)
    day = _DAY_RE.search(_YEAR_RE.sub(" ", text))
    return (
        _MONTHS[month.group(1).casefold()] if month else None,
        int(day.group(1)) if day else None,
        int(year.group(1)) if year else None,
    )


def _date_range(text: str) -> tuple[date | None, date | None]:
    """Read the date-range forms the organisers print.

    "January 11–14, 2027", "March 29–April 01, 2027", "October 14 – 16, 2026",
    "Oct 27 - Oct 31, 2026", "4–6 November 2026", "29 November – 3 December 2026"
    and single days. A range whose end precedes its start is a typo on the
    page ("31–4 May 2027"), and is rejected rather than guessed.
    """
    cleaned = re.sub(r"\s+", " ", html.unescape(text).replace("–", "-").replace("—", "-")).strip()
    sides = re.split(r"\s*-\s*|\s+to\s+", cleaned, maxsplit=1)
    r_month, r_day, r_year = _date_parts(sides[-1])
    if r_year is None or r_day is None:
        return None, None
    if len(sides) == 1:
        l_month, l_day, l_year = r_month, r_day, r_year
    else:
        l_month, l_day, l_year = _date_parts(sides[0])
        if l_day is None:
            return None, None
        if r_month is None:
            r_month = l_month
        if l_month is None:
            l_month = r_month
    if l_month is None or r_month is None:
        return None, None
    explicit_start_year = l_year is not None
    l_year = l_year or r_year
    if not explicit_start_year and l_month > r_month:
        l_year -= 1  # "28 December - 2 January 2027" crosses the year.
    try:
        start, end = date(l_year, l_month, l_day), date(r_year, r_month, r_day)
    except ValueError:
        return None, None
    if end < start:
        return None, None
    return start, end


def _single_date(text: str) -> date | None:
    start, end = _date_range(text)
    return start if start is not None and start == end else None


# Region and postal parts are dropped so a venue reads "City, Country".
_REGIONS = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware",
    "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky",
    "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
    "missouri", "montana", "nebraska", "nevada", "new hampshire", "new jersey", "new mexico",
    "new york", "north carolina", "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
    "rhode island", "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming", "district of columbia",
    "alberta", "british columbia", "ontario", "quebec",
}


def _city_country(text: str) -> str:
    parts = []
    for raw in _plain(text).split(","):
        if re.match(r"\s*\d{1,3}[a-z]?\s+\D", raw):
            continue  # a street address ("4 Pl. Jussieu"), not a place name
        part = re.sub(r"\s+", " ", re.sub(r"\b\d[\d\s-]*\b", " ", raw)).strip(" .")  # postal codes
        if part and part.casefold() not in _REGIONS:
            parts.append(part)
    # Two- or three-letter state codes sit between city and country ("Banff, AB, Canada").
    parts = [part for index, part in enumerate(parts)
             if index == len(parts) - 1 or not re.fullmatch(r"[A-Z]{2,3}", part)]
    deduped = [part for index, part in enumerate(parts)
               if index == 0 or part.casefold() != parts[index - 1].casefold()]  # "Singapore, Singapore"
    return ", ".join(deduped[-2:])


# "Georgia" is left out: it is also a country, and the bare name stays ambiguous.
_US_STATES = _REGIONS - {"alberta", "british columbia", "ontario", "quebec", "georgia"}
_US_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS",
    "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC",
    "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}
_CA_PROVINCES = {"alberta", "british columbia", "ontario", "quebec", "manitoba", "nova scotia",
                 "new brunswick", "saskatchewan", "ab", "bc", "on", "pq", "qc", "mb", "ns", "nb", "sk"}


def _place_with_country(text: str) -> str:
    """Read "Estes Park, Colorado" or "Montreal, P.Q., Canada" as "City, Country".

    Some organisers end a North American venue at its state or province; the
    country is added so location preferences ("North America") still match.
    """
    cleaned = re.sub(r"\b([A-Z])\.([A-Z])\.?", r"\1\2", _plain(text))  # "P.Q." -> "PQ"
    # "Washington" alone reads as the state, which _city_country drops.
    cleaned = re.sub(r"\bWashington,\s*DC\b", "Washington DC", cleaned)
    parts = [part.strip(" .") for part in cleaned.split(",") if part.strip(" .")]
    if not parts:
        return ""
    last = parts[-1]
    if last.casefold() in _US_STATES | {"washington dc"} or last in _US_CODES:
        parts.append("United States")
    elif last.casefold() in _CA_PROVINCES:
        parts.append("Canada")
    return _city_country(", ".join(parts))


def _format(text: str) -> str:
    lowered = text.casefold()
    if "hybrid" in lowered:
        return "hybrid"
    if not re.search(r"\b(?:virtual|online|webinar)\b", lowered):
        return "in_person"
    physical = re.sub(r"\b(?:virtual|online|webinar|only|and)\b|[^a-z]+", "", lowered)
    return "hybrid" if physical else "online"


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha1(value.encode('utf-8')).hexdigest()[:16]}"


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PERSON_RE = re.compile(
    r"\b(?:Prof|Professor|Dr|Chair(?:s|ed)?|Co-chair(?:s|ed)?|Organi[sz]ers?|Organi[sz]ed by|Contact|Speakers?)\b",
    re.IGNORECASE,
)


def _scrub(text: str) -> str:
    """Drop sentences that name people or give contact details."""
    flattened = re.sub(r"\b(Prof|Dr|Mr|Ms|Mrs)\.", r"\1", _plain(text))
    kept = [
        sentence for sentence in re.split(r"(?<=[.!?])\s+", flattened)
        if sentence and not _EMAIL_RE.search(sentence) and not _PERSON_RE.search(sentence)
    ]
    return " ".join(kept).strip()


def _in_window(start: date | None, today: date, horizon: date) -> bool:
    return start is not None and today <= start <= horizon


def _attr(tag: str, name: str) -> str:
    match = re.search(rf"\b{name}\s*=\s*([\"'])(.*?)\1", tag, flags=re.IGNORECASE | re.DOTALL)
    return html.unescape(match.group(2)).strip() if match else ""


def _first(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else ""


# -- ISSCR ----------------------------------------------------------------------

def _zone(categories: list[str], longitude: object) -> timezone | ZoneInfo:
    for category in categories:
        if not category.casefold().startswith("tz-"):
            continue
        name = category[3:]
        for candidate in (name, "/".join(part[:1].upper() + part[1:] for part in name.split("/"))):
            try:
                return ZoneInfo(candidate)
            except Exception:  # noqa: BLE001 - an unknown zone name falls back below
                continue
    try:
        # Squarespace stores the venue's map position; its longitude gives the local day.
        return timezone(timedelta(hours=max(-12, min(14, round(float(longitude) / 15)))))
    except (TypeError, ValueError):
        return timezone.utc


class ISSCREventsSource:
    """ISSCR meetings, symposia and webinars from the Squarespace collection JSON."""

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        del days
        url = spec.url or ISSCR_URL
        payload = _get_json(url, None if "format=" in url else {"format": "json"})
        rows = payload.get("upcoming") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError(f"{spec.name} returned an invalid event listing")
        today = _today()
        horizon = today + timedelta(days=spec.lookahead_days)
        items = []
        for row in rows:
            item = self._item(spec, row, today, horizon) if isinstance(row, dict) else None
            if item is not None:
                items.append(item)
        return items

    def _item(self, spec: SourceSpec, row: dict[str, Any], today: date, horizon: date) -> Item | None:
        title = _plain(row.get("title"))
        path = str(row.get("fullUrl") or "")
        event_id = str(row.get("id") or "").strip()
        location = row.get("location") if isinstance(row.get("location"), dict) else {}
        categories = [str(value) for value in row.get("categories") or []]
        tags = [str(value) for value in row.get("tags") or []]
        try:
            zone = _zone(categories, location.get("mapLng"))
            start = datetime.fromtimestamp(int(row["startDate"]) / 1000, zone).date()
            end = datetime.fromtimestamp(int(row.get("endDate") or row["startDate"]) / 1000, zone).date()
        except (KeyError, TypeError, ValueError, OverflowError, OSError):
            return None
        if not title or not path or not event_id or not _in_window(start, today, horizon):
            return None
        labels = " ".join([*categories, *tags])
        if re.search(r"\b(?:cancell?ed|postponed)\b", f"{title} {labels}", flags=re.IGNORECASE):
            return None
        markers = {value.casefold() for value in [*tags, *categories]}
        fmt = ("hybrid" if "hybrid" in markers
               else "online" if markers & {"webinar", "virtual", "online"} else "in_person")
        city = _plain(location.get("addressLine2")).split(",")[0].strip()
        place = ", ".join(part for part in (city, _plain(location.get("addressCountry"))) if part)
        kinds = [value for value in categories if not value.casefold().startswith("tz-")]
        themes = [value for value in tags if not re.fullmatch(r"in-person|webinar|virtual|hybrid|\d{4} symposia",
                                                              value, re.IGNORECASE)]
        abstract = " · ".join(part for part in (
            _scrub(row.get("excerpt") or ""), ", ".join(kinds), ", ".join(themes)) if part)
        return _event_item(
            spec, external_id=f"isscr-{event_id}", url=canonicalize_url(urljoin(ISSCR_URL, path)),
            title=title, abstract=abstract, start=start, end=end, deadlines=[],
            location=place or ("Online" if fmt == "online" else ""), fmt=fmt,
            event_type=_event_type(title, " ".join(kinds)), official_id=event_id, organizer="ISSCR",
        )


# -- Keystone Symposia ----------------------------------------------------------

class KeystoneEventsSource:
    """Keystone Symposia conferences; deadlines and topic text come from each meeting page."""

    MAX_DETAIL_PAGES = 25

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        del days
        page = _get_text(spec.url or KEYSTONE_URL)
        rows = [chunk.split("</tr>", 1)[0] for chunk in re.split(r"<tr\s+class=\"meeting-row[^\"]*\"", page)[1:]]
        if not rows:
            raise RuntimeError(f"{spec.name} returned no recognisable meeting rows")
        today = _today()
        horizon = today + timedelta(days=spec.lookahead_days)
        parsed = [row for row in (self._row(raw) for raw in rows) if row is not None]
        upcoming = sorted((row for row in parsed if _in_window(row["start"], today, horizon)),
                          key=lambda row: row["start"])
        items, seen = [], set()
        for index, row in enumerate(upcoming):
            if row["code"] in seen:
                continue
            seen.add(row["code"])
            detail = self._detail(row["url"]) if index < self.MAX_DETAIL_PAGES else None
            items.append(self._item(spec, row, detail))
        return items

    @staticmethod
    def _row(raw: str) -> dict[str, Any] | None:
        if "soldout-gradient" in raw or re.search(r"\b(?:cancell?ed|postponed)\b", _plain(raw), re.IGNORECASE):
            return None
        link = re.search(r"<div class=\"meeting-title\">\s*<a\s+([^>]*)>(.*?)</a>", raw, re.DOTALL)
        if not link:
            return None
        url = canonicalize_url(_attr(link.group(1), "href"))
        code_match = re.search(r"/meeting/([A-Za-z0-9]+)$", url)
        title = re.sub(r"\s*\([A-Z]\d{1,2}\)$", "", _plain(link.group(2)))
        start, end = _date_range(_plain(_first(r"<span class=\"date\">(.*?)</span>", raw)))
        if not url or not code_match or not title or start is None:
            return None
        early = None
        calendar = re.search(r"class=\"add-to-calendar-button\"[^>]*href=\"([^\"]+)\"|href=\"([^\"]+)\"[^>]*"
                             r"class=\"add-to-calendar-button\"", raw)
        if calendar:
            query = parse_qs(urlparse(html.unescape(calendar.group(1) or calendar.group(2))).query)
            if "early registration" in " ".join(query.get("title", [])).casefold():
                early = _iso_date(query.get("start", [""])[0])
        return {
            "url": url, "code": code_match.group(1), "title": title, "start": start, "end": end,
            "location": _city_country(_first(r"<div class=\"meeting-location\">(.*?)</div>", raw)),
            "keywords": [_plain(value).lstrip("#").strip()
                         for value in re.findall(r"<div class=\"mtg-keyword\">(.*?)</div>", raw, re.DOTALL)],
            "joint": _plain(_first(r"<div class=\"joint-title\">.*?<a[^>]*>(.*?)</a>", raw)),
            "early": early,
        }

    @staticmethod
    def _detail(url: str) -> dict[str, Any] | None:
        try:
            page = _get_text(url)
        except Exception as error:  # noqa: BLE001 - the listing still carries the core facts
            logger.debug("Keystone meeting page %s unavailable: %s", url, error)
            return None
        deadlines = []
        for style, body in re.findall(r"<div class=\"deadline-row\"([^>]*)>(.*?)</div>", page, re.DOTALL):
            if "display: none" in style or "line-through" in body:
                continue  # hidden, or struck through because it has passed
            label, _, value = _plain(body).partition(":")
            when = _single_date(value)
            if when is not None:
                kind = re.sub(r"\s*deadline$", "", label.strip(), flags=re.IGNORECASE).casefold()
                deadlines.append((kind or "deadline", when))
        summary_html = _first(r"<div class=\"section meeting-summary\"[^>]*>(.*?)(?:<div class=\"section |$)", page)
        about = _first(r"<div class=\"section-inner\">(.*)", summary_html)
        # The "Joint Meeting" preamble repeats the listing's joint-meeting link.
        summary = re.sub(r"^.*?About This Meeting:\s*", "", _scrub(about), count=1, flags=re.IGNORECASE)
        return {"deadlines": deadlines, "summary": summary[:2000]}

    @staticmethod
    def _item(spec: SourceSpec, row: dict[str, Any], detail: dict[str, Any] | None) -> Item:
        if detail is not None and detail["deadlines"]:
            deadlines = detail["deadlines"]
        else:
            deadlines = [("early registration", row["early"])] if row["early"] else []
        summary = detail["summary"] if detail else ""
        topics = ", ".join(value for value in row["keywords"] if value)
        abstract = " · ".join(part for part in (
            summary, topics and f"Topics: {topics}", row["joint"] and f"Joint with: {row['joint']}") if part)
        return _event_item(
            spec, external_id=f"keystone-{row['code']}", url=row["url"], title=row["title"],
            abstract=abstract, start=row["start"], end=row["end"], deadlines=deadlines,
            location=row["location"], fmt="in_person",
            event_type=_event_type(row["title"], summary), official_id=row["code"],
            organizer="Keystone Symposia",
        )


def _iso_date(value: str) -> date | None:
    match = re.match(r"\s*(\d{4})-(\d{2})-(\d{2})", value or "")
    if not match:
        return None
    try:
        return date(*(int(part) for part in match.groups()))
    except ValueError:
        return None


# -- Cold Spring Harbor Laboratory meetings --------------------------------------

class CSHLEventsSource:
    """CSHL scientific meetings, all held at Cold Spring Harbor, New York."""

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        del days
        page = _get_text(spec.url or CSHL_URL)
        rows = re.findall(r"<tr\s+class=\"science-type-row\"[^>]*>(.*?)</tr>", page, re.DOTALL)
        if not rows:
            raise RuntimeError(f"{spec.name} returned no recognisable meeting rows")
        today = _today()
        horizon = today + timedelta(days=spec.lookahead_days)
        items, seen = [], set()
        for row in rows:
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
            link = re.search(r"<a\s+([^>]*)>(.*?)</a>", cells[0], re.DOTALL) if cells else None
            if len(cells) < 2 or not link:
                continue
            url = canonicalize_url(urljoin(CSHL_URL, _attr(link.group(1), "href")))
            code = (parse_qs(urlparse(url).query).get("meet") or [""])[0].strip()
            title = _plain(link.group(2))
            start, end = _date_range(_plain(cells[1]))
            if not code or not title or not _in_window(start, today, horizon):
                continue
            if re.search(r"\b(?:cancell?ed|postponed)\b", _plain(row), re.IGNORECASE):
                continue
            # Meeting codes repeat every year the meeting runs.
            external_id = f"cshl-{code}-{start.year}"
            if external_id in seen:
                continue
            seen.add(external_id)
            abstracts = _single_date(_plain(cells[2])) if len(cells) > 2 else None
            items.append(_event_item(
                spec, external_id=external_id, url=url, title=title,
                abstract="Cold Spring Harbor Laboratory meeting",
                start=start, end=end, deadlines=[("abstract", abstracts)] if abstracts else [],
                location="Cold Spring Harbor, United States", fmt="in_person",
                event_type=_event_type(title), official_id=code, organizer="Cold Spring Harbor Laboratory",
            ))
        return items


# -- Nature Conferences -------------------------------------------------------------

class NatureConferencesSource:
    """Springer Nature conferences; the page keeps past entries, so dates decide."""

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        del days
        listing = spec.url or NATURE_URL
        page = _get_text(listing)
        blocks = re.findall(r"<h4[^>]*>(.*?)</h4>\s*<p[^>]*>(.*?)</p>", page, re.DOTALL)
        blocks = [(head, body) for head, body in blocks if re.search(r"Date:", _plain(body))]
        if not blocks:
            raise RuntimeError(f"{spec.name} returned no recognisable conference entries")
        today = _today()
        horizon = today + timedelta(days=spec.lookahead_days)
        items, seen = [], set()
        for head, body in blocks:
            lines = [_plain(line) for line in re.split(r"<br\s*/?>", body)]
            fields = {}
            for line in lines:
                label, sep, value = line.partition(":")
                if sep:
                    fields[label.strip().casefold()] = value.strip()
            title = _plain(head)
            start, end = _date_range(fields.get("date", ""))
            if not title or not _in_window(start, today, horizon):
                continue
            if re.search(r"\b(?:cancell?ed|postponed)\b", f"{title} {_plain(body)}", re.IGNORECASE):
                continue
            href = _attr(_first(r"<a\s+([^>]*)>", head), "href")
            url = canonicalize_url(urljoin(listing, href)) if href.startswith(("http", "/")) else canonicalize_url(listing)
            # The same registration site can serve several editions; the year separates them.
            external_id = _stable_id("nature-conf", f"{url}|{title.casefold()}|{start.year}")
            if external_id in seen:
                continue
            seen.add(external_id)
            location = fields.get("location", "")
            partner = fields.get("partner", "")
            fmt = _format(fields.get("format", "") or location)
            items.append(_event_item(
                spec, external_id=external_id, url=url, title=title,
                abstract=f"Partner: {partner}" if partner else "",
                start=start, end=end, deadlines=[],
                location="" if location.casefold() in {"tbd", "tba"} else _city_country(location),
                fmt=fmt, event_type=_event_type(title), official_id=url, organizer="Nature Conferences",
            ))
        return items


# -- Wellcome Connecting Science -------------------------------------------------------

_ROUTE_RE = re.compile(r"([A-Za-z][A-Za-z \-]*?)\s*:\s*(Closed|\d{1,2}\s+[A-Za-z]+\s+\d{4})", re.IGNORECASE)


class WellcomeConnectingScienceSource:
    """Wellcome Connecting Science conferences at the Wellcome Genome Campus."""

    MAX_DETAIL_PAGES = 15

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        del days
        listing = spec.url or WELLCOME_CS_URL
        page = _get_text(listing)
        blocks = re.findall(r"<div class=\"event-details\">(.*?)</div>", page, re.DOTALL)
        if not blocks:
            raise RuntimeError(f"{spec.name} returned no recognisable event entries")
        today = _today()
        horizon = today + timedelta(days=spec.lookahead_days)
        items, details = [], 0
        for block in blocks:
            link = re.search(r"<h3>\s*<a\s+([^>]*)>(.*?)</a>", block, re.DOTALL)
            if not link:
                continue
            url = canonicalize_url(urljoin(listing, _attr(link.group(1), "href")))
            title = _plain(link.group(2))
            start, end = _date_range(_plain(_first(r"<strong>(.*?)</strong>", block)))
            if not title or not _in_window(start, today, horizon):
                continue
            route_text = next((_plain(para) for para in re.findall(r"<p>(.*?)</p>", block, re.DOTALL)
                               if _plain(para).casefold().startswith("deadlines")), "")
            routes = [(label.strip().casefold(), value) for label, value in _ROUTE_RE.findall(route_text)]
            deadlines = [(label, when) for label, value in routes
                         if (when := _single_date(value)) is not None]
            if re.search(r"all closed|cancell?ed|postponed", route_text, re.IGNORECASE):
                continue
            if routes and not any(when >= today for _, when in deadlines):
                continue  # Every listed route to attend has closed.
            detail = None
            if details < self.MAX_DETAIL_PAGES:
                details += 1
                detail = self._detail(url)
            place = detail["location"] if detail else ""
            fmt = _format(place) if place else ("hybrid" if "virtual" in route_text.casefold() else "in_person")
            items.append(_event_item(
                spec, external_id=_stable_id("wcs", url), url=url, title=title,
                abstract=detail["summary"] if detail else "", start=start, end=end, deadlines=deadlines,
                location=self._location(place), fmt=fmt,
                # The listing page is the organiser's own category for these events.
                event_type=_event_type(title, "course" if "course" in urlparse(listing).path else "conference"),
                official_id=url, organizer="Wellcome Connecting Science",
            ))
        return items

    @staticmethod
    def _location(place: str) -> str:
        physical = re.sub(r"\s*(?:\band\b|&)?\s*\b(?:virtual|online)\b\s*", "", place, flags=re.IGNORECASE).strip(" ,")
        if not physical:
            return "Online" if place else ""
        if "wellcome genome campus" in physical.casefold():
            return "Hinxton, United Kingdom"  # The campus address given in the site footer.
        return _city_country(physical)

    @staticmethod
    def _detail(url: str) -> dict[str, str] | None:
        try:
            page = _get_text(url)
        except Exception as error:  # noqa: BLE001 - the listing still carries the core facts
            logger.debug("Wellcome Connecting Science page %s unavailable: %s", url, error)
            return None
        location = _plain(_first(r"<div class=\"event-location\">(.*?)</div>", page))
        subtitle = _plain(_first(r"</div><!-- flex-container -->\s*<h3>(.*?)</h3>", page))
        summary_html = _first(r"<h3>\s*Summary\s*</h3>(.*?)(?:<input type=\"checkbox\"|$)", page)
        paragraphs = [_plain(para) for para in re.findall(r"<p[^>]*>(.*?)</p>", summary_html, re.DOTALL)]
        summary = _scrub(" ".join(para for para in paragraphs if para))
        return {"location": location,
                "summary": " · ".join(part for part in (subtitle, summary[:2000]) if part)}


# -- Controlled Release Society ------------------------------------------------------------

class CRSEventsSource:
    """Controlled Release Society meetings, workshops and webinars (the "Upcoming Events" view)."""

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        del days
        listing = spec.url or CRS_URL
        page = _get_text(listing)
        _, found, content = page.partition("<div class=\"view-content\">")
        content = re.split(r"<div class=\"sidebar|<nav class=\"pager|<ul class=\"pager", content)[0]
        rows = re.split(r"<div class=\"views-row\">", content)[1:] if found else []
        if not rows:
            raise RuntimeError(f"{spec.name} returned no recognisable event rows")
        today = _today()
        horizon = today + timedelta(days=spec.lookahead_days)
        items = []
        for row in rows:
            link = re.search(r"views-field-title.*?<a\s+([^>]*)>(.*?)</a>", row, re.DOTALL)
            if not link:
                continue
            url = canonicalize_url(urljoin(listing, _attr(link.group(1), "href")))
            title = _plain(link.group(2))
            # The labels show the society's local calendar day; the attributes are UTC.
            days_shown = [_us_date(_plain(label)) for label in re.findall(r"<time[^>]*>(.*?)</time>", row)]
            days_shown = [value for value in days_shown if value is not None]
            start = days_shown[0] if days_shown else None
            end = days_shown[-1] if days_shown else None
            if "/events/" not in url or not title or not _in_window(start, today, horizon):
                continue
            if re.search(r"\b(?:cancell?ed|postponed)\b", title, re.IGNORECASE):
                continue
            location = _plain(_first(r"views-field-field-event-location.*?<div class=\"field-content\">(.*?)</div>", row))
            kind = _attr(_first(r"(<span[^>]*data-event-type[^>]*>)", row), "data-event-type")
            fmt = _format(f"{location} {kind}") if location or kind.casefold() == "webinar" else "in_person"
            place = "Online" if fmt == "online" else re.sub(r",\s*\d[\d\s-]*(?=,|$)", "", location).strip()
            items.append(_event_item(
                spec, external_id=_stable_id("crs", url), url=url, title=title,
                abstract=" · ".join(part for part in ("Controlled Release Society", kind) if part),
                start=start, end=max(end, start) if end else start, deadlines=[],
                location=place, fmt=fmt,
                event_type=_event_type(title, kind.replace("Annual", "annual meeting")),
                official_id=urlparse(url).path, organizer="Controlled Release Society",
            ))
        return items


def _us_date(value: str) -> date | None:
    match = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", value.strip())
    if not match:
        return None
    try:
        return date(int(match.group(3)), int(match.group(1)), int(match.group(2)))
    except ValueError:
        return None


# -- FEBS ---------------------------------------------------------------------------------

class FEBSEventsSource:
    """FEBS events and the external meetings FEBS lists, across several calendar pages."""

    MAX_PAGES = 4

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        del days
        listing = (spec.url or FEBS_URL).rstrip("/") + "/"
        today = _today()
        horizon = today + timedelta(days=spec.lookahead_days)
        items, seen = [], set()
        for number in range(1, self.MAX_PAGES + 1):
            url = listing if number == 1 else f"{listing}page/{number}/"
            try:
                page = _get_text(url)
            except httpx.HTTPStatusError as error:
                if number > 1 and error.response.status_code == 404:
                    break  # Past the last calendar page.
                raise
            articles = re.findall(r"<article\s+class=\"([^\"]*\bpost-(\d+)[^\"]*)\"[^>]*>(.*?)</article>",
                                  page, re.DOTALL)
            if not articles:
                if number == 1:
                    raise RuntimeError(f"{spec.name} returned no recognisable event entries")
                break
            beyond = 0
            for classes, post_id, body in articles:
                if "events" not in classes.split() or post_id in seen:
                    continue
                seen.add(post_id)
                item, after_horizon = self._item(spec, post_id, body, today, horizon)
                beyond += after_horizon
                if item is not None:
                    items.append(item)
            if beyond == len(articles) or f"page/{number + 1}/" not in page:
                break  # The calendar runs in date order; later pages are further out.
        return items

    @staticmethod
    def _item(spec: SourceSpec, post_id: str, body: str, today: date, horizon: date) -> tuple[Item | None, bool]:
        header = _first(r"<header[^>]*>(.*?)</header>", body)
        paragraphs = [_plain(para) for para in re.findall(r"<p[^>]*>(.*?)</p>", header, re.DOTALL)]
        title = _plain(_first(r"<h2[^>]*class=\"[^\"]*entry-title[^\"]*\"[^>]*>(.*?)</h2>", body))
        start, end = _date_range(paragraphs[0]) if paragraphs else (None, None)
        if start is not None and start > horizon:
            return None, True
        website = re.search(r"<a\s+([^>]*)>\s*Event website\s*</a>", body, re.IGNORECASE)
        url = canonicalize_url(_attr(website.group(1), "href")) if website else ""
        if not title or not url.startswith("http") or not _in_window(start, today, horizon):
            return None, False
        if re.search(r"\b(?:cancell?ed|postponed)\b", title, re.IGNORECASE):
            return None, False
        place = next((para for para in paragraphs[1:] if para and para.casefold() != "event website"), "")
        categories = [_plain(value) for value in re.findall(r"class=\"event__category[^\"]*\"[^>]*>(.*?)</span>", body)]
        febs_event = any(value.casefold() == "febs event" for value in categories)
        summary_html = _first(r"<div[^>]*class=\"entry-summary\"[^>]*>(.*?)</div>", body)
        paras = [_plain(para) for para in re.findall(r"<p[^>]*>(.*?)</p>", summary_html, re.DOTALL)]
        summary = _scrub(" ".join(para for para in paras if para))[:2000]
        kinds = " ".join(categories).replace("Hands-on", "hands-on course")
        fmt = _format(place) if re.search(r"online|virtual|hybrid", place, re.IGNORECASE) else "in_person"
        return _event_item(
            spec, external_id=f"febs-{post_id}", url=url, title=title,
            abstract=" · ".join(part for part in (", ".join(categories), summary) if part),
            start=start, end=end, deadlines=[],
            location="Online" if fmt == "online" else _city_country(place), fmt=fmt,
            event_type=_event_type(title, f"{kinds} {summary}"), official_id=post_id,
            # External meetings are listed by FEBS; their facts belong to the organiser's site.
            organizer="FEBS" if febs_event else "External organiser (listed by FEBS)",
        ), False


# -- Engineering Conferences International --------------------------------------------------

_ROMAN_RE = re.compile(r"(?=[IVXL])L?X{0,3}(?:IX|IV|V?I{0,3})")
_SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of", "on", "or", "the", "to",
                "via", "vs", "with"}


def _title_case(text: str) -> str:
    """ECI prints titles in capitals; read them in title case, keeping numerals and acronyms."""
    if re.search(r"[a-z]", text):
        return text
    words = []
    for index, word in enumerate(text.split()):
        core = word.strip("():,;–—-&")
        if not core or re.search(r"\d", core) or word.startswith("(") or _ROMAN_RE.fullmatch(core):
            words.append(word)  # "IM2AM", "(ULSIC", "XXIX" stay as printed
        elif index and core.casefold() in _SMALL_WORDS:
            words.append(word.casefold())
        else:
            words.append("-".join(part[:1] + part[1:].casefold() for part in word.split("-")))
    return " ".join(words)


class ECIEventsSource:
    """Engineering Conferences International meetings, all read from the one conference list.

    Each entry gives the dates, title and code, chairs (not kept) and venue,
    so no conference page is fetched.
    """

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        del days
        listing = spec.url or ECI_URL
        page = _get_text(listing)
        _, found, content = page.partition("<div class=\"entry-content\">")
        content = content.split("<!-- .entry-content -->", 1)[0]
        rows = [row for row in (self._row(listing, para)
                                for para in re.findall(r"<p>(.*?)</p>", content, re.DOTALL)) if row]
        if not found or not rows:
            raise RuntimeError(f"{spec.name} returned no recognisable conference entries")
        today = _today()
        horizon = today + timedelta(days=spec.lookahead_days)
        items, seen = [], set()
        for row in rows:
            if row["code"] in seen or not _in_window(row["start"], today, horizon):
                continue
            seen.add(row["code"])
            fmt = _format(row["place"]) if re.search(r"virtual|online|hybrid", row["place"], re.IGNORECASE) \
                else "in_person"
            items.append(_event_item(
                spec, external_id=f"eci-{row['code'].casefold()}", url=row["url"], title=row["title"],
                abstract=" · ".join(part for part in (
                    "Engineering Conferences International", row["category"]) if part),
                start=row["start"], end=row["end"], deadlines=[],
                location="Online" if fmt == "online" else _place_with_country(row["place"]),
                fmt=fmt, event_type="conference", official_id=row["code"],
                organizer="Engineering Conferences International",
            ))
        return items

    @staticmethod
    def _row(listing: str, para: str) -> dict[str, Any] | None:
        # Inline tags split words mid-token ("VI<strong>I</strong>"), so they are dropped, not spaced.
        inline = re.sub(r"</?(?:strong|b|em|span|a)\b[^>]*>", "", para)
        lines = [_plain(line) for line in re.split(r"<br\s*/?>", inline)]
        if len(lines) < 2:
            return None
        start, end = _date_range(lines[0])
        heading = re.match(r"(.+?)\s*[—–-]\s*(\d{2}[A-Z]{2})\s*$", lines[1])
        if start is None or not heading:
            return None  # "Date TBA" and "February 2028" are not dates yet.
        title = _title_case(heading.group(1).strip())
        if re.search(r"\b(?:cancell?ed|postponed)\b", title, re.IGNORECASE):
            return None
        href = _attr(_first(r"(<a\s[^>]*>)", para), "href")
        url = canonicalize_url(urljoin(listing, href)) if href else canonicalize_url(listing)
        path = urlparse(url).path.strip("/").split("/")
        category = path[1].replace("-", " ").capitalize() if len(path) > 2 and path[0] == "conferences" else ""
        place = _plain(_first(r"<em>(.*?)</em>", para))
        if re.fullmatch(r"(?:location\s+)?tb[ad]\b.*", place, re.IGNORECASE):
            place = ""
        return {"start": start, "end": end, "title": title, "code": heading.group(2), "url": url,
                "category": category, "place": place}


# -- Biophysical Society ----------------------------------------------------------------------

_BPS_REGION_RE = re.compile(r"africa|asia|australia|europe|north america|south america|oceania|middle east",
                            re.IGNORECASE)


class BiophysicalSocietyEventsSource:
    """Biophysical Society meetings from the "List of All Upcoming BPS Events" on its calendar."""

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        del days
        listing = spec.url or BPS_URL
        page = _get_text(listing)
        blocks = re.split(r"<div class=\"BPSEventListContainer\">", page)[1:]
        if not blocks:
            raise RuntimeError(f"{spec.name} returned no recognisable event entries")
        today = _today()
        horizon = today + timedelta(days=spec.lookahead_days)
        items, seen = [], set()
        for block in blocks:
            title_html = _first(r"<p class=\"BPSEventListTitle\">(.*?)</p>", block)
            title = _plain(title_html)
            start, end = _date_range(_plain(_first(r"<p class=\"BPSEventListDetails\">(.*?)</p>", block)))
            if not title or not _in_window(start, today, horizon):
                continue
            if re.search(r"\b(?:cancell?ed|postponed)\b", title, re.IGNORECASE):
                continue
            href = _attr(_first(r"(<a\s[^>]*>)", title_html), "href")
            url = canonicalize_url(urljoin(listing, href)) if href else canonicalize_url(listing)
            external_id = _stable_id("bps", url if href else f"{title.casefold()}|{start.year}")
            if external_id in seen:
                continue
            seen.add(external_id)
            category = _plain(_first(r"<span class=\"[^\"]*EDNcategorycolor[^\"]*\">(.*?)</span>", block))
            place = _plain(_first(r"<p class=\"BPSEventListAddress\">(.*?)</p>", block))
            fmt = "online" if re.search(r"\bvirtual\b|\bonline\b", place, re.IGNORECASE) else "in_person"
            meeting = re.search(r"annual meeting|thematic meeting|bps conference", category, re.IGNORECASE)
            if re.fullmatch(r"\d+(?:st|nd|rd|th) Annual Meeting", title):
                title = f"Biophysical Society {title}"
            items.append(_event_item(
                spec, external_id=external_id, url=url, title=title,
                abstract=" · ".join(part for part in ("Biophysical Society", category) if part),
                start=start, end=end, deadlines=[],
                location="Online" if fmt == "online" else self._location(place), fmt=fmt,
                # Annual, thematic and BPS-conference meetings are conferences; networking
                # events are typed by their own title (often a workshop), or left unknown.
                event_type=_event_type(title, "conference" if meeting else ""),
                official_id=urlparse(url).path if href else external_id, organizer="Biophysical Society",
            ))
        return items

    @staticmethod
    def _location(place: str) -> str:
        parts = [part.strip() for part in place.split(",") if part.strip()]
        if parts and re.match(r"US\b", parts[-1]):
            parts[-1] = "United States"  # "US - Mid-Atlantic" is the society's own region label.
        elif len(parts) >= 3 and _BPS_REGION_RE.fullmatch(parts[-1]):
            parts.pop()  # "Oxford, United Kingdom, Europe"
        return _place_with_country(", ".join(parts))


# -- FASEB Science Research Conferences ------------------------------------------------------------

class FASEBConferencesSource:
    """FASEB Science Research Conferences.

    The listing gives title, dates and page, but not the venue, so each
    upcoming conference page on faseb.org is read (bounded) for the venue,
    stated deadlines and summary. Only the listing's first page is read: its
    "Load More" control is a reCAPTCHA-protected search.
    """

    MAX_DETAIL_PAGES = 20

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        del days
        listing = spec.url or FASEB_SRC_URL
        page = _get_text(listing)
        cards = re.findall(r"<a\s([^>]*class=\"event-card[^\"]*\"[^>]*)>(.*?)</a>", page, re.DOTALL)
        if not cards:
            raise RuntimeError(f"{spec.name} returned no recognisable conference entries")
        today = _today()
        horizon = today + timedelta(days=spec.lookahead_days)
        items, seen, details = [], set(), 0
        for attrs, body in cards:
            url = canonicalize_url(urljoin(listing, _attr(f"<a {attrs}>", "href")))
            label = _plain(_first(r"<small>(.*?)</small>", body))
            title = _plain(_first(r"<span class=\"d-block h5\">(.*?)</span>", body))
            start, end = _date_range(_plain(_first(r"<span class=\"d-block\">(.*?)</span>", body)))
            if not title or not url.startswith("http") or url in seen or not _in_window(start, today, horizon):
                continue
            if re.search(r"\b(?:cancell?ed|postponed)\b", title, re.IGNORECASE):
                continue
            seen.add(url)
            detail = None
            # Pages hosted elsewhere (FASEB's registration platform) are not read.
            if urlparse(url).hostname in {"www.faseb.org", "faseb.org"} and details < self.MAX_DETAIL_PAGES:
                details += 1
                detail = self._detail(url)
            place = detail["place"] if detail else ""
            fmt = _format(place) if re.search(r"virtual|online|hybrid", place, re.IGNORECASE) else "in_person"
            summary = detail["summary"] if detail else ""
            items.append(_event_item(
                spec, external_id=_stable_id("faseb-src", url), url=url, title=title,
                abstract=" · ".join(part for part in (label or "FASEB Science Research Conference", summary) if part),
                start=start, end=end, deadlines=detail["deadlines"] if detail else [],
                location="Online" if fmt == "online" else _place_with_country(place), fmt=fmt,
                event_type=_event_type(title, label or "conference"), official_id=urlparse(url).path,
                organizer="FASEB",
            ))
        return items

    @staticmethod
    def _detail(url: str) -> dict[str, Any] | None:
        try:
            page = _get_text(url)
        except Exception as error:  # noqa: BLE001 - the listing still carries the core facts
            logger.debug("FASEB conference page %s unavailable: %s", url, error)
            return None
        hero = _first(r"<section class=\"secondary-hero[^\"]*\"[^>]*>(.*?)</section>", page)
        lines = [_plain(line) for line in re.split(r"<br\s*/?>", _first(r"<h4[^>]*>(.*?)</h4>", hero))]
        # The hero reads "July 19-22 2027 / Venue, City, ST #TAG / Registration ...".
        place = next((re.sub(r"\s*#\w+", "", line).strip(" ,") for line in lines[1:]
                      if "," in line and not re.search(r"registration|abstract|deadline", line, re.IGNORECASE)), "")
        deadlines = []
        for label, value in re.findall(r"<strong>([^<:]{3,120}?\bDeadlines?)\s*:\s*([^<]+)</strong>", page):
            when = _single_date(_plain(value))
            if when is not None:
                kind = re.sub(r"\s*\bdeadlines?$", "", _plain(label), flags=re.IGNORECASE).casefold()
                deadlines.append((kind or "deadline", when))
        summary_html = _first(r"Conference Summary(?:\s*</?(?:strong|span|b|br)\b[^>]*>)*(.*?)</p>", page)
        summary = _scrub(re.sub(r"<br\s*/?>", " ", summary_html))
        return {"place": place, "deadlines": deadlines, "summary": summary[:2000]}
