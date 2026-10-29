"""Official national funding calls from UKRI (United Kingdom) and the SNSF (Switzerland).

Both funders publish an RSS feed of calls without deadlines, so each recent
call's official page is read for its labelled dates. Only a deadline (or an
announced opening date) checked here decides whether a call is still open;
detail-page fetches are bounded per brew.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

import feedparser
import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ..config import user_local_date
from ..models import Item, SourceSpec
from ._retry import transient_http_error
from .events_api import _plain
from .rss import canonicalize_url

logger = logging.getLogger(__name__)

USER_AGENT = "dailydigest/0.1 (+https://github.com/Antimonyleo/daily_digests)"
UKRI_URL = "https://www.ukri.org/opportunity/feed/"
SNSF_URL = "https://www.snf.ch/api/rss/en/call/feed.xml"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(transient_http_error),
    reraise=True,
)
def _get_text(url: str) -> str:
    with httpx.Client(timeout=30.0, follow_redirects=True,
                      headers={"User-Agent": USER_AGENT}) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text


def _today() -> date:
    return user_local_date()


def _feed_entries(spec: SourceSpec, url: str) -> list[Any]:
    content = _get_text(url)
    feed = feedparser.parse(content)
    if not feed.entries:
        raise RuntimeError(f"{spec.name} returned no funding calls from its feed")
    return list(feed.entries)


def _published(entry: Any) -> datetime | None:
    value = entry.get("published_parsed") or entry.get("updated_parsed")
    try:
        return datetime(*value[:6], tzinfo=timezone.utc) if value else None
    except (TypeError, ValueError):
        return None


def _iso_date(value: object) -> date | None:
    match = re.match(r"\s*(\d{4})-(\d{2})-(\d{2})", str(value or ""))
    if not match:
        return None
    try:
        return date(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def _opportunity_type(text: str) -> str:
    lowered = text.casefold()
    if "fellowship" in lowered or "postdoc.mobility" in lowered:
        return "fellowship"
    if re.search(r"\b(?:award|prize)s?\b", lowered):
        return "award"
    return "grant"


def _amount(text: str) -> float | None:
    match = re.search(r"(?:£|CHF|GBP)\s*([\d,]+(?:\.\d+)?)\s*(million|m\b)?", text, flags=re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1).replace(",", ""))
    return value * 1_000_000 if match.group(2) else value


def _short(text: str, limit: int = 400) -> str:
    """Keep whole sentences of an official summary up to ``limit`` characters."""
    kept = ""
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # "eligible ." left by stripped links
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if kept and len(kept) + len(sentence) + 1 > limit:
            break
        kept = f"{kept} {sentence}".strip()
    return kept[:limit]


def _external_id(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha1(value.encode('utf-8')).hexdigest()[:16]}"


def _funding_item(spec: SourceSpec, *, external_id: str, url: str, title: str, abstract: str,
                  funder: str, published: datetime | None, status: str, opens: date | None,
                  deadlines: list[date], timezone_name: str, currency: str, kind: str,
                  eligibility: str, official_id: str, amounts: dict[str, float]) -> Item:
    deadline = deadlines[0] if deadlines else None
    metadata: dict[str, Any] = {
        "opportunity_type": kind,
        "status": status,
        "official": True,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "funder": funder,
        "open_date": opens.isoformat() if opens else None,
        "deadline": deadline.isoformat() if deadline else None,
        "deadline_timezone": timezone_name,
        "deadlines": [{"type": "application", "date": value.isoformat(), "timezone": timezone_name}
                      for value in deadlines],
        "currency": currency,
        # Country and career-stage rules stay with review; only the funder's own
        # labelled summary is carried, never inferred tags.
        "eligibility": "",
        "eligibility_tags": [],
        "eligibility_description": eligibility,
        "official_id": official_id,
    }
    metadata.update(amounts)
    return Item(
        source=spec.name, section=spec.section, external_id=external_id, url=url, title=title,
        abstract=abstract[:4000], authors=funder, published_at=published, metadata=metadata,
    )


# -- UKRI ------------------------------------------------------------------------

_BUSINESS_RE = re.compile(r"\b(?:business(?:es)?|SMEs?|small and medium|enterprises?|compan(?:y|ies)|farming)\b",
                          re.IGNORECASE)
_RESEARCH_RE = re.compile(r"\b(?:academic|universit|research organisation|research technology|RTOs?|catapult|"
                          r"higher education|research council)", re.IGNORECASE)


def _business_only(category: str, summary: str) -> bool:
    """Innovate UK competitions whose applicant clause names only businesses."""
    if "innovate uk" not in category.casefold():
        return False
    clause = summary.split(" can apply", 1)[0] if " can apply" in summary else summary[:160]
    return bool(_BUSINESS_RE.search(clause)) and not _RESEARCH_RE.search(clause)


def _ukri_fields(page: str) -> dict[str, str]:
    summary = re.search(r"<dl[^>]*opportunity__summary[^>]*>(.*?)</dl>", page, flags=re.DOTALL)
    if not summary:
        return {}
    return {
        _plain(label).rstrip(": ").casefold(): value
        for label, value in re.findall(r"<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>", summary.group(1), re.DOTALL)
    }


def _first_attr(fragment: str, name: str) -> str:
    match = re.search(rf"\b{name}=\"([^\"]+)\"", fragment)
    return match.group(1) if match else ""


def _labelled_section(page: str, labels: tuple[str, ...]) -> str:
    pattern = "|".join(re.escape(label) for label in labels)
    match = re.search(rf">\s*(?:{pattern})\s*<", page, flags=re.IGNORECASE)
    if not match:
        return ""
    rest = page[match.end():]
    start = rest.find("<p")
    if start < 0:
        return ""
    body = re.split(r"<h[1-4][\s>]|govuk-accordion__section-header|</section", rest[start:], maxsplit=1)[0]
    return _short(_plain(body))


def _detail_pages_checked(source: Any, spec: SourceSpec, attempted: int, failed: int) -> None:
    """Record detail-page failures; every page failing is an outage, not an
    empty official result."""
    source.last_fetch_diagnostics = {"detail_pages": attempted, "detail_failures": failed}
    if attempted and failed == attempted:
        raise RuntimeError(f"{spec.name}: none of {attempted} opportunity pages could be read")


class UKRIFundingSource:
    """UKRI research-council and UKRI-wide calls, verified on each opportunity page."""

    MAX_DETAIL_PAGES = 25

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        del days
        entries = _feed_entries(spec, spec.url or UKRI_URL)
        today = _today()
        horizon = today + timedelta(days=spec.lookahead_days)
        items, fetched, failed, seen = [], 0, 0, set()
        for entry in entries:
            url = canonicalize_url(str(entry.get("link") or ""))
            title = _plain(entry.get("title"))
            if not url or not title or url in seen:
                continue
            seen.add(url)
            description = re.sub(r"<p[^>]*aria-label=\"Councils\".*?</p>", " ",
                                 str(entry.get("summary") or ""), flags=re.DOTALL)
            summary = _plain(description)
            category = " ".join(str(tag.get("term") or "") for tag in entry.get("tags") or [])
            if _business_only(category, summary) or title.casefold().startswith("invite only"):
                continue  # Not open to a research applicant.
            if fetched >= self.MAX_DETAIL_PAGES:
                break
            fetched += 1
            try:
                page = _get_text(url)
            except Exception as error:  # noqa: BLE001 - an unverified call is skipped, not guessed
                logger.debug("UKRI opportunity page %s unavailable: %s", url, error)
                failed += 1
                continue
            item = self._item(spec, entry, url, title, summary, category, page, today, horizon)
            if item is not None:
                items.append(item)
        _detail_pages_checked(self, spec, fetched, failed)
        return items

    @staticmethod
    def _item(spec: SourceSpec, entry: Any, url: str, title: str, summary: str, category: str,
              page: str, today: date, horizon: date) -> Item | None:
        fields = _ukri_fields(page)
        status = _plain(fields.get("opportunity status")).casefold()
        if status not in {"open", "upcoming"}:
            return None  # Closed, or a page without the labelled summary.
        # "To be confirmed" and "Open - no closing date" carry no <time> element.
        opens = _iso_date(_first_attr(fields.get("opening date", ""), "datetime"))
        closes = _iso_date(_first_attr(fields.get("closing date", ""), "datetime"))
        if closes is not None and (closes < today or closes > horizon):
            return None
        forthcoming = status == "upcoming" or (opens is not None and opens > today)
        if closes is None and not (forthcoming and opens is not None and today < opens <= horizon):
            return None  # No checkable deadline or announced opening date.
        amounts = {}
        for key, label in (("amount_max", "maximum award"), ("amount_min", "minimum award")):
            value = _amount(_plain(fields.get(label)))
            if value is not None:
                amounts[key] = value
        guid = str(entry.get("id") or entry.get("guid") or "")
        post = re.search(r"[?&]p=(\d+)", guid)
        funder = _plain(fields.get("funders")) or "UK Research and Innovation"
        kind = _plain(fields.get("funding type"))
        return _funding_item(
            spec,
            external_id=f"ukri-{post.group(1)}" if post else _external_id("ukri", url),
            url=url, title=title, abstract=" · ".join(part for part in (summary, category) if part),
            funder=funder, published=_published(entry),
            status="forthcoming" if forthcoming else "open", opens=opens,
            deadlines=[closes] if closes else [], timezone_name="Europe/London", currency="GBP",
            kind=_opportunity_type(f"{kind} {title}"),
            eligibility=_labelled_section(page, ("Who can apply", "Eligibility summary")),
            official_id=post.group(1) if post else url, amounts=amounts,
        )


# -- SNSF ------------------------------------------------------------------------------

def _json_ld_deadlines(page: str) -> list[date]:
    found = []
    for block in re.findall(r"<script[^>]*application/ld\+json[^>]*>(.*?)</script>", page, re.DOTALL):
        try:
            data = json.loads(block)
        except ValueError:
            continue
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, dict):
                identifier = node.get("name"), node.get("value")
                if node.get("@type") == "PropertyValue" and "deadline" in str(identifier[0]).casefold():
                    if (when := _iso_date(identifier[1])) is not None:
                        found.append(when)
                stack.extend(value for value in node.values() if isinstance(value, (dict, list)))
    return found


def _html_deadlines(page: str) -> list[date]:
    found = []
    for day, month, year in re.findall(
        r"article__time--deadline\">[^<]*</p>\s*(\d{1,2})\.(\d{1,2})\.(\d{4})", page
    ):
        try:
            found.append(date(int(year), int(month), int(day)))
        except ValueError:
            continue
    return found


class SNSFFundingSource:
    """SNSF calls for proposals; the feed holds the full history, so only recent calls are read."""

    MAX_DETAIL_PAGES = 40
    RECENT_DAYS = 365

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        del days
        entries = _feed_entries(spec, spec.url or SNSF_URL)
        today = _today()
        horizon = today + timedelta(days=spec.lookahead_days)
        cutoff = datetime.combine(today - timedelta(days=self.RECENT_DAYS), datetime.min.time(), tzinfo=timezone.utc)
        recent = sorted(
            ((published, entry) for entry in entries
             if (published := _published(entry)) is not None and published >= cutoff),
            key=lambda pair: pair[0], reverse=True,
        )
        items, seen, attempted, failed = [], set(), 0, 0
        for published, entry in recent[: self.MAX_DETAIL_PAGES]:
            url = canonicalize_url(str(entry.get("link") or ""))
            title = _plain(entry.get("title"))
            if not url or not title or url in seen:
                continue
            seen.add(url)
            attempted += 1
            try:
                page = _get_text(url)
            except Exception as error:  # noqa: BLE001 - an unverified call is skipped, not guessed
                logger.debug("SNSF call page %s unavailable: %s", url, error)
                failed += 1
                continue
            deadlines = sorted(set(_json_ld_deadlines(page) or _html_deadlines(page)))
            upcoming = [value for value in deadlines if value >= today]
            if not upcoming or upcoming[0] > horizon:
                continue
            token = re.search(r"snf\.ch/[a-z]{2}/([A-Za-z0-9]+)/", url)
            lead_html = re.search(r"<div class=\"block-title__lead\">(.*?)</div>", page, re.DOTALL)
            lead = _plain(lead_html.group(1)) if lead_html else ""
            items.append(_funding_item(
                spec,
                external_id=f"snsf-{token.group(1)}" if token else _external_id("snsf", url),
                url=url, title=title, abstract=lead or _plain(entry.get("summary")),
                funder="Swiss National Science Foundation (SNSF)", published=published,
                status="open", opens=None, deadlines=upcoming, timezone_name="Europe/Zurich",
                currency="CHF", kind=_opportunity_type(title),
                eligibility="", official_id=token.group(1) if token else url, amounts={},
            ))
        _detail_pages_checked(self, spec, attempted, failed)
        return items
