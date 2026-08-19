"""Official Grants.gov opportunity search and detail adapter."""

from __future__ import annotations

import html
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from dateutil import parser as date_parser
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..models import Item, SourceSpec
from ._terms import profile_search_terms

_TAG_RE = re.compile(r"<[^>]+>")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
def _post_json(url: str, payload: dict[str, Any]) -> dict:
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.post(
            url,
            json=payload,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "dailydigest/0.1",
            },
        )
        response.raise_for_status()
        return response.json()


def _plain_text(value: object) -> str:
    text = _TAG_RE.sub(" ", str(value or ""))
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date_parser.parse(
            text,
            fuzzy=True,
            tzinfos={"EST": -18000, "EDT": -14400, "CST": -21600, "CDT": -18000},
        ).date()
    except (ValueError, TypeError, OverflowError):
        return None


def _datetime(value: object) -> datetime | None:
    parsed = _date(value)
    if parsed is None:
        return None
    return datetime.combine(parsed, datetime.min.time(), tzinfo=timezone.utc)


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _amount(value: object) -> int | float | None:
    text = re.sub(r"[^0-9.\-]", "", str(value or ""))
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def _slug(value: object, default: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")
    return cleaned or default


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y"}


class GrantsGovSource:
    """Search open federal opportunities and verify each through official detail."""

    SEARCH_URL = "https://api.grants.gov/v1/api/search2"
    DETAIL_URL = "https://api.grants.gov/v1/api/fetchOpportunity"
    MAX_TERMS = 10
    ROWS_PER_TERM = 25
    MAX_ITEMS = 60

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        del days  # Opportunity selection uses a future deadline horizon instead.
        if spec.profile_driven:
            terms = profile_search_terms(self.MAX_TERMS)
        else:
            # A pipe-separated `query` searches several phrases. Career awards and
            # fellowships are described by the AWARD, not by the research topic, so
            # a profile-keyword search ("DNA nanotechnology") never returns them —
            # they need their own vocabulary ("postdoctoral fellowship", "career
            # development award"). Relevance is still enforced downstream by the
            # opportunity topic floor and the eligibility gate.
            terms = [
                part.strip()
                for part in str(spec.query or "").split("|")
                if part.strip()
            ][: self.MAX_TERMS]
        if not terms:
            return []

        hits: dict[str, dict[str, Any]] = {}
        for term in terms:
            payload = _post_json(
                self.SEARCH_URL,
                {
                    "rows": self.ROWS_PER_TERM,
                    "keyword": term,
                    "oppStatuses": "forecasted|posted",
                },
            )
            for hit in (payload.get("data") or {}).get("oppHits") or []:
                identifier = str(hit.get("id") or hit.get("number") or "").strip()
                if identifier:
                    hits.setdefault(identifier, hit)
                if len(hits) >= self.MAX_ITEMS:
                    break
            if len(hits) >= self.MAX_ITEMS:
                break

        today = _today()
        horizon = today + timedelta(days=spec.lookahead_days)
        items: list[Item] = []
        for hit in hits.values():
            hit_deadline = _date(hit.get("closeDate"))
            if hit_deadline is not None and not (today <= hit_deadline <= horizon):
                continue
            detail = _post_json(self.DETAIL_URL, {"opportunityId": hit.get("id")}).get("data") or {}
            synopsis = detail.get("synopsis") or detail.get("forecast") or {}
            deadline = hit_deadline or _date(
                synopsis.get("responseDateDesc")
                or synopsis.get("responseDate")
                or synopsis.get("estApplicationResponseDate")
                or synopsis.get("estApplicationResponseDateStr")
                or detail.get("originalDueDateDesc")
            )
            if deadline is not None and not (today <= deadline <= horizon):
                continue
            number = str(
                detail.get("opportunityNumber") or hit.get("number") or hit.get("id")
            ).strip()
            title = str(detail.get("opportunityTitle") or hit.get("title") or "").strip()
            if not number or not title:
                continue
            agency_details = synopsis.get("agencyDetails") or detail.get("agencyDetails") or {}
            agency = str(
                agency_details.get("agencyName")
                or synopsis.get("agencyName")
                or hit.get("agencyName")
                or hit.get("agency")
                or "Grants.gov"
            ).strip()
            # Sorted because the API returns this list in a different order on
            # every fetch. The order leaks into both `eligibility_tags` and the
            # joined `eligibility` string, which the opportunity snapshot hashes
            # to decide whether a call materially changed — unsorted, every
            # grant looked "updated" daily and was re-surfaced in every digest.
            applicant_types = sorted(
                {
                    str(row.get("description") or "").strip()
                    for row in synopsis.get("applicantTypes") or []
                    if str(row.get("description") or "").strip()
                }
            )
            eligibility_description = _plain_text(
                synopsis.get("applicantEligibilityDesc")
                or synopsis.get("additionalInformationOnEligibility")
            )
            instruments = [
                str(row.get("description") or "").strip()
                for row in synopsis.get("fundingInstruments") or []
                if str(row.get("description") or "").strip()
            ]
            raw_status = str(hit.get("oppStatus") or "posted").casefold()
            status = "forthcoming" if raw_status == "forecasted" else "open"
            metadata: dict[str, Any] = {
                "opportunity_type": _slug(instruments[0] if instruments else "grant", "grant"),
                "status": status,
                "official": True,
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "funder": agency,
                "open_date": (
                    _date(hit.get("openDate")).isoformat() if _date(hit.get("openDate")) else None
                ),
                "deadline": deadline.isoformat() if deadline else None,
                "deadline_timezone": "",
                "deadlines": (
                    [{"type": "application", "date": deadline.isoformat(), "timezone": ""}]
                    if deadline
                    else []
                ),
                "amount_min": _amount(synopsis.get("awardFloor")),
                "amount_max": _amount(synopsis.get("awardCeiling")),
                "currency": "USD",
                "eligibility": "; ".join(applicant_types),
                "eligibility_tags": applicant_types,
                "eligibility_description": eligibility_description,
                "cost_sharing": _bool(synopsis.get("costSharing", False)),
                "official_id": number,
            }
            items.append(
                Item(
                    source=spec.name,
                    section=spec.section,
                    external_id=number,
                    url=f"https://www.grants.gov/search-results-detail/{hit.get('id')}",
                    title=title,
                    abstract=_plain_text(
                        synopsis.get("synopsisDesc") or synopsis.get("forecastDesc")
                    ),
                    authors=agency,
                    published_at=_datetime(
                        synopsis.get("postingDate")
                        or synopsis.get("postingDateStr")
                        or hit.get("openDate")
                    ),
                    metadata=metadata,
                )
            )
        return items
