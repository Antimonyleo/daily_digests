"""Official EU Funding & Tenders Portal topics (Horizon Europe, ERC, MSCA, EIC)."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ..models import Item, SourceSpec
from ._retry import transient_http_error

SEARCH_URL = "https://api.tech.ec.europa.eu/search-api/prod/rest/search"
TOPIC_URL = (
    "https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/"
    "opportunities/topic-details/{identifier}"
)
# Portal codes: topic records (type 1) that are forthcoming or open.
_TOPIC = "1"
_FORTHCOMING, _OPEN = "31094501", "31094502"
HORIZON_EUROPE = "43108390"
MSCA = "43108473"
# Individual fellowships; MSCA doctoral networks and staff exchanges are
# institutional and stay out of a fellowship source.
_FELLOWSHIP_TOPIC = re.compile(r"-PF-|fellowship", re.IGNORECASE)
_FIELDS = [
    "title", "identifier", "status", "deadlineDate", "startDate", "callIdentifier",
    "callTitle", "typesOfAction", "deadlineModel", "url",
]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(transient_http_error),
    reraise=True,
)
def _search(page: int, programme: str, division: str = "") -> dict:
    must: list[dict[str, Any]] = [
        {"terms": {"type": [_TOPIC]}},
        {"terms": {"status": [_FORTHCOMING, _OPEN]}},
    ]
    if programme:
        must.append({"terms": {"frameworkProgramme": [programme]}})
    if division:
        must.append({"terms": {"programmeDivision": [division]}})
    # Multipart is a CORS "simple" request, so browsers need no preflight.
    parts = {
        "query": ("blob", json.dumps({"bool": {"must": must}}), "application/json"),
        "languages": ("blob", json.dumps(["en"]), "application/json"),
        "displayFields": ("blob", json.dumps(_FIELDS), "application/json"),
        "sort": ("blob", json.dumps({"field": "sortStatus", "order": "ASC"}), "application/json"),
    }
    with httpx.Client(timeout=40.0, follow_redirects=True) as client:
        response = client.post(
            SEARCH_URL,
            params={"apiKey": "SEDIA", "text": "***", "pageSize": "100", "pageNumber": str(page)},
            files=parts,
            headers={"Accept": "application/json", "User-Agent": "dailydigest/0.1"},
        )
        response.raise_for_status()
        return response.json()


def _first(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()


def _date(value: object) -> date | None:
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(value or ""))
    if not match:
        return None
    try:
        return date(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _opportunity_type(title: str, action: str, identifier: str) -> str:
    text = f"{title} {action} {identifier}".casefold()
    if "fellowship" in text or "-msca-" in text and "-pf-" in text:
        return "fellowship"
    if "prize" in text or "award" in text:
        return "award"
    return "grant"


class EUFundingSource:
    """Read every open or forthcoming topic and keep those with a future deadline.

    The portal's status flag lags: topics whose deadlines all passed years ago
    still say "open". Only a deadline (or opening date) checked here decides.
    Keyword search on the portal misses most topics, so relevance is left to
    the opportunity ranker rather than a server-side query.
    """

    MAX_PAGES = 12

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        del days
        programme = HORIZON_EUROPE if spec.query is None else str(spec.query).strip()
        # ``condition: msca_fellowships`` reads the MSCA programme division and
        # keeps its individual fellowships only.
        fellowships_only = str(spec.condition or "").casefold() == "msca_fellowships"
        division = MSCA if fellowships_only else ""
        today = _today()
        horizon = today + timedelta(days=spec.lookahead_days)
        seen: set[str] = set()
        items: list[Item] = []
        for page in range(1, self.MAX_PAGES + 1):
            payload = _search(page, programme, division)
            results = payload.get("results")
            if not isinstance(results, list):
                raise RuntimeError(f"{spec.name} returned an invalid EU portal response")
            for row in results:
                item = self._item(spec, row.get("metadata") or {}, today, horizon, seen)
                if item is not None and fellowships_only and not _FELLOWSHIP_TOPIC.search(
                    f"{item.external_id} {item.title}"
                ):
                    continue
                if item is not None:
                    items.append(item)
            total = int(payload.get("totalResults") or 0)
            if not results or page * 100 >= total:
                break
        return items

    def _item(self, spec: SourceSpec, metadata: dict[str, Any], today: date,
              horizon: date, seen: set[str]) -> Item | None:
        identifier = _first(metadata, "identifier")
        title = _first(metadata, "title")
        if not identifier or not title or identifier in seen:
            return None
        deadlines = sorted(
            parsed for parsed in (_date(value) for value in metadata.get("deadlineDate") or [])
            if parsed is not None
        )
        upcoming = [value for value in deadlines if value >= today]
        if not upcoming or upcoming[0] > horizon:
            return None
        seen.add(identifier)
        opens = _date(_first(metadata, "startDate"))
        action = re.sub(r"\s+", " ", _first(metadata, "typesOfAction"))
        call = _first(metadata, "callTitle")
        stage = _first(metadata, "deadlineModel")
        deadline = upcoming[0]
        link = _first(metadata, "url")
        if "/topic-details/" not in link:
            link = TOPIC_URL.format(identifier=identifier)
        summary = " · ".join(part for part in (call and f"Call: {call}", action, stage) if part)
        return Item(
            source=spec.name,
            section=spec.section,
            external_id=identifier,
            url=link,
            title=title,
            abstract=summary,
            authors="European Commission",
            published_at=(
                datetime.combine(opens, datetime.min.time(), tzinfo=timezone.utc) if opens else None
            ),
            metadata={
                "opportunity_type": _opportunity_type(title, action, identifier),
                "status": "forthcoming" if opens and opens > today else "open",
                "official": True,
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "funder": "European Commission",
                "programme": "Horizon Europe" if identifier.startswith("HORIZON") or "ERC" in identifier else "",
                "open_date": opens.isoformat() if opens else None,
                "deadline": deadline.isoformat(),
                "deadline_timezone": "Europe/Brussels",
                "deadlines": [
                    {"type": "stage" if len(deadlines) > 1 else "application",
                     "date": value.isoformat(), "timezone": "Europe/Brussels"}
                    for value in upcoming
                ],
                "currency": "EUR",
                # Country rules sit in each topic's conditions; leave them to review.
                "eligibility": "",
                "eligibility_tags": [],
                "eligibility_description": "",
                "official_id": identifier,
            },
        )
