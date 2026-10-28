"""NIH Guide funding announcements (RFA, PA, PAR, PAS) from the NIH Guide search.

The service is the backend of the official NIH Guide search page. It serves the
same notices as Grants.gov from a separate host, so US funding stays covered
when Grants.gov is unreachable.
"""

from __future__ import annotations

import html
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ..models import Item, SourceSpec
from ._retry import transient_http_error
from ._terms import profile_search_terms

SEARCH_URL = "https://search.grants.nih.gov/guide/api/data"
GUIDE_URL = "https://grants.nih.gov/grants/guide/{folder}/{filename}"
GRANTS_GOV_URL = "https://www.grants.gov/search-results-detail/{ggid}"
FUNDING_DOCTYPES = {"RFA", "PA", "PAR", "PAS"}
_FOLDERS = {"RFA": "rfa-files", "PA": "pa-files", "PAR": "pa-files", "PAS": "pa-files"}
_TAG_RE = re.compile(r"<[^>]+>")
# Fellowship and career-development activity codes.
_FELLOWSHIP_CODES = re.compile(r"^(?:F\d\d|K\d\d|K99|T\d\d|R00)$")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(transient_http_error),
    reraise=True,
)
def _search(term: str, rows: int, today: date) -> dict:
    # The service rejects some parameter orders with HTTP 500; keep this one.
    params = [
        ("perpage", str(rows)), ("sort", "reldate:desc"), ("from", "0"), ("query", term),
        ("type", "active,activenosis"), ("parentic", "all"), ("primaryic", "all"),
        ("activitycodes", "all"), ("doctype", "all"), ("parentfoa", "all"),
        ("clinicaltrials", "all"), ("fields", "all"), ("spons", "true"),
        ("daterange", f"01011991-{today:%m%d%Y}"),
    ]
    with httpx.Client(timeout=30.0, follow_redirects=True,
                      headers={"Accept": "application/json", "User-Agent": "dailydigest/0.1"}) as client:
        response = client.get(SEARCH_URL, params=params)
        response.raise_for_status()
        return response.json()


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


def _plain(value: object) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG_RE.sub(" ", str(value or "")))).strip()


class NIHGuideSource:
    MAX_TERMS = 8
    ROWS_PER_TERM = 15
    MAX_ITEMS = 60

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        del days
        if spec.profile_driven:
            terms = profile_search_terms(self.MAX_TERMS)
        else:
            terms = [part.strip() for part in str(spec.query or "").split("|") if part.strip()]
        terms = terms[: self.MAX_TERMS]
        if not terms:
            return []
        today = _today()
        horizon = today + timedelta(days=spec.lookahead_days)
        hits: dict[str, dict[str, Any]] = {}
        for term in terms:
            payload = _search(term, self.ROWS_PER_TERM, today)
            rows = ((payload.get("data") or {}).get("hits") or {}).get("hits")
            if not isinstance(rows, list):
                raise RuntimeError(f"{spec.name} returned an invalid NIH Guide response")
            for row in rows:
                source = row.get("_source") or {}
                number = str(source.get("docnum") or "").strip()
                if number and str(source.get("doctype") or "").upper() in FUNDING_DOCTYPES:
                    hits.setdefault(number, source)
            if len(hits) >= self.MAX_ITEMS:
                break
        items = []
        for number, source in list(hits.items())[: self.MAX_ITEMS]:
            item = self._item(spec, number, source, today, horizon)
            if item is not None:
                items.append(item)
        return items

    def _item(self, spec: SourceSpec, number: str, source: dict[str, Any],
              today: date, horizon: date) -> Item | None:
        title = _plain(source.get("title"))
        expires = _date(source.get("expdate"))
        if not title or expires is None or expires < today:
            return None
        receipt = _date(source.get("appreceiptdate"))
        if receipt is not None and receipt < today:
            receipt = None
        opens = _date(source.get("opendate"))
        if receipt is not None and receipt > horizon or (opens and opens > horizon):
            return None
        doctype = str(source.get("doctype") or "").upper()
        filename = str(source.get("filename") or "").strip()
        ggid = str(source.get("ggid") or "").strip()
        if filename and filename.lower() != "none":
            url = GUIDE_URL.format(folder=_FOLDERS.get(doctype, "pa-files"), filename=filename)
        elif ggid.isdigit():
            # Same official record on Grants.gov; dedupe merges the two copies.
            url = GRANTS_GOV_URL.format(ggid=ggid)
        else:
            return None
        codes = [str(code).strip() for code in source.get("ac") or [] if str(code).strip()]
        institute = str(source.get("primaryIC") or "NIH").strip()
        return Item(
            source=spec.name,
            section=spec.section,
            external_id=number,
            url=url,
            title=title,
            abstract=_plain(source.get("purpose"))[:4000],
            authors=f"NIH {institute}".strip(),
            published_at=(
                datetime.combine(_date(source.get("reldate")), datetime.min.time(), tzinfo=timezone.utc)
                if _date(source.get("reldate")) else None
            ),
            metadata={
                "opportunity_type": (
                    "fellowship" if any(_FELLOWSHIP_CODES.match(code) for code in codes) else "grant"
                ),
                "status": "forthcoming" if opens and opens > today else "open",
                "official": True,
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "funder": f"National Institutes of Health ({institute})",
                "activity_codes": codes,
                "open_date": opens.isoformat() if opens else None,
                # Most announcements accept several due dates; only a single
                # stated receipt date is a deadline. Otherwise show the expiry.
                "deadline": receipt.isoformat() if receipt else None,
                "expires": expires.isoformat(),
                "deadline_timezone": "",
                "deadlines": (
                    [{"type": "application", "date": receipt.isoformat(), "timezone": ""}]
                    if receipt else []
                ),
                "currency": "USD",
                "eligibility": "",
                "eligibility_tags": [],
                "eligibility_description": "",
                "official_id": number,
                "grants_gov_id": ggid if ggid.isdigit() else "",
            },
        )
