from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..models import Item, SourceSpec

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=False,
)
def _get_json(url: str, params: dict[str, str]) -> dict:
    with httpx.Client(timeout=20.0, follow_redirects=True, headers={"User-Agent": "dailydigest/0.1"}) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


class FDASource:
    """openFDA structured-data adapter (separate from the FDA RSS feeds).

    Default endpoint: ``drug/drugsfda.json`` filtered by recent
    ``submissions.submission_status_date``. Spec fields:
      - ``endpoint`` (default ``drug/drugsfda.json``)
      - ``query`` (raw openFDA ``search`` clause; default is a 2-day date range)
    The openFDA schema is messy; we skip records lacking enough info to render.
    """

    BASE = "https://api.fda.gov"
    MAX_ITEMS = 100
    LIMIT = 50

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        endpoint = (spec.endpoint or "drug/drugsfda.json").lstrip("/")
        today = datetime.now(timezone.utc).date()
        window_start = today - timedelta(days=max(1, days))
        date_query = (
            f"submissions.submission_status_date:"
            f"[{window_start.strftime('%Y%m%d')} TO {today.strftime('%Y%m%d')}]"
        )
        search = self._search_query(date_query, spec.query)

        url = f"{self.BASE}/{endpoint}"
        params = {"search": search, "limit": str(self.LIMIT)}

        out: list[Item] = []
        try:
            payload = _get_json(url, params)
        except Exception as e:
            name = getattr(spec, "name", "FDASource")
            raise RuntimeError(
                f"{name} fetch failed: {type(e).__name__}: {str(e)[:200]}"
            ) from e
        if not payload:
            return out

        results: list[dict[str, Any]] = payload.get("results") or []
        for entry in results[: self.MAX_ITEMS]:
            item = self._parse_entry(entry, spec)
            if item is not None:
                out.append(item)
        return out

    def _search_query(self, date_query: str, custom_query: str | None) -> str:
        custom = (custom_query or "").strip()
        if not custom:
            return date_query
        if "submission_status_date" in custom:
            return custom
        return f"({date_query}) AND ({custom})"

    def _parse_entry(self, entry: dict[str, Any], spec: SourceSpec) -> Item | None:
        app_no = entry.get("application_number") or ""
        sponsor = entry.get("sponsor_name") or ""

        products = entry.get("products") or []
        brand = ""
        generic = ""
        if products:
            first = products[0] or {}
            brand = (first.get("brand_name") or "").strip()
            ingredients = first.get("active_ingredients") or []
            generic = (ingredients[0].get("name", "") if ingredients else "") or ""

        title_bits = [b for b in (sponsor, brand or generic, app_no) if b]
        if not title_bits:
            return None
        title = " - ".join(title_bits).strip()

        # Build a synthetic abstract from submission status info.
        # Pick the submission with the most recent status date; default to last.
        submissions = entry.get("submissions") or []
        latest = max(
            submissions,
            key=lambda s: s.get("submission_status_date") or "",
            default={},
        )
        abstract_bits = []
        for k in ("submission_type", "submission_status", "submission_status_date",
                  "submission_class_code_description"):
            v = latest.get(k)
            if v:
                abstract_bits.append(f"{k.replace('_', ' ')}: {v}")
        abstract = "; ".join(abstract_bits)

        pub_dt: datetime | None = None
        date_str = latest.get("submission_status_date")
        if date_str:
            try:
                pub_dt = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
            except Exception:
                pub_dt = None

        if app_no:
            url = (
                "https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm"
                f"?event=overview.process&ApplNo={app_no}"
            )
            ext = app_no
        else:
            url = "https://api.fda.gov/" + (spec.endpoint or "drug/drugsfda.json")
            ext = hashlib.sha1(title.encode("utf-8")).hexdigest()[:16]

        return Item(
            source=spec.name,
            section=spec.section,
            external_id=str(ext),
            url=url,
            title=title,
            abstract=abstract,
            authors=sponsor,
            published_at=pub_dt,
        )
