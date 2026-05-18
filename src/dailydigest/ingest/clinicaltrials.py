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


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    except Exception:
        return None


class ClinicalTrialsSource:
    """ClinicalTrials.gov v2 API ingest, filtered to recently updated studies.

    API: https://clinicaltrials.gov/api/v2/studies
    Spec field: ``condition`` (optional; e.g. ``"cancer"``).
    """

    BASE = "https://clinicaltrials.gov/api/v2/studies"
    PAGE_SIZE = 50
    MAX_ITEMS = 100

    def fetch(self, spec: SourceSpec) -> list[Item]:
        today = datetime.now(timezone.utc).date()
        from_d = (today - timedelta(days=2)).isoformat()
        to_d = today.isoformat()
        params: dict[str, str] = {
            "filter.advanced": f"AREA[LastUpdatePostDate]RANGE[{from_d},{to_d}]",
            "pageSize": str(self.PAGE_SIZE),
            "format": "json",
        }
        if spec.condition:
            params["query.cond"] = spec.condition

        out: list[Item] = []
        page_token: str | None = None
        while len(out) < self.MAX_ITEMS:
            if page_token:
                params["pageToken"] = page_token
            try:
                payload = _get_json(self.BASE, params)
            except Exception as e:
                logger.warning("%s fetch failed: %s: %s", getattr(spec, "name", "ClinicalTrialsSource"), type(e).__name__, str(e)[:200])
                break
            if not payload:
                break

            studies: list[dict[str, Any]] = payload.get("studies") or []
            for study in studies:
                if len(out) >= self.MAX_ITEMS:
                    break
                item = self._parse_study(study, spec)
                if item is not None:
                    out.append(item)

            next_token = payload.get("nextPageToken")
            if not next_token or next_token == page_token:
                break
            page_token = next_token
        return out

    def _parse_study(self, study: dict[str, Any], spec: SourceSpec) -> Item | None:
        proto = study.get("protocolSection") or {}
        ident = proto.get("identificationModule") or {}
        nct_id = ident.get("nctId") or ""
        title = (ident.get("briefTitle") or ident.get("officialTitle") or "").strip()
        if not nct_id or not title:
            return None

        desc = proto.get("descriptionModule") or {}
        abstract = (desc.get("briefSummary") or "").strip()

        sponsor_mod = proto.get("sponsorCollaboratorsModule") or {}
        lead = (sponsor_mod.get("leadSponsor") or {}).get("name", "") or ""

        status_mod = proto.get("statusModule") or {}
        last_update = (
            (status_mod.get("lastUpdatePostDateStruct") or {}).get("date")
            or status_mod.get("lastUpdatePostDate")
            or (status_mod.get("lastUpdateSubmitDateStruct") or {}).get("date")
            or status_mod.get("lastUpdateSubmitDate")
        )
        pub_dt = _parse_date(last_update)

        url = f"https://clinicaltrials.gov/study/{nct_id}"
        ext = nct_id or hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
        return Item(
            source=spec.name,
            section=spec.section,
            external_id=ext,
            url=url,
            title=title,
            abstract=abstract,
            authors=lead,
            published_at=pub_dt,
        )
