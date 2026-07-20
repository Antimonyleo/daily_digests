from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET  # types/ParseError only; parsing uses defusedxml
from datetime import datetime, timezone

import httpx
from defusedxml.ElementTree import fromstring as ET_fromstring
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..models import Item, SourceSpec

logger = logging.getLogger(__name__)

_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=False,
)


class PubMedSource:
    """Two-step PubMed ingest: esearch -> efetch (XML).

    Spec field: ``query`` (a PubMed search query, e.g. ``"CAR-T"``).
    Sleeps ~0.34s between calls to stay under NCBI's 3 req/s public limit.
    """

    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    HEADERS = {"User-Agent": "dailydigest/0.1 (mailto:noreply@example.com)"}
    SLEEP = 0.34
    RETMAX = 200
    MAX_ITEMS = 100

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        query = spec.query or ""
        if spec.profile_driven:
            from ._terms import profile_search_terms

            terms = profile_search_terms(12)
            if terms:
                # OR the profile keywords over title/abstract so PubMed actively
                # searches for the user's topics across all indexed journals.
                clauses = " OR ".join(f'"{t}"[tiab]' for t in terms)
                query = f"({clauses})" if not query else f"({clauses}) OR ({query})"
        if not query:
            return []
        self._reldate = max(1, days)
        out: list[Item] = []
        try:
            with httpx.Client(timeout=20.0, headers=self.HEADERS) as client:
                ids = self._esearch(client, query)
                if not ids:
                    return out
                time.sleep(self.SLEEP)
                xml_text = self._efetch(client, ids)
        except Exception as e:
            logger.warning("%s fetch failed: %s: %s", getattr(spec, "name", "PubMedSource"), type(e).__name__, str(e)[:200])
            return out

        if not xml_text:
            return out

        try:
            root = ET_fromstring(xml_text)
        except Exception:
            return out

        for art in root.findall(".//PubmedArticle"):
            try:
                item = self._parse_article(art, spec)
            except Exception:
                continue
            if item is not None:
                out.append(item)
            if len(out) >= self.MAX_ITEMS:
                break
        return out

    @_RETRY
    def _esearch(self, client: httpx.Client, query: str) -> list[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "datetype": "pdat",
            "reldate": str(getattr(self, "_reldate", 2)),
            "retmax": str(self.RETMAX),
            "sort": "date",
        }
        resp = client.get(f"{self.BASE}/esearch.fcgi", params=params)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("esearchresult") or {}).get("idlist") or []

    @_RETRY
    def _efetch(self, client: httpx.Client, ids: list[str]) -> str:
        params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "rettype": "abstract",
            "retmode": "xml",
        }
        resp = client.get(f"{self.BASE}/efetch.fcgi", params=params)
        resp.raise_for_status()
        return resp.text

    def _parse_article(self, art: ET.Element, spec: SourceSpec) -> Item | None:
        pmid_el = art.find(".//PMID")
        pmid = (pmid_el.text or "").strip() if pmid_el is not None else ""
        if not pmid:
            return None
        title_el = art.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""
        if not title:
            return None
        abstract_parts: list[str] = []
        for ab in art.findall(".//Abstract/AbstractText"):
            label = ab.attrib.get("Label")
            text = "".join(ab.itertext()).strip()
            if not text:
                continue
            abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(abstract_parts)

        authors_list: list[str] = []
        for au in art.findall(".//Author"):
            last = au.findtext("LastName") or ""
            initials = au.findtext("Initials") or ""
            collective = au.findtext("CollectiveName") or ""
            name = collective or f"{last} {initials}".strip()
            if name:
                authors_list.append(name)
        authors = ", ".join(authors_list)

        pub_dt: datetime | None = None
        pubdate = art.find(".//PubDate")
        if pubdate is not None:
            year = pubdate.findtext("Year")
            month = pubdate.findtext("Month") or "1"
            day = pubdate.findtext("Day") or "1"
            if year is None:
                medline = pubdate.findtext("MedlineDate") or ""
                m_year = re.search(r"(\d{4})", medline)
                if m_year:
                    year = m_year.group(1)
            if year:
                try:
                    m = int(month) if month.isdigit() else _MONTH_ABBR.get(month.lower()[:3], 1)
                    d = int(day) if day.isdigit() else 1
                    pub_dt = datetime(int(year), m, d, 12, 0, 0, tzinfo=timezone.utc)
                except Exception:
                    pub_dt = None

        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        return Item(
            source=spec.name,
            section=spec.section,
            external_id=pmid,
            url=url,
            title=title,
            abstract=abstract,
            authors=authors,
            published_at=pub_dt,
        )
