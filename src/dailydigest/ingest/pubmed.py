from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET  # types/ParseError only; parsing uses defusedxml
from datetime import datetime, timezone
from typing import Any

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


def _profile_query(term: str, aliases: list[str]) -> str:
    """Give exact interest phrases and explicit aliases first access to slots."""
    phrases = [term, *aliases[:3]]
    return "(" + " OR ".join(dict.fromkeys(
        f'"{phrase.replace(chr(34), "")}"[tiab]' for phrase in phrases if phrase.strip()
    )) + ")"


def _profile_fallback_query(term: str, precise_query: str) -> str | None:
    """Broaden original words only; aliases remain in the exact first pass.

    PubMed supports prefix wildcards. Use at least five characters and never
    truncate short terms such as gene, drug, DNA or RNA. This is prefix matching,
    not a synonym/stemming model. Exclude precise matches so a second query
    cannot spend the remaining reserved budget retrieving them again.
    """
    words = list(dict.fromkeys(re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", term.casefold())))
    if not words or len(words) == 1 and len(words[0]) < 5:
        return None
    clauses = [f'{word + "*" if len(word) >= 5 else word}[tiab]' for word in words]
    return "(" + " AND ".join(clauses) + ") NOT " + precise_query


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
    MIN_PRECISE_RESULTS = 3

    def __init__(self) -> None:
        self.last_fetch_diagnostics: dict[str, Any] = {"queries": []}

    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        if spec.profile_driven:
            from ..config import load_profile
            from ._terms import profile_search_term_groups

            core, context = profile_search_term_groups(
                12, canonical_order=True, include_aliases=False
            )
            try:
                profile = load_profile()
            except Exception:  # profile_search_term_groups already logs failure
                profile = None
            canonical = getattr(profile, "canonical_facets", None) or {}
            # Search each primary interest separately, reserving a bounded share
            # before broad topics or context can exhaust the result pool.
            terms = core + context
            if terms:
                self.last_fetch_diagnostics = {"queries": [], "returned_count": 0,
                                               "item_budget": self.MAX_ITEMS}
                out: list[Item] = []
                seen: set[str] = set()
                remaining = self.MAX_ITEMS
                failures = 0
                attempted = 0
                core_attempted = core_failures = 0
                context_budget = min(20, len(context) * 5) if core else 0
                for index, term in enumerate(terms):
                    if index:
                        time.sleep(self.SLEEP)
                    remaining_group = (len(core) if index < len(core) else len(terms)) - index
                    available = remaining - context_budget if index < len(core) else remaining
                    cap = (available + remaining_group - 1) // remaining_group
                    if not cap:
                        self.last_fetch_diagnostics["queries"].append({
                            "interest": term, "query": term, "attempted": False,
                            "truncated": True, "stop_reason": "item_budget", "failure": None,
                        })
                        continue
                    facet = next((value for name, value in canonical.items() if name.casefold() == term.casefold()), None)
                    query = _profile_query(term, getattr(facet, "aliases", []))
                    attempted += 1
                    if index < len(core):
                        core_attempted += 1
                    child = type(self)()
                    child.RETMAX = child.MAX_ITEMS = cap
                    primary_failed = False
                    try:
                        fetched = child.fetch(spec.model_copy(update={"profile_driven": False, "query": query}), days)
                    except RuntimeError:
                        primary_failed = True
                        failures += 1
                        if index < len(core):
                            core_failures += 1
                        fetched = []
                    diagnostics = child.last_fetch_diagnostics.get("queries", [])
                    for diagnostic in diagnostics:
                        diagnostic.update(interest=term, search_stage="phrase")
                        self.last_fetch_diagnostics["queries"].append(diagnostic)
                    raw_used = max(len(fetched), sum(d.get("fetched_count", 0) for d in diagnostics))
                    room = cap - raw_used
                    fallback_query = _profile_fallback_query(term, query)
                    if (not primary_failed and len(fetched) < min(self.MIN_PRECISE_RESULTS, cap)
                            and room > 0 and fallback_query):
                        time.sleep(self.SLEEP)
                        fallback = type(self)()
                        fallback.RETMAX = fallback.MAX_ITEMS = room
                        try:
                            fetched.extend(fallback.fetch(spec.model_copy(update={
                                "profile_driven": False, "query": fallback_query,
                            }), days))
                        except RuntimeError:
                            # Keep precise results; expose optional expansion failure
                            # in its own query diagnostic rather than hiding it.
                            pass
                        for diagnostic in fallback.last_fetch_diagnostics.get("queries", []):
                            diagnostic.update(interest=term, search_stage="word_fallback")
                            self.last_fetch_diagnostics["queries"].append(diagnostic)
                    # Reserve the raw result budget, not only unique parsed rows.
                    remaining -= cap
                    for item in fetched:
                        if item.external_id not in seen:
                            seen.add(item.external_id)
                            out.append(item)
                self.last_fetch_diagnostics["returned_count"] = len(out)
                if core_attempted and core_failures == core_attempted:
                    raise RuntimeError(f"{spec.name}: all core profile searches failed")
                if attempted and failures == attempted:
                    raise RuntimeError(f"{spec.name}: all profile searches failed")
                return out
        self.last_fetch_diagnostics = {"queries": [], "returned_count": 0}
        query = spec.query or ""
        if not query:
            return []
        diagnostic: dict[str, Any] = {
            "query": query, "attempted": True, "request_count": 0,
            "fetched_count": 0, "returned_count": 0, "unique_yield": 0,
            "total_available": None, "exhausted": False, "truncated": False,
            "stop_reason": None, "failure": None,
            "oldest_publication_date": None, "newest_publication_date": None,
            "request_count_includes_retries": False,
            "indexing_delay": None,
            "skipped_invalid": 0,
        }
        self.last_fetch_diagnostics["queries"].append(diagnostic)
        self._reldate = max(1, days)
        out: list[Item] = []
        try:
            with httpx.Client(timeout=20.0, headers=self.HEADERS) as client:
                diagnostic["request_count"] += 1
                ids = self._esearch(client, query)
                diagnostic["fetched_count"] = len(ids)
                if not ids:
                    if diagnostic["total_available"]:
                        raise RuntimeError("PubMed returned no IDs for a nonempty result count")
                    diagnostic.update(
                        exhausted=not diagnostic["truncated"],
                        stop_reason="search_warning" if diagnostic["truncated"] else "exhausted",
                    )
                    return out
                time.sleep(self.SLEEP)
                diagnostic["request_count"] += 1
                xml_text = self._efetch(client, ids)
        except Exception as e:
            diagnostic.update(stop_reason="failure", failure=type(e).__name__, truncated=True)
            name = getattr(spec, "name", "PubMedSource")
            raise RuntimeError(
                f"{name} fetch failed: {type(e).__name__}: {str(e)[:200]}"
            ) from e

        if not xml_text:
            diagnostic.update(stop_reason="failure", failure="empty_xml", truncated=True)
            raise RuntimeError(f"{spec.name} returned empty PubMed XML for nonempty search")

        try:
            root = ET_fromstring(xml_text)
        except Exception as e:
            diagnostic.update(stop_reason="failure", failure="invalid_xml", truncated=True)
            raise RuntimeError(f"{spec.name} returned invalid PubMed XML") from e

        if root.tag != "PubmedArticleSet" or root.find(".//ERROR") is not None:
            diagnostic.update(stop_reason="failure", failure="invalid_response", truncated=True)
            raise RuntimeError(f"{spec.name} returned an invalid PubMed article response")

        articles = root.findall(".//PubmedArticle")
        if not articles:
            diagnostic.update(stop_reason="failure", failure="missing_articles", truncated=True)
            raise RuntimeError(f"{spec.name} returned no PubMed articles for nonempty search")
        for art in articles:
            try:
                item = self._parse_article(art, spec)
            except Exception:
                diagnostic["skipped_invalid"] += 1
                continue
            if item is not None:
                out.append(item)
            else:
                diagnostic["skipped_invalid"] += 1
            if len(out) >= self.MAX_ITEMS:
                break
        total = diagnostic["total_available"]
        if not out:
            diagnostic.update(stop_reason="failure", failure="unusable_articles", truncated=True)
            raise RuntimeError(f"{spec.name} returned no usable PubMed articles for nonempty search")
        if len(out) >= self.MAX_ITEMS:
            diagnostic.update(truncated=True, stop_reason="item_cap")
        elif diagnostic["skipped_invalid"]:
            diagnostic.update(truncated=True, stop_reason="parsing_loss")
        elif len(ids) >= self.RETMAX or (total is not None and total > len(ids)):
            diagnostic.update(truncated=True, stop_reason="search_cap")
        elif len(articles) < len(ids):
            diagnostic.update(truncated=True, stop_reason="incomplete_fetch")
        elif diagnostic["truncated"]:
            diagnostic.update(stop_reason="search_warning")
        elif total is not None and total <= len(ids):
            diagnostic.update(exhausted=True, stop_reason="exhausted")
        else:
            diagnostic.update(stop_reason="unknown_total")
        diagnostic.update(returned_count=len(out), unique_yield=len({item.external_id for item in out}))
        dates = [item.published_at.date().isoformat() for item in out if item.published_at]
        if dates:
            diagnostic.update(oldest_publication_date=min(dates), newest_publication_date=max(dates))
        self.last_fetch_diagnostics["returned_count"] = len(out)
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
            "sort": "pub date",
        }
        resp = client.get(f"{self.BASE}/esearch.fcgi", params=params)
        resp.raise_for_status()
        data = resp.json()
        result = data.get("esearchresult") if isinstance(data, dict) else None
        if not isinstance(result, dict) or not isinstance(result.get("idlist"), list):
            raise RuntimeError("Invalid PubMed search response")
        if result.get("ERROR") or result.get("errorlist") or data.get("error"):
            raise RuntimeError("PubMed search reported an error")
        count = result.get("count")
        if count is not None and self.last_fetch_diagnostics["queries"]:
            self.last_fetch_diagnostics["queries"][-1]["total_available"] = int(count)
        warninglist = result.get("warninglist") or {}
        warnings = {key: messages for key, messages in warninglist.items() if messages}
        if warnings:
            # Keep provider wording: limits and ignored phrases change what was
            # searched. A plain empty-result notice is informational instead.
            empty_notice = (
                not result["idlist"] and str(count) == "0"
                and set(warnings) == {"outputmessages"}
                and all(message == "No items found." for message in warnings["outputmessages"])
            )
            if self.last_fetch_diagnostics["queries"]:
                self.last_fetch_diagnostics["queries"][-1].update(
                    warnings=warnings, truncated=not empty_notice,
                )
            if not empty_notice:
                logger.warning("PubMed search warnings: %r", warnings)
        return result["idlist"]

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

        venue = (art.findtext(".//Journal/Title") or "").strip()
        publication_types = list(dict.fromkeys(
            text for element in art.findall("./MedlineCitation/Article/PublicationTypeList/PublicationType")
            if (text := "".join(element.itertext()).strip())
        ))
        retraction_relations = {
            element.attrib.get("RefType")
            for element in art.findall("./MedlineCitation/CommentsCorrectionsList/CommentsCorrections")
        }
        if "RetractionOf" in retraction_relations and "Retraction of Publication" not in publication_types:
            publication_types.append("Retraction of Publication")
        publication_metadata = {}
        if publication_types:
            publication_metadata["publication_types"] = publication_types
        # NLM's "In" relationship marks the withdrawn original; "From"
        # belongs to its corrected replacement and must not exclude that work.
        if retraction_relations & {"RetractionIn", "RetractedandRepublishedIn"} or any(
            value.casefold() == "retracted publication" for value in publication_types
        ):
            publication_metadata["is_retracted"] = True
        doi = ""
        for article_id in art.findall("./PubmedData/ArticleIdList/ArticleId"):
            if str(article_id.attrib.get("IdType") or "").lower() == "doi":
                doi = (article_id.text or "").strip().lower()
                if doi:
                    break
        if not doi:
            for location_id in art.findall(".//ELocationID"):
                if str(location_id.attrib.get("EIdType") or "").lower() == "doi":
                    doi = (location_id.text or "").strip().lower()
                    if doi:
                        break

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
            metadata={
                **{key: value for key, value in (("venue", venue), ("doi", doi)) if value},
                **publication_metadata,
            },
        )
