from __future__ import annotations

from typing import Protocol

from ..models import Item, SourceSpec


class Source(Protocol):
    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        """Fetch items for ``spec``.

        ``days`` is the look-back window in days for date-capable APIs (bioRxiv,
        arXiv, OpenAlex, PubMed, FDA, ClinicalTrials); it widens after a usage
        gap so a backlog is covered. RSS feeds ignore it (they return whatever
        the feed currently exposes).
        """
        ...


_NATIONAL_FUNDING = {"ukri_funding": "UKRIFundingSource", "snsf_funding": "SNSFFundingSource"}
_EVENT_PAGES = {
    "isscr_events": "ISSCREventsSource", "keystone_events": "KeystoneEventsSource",
    "cshl_events": "CSHLEventsSource", "nature_conferences": "NatureConferencesSource",
    "wellcome_cs_events": "WellcomeConnectingScienceSource", "crs_events": "CRSEventsSource",
    "febs_events": "FEBSEventsSource", "eci_events": "ECIEventsSource",
    "biophysics_events": "BiophysicalSocietyEventsSource", "faseb_src_events": "FASEBConferencesSource",
}


def dispatch_source(spec: SourceSpec) -> Source:
    # Lazy imports keep optional adapter deps off the hot import path.
    if spec.kind == "rss":
        from .rss import RSSSource

        return RSSSource()
    if spec.kind == "biorxiv":
        from .biorxiv import BiorxivSource

        return BiorxivSource()
    if spec.kind == "arxiv":
        from .arxiv import ArxivSource

        return ArxivSource()
    if spec.kind == "openalex":
        from .openalex import OpenAlexSource

        return OpenAlexSource()
    if spec.kind == "openalex_authors":
        from .openalex import OpenAlexAuthorsSource

        return OpenAlexAuthorsSource()
    if spec.kind == "openalex_venues":
        from .openalex import OpenAlexVenuesSource

        return OpenAlexVenuesSource()
    if spec.kind == "pubmed":
        from .pubmed import PubMedSource

        return PubMedSource()
    if spec.kind == "fda_api":
        from .fda import FDASource

        return FDASource()
    if spec.kind == "clinicaltrials":
        from .clinicaltrials import ClinicalTrialsSource

        return ClinicalTrialsSource()
    if spec.kind == "grants_gov":
        from .grants_gov import GrantsGovSource

        return GrantsGovSource()
    if spec.kind == "events_rss":
        from .events_rss import EventsRSSSource

        return EventsRSSSource()
    if spec.kind == "eu_funding":
        from .eu_funding import EUFundingSource

        return EUFundingSource()
    if spec.kind == "nih_guide":
        from .nih_guide import NIHGuideSource

        return NIHGuideSource()
    if spec.kind == "embl_events":
        from .events_api import EMBLEventsSource

        return EMBLEventsSource()
    if spec.kind == "embo_events":
        from .events_api import EMBOEventsSource

        return EMBOEventsSource()
    if spec.kind == "curated_calls":
        from .curated import CuratedCallsSource

        return CuratedCallsSource()
    if spec.kind == "curated_events":
        from .curated import CuratedEventsSource

        return CuratedEventsSource()
    if spec.kind in _NATIONAL_FUNDING:
        from . import national_funding

        return getattr(national_funding, _NATIONAL_FUNDING[spec.kind])()
    if spec.kind in _EVENT_PAGES:
        from . import event_pages

        return getattr(event_pages, _EVENT_PAGES[spec.kind])()
    raise ValueError(f"Unknown source kind: {spec.kind}")
