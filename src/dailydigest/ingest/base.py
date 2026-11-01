from __future__ import annotations

from importlib import import_module
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


# kind -> (module, class). Imports stay lazy so optional adapter dependencies
# remain off the hot import path.
_ADAPTERS = {
    "rss": ("rss", "RSSSource"), "biorxiv": ("biorxiv", "BiorxivSource"),
    "arxiv": ("arxiv", "ArxivSource"), "openalex": ("openalex", "OpenAlexSource"),
    "openalex_authors": ("openalex", "OpenAlexAuthorsSource"),
    "openalex_venues": ("openalex", "OpenAlexVenuesSource"), "pubmed": ("pubmed", "PubMedSource"),
    "fda_api": ("fda", "FDASource"), "clinicaltrials": ("clinicaltrials", "ClinicalTrialsSource"),
    "grants_gov": ("grants_gov", "GrantsGovSource"), "events_rss": ("events_rss", "EventsRSSSource"),
    "eu_funding": ("eu_funding", "EUFundingSource"), "nih_guide": ("nih_guide", "NIHGuideSource"),
    "embl_events": ("events_api", "EMBLEventsSource"), "embo_events": ("events_api", "EMBOEventsSource"),
    "curated_calls": ("curated", "CuratedCallsSource"), "curated_events": ("curated", "CuratedEventsSource"),
    "ukri_funding": ("national_funding", "UKRIFundingSource"),
    "snsf_funding": ("national_funding", "SNSFFundingSource"),
    "isscr_events": ("event_pages", "ISSCREventsSource"),
    "keystone_events": ("event_pages", "KeystoneEventsSource"),
    "cshl_events": ("event_pages", "CSHLEventsSource"),
    "nature_conferences": ("event_pages", "NatureConferencesSource"),
    "wellcome_cs_events": ("event_pages", "WellcomeConnectingScienceSource"),
    "crs_events": ("event_pages", "CRSEventsSource"), "febs_events": ("event_pages", "FEBSEventsSource"),
    "eci_events": ("event_pages", "ECIEventsSource"),
    "biophysics_events": ("event_pages", "BiophysicalSocietyEventsSource"),
    "faseb_src_events": ("event_pages", "FASEBConferencesSource"),
}

# Public funding and event listings that are identical for every reader: they
# take no profile and ignore the look-back window. The browser deployment's
# daily snapshot carries them, so a site that changes or drops its CORS headers
# does not break browser brews; readers fall back to a direct fetch when the
# snapshot record is missing, failed or stale. Sources marked ``local_only``
# are still never collected.
SNAPSHOT_PAGE_KINDS = frozenset({
    "ukri_funding", "snsf_funding", "embl_events", "embo_events", "isscr_events",
    "keystone_events", "cshl_events", "nature_conferences", "wellcome_cs_events",
    "crs_events", "febs_events", "eci_events", "faseb_src_events",
})


def source_class(kind: str) -> type:
    try:
        module, name = _ADAPTERS[kind]
    except KeyError:
        raise ValueError(f"Unknown source kind: {kind}") from None
    return getattr(import_module(f"{__package__}.{module}"), name)


def dispatch_source(spec: SourceSpec) -> Source:
    return source_class(spec.kind)()
