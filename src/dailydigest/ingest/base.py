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
    raise ValueError(f"Unknown source kind: {spec.kind}")
