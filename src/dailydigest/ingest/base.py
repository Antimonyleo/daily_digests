from __future__ import annotations

from typing import Protocol

from ..models import Item, SourceSpec


class Source(Protocol):
    def fetch(self, spec: SourceSpec) -> list[Item]: ...


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
