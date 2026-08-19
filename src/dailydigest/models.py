from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

_VALID_SECTIONS = {
    "research",
    "industry",
    "ai",
    "regulatory",
    "world",
    "opportunities",
    "events",
}


class Item(BaseModel):
    """Normalized article/paper item flowing through the pipeline."""

    source: str
    section: str
    external_id: str
    url: str
    title: str
    abstract: str = ""
    authors: str = ""
    published_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    score: float | None = None
    digest_id: str | None = None

    @field_validator("abstract", mode="before")
    @classmethod
    def _cap_abstract(cls, v: object) -> str:
        if not v:
            return ""
        return str(v)[:4000]

    @field_validator("section")
    @classmethod
    def _validate_section(cls, v: str) -> str:
        if v not in _VALID_SECTIONS:
            import logging
            logging.getLogger(__name__).warning(
                "Unknown section %r — item will not appear in rendered email. "
                "Valid sections: %s", v, sorted(_VALID_SECTIONS)
            )
        return v


class CanonicalFacet(BaseModel):
    """One named research interest used for attribution, not retrieval.

    ``anchors`` are specific descriptions of the work the reader wants.
    ``aliases`` retain related retrieval vocabulary. ``priority`` is a raw
    relative ordering preference and never changes relevance.
    """

    anchors: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    priority: float | None = Field(default=None, ge=0.0)


class Profile(BaseModel):
    name: str = ""
    bio: str
    keywords: list[str] = Field(default_factory=list)
    # "Keep-me-informed" interests that are context, not core research: they drive
    # retrieval and the news/industry sections, but are DOWN-WEIGHTED for the
    # research-relevance gate so a paper matching only a peripheral term (e.g. a
    # single-cell cancer-transcriptomics paper matching "single-cell RNA
    # sequencing") does not take a research slot from the reader's actual field.
    context_keywords: list[str] = Field(default_factory=list)
    # Optional per-interest weights. Keys are embedded as additional profile
    # facets, so broad interests can be strengthened or softened without
    # duplicating keywords. Values below 1.0 downweight; above 1.0 upweight.
    interest_weights: dict[str, float] = Field(default_factory=dict)
    # Backward-compatible alias used in older docs/conversations.
    facet_weights: dict[str, float] = Field(default_factory=dict)
    downweight: list[str] = Field(default_factory=list)
    # Authors / labs / institutions to follow. Items whose author byline matches
    # any entry are boosted in ranking and exposed as a learnable feature.
    authors_of_interest: list[str] = Field(default_factory=list)
    # Representative texts (your own paper titles+abstracts, or exemplar papers)
    # used as high-weight positive anchors for the interest vector — a far
    # stronger query than a keyword bag.
    seed_works: list[str] = Field(default_factory=list)
    # Interests to suppress: embedded as negative query vectors during ranking.
    # Items semantically similar to these topics will be penalized.
    # Example: {"cryptocurrency": 1.0, "sports": 1.0}
    negative_interests: dict[str, float] = Field(default_factory=dict)
    # Raw relative priority per core interest (keys should match `keywords`).
    # ONLY affects ordering (a small nudge to the final score), NEVER the
    # relevance gate. Normalized so the top interest = 1.0; unlisted keywords
    # default to a moderate 0.5. Empty dict = uniform (inert on ordering).
    topic_priorities: dict[str, float] = Field(default_factory=dict)
    # Named, non-overlapping interests for attribution and coverage. They do
    # not replace ``keywords``: source retrieval continues to use those terms.
    # Empty preserves the historical one-facet-per-keyword attribution.
    canonical_facets: dict[str, CanonicalFacet] = Field(default_factory=dict)


class SourceSpec(BaseModel):
    name: str
    kind: str
    url: str | None = None
    server: str | None = None
    section: str = "research"

    @field_validator("section")
    @classmethod
    def _validate_section(cls, v: str) -> str:
        if v not in _VALID_SECTIONS:
            raise ValueError(
                f"unknown section {v!r}; expected one of {sorted(_VALID_SECTIONS)}"
            )
        return v
    # Optional ranking metadata. These are deliberately coarse, stable knobs:
    # exact impact factors drift yearly, while source tiers are maintainable.
    quality_tier: str | None = None
    prestige_score: float | None = Field(default=None, ge=0.0, le=1.0)
    impact_floor: float | None = Field(default=None, ge=0.0)
    promo_risk: float | None = Field(default=None, ge=0.0, le=1.0)
    # Mark a source whose articles sit behind a paywall / subscription. Such
    # items take a ranking penalty so they must be more relevant to earn a slot —
    # most can't be read without a subscription.
    paywalled: bool = False
    # Drive this source's query from the user's profile keywords instead of a
    # static ``query``. For OpenAlex/PubMed this turns the source into an active
    # topic search across all venues (not just the curated feed list).
    profile_driven: bool = False
    # Phase 3 optional fields used by additional ingest adapters.
    category: str | None = None
    query: str | None = None
    condition: str | None = None
    endpoint: str | None = None
    polite_email: str | None = None
    lookahead_days: int = Field(default=180, ge=1, le=730)
    # OpenAlex source (venue) ids for the ``openalex_venues`` kind — a retrieval
    # channel that pulls recent articles directly from specific high-value
    # journals (e.g. ACS Nano, JACS) whose native RSS is Cloudflare-blocked.
    # Each fetched item is tagged with its real journal name so it earns the
    # correct venue prestige rather than aggregator tier.
    venue_ids: list[str] = Field(default_factory=list)
    # OpenAlex work types to accept, pipe-separated (OpenAlex filter syntax).
    # Defaults to "article". Preprint repositories index their output as
    # ``type:preprint``, so a ChemRxiv/OSF venue channel returns nothing without
    # this override.
    openalex_types: str = "article"
