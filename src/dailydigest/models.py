from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


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

    score: float | None = None
    digest_id: str | None = None


class Profile(BaseModel):
    name: str = ""
    bio: str
    keywords: list[str] = Field(default_factory=list)
    downweight: list[str] = Field(default_factory=list)


class SourceSpec(BaseModel):
    name: str
    kind: str
    url: str | None = None
    server: str | None = None
    section: str = "research"
    # Optional ranking metadata. These are deliberately coarse, stable knobs:
    # exact impact factors drift yearly, while source tiers are maintainable.
    quality_tier: str | None = None
    prestige_score: float | None = Field(default=None, ge=0.0, le=1.0)
    impact_floor: float | None = Field(default=None, ge=0.0)
    promo_risk: float | None = Field(default=None, ge=0.0, le=1.0)
    # Phase 3 optional fields used by additional ingest adapters.
    category: str | None = None
    query: str | None = None
    condition: str | None = None
    endpoint: str | None = None
    polite_email: str | None = None
