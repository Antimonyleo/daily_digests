"""Lightweight source-quality and novelty heuristics for ranking.

The ranker already captures personal topic fit via embeddings and, after
enough feedback, a learned LR model. This module adds stable editorial signals:
journal/source reputation, novelty/urgency language, and promotional-language
penalties. It intentionally avoids live impact-factor lookups so daily brewing
stays local, fast, and reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ..models import SourceSpec


@dataclass(frozen=True)
class SourceQuality:
    prestige_score: float
    quality_tier: str
    impact_floor: float | None = None
    promo_risk: float = 0.0


DEFAULT_RESEARCH_PRESTIGE = 0.42
DEFAULT_NEWS_PRESTIGE = 0.58
LOW_RESEARCH_PRESTIGE = 0.50

TOP_TIER_NAMES = {
    "nature",
    "science",
    "cell",
    "nejm",
    "the lancet",
}

HIGH_TIER_PATTERNS = (
    "nature biotechnology",
    "nature methods",
    "nature medicine",
    "nature nanotechnology",
    "nature materials",
    "nature chemistry",
    "nature physics",
    "nature photonics",
    "nature catalysis",
    "nature energy",
    "nature sustainability",
    "science translational medicine",
    "science immunology",
    "science robotics",
    "cell host and microbe",
    "cell stem cell",
    "cell metabolism",
    "cancer cell",
    "immunity",
    "neuron",
    "molecular cell",
    "pnas",
    "nucleic acids research",
)

STRONG_TIER_PATTERNS = (
    "nature communications",
    "nature reviews",
    "science advances",
    "science signaling",
    "cell reports",
    "cell chemical biology",
    "cell systems",
    "cell genomics",
    "med (cell press)",
    "chem (cell press)",
    "jacs",
    "acs nano",
    "nano letters",
    "acs catalysis",
    "acs central science",
    "acs chemical biology",
    "chemistry of materials",
    "chemical science",
    "chem. soc. rev.",
    "energy and environmental science",
    "angew. chem.",
    "advanced materials",
    "advanced functional materials",
    "advanced science",
    "small (wiley)",
)

REPOSITORY_PATTERNS = (
    "biorxiv",
    "medrxiv",
    "arxiv",
    "openalex",
    "pubmed",
)

CREDIBLE_NEWS_PATTERNS = (
    "stat news",
    "endpoints",
    "fiercebiotech",
    "biopharma dive",
    "bbc",
    "al jazeera",
    "mit technology review",
)

HIGH_NOVELTY_TERMS = (
    "first-in-class",
    "first in class",
    "breakthrough",
    "pivotal phase 3",
    "phase 3",
    "phase iii",
    "approved",
    "approval",
    "survival benefit",
    "clinically meaningful",
    "practice-changing",
    "landmark",
    "curative",
    "resolved structure",
    "de novo",
    "single-cell atlas",
)

MODERATE_NOVELTY_TERMS = (
    "clinical trial",
    "trial results",
    "efficacy",
    "safety",
    "fda",
    "ema",
    "crispr",
    "base editing",
    "prime editing",
    "rna therapy",
    "gene therapy",
    "drug discovery",
    "new mechanism",
    "novel",
)

PROMOTIONAL_TERMS = (
    "sponsored",
    "sponsored content",
    "webinar",
    "white paper",
    "whitepaper",
    "partner content",
    "advertorial",
    "commercial launch",
    "product launch",
    "available now",
    "showcase",
    "booth",
    "register now",
    "pleased to announce",
    "today announced",
    "announces the launch",
    "ai discovery platform",
)


def _safe_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _row_text(row: Any) -> str:
    title = _safe_str(getattr(row, "title", ""))
    abstract = _safe_str(getattr(row, "abstract", ""))
    return f"{title} {abstract}".strip()


def _row_source(row: Any) -> str:
    return _safe_str(getattr(row, "source", ""))


def _row_section(row: Any) -> str:
    return _safe_str(getattr(row, "section", ""))


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _configured_sources() -> dict[str, SourceSpec]:
    try:
        from ..config import load_sources

        return {s.name.lower(): s for s in load_sources()}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _configured_sources_cached() -> dict[str, SourceSpec]:
    return _configured_sources()


def _tier_to_prestige(tier: str | None, section: str) -> float | None:
    if not tier:
        return None
    tier_lc = tier.lower().strip()
    if tier_lc in {"top", "elite"}:
        return 0.98
    if tier_lc in {"high", "flagship"}:
        return 0.88
    if tier_lc in {"strong", "reputable"}:
        return 0.74
    if tier_lc in {"repository", "preprint"}:
        return 0.55
    if tier_lc == "trusted-news":
        return 0.72
    if tier_lc == "news":
        return DEFAULT_NEWS_PRESTIGE
    if tier_lc == "self-published":
        return 0.42
    if tier_lc == "low":
        return 0.35
    return DEFAULT_RESEARCH_PRESTIGE if section == "research" else DEFAULT_NEWS_PRESTIGE


def infer_source_quality(source: str, section: str) -> SourceQuality:
    """Return stable quality metadata for a configured or ad-hoc source."""
    source_lc = source.lower().strip()
    section_lc = section.lower().strip()

    configured = _configured_sources_cached().get(source_lc)
    if configured is not None:
        prestige = configured.prestige_score
        if prestige is None:
            prestige = _tier_to_prestige(configured.quality_tier, section_lc)
        inferred = _infer_source_quality_by_name(source_lc, section_lc)
        return SourceQuality(
            prestige_score=_clip(float(prestige if prestige is not None else inferred.prestige_score)),
            quality_tier=configured.quality_tier or inferred.quality_tier,
            impact_floor=configured.impact_floor if configured.impact_floor is not None else inferred.impact_floor,
            promo_risk=_clip(
                float(
                    configured.promo_risk
                    if configured.promo_risk is not None
                    else inferred.promo_risk
                )
            ),
        )

    return _infer_source_quality_by_name(source_lc, section_lc)


def _infer_source_quality_by_name(source_lc: str, section_lc: str) -> SourceQuality:
    if section_lc == "research":
        if source_lc in TOP_TIER_NAMES:
            return SourceQuality(0.99, "top", 7.0)
        if any(pattern in source_lc for pattern in HIGH_TIER_PATTERNS):
            return SourceQuality(0.90, "high", 7.0)
        if any(pattern in source_lc for pattern in STRONG_TIER_PATTERNS):
            return SourceQuality(0.76, "strong", 7.0)
        if any(pattern in source_lc for pattern in REPOSITORY_PATTERNS):
            return SourceQuality(0.56, "repository", None)
        return SourceQuality(DEFAULT_RESEARCH_PRESTIGE, "unknown", None)

    if any(pattern in source_lc for pattern in CREDIBLE_NEWS_PATTERNS):
        return SourceQuality(0.72, "trusted-news", None)
    if "press release" in source_lc or "company" in source_lc:
        return SourceQuality(0.42, "self-published", None, promo_risk=0.35)
    return SourceQuality(DEFAULT_NEWS_PRESTIGE, "news", None)


def novelty_score(row: Any) -> float:
    text_lc = _row_text(row).lower()
    if not text_lc:
        return 0.0

    high_hits = sum(1 for term in HIGH_NOVELTY_TERMS if term in text_lc)
    moderate_hits = sum(1 for term in MODERATE_NOVELTY_TERMS if term in text_lc)
    score = min(0.75, high_hits * 0.24) + min(0.35, moderate_hits * 0.08)

    # Reviews can be valuable, but they are usually less urgent than primary
    # results unless they include clearly novel or clinical terms.
    source_lc = _row_source(row).lower()
    title_lc = _safe_str(getattr(row, "title", "")).lower()
    if ("review" in source_lc or re.search(r"\breview\b", title_lc)) and high_hits == 0:
        score -= 0.12

    return _clip(score)


def promotional_score(row: Any) -> float:
    text_lc = _row_text(row).lower()
    source_quality = infer_source_quality(_row_source(row), _row_section(row))
    if not text_lc:
        return _clip(source_quality.promo_risk)

    hits = sum(1 for term in PROMOTIONAL_TERMS if term in text_lc)
    score = min(0.75, hits * 0.22)
    score += source_quality.promo_risk
    return _clip(score)


def quality_adjusted_score(row: Any, base_score: float) -> float:
    """Blend topic fit with stable quality, novelty, and promo signals."""
    section = _row_section(row).lower()
    source_quality = infer_source_quality(_row_source(row), section)
    novelty = novelty_score(row)
    promo = promotional_score(row)
    base = float(base_score)

    if section == "research":
        score = (
            base
            + (0.18 * source_quality.prestige_score)
            + (0.12 * novelty)
            - (0.35 * promo)
        )
        low_prestige = source_quality.prestige_score < LOW_RESEARCH_PRESTIGE
        exceptional = base >= 0.78 and novelty >= 0.50
        if low_prestige and not exceptional:
            score -= 0.22
        return score

    if section in {"industry", "world"}:
        return base + (0.06 * source_quality.prestige_score) + (0.14 * novelty) - (0.45 * promo)

    if section == "regulatory":
        return base + (0.06 * source_quality.prestige_score) + (0.10 * novelty) - (0.45 * promo)

    return base + (0.06 * source_quality.prestige_score) + (0.10 * novelty) - (0.45 * promo)
