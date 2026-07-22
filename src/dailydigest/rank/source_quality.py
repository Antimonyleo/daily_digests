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

RANKER_VERSION = "2026-05-15-ranking-improvements-v5"

_ARXIV_CS_RE = re.compile(r"\barxiv[\s:_\-]*cs[\s:./\-]", re.IGNORECASE)


@dataclass(frozen=True)
class SourceQuality:
    prestige_score: float
    quality_tier: str
    impact_floor: float | None = None
    promo_risk: float = 0.0
    paywalled: bool = False


# Ranking penalty applied to paywalled industry/world items so they must clear a
# higher relevance bar to appear (most can't be read without a subscription).
PAYWALL_PENALTY = 0.15


@dataclass(frozen=True)
class ScoreBreakdown:
    topic: float
    source: float
    novelty: float
    learned: float
    penalty: float
    final: float
    tags: tuple[str, ...]
    promo_penalty: float = 0.0
    access_penalty: float = 0.0
    reason_penalty: float = 0.0
    content_type: str = "article"
    freshness_tags: tuple[str, ...] = ()
    quality_tags: tuple[str, ...] = ()
    why_shown: tuple[str, ...] = ()


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
    # Flagship physical-science journals whose real impact rivals the Nature
    # sub-journals above (ACS Nano & JACS 2yr-mean-citedness ≈ 15.6, matching
    # Nature; Nano Letters ≈ 8.4). Previously mis-tiered as merely "strong",
    # which — under score compression at the top of the research section — left
    # them just outside the cut every day despite being core venues for the
    # reader's nanoscience / self-assembly / materials field.
    "acs nano",
    "nano letters",
    "journal of the american chemical society",
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
    "jacs",  # bare token keeps JACS Au (2yr ≈ 7.1) at strong; JACS itself is high-tier
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
    # Domain-neutral advance markers so a physical-science / biomolecular-design
    # result (self-assembly, photonics, protein design) gets fair novelty credit
    # instead of the previous clinical-only vocabulary, which starved this
    # reader's field of novelty and skewed the bonus toward the very clinical
    # content they set as negative interests.
    "first demonstration",
    "record efficiency",
    "unprecedented",
    "orders of magnitude",
    "first realization",
    "long-sought",
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

ACCESS_FRICTION_TERMS = (
    "sign up",
    "signup",
    "log in",
    "login",
    "subscribe",
    "subscription required",
    "requires subscription",
    "member-only",
    "members only",
    "membership required",
    "purchase access",
)

COMMENTARY_TERMS = (
    "commentary",
    "editorial",
    "viewpoint",
    "opinion",
    "perspective",
    "news and views",
    "news & views",
    "correspondence",
    "letter to the editor",
    "podcast",
)

CONTENT_TYPE_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("editorial", ("editorial", "opinion", "viewpoint", "perspective", "news and views", "news & views")),
    ("commentary", ("commentary", "correspondence", "letter to the editor")),
    ("podcast", ("podcast",)),
    ("review", ("review", "primer", "overview")),
    ("clinical", ("clinical trial", "trial results", "phase 3", "phase iii", "approval", "approved")),
    ("method", ("method", "methods", "protocol", "platform", "assay")),
    ("dataset", ("dataset", "atlas")),
    ("structure", ("structure", "cryo-em")),
)

METHOD_OR_RESULT_TERMS = (
    "method",
    "methods",
    "protocol",
    "platform",
    "screen",
    "assay",
    "dataset",
    "atlas",
    "structure",
    "cryo-em",
    "single-cell",
    "approved",
    "approval",
    "efficacy",
    "survival",
    "mechanism",
)

SKIP_PATTERNS = (
    re.compile(r"^\s*issue\s+(?:publication\s+information|editorial\s+masthead)\s*$", re.IGNORECASE),
    re.compile(r"^\s*introducing\s+our\s+authors\s*$", re.IGNORECASE),
    re.compile(r"\bfront\s+cover\b", re.IGNORECASE),
    re.compile(r"\binside\s+cover\b", re.IGNORECASE),
    re.compile(r"\bback\s+cover\b", re.IGNORECASE),
    re.compile(r"\bcover\s+(picture|image|profile|feature|art|story)\b", re.IGNORECASE),
)

# Front-matter / non-primary-research content that high-impact journal RSS feeds
# (Nature, Nature Biotechnology, Science, …) publish alongside real papers. These
# ride the journal's prestige into the research section despite not being primary
# research. Matched on the TITLE (structural markers), so genuine papers whose
# abstract merely mentions e.g. "correction" are unaffected.
NON_RESEARCH_TITLE_PATTERNS = (
    re.compile(r"^\s*(author\s+)?correction(\s+to)?\b[:.]?", re.IGNORECASE),
    re.compile(r"^\s*publisher\s+correction\b", re.IGNORECASE),
    re.compile(r"^\s*erratum\b", re.IGNORECASE),
    re.compile(r"^\s*retraction(\s+note)?\b", re.IGNORECASE),
    re.compile(r"^\s*addendum\b", re.IGNORECASE),
    re.compile(r"^\s*editorial\b[:.]?", re.IGNORECASE),
    re.compile(r"\bnews\s*(&|and)\s*views\b", re.IGNORECASE),
    re.compile(r"\bresearch\s+highlights?\b", re.IGNORECASE),
    re.compile(r"\bnews\s+(feature|in\s+brief|round[\s-]?up)\b", re.IGNORECASE),
    re.compile(r"\bnews\s+from\s+around\s+the\s+world\b", re.IGNORECASE),
    re.compile(r"^\s*in\s+this\s+issue\b", re.IGNORECASE),
    re.compile(r"^\s*(this\s+(week|month)\s+in|the\s+week\s+in)\b", re.IGNORECASE),
    re.compile(r"^\s*(technology\s+feature|toolbox|outlook|the\s+last\s+word)\b", re.IGNORECASE),
    re.compile(r"\bpodcast\b", re.IGNORECASE),
    # Career-advice columns (Nature/Science Careers) — first-person Q&A and
    # "how to ... your career/research" advice. High-precision: these phrasings
    # never occur in a primary-research title.
    re.compile(r"\bhow\s+do\s+i\b", re.IGNORECASE),
    re.compile(r"\bfuture[\s-]?proof\b", re.IGNORECASE),
    re.compile(r"\bjob\s+interviews?\b", re.IGNORECASE),
    re.compile(r"\bmentorship\b", re.IGNORECASE),
    re.compile(r"\brecipe\s+for\s+success\b", re.IGNORECASE),
    re.compile(r"\bwork[\s–-]life\b", re.IGNORECASE),
    re.compile(r"\ba\s+day\s+in\s+the\s+life\b", re.IGNORECASE),
    re.compile(
        r"\bhow\s+to\s+(write|get|land|choose|ace|survive|balance|manage|make|build|network|negotiate)\b.*"
        r"\b(job|career|cv|r[eé]sum[eé]|manuscript|paper|research|lab|grant|postdoc|ph\.?d|supervisor|mentor)\b",
        re.IGNORECASE,
    ),
    # News-desk items that ride a research journal's feed (politics/legal, not
    # science). Research-section only, so these markers won't hit real papers.
    re.compile(r"\b(criminal|smuggling|fraud)\s+charges\b", re.IGNORECASE),
    re.compile(r"\b(political|public)\s+(uproar|outcry|backlash)\b", re.IGNORECASE),
    re.compile(r"\bindicted\b", re.IGNORECASE),
)


def is_non_research_content(row: Any) -> bool:
    """Return True for research-feed items that are not primary research.

    News, News & Views, research highlights, editorials, corrections, retractions,
    and news round-ups are dropped from the research section (they can still reach
    the news sections). Only applies to the research section.
    """
    if _row_section(row).lower() != "research":
        return False
    title = _safe_str(getattr(row, "title", "")).strip()
    if not title:
        return False
    return any(p.search(title) for p in NON_RESEARCH_TITLE_PATTERNS)


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
    if tier_lc == "aggregator":
        return 0.34
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
            paywalled=bool(getattr(configured, "paywalled", False)),
        )

    return _infer_source_quality_by_name(source_lc, section_lc)


def recognized_research_venue(venue_name: str | None) -> str | None:
    """Return ``venue_name`` when it names a known top/high/strong research venue.

    Used to *upgrade* aggregator-sourced items (OpenAlex/PubMed) to their real
    journal identity so an ACS Nano or Nature paper arriving via a topic search is
    scored as the prestigious venue it is, not as a generic aggregator hit. Returns
    None for unknown/low-impact venues so their aggregator attribution is kept.
    """
    if not venue_name:
        return None
    v = venue_name.lower().strip()
    if not v:
        return None
    if v in TOP_TIER_NAMES:
        return venue_name
    if any(p in v for p in HIGH_TIER_PATTERNS) or any(p in v for p in STRONG_TIER_PATTERNS):
        return venue_name
    return None


def _infer_source_quality_by_name(source_lc: str, section_lc: str) -> SourceQuality:
    if section_lc == "research":
        if "openalex" in source_lc:
            return SourceQuality(0.34, "aggregator", None)
        if is_arxiv_cs_source(source_lc):
            return SourceQuality(0.48, "repository", None)
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

    high_contrib = min(0.75, high_hits * 0.24)
    # Hype words ("breakthrough", "landmark", "practice-changing") overlap with
    # promotional language and are easily gamed. Only let them count fully when
    # the text also carries structured substance — a method/result signal or
    # numbers. Substance-free hype is halved so marketing copy cannot masquerade
    # as a high-novelty result.
    has_substance = any(term in text_lc for term in METHOD_OR_RESULT_TERMS) or any(
        ch.isdigit() for ch in text_lc
    )
    if high_hits > 0 and not has_substance:
        high_contrib *= 0.5
    score = high_contrib + min(0.35, moderate_hits * 0.08)

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
    access_hits = sum(1 for term in ACCESS_FRICTION_TERMS if term in text_lc)
    score = min(0.75, hits * 0.22) + min(0.35, access_hits * 0.14)
    score += source_quality.promo_risk
    return _clip(score)


def access_friction_score(row: Any) -> float:
    text_lc = _row_text(row).lower()
    if not text_lc:
        return 0.0
    hits = sum(1 for term in ACCESS_FRICTION_TERMS if term in text_lc)
    return _clip(min(0.35, hits * 0.14))


def is_arxiv_cs_source(row_or_source) -> bool:
    """Return True when the source string looks like an arXiv CS feed."""
    source = (
        row_or_source
        if isinstance(row_or_source, str)
        else (getattr(row_or_source, "source", None) or "")
    )
    return bool(_ARXIV_CS_RE.search(source))


def source_bucket(row: Any) -> str:
    """Return a coarse source class for final digest balancing."""
    source_lc = _row_source(row).lower()
    section = _row_section(row).lower()
    if section != "research":
        return section or "other"
    # Live venue-impact enrichment (when enabled) can flag an item whose actual
    # publication venue is low impact — this is the only way to catch low-impact
    # papers arriving through aggregators (OpenAlex/PubMed) whose source name
    # hides the real journal. Honor it so the low-impact frequency cap applies.
    if getattr(row, "venue_low_impact", False) is True:
        return "low_impact_journal"
    if is_arxiv_cs_source(source_lc):
        return "arxiv_cs"
    if "arxiv" in source_lc:
        return "arxiv_other"
    if "biorxiv" in source_lc or "medrxiv" in source_lc:
        return "bio_med_preprint"
    if "openalex" in source_lc:
        return "aggregator"
    if "pubmed" in source_lc:
        return "published_database"

    quality = infer_source_quality(_row_source(row), section)
    if quality.quality_tier in {"top", "high", "strong"}:
        return "published_journal"
    # Catch remaining preprint servers (ChemRxiv, SSRN, Research Square, …)
    # that aren't matched by the explicit string checks above.
    if quality.quality_tier in {"repository", "preprint"}:
        return "preprint_other"
    # Unknown / low-prestige peer-reviewed venues. These are the low-impact
    # journals that should appear only rarely even when topically relevant.
    if quality.prestige_score < LOW_RESEARCH_PRESTIGE:
        return "low_impact_journal"
    return "other_research"


def is_low_impact_research(row: Any) -> bool:
    """Return True for low-impact / unknown peer-reviewed research venues."""
    return source_bucket(row) == "low_impact_journal"


def is_preprint_source(row: Any) -> bool:
    return source_bucket(row) in {"arxiv_cs", "arxiv_other", "bio_med_preprint", "preprint_other"}


def is_published_journal_source(row: Any) -> bool:
    return source_bucket(row) == "published_journal"


def is_high_quality_journal_source(row: Any) -> bool:
    if _row_section(row).lower() != "research":
        return False
    quality = infer_source_quality(_row_source(row), "research")
    return quality.quality_tier in {"top", "high", "strong"}


def content_type(row: Any) -> str:
    """Return a compact content-type label inferred from source text."""
    text_lc = _row_text(row).lower()
    for label, terms in CONTENT_TYPE_TERMS:
        if any(term in text_lc for term in terms):
            return label
    return "research" if _row_section(row).lower() == "research" else "article"


def should_skip_item(row: Any) -> bool:
    """Return True for feed entries that should not enter ranking at all."""
    section = (getattr(row, "section", None) or "").lower()
    title = (getattr(row, "title", None) or "").strip()
    if not title:
        return True
    if section == "research":
        source_lc = _row_source(row).lower()
        text = _row_text(row)
        title_lc = title.lower()
        if title_lc in {
            "issue publication information",
            "issue editorial masthead",
            "introducing our authors",
        }:
            return True
        if any(pattern.search(text) for pattern in SKIP_PATTERNS):
            return True
        # News / News & Views / highlights / corrections / editorials that ride a
        # journal's prestige into the research section but are not primary research.
        if is_non_research_content(row):
            return True
        text_lc = text.lower()
        is_commentary = any(term in text_lc for term in COMMENTARY_TERMS)
        has_new_information = (
            novelty_score(row) >= 0.24
            or any(term in text_lc for term in METHOD_OR_RESULT_TERMS)
        )
        if is_commentary and not has_new_information:
            return True
    return False


def _reason_tags(
    row: Any,
    source_quality: SourceQuality,
    novelty: float,
    promo: float,
    access: float,
    learned: float,
) -> tuple[str, ...]:
    tags: list[str] = []
    section = _row_section(row).lower()
    tier = source_quality.quality_tier
    if tier in {"top", "high", "strong"}:
        tags.append("High-quality source")
    elif tier in {"repository", "preprint"}:
        tags.append("Preprint / repository")
    elif tier == "aggregator":
        tags.append("Aggregator source")
    elif tier == "trusted-news":
        tags.append("Trusted news")

    if novelty >= 0.55:
        tags.append("High novelty")
    elif novelty >= 0.24:
        tags.append("Fresh signal")

    if section == "regulatory":
        tags.append("Regulatory update")
    if learned >= 0.65:
        tags.append("Matches learned preferences")
    if promo >= 0.35:
        tags.append("Promo risk")
    if access >= 0.14:
        tags.append("Access friction")
    if is_arxiv_cs_source(row):
        tags.append("arXiv CS")
    return tuple(tags[:5])


def _quality_tags(source_quality: SourceQuality, promo: float, access: float, row: Any) -> tuple[str, ...]:
    tags: list[str] = []
    tier = source_quality.quality_tier
    if tier in {"top", "high", "strong"}:
        tags.append("high_quality_source")
    elif tier in {"repository", "preprint"}:
        tags.append("preprint_repository")
    elif tier == "aggregator":
        tags.append("aggregator_source")
    elif tier == "trusted-news":
        tags.append("trusted_news")
    if promo >= 0.35:
        tags.append("promo_risk")
    if access >= 0.14:
        tags.append("access_friction")
    if is_arxiv_cs_source(row):
        tags.append("arxiv_cs")
    return tuple(tags)


def _freshness_tags(novelty: float) -> tuple[str, ...]:
    if novelty >= 0.55:
        return ("high_novelty",)
    if novelty >= 0.24:
        return ("fresh_signal",)
    return ()


def _why_shown(
    *,
    topic: float,
    learned: float,
    source_quality: SourceQuality,
    novelty: float,
    promo: float,
    access: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if topic >= 0.65:
        reasons.append("Strong topic match")
    elif topic >= 0.45:
        reasons.append("Relevant to your profile")
    if learned >= 0.65:
        reasons.append("Matches your feedback")
    if source_quality.quality_tier in {"top", "high", "strong", "trusted-news"}:
        reasons.append("Reliable source")
    if novelty >= 0.55:
        reasons.append("High-impact new signal")
    elif novelty >= 0.24:
        reasons.append("Fresh development")
    if promo >= 0.35:
        reasons.append("Downweighted for promotional language")
    if access >= 0.14:
        reasons.append("Downweighted for access friction")
    return tuple(reasons[:5])


def score_breakdown(
    row: Any,
    base_score: float,
    learned_score: float = 0.0,
    reason_penalty: float = 0.0,
    negative_interest_penalty: float = 0.0,
) -> ScoreBreakdown:
    """Return display/debug components for a ranked item.

    ``base_score`` is the profile/topic score before quality adjustments. The
    returned components are clipped to 0..1 so they work directly as bar widths.
    """
    section = _row_section(row).lower()
    source_quality = infer_source_quality(_row_source(row), section)
    novelty = novelty_score(row)
    promo = promotional_score(row)
    access = access_friction_score(row)
    final = quality_adjusted_score(row, base_score)
    reason_penalty = _clip(float(reason_penalty))
    negative_interest_penalty = _clip(float(negative_interest_penalty))
    if reason_penalty:
        final -= reason_penalty
    if negative_interest_penalty:
        final -= negative_interest_penalty
    penalty = _clip(promo + reason_penalty + negative_interest_penalty)
    tags = _reason_tags(row, source_quality, novelty, promo, access, learned_score)
    quality_tags = _quality_tags(source_quality, promo, access, row)
    freshness_tags = _freshness_tags(novelty)
    topic = _clip(float(base_score))
    learned = _clip(float(learned_score))
    return ScoreBreakdown(
        topic=topic,
        source=_clip(source_quality.prestige_score),
        novelty=_clip(novelty),
        learned=learned,
        penalty=penalty,
        final=_clip(float(final)),
        tags=tags,
        promo_penalty=_clip(promo),
        access_penalty=_clip(access),
        reason_penalty=reason_penalty,
        content_type=content_type(row),
        freshness_tags=freshness_tags,
        quality_tags=quality_tags,
        why_shown=_why_shown(
            topic=topic,
            learned=learned,
            source_quality=source_quality,
            novelty=novelty,
            promo=promo,
            access=access,
        ),
    )


def display_breakdown(row: Any) -> ScoreBreakdown:
    """Best-effort breakdown for already persisted digest rows.

    We persist the final score but not the original profile vector similarity.
    For the UI, use the final score as the topic-match proxy and recompute the
    stable source/novelty/penalty components from item metadata.
    """
    raw_score = getattr(row, "score", None)
    base = float(raw_score) if isinstance(raw_score, (int, float)) else 0.5
    breakdown = score_breakdown(row, _clip(base))
    return ScoreBreakdown(
        topic=breakdown.topic,
        source=breakdown.source,
        novelty=breakdown.novelty,
        learned=breakdown.learned,
        penalty=breakdown.penalty,
        final=base,
        tags=breakdown.tags,
        promo_penalty=breakdown.promo_penalty,
        access_penalty=breakdown.access_penalty,
        reason_penalty=breakdown.reason_penalty,
        content_type=breakdown.content_type,
        freshness_tags=breakdown.freshness_tags,
        quality_tags=breakdown.quality_tags,
        why_shown=breakdown.why_shown,
    )


def breakdown_payload(
    row: Any,
    base_score: float,
    learned_score: float = 0.0,
    reason_penalty: float = 0.0,
    *,
    final_score: float | None = None,
    rank_score: float | None = None,
    confidence_score: float | None = None,
    negative_interest_penalty: float = 0.0,
    selection_reason: str | None = None,
    scoring_mode: str | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable rank explanation for persistence/UI."""
    breakdown = score_breakdown(
        row,
        base_score,
        learned_score=learned_score,
        reason_penalty=reason_penalty,
        negative_interest_penalty=negative_interest_penalty,
    )
    display_score = float(final_score if final_score is not None else breakdown.final)
    confidence = float(confidence_score if confidence_score is not None else display_score)
    return {
        "ranker_version": RANKER_VERSION,
        "score": round(display_score, 4),
        "rank_score": round(float(rank_score if rank_score is not None else display_score), 4),
        "confidence_score": round(confidence, 4),
        "topic": round(float(breakdown.topic), 4),
        "source": round(float(breakdown.source), 4),
        "novelty": round(float(breakdown.novelty), 4),
        "learned": round(float(breakdown.learned), 4),
        "penalty": round(float(breakdown.penalty), 4),
        "promo_penalty": round(float(breakdown.promo_penalty), 4),
        "access_penalty": round(float(breakdown.access_penalty), 4),
        "reason_penalty": round(float(breakdown.reason_penalty), 4),
        "negative_interest_penalty": round(float(negative_interest_penalty), 4),
        "content_type": breakdown.content_type,
        "source_bucket": source_bucket(row),
        "selection_reason": selection_reason or "",
        "scoring_mode": scoring_mode or "cosine",
        "tags": list(breakdown.tags),
        "quality_tags": list(breakdown.quality_tags),
        "freshness_tags": list(breakdown.freshness_tags),
        "why_shown": list(breakdown.why_shown),
    }


def _research_quality_weight() -> float:
    """Return the configured venue-quality weight (1.0 = legacy tie-breaker)."""
    try:
        from ..config import get_settings

        return float(get_settings().research_quality_weight)
    except Exception:  # noqa: BLE001
        return 1.4


def _venue_relevance_credit_coeff() -> float:
    try:
        from ..config import get_settings

        return float(get_settings().venue_relevance_credit)
    except Exception:  # noqa: BLE001
        return 0.10


def venue_relevance_credit(row: Any) -> float:
    """Topic-relevance credit granted to a high-quality research venue at the gate.

    A prestigious journal in the reader's field is a strong quality prior, so it
    should clear the ``min_topic_relevance`` floor at a slightly lower raw cosine
    than an anonymous preprint/aggregator hit. Returns ``coeff * prestige_excess``
    for research items (0 for news/preprints/aggregators, whose prestige_excess is
    small or zero), where ``prestige_excess = prestige - LOW_RESEARCH_PRESTIGE``.

    The credit is intentionally small so it only rescues *borderline* top-venue
    work (cosine just under the floor); a genuinely off-topic prestigious paper
    still fails the gate.
    """
    if _row_section(row).lower() != "research":
        return 0.0
    quality = infer_source_quality(_row_source(row), "research")
    # Preprints/aggregators do not get venue relevance credit even if their
    # nominal prestige sits above the low-impact line — the credit is a
    # peer-reviewed-venue signal.
    if quality.quality_tier not in {"top", "high", "strong"}:
        return 0.0
    prestige_excess = max(quality.prestige_score - LOW_RESEARCH_PRESTIGE, 0.0)
    return _venue_relevance_credit_coeff() * prestige_excess


def quality_adjusted_score(row: Any, base_score: float) -> float:
    """Blend topic fit with stable quality, novelty, and promo signals.

    Quality weighting (``research_quality_weight``, ``Q``) controls how much
    venue reputation matters. High-quality *and* relevant work is rewarded via a
    quality×relevance interaction term, while low-impact venues take a penalty
    that — unlike the legacy version — does not fully fade at high relevance, so
    a low-impact paper does not rank alongside strong-venue work merely for being
    on-topic. Truly exceptional results (high relevance + novelty) are exempt.

    The returned score is clipped to [0, 1] so the absolute thresholds that gate
    selection downstream (``adaptive_size_bar``, ``low_impact_relevance_floor``,
    the exceptional-preprint cutoff) operate on a stable, bounded scale.
    """
    section = _row_section(row).lower()
    source_quality = infer_source_quality(_row_source(row), section)
    novelty = novelty_score(row)
    promo = promotional_score(row)
    base = float(base_score)

    if section == "research":
        Q = _research_quality_weight()
        prestige_excess = max(source_quality.prestige_score - LOW_RESEARCH_PRESTIGE, 0.0)
        # Venue quality is applied ONLY as a RELEVANCE-GATED interaction: a
        # high-impact venue amplifies work that is already on-topic, but grants
        # little/no lift to an off-topic prestigious paper — so quality reorders
        # among comparably-relevant items yet can never let an off-niche journal
        # outrank a more on-topic one. This encodes the reader's rule "high-quality
        # AND relevant ranks higher; quality never substitutes for relevance."
        #
        # The prior formula added a relevance-INDEPENDENT source_bonus
        # (0.18·Q·prestige_excess, up to ~+0.13 unconditionally), which lifted
        # off-niche Nature/Angew papers (base≈0.69) above the reader's on-field
        # preprints (base≈0.78) — a direct topic inversion. The gate ramps from 0 at
        # the retrieval floor (~0.68) to full by ~0.80, exactly across the band where
        # research items cluster, so it discriminates where it matters.
        rel_gate = max(0.0, min(1.0, (float(base) - 0.68) / 0.12))
        quality_relevance = 0.36 * Q * prestige_excess * rel_gate
        score = base + quality_relevance + (0.08 * novelty) - (0.35 * promo)
        low_prestige = source_quality.prestige_score < LOW_RESEARCH_PRESTIGE
        exceptional = base >= 0.80 and novelty >= 0.50
        if low_prestige and not exceptional:
            # Relevance-scaled component fades as the paper gets more on-topic, but
            # a larger persistent residual (0.10, was 0.05) keeps a low-impact venue
            # meaningfully below comparable strong-venue work even when it is highly
            # on-topic — so "low-impact even if quite related" stays infrequent and
            # low in the ordering, not shoulder-to-shoulder with flagship papers.
            smooth = max(0.0, min(1.0, (0.82 - base) / 0.32))
            residual = 0.10
            score -= Q * 0.16 * smooth + residual
        # Preprints (bioRxiv, ChemRxiv, SSRN, …) have not been peer-reviewed.
        # Apply a mild penalty so a peer-reviewed paper with similar topic fit
        # is preferred. Fades to zero at high base scores so a highly relevant
        # preprint can still surface over a weakly matched journal paper.
        # arXiv CS gets a slightly higher penalty but we take the MAX of the two,
        # not both, to avoid double-stacking.
        if (source_quality.quality_tier in {"repository", "preprint"} or is_arxiv_cs_source(row)) and not exceptional:
            preprint_smooth = max(0.0, min(1.0, (0.76 - base) / 0.26))
            arxiv_smooth = max(0.0, min(1.0, (0.76 - base) / 0.20)) if is_arxiv_cs_source(row) else 0.0
            # Instead of stacking both penalties, use the larger of the two
            preprint_or_arxiv_penalty = max(0.15 * preprint_smooth, 0.18 * arxiv_smooth)
            score -= preprint_or_arxiv_penalty
        # Penalize editorial/commentary pieces that aren't primary research
        ctype = content_type(row)
        if ctype in {"editorial", "commentary"} and not exceptional:
            score -= 0.10
        return _clip(score)

    if section in {"industry", "world"}:
        score = base + (0.06 * source_quality.prestige_score) + (0.14 * novelty) - (0.45 * promo)
        # Smooth low-prestige penalty for self-published industry sources.
        quality_tier_val = source_quality.quality_tier or ""
        if quality_tier_val in ("self-published", "low") and base < 0.72:
            smooth = max(0.0, min(1.0, (0.72 - base) / 0.24))
            score -= 0.12 * smooth
        # Opinion / editorial / commentary pieces ("Opinion:", "Perspective")
        # carry little hard news — push them below straight reporting.
        if content_type(row) in {"editorial", "commentary"}:
            score -= 0.12
        # Paywalled sources must be more relevant to earn a slot.
        if source_quality.paywalled:
            score -= PAYWALL_PENALTY
        return _clip(score)

    if section == "regulatory":
        # FDA/EMA items often contain "today announced", "approved", "pleased to
        # announce" — all promo-flagged but editorially legitimate. Use a light
        # 0.15× penalty rather than the full 0.45× applied to industry.
        score = base + (0.06 * source_quality.prestige_score) + (0.10 * novelty) - (0.15 * promo)
        if source_quality.paywalled:
            score -= PAYWALL_PENALTY
        return _clip(score)

    return _clip(base + (0.06 * source_quality.prestige_score) + (0.10 * novelty) - (0.15 * promo))
