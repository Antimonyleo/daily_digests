from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import Profile, SourceSpec


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    profile_path: str = "data/profile.yaml"
    sources_path: str = "config/sources.yaml"

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    # Backend selector: "api" | "extractive"
    llm_backend: str = "extractive"

    resend_api_key: str = ""
    digest_from: str = "onboarding@resend.dev"
    digest_to: str = ""
    reply_to_email: str = ""

    user_tz: str = ""
    digest_hour: int = Field(default=8, ge=0, le=23)

    db_path: str = "data/digest.db"

    # --- Embedding / ranking model configuration -------------------------- #
    # Swap in a stronger scientific encoder (e.g. "allenai/specter2_base",
    # "ncbi/MedCPT-Query-Encoder", "BAAI/bge-large-en-v1.5") without code
    # changes. The item embedding cache keys on the model name, so changing
    # this transparently re-embeds. Device defaults to CPU (the intended
    # local / CI mode); set "cuda" only on a supported GPU.
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_device: str = "cpu"
    embed_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embed_doc_prefix: str = ""
    # Embedding backend: "" (auto: fastembed/ONNX, no torch), "fastembed", or
    # "sentence-transformers" (needs the optional `hf` extra; required for models
    # outside fastembed's catalog, e.g. SPECTER2). Auto falls back to
    # sentence-transformers when fastembed cannot load the configured model.
    embed_backend: str = ""

    # --- Quality weighting --------------------------------------------------- #
    # How strongly venue quality influences research ranking. 1.0 = legacy
    # (prestige is a mild tie-breaker). Higher values reward high-quality AND
    # relevant work more and suppress low-impact venues harder.
    research_quality_weight: float = Field(default=1.4, ge=0.0, le=5.0)
    # Scale of the topic-priority selection nudge. Multiplied by a normalized
    # per-interest priority (0..1) after the relevance and quality gates. It
    # never changes stored final scores or eligibility. Max nudge = scale*1.0.
    # Small by design: it reorders near-ties but cannot override clearly better
    # work.
    topic_priority_bonus_scale: float = Field(default=0.06, ge=0.0, le=1.0)
    # Small selection-only boost for an interest absent from recent viewed
    # digests. It never changes semantic eligibility or quality thresholds.
    topic_coverage_bonus_scale: float = Field(default=0.03, ge=0.0, le=0.2)
    # A low-impact-venue research item must clear this base topic relevance to be
    # eligible for the digest at all — so the few that appear are strongly on-topic.
    low_impact_relevance_floor: float = Field(default=0.58, ge=0.0, le=1.0)
    # When a score calibrator has been fit from vote history, derive the
    # low-impact floor from it (the score at which P(relevant) ~ 0.5), clamped
    # near the configured default. Falls back to the default when uncalibrated.
    adaptive_relevance_floor: bool = True
    # Max fraction of the research section that may be filled by low-impact-venue
    # items, so they cannot appear frequently even when many are related.
    max_low_impact_research_frac: float = Field(default=0.15, ge=0.0, le=1.0)
    # Negative-interest penalty (see profile.negative_interests). DISCRIMINATIVE
    # (relative) formulation: penalize by ``negative_interest_weight *
    # max(0, sim_to_nearest_negative − topic_relevance − negative_interest_margin)``
    # — i.e. only when an item is CLOSER to an explicit negative topic than to the
    # reader's own profile. CRITICAL: bge-small similarities to the verbose
    # negative phrases sit in the SAME ~0.55–0.68 band as genuine topic relevance
    # AND do not separate the reader's field from off-field biomedical content on
    # their own (both are ~0.6), so any ABSOLUTE threshold either taxes the whole
    # pool (low cut → collapses the section, penalizes the reader's own field) or
    # does nothing (high cut). The relative (neg − topic) signal cleanly separates
    # them: clinical/epidemiology/GWAS items are neg-dominant, materials/design
    # items are profile-dominant. margin ≥ 0 requires the negative to win by that
    # much before any penalty applies.
    negative_interest_margin: float = Field(default=0.0, ge=-0.5, le=0.5)
    negative_interest_weight: float = Field(default=0.80, ge=0.0, le=2.0)
    # Magnitude of the OpenAlex venue-impact adjustment (boost high-impact venues,
    # penalize low-impact ones) applied when citation_enrichment is enabled.
    venue_quality_weight: float = Field(default=0.18, ge=0.0, le=1.0)

    # Active learning: reserve up to N research slots for the most LR-uncertain
    # HIGH-QUALITY items, to gather informative feedback. Off by default; only
    # high-quality venues are eligible, so exploration never shows low-impact work.
    exploration_slots: int = Field(default=0, ge=0, le=3)

    # Live citation/impact enrichment via OpenAlex (network at runtime). Off by
    # default to keep the daily run local and reproducible.
    citation_enrichment: bool = False
    citation_polite_email: str = ""

    # Cross-day near-duplicate suppression: drop a candidate whose embedding is
    # within `threshold` cosine of an item shown in a sent digest over the last
    # `days` days — catches the same paper/story re-syndicated via another source.
    cross_day_dedupe: bool = True
    cross_day_dedupe_days: int = Field(default=7, ge=1, le=60)
    cross_day_dedupe_threshold: float = Field(default=0.93, ge=0.5, le=1.0)

    # Within-day semantic suppression: inside a SINGLE digest, collapse similar
    # research candidates so one representative survives. Disabled by default:
    # the current 0.86 threshold can also suppress distinct papers. It remains
    # available as an explicit opt-in; the highest-scored item of each cluster is
    # always kept.
    within_day_dedupe: bool = False
    within_day_dedupe_threshold: float = Field(default=0.86, ge=0.5, le=1.0)

    # Per-section size. When adaptive_section_sizes is on, top_* is the CEILING
    # and min_* is the floor; the actual count for a section flexes between them
    # with the day's supply of on-topic items (see min_topic_relevance).
    # When adaptive sizing is off, top_* is a fixed cap (legacy behavior).
    top_research: int = Field(default=12, ge=0, le=100)
    top_industry: int = Field(default=6, ge=0, le=100)
    top_ai: int = Field(default=4, ge=0, le=100)
    top_regulatory: int = Field(default=3, ge=0, le=100)
    top_world: int = Field(default=3, ge=0, le=100)

    # Adaptive section sizing: size each section to the number of items that
    # clear `min_topic_relevance` (the absolute topic-cosine floor below),
    # clamped to [min_*, top_*]. A day rich in on-topic items shows more; a quiet
    # day fewer.
    adaptive_section_sizes: bool = True
    # Soft floor for the research section on a low-supply day. 3 (was 5) so the
    # digest shrinks to only the genuinely strong papers rather than padding two
    # extra weak slots — the cutoff is dynamic, driven by how many items clear the
    # relevance/quality bar, not a fixed count.
    min_research: int = Field(default=3, ge=0, le=100)
    min_industry: int = Field(default=3, ge=0, le=100)
    min_ai: int = Field(default=2, ge=0, le=100)
    min_regulatory: int = Field(default=2, ge=0, le=100)
    min_world: int = Field(default=2, ge=0, le=100)
    # Minimum topic relevance (true profile cosine, 0..1) an item must clear to
    # earn a slot. A section shows as many items as clear this bar, clamped to
    # [min_*, top_*] — so off-topic-but-prestigious items are excluded outright
    # rather than filling the section. Raise it to show fewer, stricter items;
    # lower it to be more inclusive. (Stable now that profile cosine is a true
    # unit cosine; ~0.5–0.6 is weak/off-topic, ~0.65+ is a solid match.)
    min_topic_relevance: float = Field(default=0.65, ge=0.0, le=1.0)
    # Venue-aware relaxation of the topic floor. A top/high/strong-tier journal in
    # the reader's field is itself a strong quality prior, so it should clear the
    # relevance gate at a slightly lower raw topic cosine than an anonymous
    # preprint or aggregator hit. The effective topic score used at the gate is
    # ``topic + venue_relevance_credit * prestige_excess`` (research only), where
    # prestige_excess = prestige - 0.50. At 0.10, a high-tier venue (prestige
    # 0.90, excess 0.40) gets +0.04; a top-tier venue (0.99) gets ~+0.049.
    # Because the credit is small, an OFF-topic prestigious paper (low cosine)
    # still fails the gate — it only rescues genuinely borderline top-venue work
    # that bge-small's compressed cosine scores just under the floor. Set to 0 to
    # restore the pure venue-blind floor.
    venue_relevance_credit: float = Field(default=0.10, ge=0.0, le=0.5)
    # FINAL-fused-score cutoff for the research section. Section SIZING gates on
    # topic cosine (min_topic_relevance), but the picker then fills those slots by
    # the FINAL learned/fused score — so a prestigious-but-personally-disliked
    # paper the model scored near ZERO could still pad a slot. This gate drops
    # research picks whose final fused score is below ``frac`` of the section's
    # top score. RELATIVE (a fraction of the top) rather than absolute because the
    # fused score is RRF min-maxed to [0,1] PER RUN — the top is always ~1.0 and
    # the floor ~0.0, so a fixed absolute cut would mean different things on
    # different days, whereas "at least X% as good as the best pick" is stable.
    # A small hard-minimum (research_final_score_min_keep) guarantees the digest
    # is never emptied by this gate. Set frac to 0 to disable.
    research_final_score_floor_frac: float = Field(default=0.35, ge=0.0, le=1.0)
    research_final_score_min_keep: int = Field(default=3, ge=0, le=100)

    # Quality floor (final confidence, 0..1) for the news sections (industry,
    # world). Opinion columns, paywalled "STAT+:" teasers and weak items fall
    # below it and are dropped, so a section shrinks to its genuinely useful
    # items rather than padding to the cap with filler. Regulatory (FDA/EMA) is
    # exempt — those are wanted regardless. Raise for stricter news curation.
    min_news_quality: float = Field(default=0.45, ge=0.0, le=1.0)

    # Catch-up after a usage gap. When the last digest was N days ago, both the
    # ingest fetch window and the ranking window widen to cover the gap (capped at
    # max_backfill_days), and the research ceiling scales up so a backlog of
    # relevant papers can surface instead of only the last day or two.
    max_backfill_days: int = Field(default=21, ge=1, le=90)
    # Research ceiling on a full catch-up run (grows from top_research toward this
    # as the gap widens). Set equal to top_research to disable catch-up growth.
    max_research_backlog: int = Field(default=40, ge=0, le=200)

    retention_days: int = Field(default=30, ge=1, le=3650)

    imap_host: str = ""
    imap_port: int = 993
    imap_user: str = ""
    imap_password: str = ""

    @field_validator("user_tz")
    @classmethod
    def _validate_tz(cls, v: str) -> str:
        if not v:
            return v
        try:
            from zoneinfo import ZoneInfo
            ZoneInfo(v)
            return v
        except Exception:
            import logging
            logging.getLogger(__name__).error(
                "USER_TZ=%r is not a valid timezone; using UTC instead. "
                "This will cause the digest to run at 8am UTC, not your local 8am.", v
            )
            return "UTC"


def load_settings() -> Settings:
    s = Settings()
    if not s.user_tz:
        try:
            import tzlocal

            detected = tzlocal.get_localzone_name()
            if detected:
                s = s.model_copy(update={"user_tz": detected})
        except Exception:
            pass
    if not s.user_tz:
        s = s.model_copy(update={"user_tz": "UTC"})
    return s


def load_profile(path: str | None = None) -> Profile:
    settings = get_settings()
    p = Path(path or settings.profile_path)
    if not p.exists():
        # The example is documentation, not a default reader identity. A new
        # user must complete local setup before retrieval/ranking can run.
        raise FileNotFoundError(f"Profile not found: {p}; complete setup at /setup")
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Profile at {p} must be a YAML mapping, got {type(data).__name__}")
    try:
        return Profile(**data)
    except Exception as exc:
        raise ValueError(f"Profile at {p} is invalid: {exc}") from exc


def load_sources(path: str | None = None) -> list[SourceSpec]:
    settings = get_settings()
    p = Path(path or settings.sources_path)
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"sources.yaml must be a YAML mapping, got {type(data).__name__}")
    out: list[SourceSpec] = []
    for section, entries in data.items():
        for entry in entries or []:
            out.append(SourceSpec(section=section, **entry))
    return out


def ensure_data_dir() -> None:
    Path(load_settings().db_path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Preferred lazy accessor for settings (cached). Use this in new code."""
    return load_settings()


def reload_settings() -> None:
    """Re-read .env and refresh in-memory singletons.

    Call this after programmatically updating .env (e.g. from the web setup
    wizard) so the running pipeline picks up the new backend immediately.
    """
    global SETTINGS
    get_settings.cache_clear()
    SETTINGS = load_settings()


# Legacy access pattern: many existing modules import SETTINGS directly.
# Kept for backwards compatibility. Prefer get_settings() in new code.
SETTINGS = load_settings()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
