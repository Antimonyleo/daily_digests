from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import Profile, SourceSpec


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    profile_path: str = "data/profile.yaml"
    sources_path: str = "config/sources.yaml"

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    # Backend selector: "api" | "claude_code" | "codex" | "extractive"
    llm_backend: str = "extractive"
    # Optional model pin for CLI backends (claude_code, codex). Empty = inherit
    # the CLI's default. e.g. "claude-haiku-4-5-20251001" or "gpt-5-mini".
    llm_cli_model: str = ""

    resend_api_key: str = ""
    digest_from: str = "onboarding@resend.dev"
    digest_to: str = ""
    reply_to_email: str = ""

    user_tz: str = ""
    digest_hour: int = 8

    db_path: str = "data/digest.db"

    top_research: int = 12
    top_industry: int = 6
    top_regulatory: int = 3
    top_world: int = 3
    retention_days: int = 30

    candidates_for_summary: int = Field(default=60, description="top-K after Stage A ranking")


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
    settings = load_settings()
    p = Path(path or settings.profile_path)
    if not p.exists():
        # fall back to example so a fresh checkout still runs
        example = Path("config/profile.example.yaml")
        if example.exists():
            p = example
        else:
            raise FileNotFoundError(f"Profile not found: {p}")
    data = yaml.safe_load(p.read_text())
    return Profile(**data)


def load_sources(path: str | None = None) -> list[SourceSpec]:
    settings = load_settings()
    p = Path(path or settings.sources_path)
    data = yaml.safe_load(p.read_text())
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
