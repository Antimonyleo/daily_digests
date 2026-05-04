"""Shared pytest fixtures for DailyDigest tests.

- DB_PATH is redirected to a temp file so tests never touch data/digest.db.
- LLM_API_KEY is blanked to force extractive summarizer fallback.
- httpx network calls raise if accidentally triggered (no live fetches in unit tests).
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    """Point DB_PATH at a throwaway sqlite file and blank the LLM key."""
    db_file = tmp_path / "test_digest.db"
    monkeypatch.setenv("DB_PATH", str(db_file))
    monkeypatch.setenv("LLM_API_KEY", "")
    # Make sure pydantic-settings picks up overridden env for SETTINGS singletons
    # by clearing the lru_cache on get_settings if it exists.
    try:
        from dailydigest.config import get_settings
        get_settings.cache_clear()
    except Exception:
        pass
    yield
    try:
        from dailydigest.config import get_settings
        get_settings.cache_clear()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Raise if any test accidentally makes a real HTTP request via httpx."""
    import httpx

    def _blocked(*args, **kwargs):
        raise RuntimeError(
            "Network call blocked in unit tests. "
            "Use monkeypatch or a fixture to mock HTTP."
        )

    monkeypatch.setattr(httpx.Client, "get", _blocked)
    monkeypatch.setattr(httpx.Client, "post", _blocked)
    monkeypatch.setattr(httpx.AsyncClient, "get", _blocked)
    monkeypatch.setattr(httpx.AsyncClient, "post", _blocked)
