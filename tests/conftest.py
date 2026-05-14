"""Shared pytest fixtures for DailyDigest tests.

- DB_PATH is redirected to a temp file so tests never touch data/digest.db.
- LLM_API_KEY is blanked to force extractive summarizer fallback.
- httpx network calls raise if accidentally triggered (no live fetches in unit tests).
- SQLAlchemy engine is reset between tests so each test gets a fresh DB.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    """Point DB_PATH at a throwaway sqlite file, blank the LLM key, and reset
    the store engine so each test starts with a fresh database."""
    db_file = tmp_path / "test_digest.db"
    monkeypatch.setenv("DB_PATH", str(db_file))
    monkeypatch.setenv("LLM_API_KEY", "")
    # Clear settings cache so pydantic-settings picks up the new DB_PATH.
    try:
        from dailydigest.config import get_settings
        get_settings.cache_clear()
    except Exception:
        pass
    # Reset store engine globals so the next init_db() points to the new file.
    try:
        import dailydigest.store as _store
        if _store._ENGINE is not None:
            try:
                _store._ENGINE.dispose()
            except Exception:
                pass
        _store._ENGINE = None
        _store._SessionLocal = None
    except Exception:
        pass

    yield

    # Teardown: clear caches for the next test.
    try:
        from dailydigest.config import get_settings
        get_settings.cache_clear()
    except Exception:
        pass
    try:
        import dailydigest.store as _store
        if _store._ENGINE is not None:
            try:
                _store._ENGINE.dispose()
            except Exception:
                pass
        _store._ENGINE = None
        _store._SessionLocal = None
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
