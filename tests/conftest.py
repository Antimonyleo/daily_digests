"""Shared pytest fixtures for DailyDigest tests.

- DB_PATH is redirected to a temp file so tests never touch data/digest.db.
- LLM_API_KEY is blanked to force extractive summarizer fallback.
- httpx network calls raise if accidentally triggered (no live fetches in unit tests).
- SQLAlchemy engine is reset between tests so each test gets a fresh DB.
"""

from __future__ import annotations

import os
import sys

import pytest

_SETUP_ENV_KEYS = (
    "LLM_BACKEND",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "USER_TZ",
    "TOP_RESEARCH",
    "INCLUDE_INDUSTRY",
    "TOP_INDUSTRY",
    "INCLUDE_AI",
    "TOP_AI",
    "INCLUDE_REGULATORY",
    "TOP_REGULATORY",
    "INCLUDE_WORLD",
    "TOP_WORLD",
    "INCLUDE_OPPORTUNITIES",
    "TOP_OPPORTUNITIES",
    "INCLUDE_EVENTS",
    "TOP_EVENTS",
)


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    """Point DB_PATH at a throwaway sqlite file, blank the LLM key, and reset
    the store engine so each test starts with a fresh database."""
    original_setup_env = {key: os.environ.get(key) for key in _SETUP_ENV_KEYS}
    db_file = tmp_path / "test_digest.db"
    monkeypatch.setenv("DB_PATH", str(db_file))
    monkeypatch.setenv("LLM_API_KEY", "")
    # A developer's private opportunity settings must not change validation of
    # ordinary setup-form tests. Tests that exercise these sections opt in.
    monkeypatch.setenv("INCLUDE_OPPORTUNITIES", "false")
    monkeypatch.setenv("INCLUDE_EVENTS", "false")
    # Clear settings cache so pydantic-settings picks up the new DB_PATH.
    try:
        from dailydigest import config as _config

        _config.get_settings.cache_clear()
        isolated_settings = _config.get_settings()
        monkeypatch.setattr(_config, "SETTINGS", isolated_settings)
    except Exception:
        pass
    web_module = sys.modules.get("dailydigest.web")
    if web_module is not None:
        try:
            monkeypatch.setattr(web_module, "SETTINGS", isolated_settings)
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
        _store._INITIALIZED = False
    except Exception:
        pass
    # Reset LR ranker cache so tests don't inherit a loaded model
    try:
        from dailydigest.rank.ranker import reset_lr_cache
        reset_lr_cache()
    except Exception:
        pass

    yield

    # setup_post deliberately updates os.environ so the running app sees saved
    # settings immediately; restore those direct writes between tests.
    for key, value in original_setup_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

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
        _store._INITIALIZED = False
    except Exception:
        pass
    # Reset LR ranker cache so tests don't inherit a loaded model
    try:
        from dailydigest.rank.ranker import reset_lr_cache
        reset_lr_cache()
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
    monkeypatch.setattr(httpx.Client, "stream", _blocked)
    monkeypatch.setattr(httpx.AsyncClient, "get", _blocked)
    monkeypatch.setattr(httpx.AsyncClient, "post", _blocked)
