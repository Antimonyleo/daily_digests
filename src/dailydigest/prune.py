"""Prune wrapper for CLI."""

from .config import load_settings
from .store import prune as _prune


def prune_old() -> int:
    """Delete items older than retention_days. Returns count deleted."""
    settings = load_settings()
    return _prune(settings.retention_days)
