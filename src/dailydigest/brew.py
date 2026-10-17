"""Lightweight brew controls shared by the reader UI and pipeline.

Importing these controls must not initialize ranking or source adapters.
"""
from __future__ import annotations

from .config import user_local_date


class BrewCancelled(RuntimeError):
    """A cooperative cancellation stopped a brew before publication."""


READING_MODE_LIMITS: dict[str, int | None] = {
    'full': None,
    'usual': 15,
    'minimal': None,
}


def normalize_reading_mode(value: str | None, *, default: str = 'full') -> str:
    """Return a supported per-brew reading mode or raise for invalid input."""
    mode = str(value or default).strip().lower()
    if mode not in READING_MODE_LIMITS:
        choices = ', '.join(READING_MODE_LIMITS)
        raise ValueError(f'unknown reading mode {mode!r}; choose one of: {choices}')
    return mode


def _digest_id() -> str:
    """Anchor today's digest to the reader's configured local timezone."""
    return user_local_date().isoformat()
