"""Per-source ingest health stats: latest snapshot + 7-day rolling history.

Persisted as JSON at ``data/health.json`` next to the SQLite db. Schema::

    {
        "updated_at": "<ISO8601>",
        "latest": [ {source, items, ok, error, duration_ms, diagnostics}, ... ],
        "history": [
            {"recorded_at": "<ISO8601>", "stats": [ ... same shape as latest ... ]},
            ...  # capped at 7 entries (one per run-day)
        ]
    }
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .config import get_settings, ensure_data_dir

logger = logging.getLogger(__name__)


class IngestStats(BaseModel):
    source: str
    items: int = 0
    ok: bool = True
    error: str | None = None
    duration_ms: int = 0
    # Adapter-specific query coverage, including valid emptiness and saturation.
    # Old records omit this field; no historical coverage is inferred for them.
    diagnostics: dict = Field(default_factory=dict)


def _health_path() -> Path:
    return Path(get_settings().db_path).parent / "health.json"


def _load_existing(path: Path) -> dict:
    if not path.exists():
        return {"updated_at": None, "latest": [], "history": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning("health.json unreadable, resetting: %s", e)
        return {"updated_at": None, "latest": [], "history": []}


def record(stats: list[IngestStats]) -> None:
    """Idempotent JSON write. Keeps a rolling 7-entry history (one per run)."""
    ensure_data_dir()
    path = _health_path()
    data = _load_existing(path)
    now = datetime.now(timezone.utc)
    snapshot = [s.model_dump() for s in stats]

    history: list[dict] = list(data.get("history") or [])
    history.append({"recorded_at": now.isoformat(), "stats": snapshot})
    # Drop entries older than 7 days, then cap at 7 most recent.
    cutoff = now - timedelta(days=7)
    history = [
        h for h in history
        if _parse_iso(h.get("recorded_at")) and _parse_iso(h["recorded_at"]) >= cutoff
    ]
    history = history[-7:]

    payload = {
        "updated_at": now.isoformat(),
        "latest": snapshot,
        "history": history,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def weekly_summary() -> list[dict]:
    """Return a per-source 7-day summary suitable for the email footer.

    Output shape: ``[{source, items_7d, failures_7d, last_error}, ...]``
    sorted alphabetically by source. Returns ``[]`` when no history file
    exists yet. Callers decide whether to render — see :func:`should_show`.
    """
    path = _health_path()
    if not path.exists():
        return []
    data = _load_existing(path)
    history = data.get("history") or []
    if not history:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    agg: dict[str, dict] = {}
    for entry in history:
        ts = _parse_iso(entry.get("recorded_at"))
        if ts is None or ts < cutoff:
            continue
        for s in entry.get("stats") or []:
            name = s.get("source") or "(unknown)"
            slot = agg.setdefault(
                name,
                {"source": name, "items_7d": 0, "failures_7d": 0, "last_error": None},
            )
            slot["items_7d"] += int(s.get("items") or 0)
            if not s.get("ok", True):
                slot["failures_7d"] += 1
                if s.get("error"):
                    slot["last_error"] = s["error"]
    return sorted(agg.values(), key=lambda d: d["source"].lower())


def latest_snapshot() -> list[dict]:
    """Return the latest per-source ingest snapshot, or [] when unavailable."""
    path = _health_path()
    if not path.exists():
        return []
    data = _load_existing(path)
    latest = data.get("latest") or []
    return list(latest) if isinstance(latest, list) else []


def coverage_summary() -> dict:
    """Describe the last attempted source check, not the saved digest's quality."""
    data = _load_existing(_health_path())
    if not isinstance(data, dict):
        data = {}
    checked = _parse_iso(data.get("updated_at"))
    checked = checked.astimezone(timezone.utc) if checked else None
    rows = data.get("latest") if isinstance(data.get("latest"), list) else []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    summary = {"total": 0, "responded": 0, "unavailable": 0, "limited": 0,
               "unknown": 0, "checked_at": checked.isoformat() if checked else "",
               "checked_label": checked.strftime("%b %d, %Y %H:%M UTC") if checked else "Time not recorded",
               "checked_stale": bool(checked and checked < cutoff),
               "public_oldest_label": "", "public_stale": False, "rows": []}
    public_times = []
    for row in rows:
        row = row if isinstance(row, dict) else {}
        diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
        queries = diagnostics.get("queries") if isinstance(diagnostics.get("queries"), list) else []
        limited = bool(diagnostics.get("truncated") or diagnostics.get("snapshot_refresh_error")
                       or diagnostics.get("snapshot_carried_forward")
                       or diagnostics.get("snapshot_latest_attempt_failed")
                       or diagnostics.get("snapshot_stale_warning")
                       or diagnostics.get("author_resolution_complete") is False
                       or any(q.get("failure") or q.get("truncated") for q in queries if isinstance(q, dict)))
        if row.get("ok") is True:
            summary["responded"] += 1
            status = "Limited results" if limited else "No items returned" if not row.get("items") else "Responded"
            if diagnostics.get("snapshot_carried_forward"):
                status = "Earlier snapshot"
            summary["limited"] += int(limited)
        elif row.get("ok") is False:
            status = "Unavailable"
            summary["unavailable"] += 1
        else:
            status = "Status unknown"
            summary["unknown"] += 1
        if diagnostics.get("transport") == "public_snapshot":
            fetched = _parse_iso(diagnostics.get("snapshot_fetched_at"))
            fetched = fetched.astimezone(timezone.utc) if fetched else None
            if fetched:
                public_times.append(fetched)
            freshness = f"Public feed: {fetched.strftime('%b %d, %Y %H:%M UTC')}" if fetched else "Public feed time not recorded"
            if diagnostics.get("snapshot_latest_attempt_failed"):
                freshness += " · Latest collection failed; showing earlier public data"
            elif diagnostics.get("snapshot_carried_forward"):
                freshness += " · Earlier validated snapshot retained for code update"
        else:
            freshness = f"Checked: {summary['checked_label']}"
        summary["rows"].append({"name": row.get("source") or "Unnamed source",
                                "status": status, "freshness": freshness})
    summary["total"] = len(summary["rows"])
    if public_times:
        summary["public_oldest_label"] = min(public_times).strftime("%b %d, %Y %H:%M UTC")
        summary["public_stale"] = min(public_times) < cutoff
    return summary


def should_show(summary: list[dict]) -> bool:
    """Heuristic: include the footer only when at least one source had >=1
    failure in the last 7 days. Keeps successful weeks visually clean."""
    return any((row.get("failures_7d") or 0) >= 1 for row in summary)
