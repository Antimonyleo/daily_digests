"""Render the digest HTML email via Jinja2."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from jinja2 import Environment, FileSystemLoader

from .config import get_settings
from .opportunities import load_opportunity_profile, opportunity_display
from .rank.source_quality import display_breakdown, source_bucket
from .store import ItemRow, item_metadata, load_digest_features

_PACKAGE_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_TEMPLATE_DIR = (
    _PACKAGE_TEMPLATE_DIR
    if _PACKAGE_TEMPLATE_DIR.is_dir()
    else Path(__file__).resolve().parents[2] / "templates"
)
_ALLOWED_LINK_SCHEMES = {"http", "https"}

# Content types that are NOT primary research get a visible badge so a Nature
# Reviews piece is not silently mistaken for a primary-research slot. Plain
# "research"/"article" (and other non-mapped types) render no badge.
_CONTENT_TYPE_LABELS: dict[str, str] = {
    "review": "Review",
    "method": "Method",
    "dataset": "Dataset",
    "clinical": "Clinical",
}


def content_type_label(content_type: str | None) -> str:
    """Map a source_quality content_type to a short badge, or "" for none."""
    return _CONTENT_TYPE_LABELS.get(str(content_type or "").strip().lower(), "")


def _is_high_profile(source_bucket_value: str | None, tags: list[str]) -> bool:
    """A published-journal venue or an explicit prestige/quality tag."""
    if str(source_bucket_value or "") == "published_journal":
        return True
    lowered = {str(t or "").strip().lower().replace(" ", "_") for t in tags}
    return "high_quality_source" in lowered


def reason_line(
    primary_facet: str | None,
    *,
    high_profile: bool = False,
    journal: str | None = None,
    why_shown: list[str] | None = None,
    tags: list[str] | None = None,
) -> str:
    """Build a compact, human-readable "why shown" reason line.

    ``Shown for {primary_facet}`` with an optional ``· high-profile journal``
    (or the journal name) suffix. When ``primary_facet`` is empty, fall back to
    the first ``why_shown`` entry, then the first tag; otherwise return "" so the
    caller omits the line rather than rendering "Shown for .".
    """
    facet = str(primary_facet or "").strip()
    if facet:
        line = f"Shown for {facet}"
    else:
        fallback = ""
        for candidate in list(why_shown or []) + list(tags or []):
            text = str(candidate or "").replace("_", " ").strip()
            if text:
                fallback = text
                break
        if not fallback:
            return ""
        line = f"Shown for {fallback}"
    if high_profile:
        venue = str(journal or "").strip()
        line += f" · {venue}" if venue else " · high-profile journal"
    return line


SECTION_META: dict[str, dict[str, str]] = {
    "research": {"title": "Research", "emoji": "🧬"},
    "industry": {"title": "Industry", "emoji": "💊"},
    "ai": {"title": "AI Tools & Methods", "emoji": "🤖"},
    "regulatory": {"title": "Clinical & Regulatory", "emoji": "📋"},
    "world": {"title": "World", "emoji": "🌍"},
    "opportunities": {"title": "Funding & Opportunities", "emoji": "💰"},
    "events": {"title": "Events & Calls", "emoji": "📅"},
}

SECTION_ORDER = [
    "research",
    "opportunities",
    "events",
    "industry",
    "ai",
    "regulatory",
    "world",
]


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        # Template names end in .html.j2, so select_autoescape(["html"]) would
        # miss them. Force escaping for feed-controlled email content.
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def safe_url(url: str | None) -> str:
    raw = (url or "").strip()
    parsed = urlsplit(raw)
    if parsed.scheme.lower() in _ALLOWED_LINK_SCHEMES and parsed.netloc:
        return raw
    return "#"


def _format_date(row: ItemRow) -> str:
    if row.published_at is None:
        return ""
    return row.published_at.strftime("%Y-%m-%d")


def render_digest(
    digest_id: str,
    sections: dict[str, list[tuple[ItemRow, float, str]]],
    health_summary: list[dict] | None = None,
) -> str:
    """sections[section] = list of (row, score, summary).

    ``health_summary`` (optional) is a per-source 7-day rollup produced by
    :func:`dailydigest.health.weekly_summary`. When provided, the template
    renders a small grey footer table.
    """
    env = _env()
    tpl = env.get_template("digest.html.j2")

    # Persisted per-item features (primary_facet, content_type, ...) keyed by
    # item id. Best-effort: if the store is unavailable we degrade to whatever
    # display_breakdown(row) can recompute from the row alone.
    try:
        persisted_features = load_digest_features(digest_id)
    except Exception:  # noqa: BLE001
        persisted_features = {}

    rendered_sections = []
    opportunity_profile = None
    if any(sections.get(key) for key in ("opportunities", "events")):
        try:
            opportunity_profile = load_opportunity_profile(
                get_settings().opportunity_profile_path
            )
        except Exception:  # noqa: BLE001
            pass
    for key in SECTION_ORDER:
        items = sections.get(key) or []
        if not items:
            continue
        meta = SECTION_META.get(key, {"title": key.title(), "emoji": ""})
        rendered_items = []
        for row, score, summary in items:
            features = persisted_features.get(int(row.id)) if row.id is not None else None
            features = features or {}
            # content_type / quality signals: prefer persisted, else recompute.
            content_type = str(features.get("content_type") or "")
            bucket = str(features.get("source_bucket") or "")
            tags = list(features.get("tags") or [])
            why = list(features.get("why_shown") or [])
            if not content_type or not bucket:
                try:
                    bd = display_breakdown(row)
                    content_type = content_type or bd.content_type
                    bucket = bucket or source_bucket(row)
                    tags = tags or list(bd.tags) + list(bd.quality_tags)
                    why = why or list(bd.why_shown)
                except Exception:  # noqa: BLE001
                    pass
            reason = reason_line(
                features.get("primary_facet"),
                high_profile=_is_high_profile(bucket, tags),
                journal=row.source or "",
                why_shown=why,
                tags=tags,
            )
            rendered_items.append(
                {
                    "label": row.item_label or "",
                    "title": row.title or "",
                    "url": safe_url(row.url),
                    "source": row.source or "",
                    "published": _format_date(row),
                    "summary": summary or "",
                    "score": score,
                    "reason": reason,
                    "type_label": content_type_label(content_type),
                    "opportunity": (
                        opportunity_display(item_metadata(row), opportunity_profile)
                        if key in {"opportunities", "events"}
                        else None
                    ),
                }
            )
        rendered_sections.append(
            {
                "key": key,
                "title": meta["title"],
                "emoji": meta["emoji"],
                "entries": rendered_items,
            }
        )

    return tpl.render(
        digest_id=digest_id,
        sections=rendered_sections,
        health_summary=health_summary or None,
    )
