"""Render the digest HTML email via Jinja2."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .store import ItemRow

_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates"

SECTION_META: dict[str, dict[str, str]] = {
    "research": {"title": "Research", "emoji": "🧬"},
    "industry": {"title": "Industry", "emoji": "💊"},
    "regulatory": {"title": "Regulatory", "emoji": "📋"},
    "world": {"title": "World", "emoji": "🌍"},
}

SECTION_ORDER = ["research", "industry", "regulatory", "world"]


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


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

    rendered_sections = []
    for key in SECTION_ORDER:
        items = sections.get(key) or []
        if not items:
            continue
        meta = SECTION_META.get(key, {"title": key.title(), "emoji": ""})
        rendered_items = []
        for row, score, summary in items:
            rendered_items.append(
                {
                    "label": row.item_label or "",
                    "title": row.title or "",
                    "url": row.url or "",
                    "source": row.source or "",
                    "published": _format_date(row),
                    "summary": summary or "",
                    "score": score,
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
