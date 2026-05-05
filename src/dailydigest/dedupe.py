from __future__ import annotations

from .ingest.rss import canonicalize_url
from .models import Item


def dedupe_by_url(items: list[Item]) -> list[Item]:
    seen: set[str] = set()
    out: list[Item] = []
    for it in items:
        canon = canonicalize_url(it.url)
        if not canon:
            out.append(it)
            continue
        if canon in seen:
            continue
        seen.add(canon)
        out.append(it)
    return out


def filter_english(items: list[Item]) -> list[Item]:
    """Best-effort English filter using langdetect on title+abstract."""
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
    except Exception:
        return items

    out: list[Item] = []
    for it in items:
        sample = (it.title + " " + it.abstract).strip()
        if len(sample) < 20:
            out.append(it)
            continue
        try:
            if detect(sample) == "en":
                out.append(it)
        except Exception:
            out.append(it)
    return out
