"""Offline coverage audit of independently supplied targets and captured requests.

This module never fetches papers, loads a reader profile, or opens user state.
Targets must be chosen independently of the captured pool. That declaration is
auditable provenance, not proof of independence. Target coverage is not recall
over all relevant literature, and a complete HTTP response is not a complete
literature search. No ranking scores or provisional pool labels are accepted as
target-selection evidence.

Input: window_start/window_end (inclusive ISO dates), topic_ids, targets, captures.
Each target needs id, topic_ids, published_at, doi/url/aliases, and provenance
with method (independent_reader or independent_reference_search) and reference.
Each capture needs id, source, topic_ids, items, optional status
(complete/partial/failed/unknown), truncated (bool/null), and official (bool).
Optional total_available/next_cursor supply additional truncation evidence;
query is a display label. Alias queries must carry their parent topic_ids.
Item identity uses doi, metadata.doi, url, or explicit aliases; titles never match.
Native arXiv links and explicit 10.48550/arXiv DOIs share a version-family key.

Run: python -m dailydigest.rank.retrieval_evaluation MANIFEST.json
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date, datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlsplit

from ..dedupe import _extract_arxiv_id, _extract_doi, canonicalize_url


def _date(value: str | None) -> date | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("Publication date must be an ISO date or timestamp")
    try:
        if len(value) == 10:
            return date.fromisoformat(value)
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("Timestamp needs an explicit timezone")
        return stamp.astimezone(timezone.utc).date()
    except ValueError as exc:
        raise ValueError(f"Invalid publication date: {value}") from exc


def _identities(record: dict) -> set[str]:
    aliases = record.get("aliases", [])
    if not isinstance(aliases, list) or any(not isinstance(value, str) for value in aliases):
        raise ValueError("Identity aliases must be strings")
    metadata = record.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("Paper metadata must be an object")
    values = [record.get("doi", ""), metadata.get("doi", ""),
              record.get("url", ""), *aliases]
    keys = set()
    for value in values:
        if not value:
            continue
        if not isinstance(value, str):
            raise ValueError("Paper identities must be strings")
        doi = _extract_doi(value)
        if doi:
            keys.add("doi:" + doi)
        # Native arXiv abstract/PDF links and explicit arXiv DOIs identify the
        # same version family. Reuse the host-aware parser; do not infer an
        # arXiv paper merely from digits in an unrelated publisher's URL.
        arxiv = _extract_arxiv_id(value, "", "")
        if doi and doi.startswith("10.48550/arxiv."):
            arxiv = _extract_arxiv_id("", doi.removeprefix("10.48550/arxiv."), "arxiv") or arxiv
        if value.lower().startswith("arxiv:"):
            arxiv = _extract_arxiv_id("", value, "arxiv") or arxiv
        if arxiv:
            keys.add("arxiv:" + arxiv)
        parsed = urlsplit(value)
        if parsed.scheme.casefold() in {"http", "https"} and parsed.netloc:
            keys.add("url:" + canonicalize_url(value))
    return keys


def _topics(record: dict, available: set[str]) -> list[str]:
    values = record.get("topic_ids")
    if (not isinstance(values, list) or not values
            or any(not isinstance(value, str) for value in values)
            or len(values) != len(set(values)) or not set(values) <= available):
        raise ValueError("Each target/request needs distinct declared topic IDs")
    return values


def _fraction(found: int, total: int) -> dict:
    return {"found": found, "total": total, "fraction": found / total if total else None}


def _health(states: list[str]) -> str:
    if "degraded" in states:
        return "degraded"
    if not states or "unknown" in states:
        return "unknown"
    return "complete"


def evaluate_retrieval(manifest: dict) -> dict:
    """Account for each paper once overall and once per intended topic.

    Found-anywhere and found-via-own-query are separate: a paper encountered by
    another topic's query is present in the captured responses, but does not
    prove the intended query channel works or downstream ranking retained it.
    Missing status/truncation metadata remains unknown.
    """
    start, end = _date(manifest.get("window_start")), _date(manifest.get("window_end"))
    if start is None or end is None or start > end:
        raise ValueError("An ordered evaluation date window is required")
    topic_ids = manifest.get("topic_ids", [])
    if (not isinstance(topic_ids, list) or not topic_ids
            or any(not isinstance(topic, str) or not topic.strip() for topic in topic_ids)
            or len(topic_ids) != len(set(topic_ids))):
        raise ValueError("Distinct nonempty topic IDs are required")
    topics = set(topic_ids)
    captures, capture_keys = [], {}
    for capture in manifest.get("captures", []):
        cid = capture.get("id")
        if not isinstance(cid, str) or not cid.strip() or cid in capture_keys:
            raise ValueError("Duplicate or missing capture ID")
        if not isinstance(capture.get("source"), str) or not capture["source"].strip():
            raise ValueError("Each capture needs a source")
        attributed = _topics(capture, topics)
        query = capture.get("query")
        if query is not None and not isinstance(query, str):
            raise ValueError("Query labels must be strings")
        status = capture.get("status") or "unknown"
        truncated = capture.get("truncated")
        if status not in {"complete", "partial", "failed", "unknown"}:
            raise ValueError("Invalid request status")
        if truncated is not None and type(truncated) is not bool:
            raise ValueError("truncated must be a boolean or null")
        official = capture.get("official", False)
        if type(official) is not bool:
            raise ValueError("official must be a boolean")
        items = capture.get("items")
        if not isinstance(items, list):
            raise ValueError("Each capture needs its returned items (possibly empty)")
        total = capture.get("total_available")
        if total is not None and (type(total) is not int or total < len(items)):
            raise ValueError("total_available must cover all returned items")
        has_more = bool(capture.get("next_cursor")) or (total is not None and total > len(items))
        if has_more and truncated is False:
            raise ValueError("truncated=false conflicts with remaining results")
        if has_more:
            truncated = True
        health = ("degraded" if status in {"partial", "failed"} or truncated is True
                  else "unknown" if status == "unknown" or truncated is None else "complete")
        indexed = [(_identities(item), _date(item.get("published_at"))) for item in items]
        capture_keys[cid] = indexed
        captures.append({
            "id": cid, "source": capture["source"], "topic_ids": attributed, "query": query,
            "status": status, "truncated": truncated, "health": health,
            "returned_count": len(items),
            "total_available": total,
            "items_without_identity": sum(not keys for keys, _ in indexed),
            "items_without_date": sum(day is None for _, day in indexed),
            "abstract_completeness": _fraction(sum(isinstance(item.get("abstract"), str)
                                                   and bool(item["abstract"].strip())
                                                   for item in items), len(items)),
            "official_empty_result": official and health == "complete" and not items,
        })
    targets, target_ids, seen_keys = [], set(), set()
    outside_window, unresolved_date = [], []
    for target in manifest.get("targets", []):
        tid = target.get("id")
        if not isinstance(tid, str) or not tid.strip() or tid in target_ids:
            raise ValueError("Duplicate or missing target ID")
        target_ids.add(tid)
        attributed = _topics(target, topics)
        provenance = target.get("provenance", {})
        if (not isinstance(provenance, dict)
                or provenance.get("method") not in {"independent_reader", "independent_reference_search"}
                or not isinstance(provenance.get("reference"), str)
                or not provenance["reference"].strip()):
            raise ValueError("Targets require independent selection and a provenance reference")
        keys = _identities(target)
        if not keys or keys & seen_keys:
            raise ValueError("Missing or repeated target identity; merge topics on one target")
        seen_keys.update(keys)
        published = _date(target.get("published_at"))
        if published is None:
            unresolved_date.append(tid)
            continue
        if not start <= published <= end:
            outside_window.append(tid)
            continue
        matched, conflicts = [], []
        for cid, indexed in capture_keys.items():
            hits = [(item_keys, day) for item_keys, day in indexed if item_keys & keys]
            if hits:
                matched.append(cid)
                if any(day is not None and day != published for _, day in hits):
                    conflicts.append(cid)
        targets.append({"id": tid, "topic_ids": attributed, "published_at": published.isoformat(),
                        "provenance": dict(provenance), "found": bool(matched),
                        "matched_capture_ids": matched, "publication_date_conflicts": conflicts})
    by_topic = {}
    for topic in topic_ids:
        rows = [target for target in targets if topic in target["topic_ids"]]
        queries = [capture for capture in captures if topic in capture["topic_ids"]]
        query_ids = {capture["id"] for capture in queries}
        by_topic[topic] = {
            "target_coverage": _fraction(sum(row["found"] for row in rows), len(rows)),
            "found_via_own_queries": sum(bool(set(row["matched_capture_ids"]) & query_ids) for row in rows),
            "missing_target_ids": [row["id"] for row in rows if not row["found"]],
            "request_count": len(queries), "request_health": _health([q["health"] for q in queries]),
        }
    return {
        "scope": "Coverage of independently declared in-window targets, not all-literature recall",
        "provenance_limit": "Independence is declared by the manifest author; this audit cannot verify it",
        "health_limit": "Declared response completion does not establish exhaustive source coverage",
        "release_approved": False, "window_start": start.isoformat(), "window_end": end.isoformat(),
        "target_coverage": _fraction(sum(row["found"] for row in targets), len(targets)),
        "missing_target_ids": [row["id"] for row in targets if not row["found"]],
        "excluded_outside_window": outside_window, "unresolved_date": unresolved_date,
        "request_health": _health([topic["request_health"] for topic in by_topic.values()]),
        "topics": by_topic, "captures": captures, "targets": targets,
    }


def prepare_openalex_manifest(
    source_health: list[dict], *, topic_map: dict[str, list[str]],
    targets: list[dict] | None = None, query_items: dict[str, list[dict]] | None = None,
) -> dict:
    """Adapt actual OpenAlex health rows without inventing unrecorded response IDs.

    ``source_health`` is the selected OpenAlex subset of a single ingest snapshot,
    containing ``source`` and ``diagnostics.queries``. ``topic_map`` explicitly maps
    recorded interest/query labels to parent topic IDs. ``query_items`` optionally
    supplies the SAME query's normalized pre-cross-query-dedupe Item dictionaries,
    keyed by ``SOURCE:QUERY_INDEX``. Persisted diagnostics alone cannot reconstruct
    these lists. Empty lists are inferred only when zero fetched and returned rows
    were actually recorded, or the query was explicitly not attempted.

    Return a usable manifest only when query items, attribution, and original
    request windows are known; otherwise return precise preparation gaps. Never
    derive the request window from rebased candidate publication dates.
    """
    query_items = query_items or {}
    if not topic_map or any(not isinstance(key, str) or not key for key in topic_map):
        raise ValueError("Explicit interest/query to topic mapping is required")
    topic_ids = sorted({topic for values in topic_map.values() for topic in values})
    for values in topic_map.values():
        _topics({"topic_ids": values}, set(topic_ids))
    captures, gaps, windows, sources, used_items = [], [], set(), set(), set()
    for row in source_health:
        source = row.get("source")
        if not isinstance(source, str) or not source.strip() or source in sources:
            raise ValueError("Choose one ingest snapshot with distinct source rows")
        sources.add(source)
        diagnostics = row.get("diagnostics", {}).get("queries")
        if not isinstance(diagnostics, list) or not diagnostics:
            gaps.append({"source": source, "reason": "query_diagnostics_unavailable"})
            continue
        for index, diagnostic in enumerate(diagnostics):
            cid = f"{source}:{index}"
            query = diagnostic.get("query", "")
            parent = diagnostic.get("interest") or query
            attribution = topic_map.get(parent)
            if not attribution:
                gaps.append({"capture_id": cid, "reason": "parent_topic_unmapped", "label": parent})
            filters = dict(part.split(":", 1) for part in diagnostic.get("filter", "").split(",")
                           if ":" in part)
            start = _date(filters.get("from_publication_date"))
            end = _date(filters.get("to_publication_date"))
            if start is not None and end is not None and start <= end:
                windows.add((start.isoformat(), end.isoformat()))
            elif diagnostic.get("attempted") is not False:
                gaps.append({"capture_id": cid, "reason": "original_request_window_unavailable"})
            failed = bool(diagnostic.get("failure"))
            skipped = diagnostic.get("attempted") is False
            truncated = diagnostic.get("truncated")
            if truncated is not None and type(truncated) is not bool:
                raise ValueError("OpenAlex diagnostic truncated must be a boolean or null")
            status = ("failed" if failed else "partial" if skipped or truncated is True
                      else "complete" if diagnostic.get("attempted") is True
                      and diagnostic.get("exhausted") is True else "unknown")
            if cid in query_items:
                items = deepcopy(query_items[cid])
                used_items.add(cid)
                if not isinstance(items, list):
                    raise ValueError("Query items must be lists of normalized Item dictionaries")
                returned = diagnostic.get("returned_count")
                if type(returned) is int and len(items) != returned:
                    raise ValueError("Query item count differs from recorded pre-deduplication returned_count")
            elif skipped or (diagnostic.get("fetched_count") == 0
                             and diagnostic.get("returned_count") == 0):
                items = []
            else:
                items = None
                gaps.append({"capture_id": cid, "reason": "per_query_item_identities_unavailable"})
            captures.append({
                "id": cid, "source": source, "topic_ids": attribution,
                "query": query, "status": status,
                "truncated": True if skipped else truncated, "official": True,
                "items": items, "diagnostic": deepcopy(diagnostic),
                # Provider totals count raw works, unlike normalized returned
                # items. Preserve them without falsely deriving truncation from
                # parse exclusions or cross-query deduplication.
                "provider_total_available": diagnostic.get("total_available"),
                "requested_window": [start.isoformat(), end.isoformat()] if start and end else None,
            })
    if set(query_items) != used_items:
        raise ValueError("Query items include IDs absent from the selected diagnostics")
    if len(windows) != 1:
        gaps.append({"reason": "mixed_or_unknown_original_request_windows", "windows": sorted(windows)})
    manifest = None
    if not gaps:
        start, end = next(iter(windows))
        manifest = {"window_start": start, "window_end": end, "topic_ids": topic_ids,
                    "targets": deepcopy(targets or []), "captures": captures}
        evaluate_retrieval(manifest)  # Validate the complete adapter product.
    return {"status": "ready" if manifest is not None else "awaiting_capture_evidence",
            "manifest": manifest, "gaps": gaps, "prepared_captures": captures,
            "limit": "Source health retains counts and diagnostics, not per-query paper identities"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--prepare-openalex", action="store_true",
                        help="Adapt a bundle with source_health, topic_map, optional targets/query_items")
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text())
    result = prepare_openalex_manifest(**payload) if args.prepare_openalex else evaluate_retrieval(payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
