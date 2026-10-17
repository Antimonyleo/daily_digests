"""Pure audits of frozen candidate pools; local absence is not retrieval recall."""

from __future__ import annotations

from collections import Counter, defaultdict
import json


def _metadata_counts(rows: list[dict], short_abstract_chars: int) -> dict:
    counts = Counter(count=len(rows))
    for row in rows:
        metadata = row.get("metadata") or {}
        abstract = str(metadata.get("abstract") or "").strip()
        counts["missing_abstract"] += not abstract
        counts["short_nonempty_abstract"] += bool(abstract) and len(abstract) < short_abstract_chars
        counts["missing_title"] += not str(metadata.get("title") or "").strip()
        counts["missing_authors"] += not str(metadata.get("authors") or "").strip()
        counts["missing_date"] += not metadata.get("published_at")
        try:
            publication = json.loads(metadata.get("metadata_json") or "{}")
            if not isinstance(publication, dict):
                raise ValueError("Metadata is not an object")
        except (TypeError, ValueError):
            publication = {}
            counts["invalid_metadata_json"] += 1
        counts["missing_venue_metadata"] += not publication.get("venue")
        counts["missing_publication_type"] += not (
            publication.get("work_type") or publication.get("publication_types")
            or publication.get("preprint_type")
        )
        counts["unknown_retraction_metadata"] += not isinstance(publication.get("is_retracted"), bool)
    return dict(counts)


def audit_coverage(
    snapshot: dict, *, known_papers: list[dict] | None = None,
    short_abstract_chars: int = 80,
) -> dict:
    """Describe metadata, observed stage losses, and recorded query health.

    ``known_papers`` accepts item IDs and optional frozen identity keys. It may
    include liked historical papers, which need not belong in this day's pool.
    Short abstracts are a review flag, never proof of truncation. This function
    neither opens databases nor calls sources, encoders, or configured providers.
    """
    if short_abstract_chars < 1:
        raise ValueError("short_abstract_chars must be positive")
    candidates = snapshot.get("candidates") or []
    research = [row for row in candidates if row.get("section") == "research"]
    by_source = defaultdict(list)
    profile = snapshot.get("profile") or {}
    configured_facets = profile.get("canonical_facets") or profile.get("keywords") or []
    by_facet = {str(label): [] for label in configured_facets if isinstance(label, str) and label.strip()}
    by_facet.setdefault("unknown", [])
    for row in research:
        by_source[str((row.get("metadata") or {}).get("source") or "unknown")].append(row)
        features = row.get("features") or {}
        secondary = features.get("secondary_facets") or []
        if not isinstance(secondary, list):
            secondary = []
        labels = {label for label in [features.get("primary_facet"), *secondary]
                  if isinstance(label, str) and label.strip()}
        for label in labels or {"unknown"}:
            by_facet.setdefault(label, []).append(row)
    research_ids = {row["item_id"] for row in research}
    health = snapshot.get("source_health") or []
    queries = [query for source in health
               for query in (source.get("diagnostics") or {}).get("queries", [])]
    statuses = Counter()
    paper_results = []
    for paper in known_papers or []:
        iid = paper.get("item_id")
        initial = [row for row in candidates if row.get("item_id") == iid]
        keys = set(paper.get("identity_keys") or [])
        keys.update(key for row in initial for key in row.get("identity_keys") or [])
        families = {row.get("family_id") for row in initial if row.get("family_id")}
        matches = [row for row in candidates if row in initial
                   or keys.intersection(row.get("identity_keys") or [])
                   or (row.get("family_id") and row.get("family_id") in families)]
        if any(row.get("selected") for row in matches):
            status = "selected"
        elif any(row.get("eligible") for row in matches):
            status = "eligible_unselected"
        elif matches:
            status = "rejected"
        else:
            status = "absent_from_local_pool"
        statuses[status] += 1
        paper_results.append({
            "item_id": iid, "status": status,
            "matched_item_ids": sorted(row["item_id"] for row in matches),
            "first_rejected_stage": matches[0].get("first_rejected_stage") if status == "rejected" else None,
            "retrieval_miss_confirmed": False,
        })
    return {
        "scope": "observed local snapshot; does not measure upstream recall",
        "decision_at": snapshot.get("decision_at"),
        "research_candidates": len(research),
        "research_eligible": sum(bool(row.get("eligible")) for row in research),
        "research_selected": sum(bool(row.get("selected")) for row in research),
        "research_first_rejected_stage": dict(Counter(
            row["first_rejected_stage"] for row in research if row.get("first_rejected_stage")
        )),
        "research_stage_counts": {stage: len(research_ids.intersection(ids))
                                  for stage, ids in (snapshot.get("stage_results") or {}).items()},
        "research_metadata": _metadata_counts(research, short_abstract_chars),
        "by_source": {source: _metadata_counts(rows, short_abstract_chars)
                      for source, rows in sorted(by_source.items())},
        "by_facet": {
            facet: {"raw": _metadata_counts(rows, short_abstract_chars),
                    "eligible": _metadata_counts([row for row in rows if row.get("eligible")], short_abstract_chars),
                    "selected": _metadata_counts([row for row in rows if row.get("selected")], short_abstract_chars)}
            for facet, rows in sorted(by_facet.items())
        },
        "facet_limit": "Captured ranking attribution, not independent scope labels or retrieval recall. Early rejects often lack attribution; they remain unknown. Multi-facet rows count once in each facet, so facet totals need not sum to pool size.",
        "short_abstract_chars": short_abstract_chars,
        "abstract_limit": "Missing and short text reduce available evidence; truncation is not established.",
        "publication_metadata_limit": "Missing venue metadata can coexist with a journal source name; unknown retraction metadata does not establish retraction or poor quality.",
        "source_health": {
            "status": "recorded" if health else "unavailable",
            "source_count": len(health),
            "source_failures": sum(source.get("ok") is False for source in health),
            "query_count": len(queries),
            "query_failures": sum(bool(query.get("failure")) for query in queries),
            "truncated_queries": sum(bool(query.get("truncated")) for query in queries),
            "unattempted_queries": sum(query.get("attempted") is False for query in queries),
            "indexing_delay_measured_queries": sum(query.get("indexing_delay") is not None for query in queries),
            "sources_without_query_diagnostics": [source.get("source") for source in health
                                                 if not (source.get("diagnostics") or {}).get("queries")],
        },
        "known_papers": {
            "count": len(paper_results), "status_counts": dict(statuses),
            "papers": paper_results,
            "limit": "Absence can reflect age, prior feedback, retention, or retrieval; no cause is inferred.",
        },
    }
