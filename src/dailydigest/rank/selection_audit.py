"""Compact diagnostics for captured research selection, with no state access.

These describe score distributions and omissions, not scientific quality or the
cause of an individual omission. Facet labels retain the captured classifier's
uncertainty; source buckets describe editorial policy, not article merit.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np


def _scores(candidates: list[dict], key: str) -> dict:
    values = []
    for candidate in candidates:
        value = (candidate.get("features", {}).get("topic_score") if key == "topic_score"
                 else candidate.get("ranking_score"))
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            values.append(float(value))
    quantiles = np.quantile(values, [.05, .5, .95]) if values else [None] * 3
    return {"observed": len(values), "missing": len(candidates) - len(values),
            **dict(zip(("p05", "median", "p95"), quantiles, strict=True))}


def audit_selection_snapshot(snapshot: dict) -> dict:
    """Summarize research coverage and score/source/facet composition.

    Every group reports both the available pool and chosen subset. This keeps a
    large source's high raw selection count from being mistaken for a high rate.
    Ranking scores are run-relative ordering signals, never probabilities.
    """
    research = [candidate for candidate in snapshot["candidates"]
                if candidate["section"] == "research"]
    eligible = [candidate for candidate in research if candidate["eligible"]]
    selected = [candidate for candidate in research if candidate["selected"]]
    groups = {}
    for dimension in ("source", "source_bucket", "primary_facet"):
        entries = defaultdict(list)
        for candidate in research:
            if dimension == "source":
                value = candidate.get("metadata", {}).get("source")
            elif dimension == "source_bucket":
                value = candidate.get("selection_facts", {}).get("source_bucket")
            else:
                value = candidate.get("features", {}).get("primary_facet")
            entries[str(value or "unknown")].append(candidate)
        summary = {}
        for value, candidates in sorted(entries.items()):
            available = [candidate for candidate in candidates if candidate["eligible"]]
            chosen = [candidate for candidate in candidates if candidate["selected"]]
            summary[value] = {
                "pool_count": len(candidates), "eligible_count": len(available),
                "selected_count": len(chosen),
                "selection_rate": len(chosen) / len(available) if available else None,
                "eligible_scores": {key: _scores(available, key)
                                    for key in ("topic_score", "ranking_score")},
                "selected_scores": {key: _scores(chosen, key)
                                    for key in ("topic_score", "ranking_score")},
            }
        groups[dimension] = summary
    weakest_selected = min((float(candidate["ranking_score"]) for candidate in selected),
                           default=None)
    omissions = [candidate for candidate in eligible if not candidate["selected"]
                 and weakest_selected is not None
                 and candidate["ranking_score"] > weakest_selected]
    omissions.sort(key=lambda candidate: (-candidate["ranking_score"], candidate["item_id"]))
    return {
        "run_id": snapshot.get("run_id"), "research_pool_count": len(research),
        "eligible_research_count": len(eligible), "selected_research_count": len(selected),
        "research_budget": snapshot["section_budgets"].get("research", 0),
        "reading_mode": snapshot.get("reading_mode"),
        "source_health": snapshot.get("source_health", {}),
        "rejection_stages": dict(Counter(candidate.get("first_rejected_stage") or "none"
                                         for candidate in research)),
        "groups": groups,
        "higher_rank_score_omissions": [{"item_id": candidate["item_id"],
                                         "rank_score": candidate["ranking_score"],
                                         "source": candidate["metadata"].get("source"),
                                         "primary_facet": candidate["features"].get("primary_facet")}
                                        for candidate in omissions],
        "limitations": [
            "Captured topic assignments are model judgments, not validated interest labels.",
            "Omissions may reflect source balance, ordering preferences, or reading budgets; "
            "a higher omitted score alone does not establish a defect.",
            "Venue/source buckets are priors and do not measure article-level scientific quality.",
            "One day's score distributions cannot establish stable score calibration.",
        ],
    }
