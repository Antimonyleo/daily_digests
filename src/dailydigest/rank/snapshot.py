"""Freeze local decision evidence without calling providers or loading encoders."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
import hashlib
import json

from ..dedupe import candidate_identity_keys
from .embed import active_embedding_provenance, active_embedding_signature
from . import source_quality as quality


SELECTION_SETTINGS = (
    "max_preprint_research_frac", "max_low_impact_research_frac",
    "low_impact_relevance_floor", "research_final_score_floor_frac",
    "research_final_score_min_keep", "adaptive_relevance_floor",
)
RANKING_SETTINGS = SELECTION_SETTINGS + (
    "min_topic_relevance", "negative_interest_weight", "negative_interest_margin",
    "research_quality_weight", "venue_relevance_credit", "topic_priority_bonus_scale",
    "topic_coverage_bonus_scale", "within_day_dedupe", "within_day_dedupe_threshold",
    "citation_enrichment", "venue_quality_weight", "exploration_slots", "embed_model", "embed_backend",
    "contribution_order_strength",
    "contribution_reranker_weight",
    "embed_query_prefix", "embed_doc_prefix",
)


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def capture_selection_facts(rows) -> dict[int, dict]:
    """Resolve configurable venue classifications once for serving and replay."""
    return {int(row.id): {
        "source_bucket": quality.source_bucket(row),
        "is_arxiv_cs": quality.is_arxiv_cs_source(row),
        "is_preprint": quality.is_preprint_source(row),
        "is_published": quality.is_published_journal_source(row),
        "is_high_quality": quality.is_high_quality_journal_source(row),
        "is_low_impact": quality.is_low_impact_research(row),
        "source_diversity_key": quality.source_diversity_key(row),
    } for row in rows}


def _families(rows) -> dict[int, str]:
    """Transitive identity union; metadata-only aliases do not split a paper."""
    parents = {int(row.id): int(row.id) for row in rows}
    aliases: dict[str, int] = {}

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for row in rows:
        iid = int(row.id)
        for key in candidate_identity_keys(row):
            if key in aliases:
                a, b = root(iid), root(aliases[key])
                parents[max(a, b)] = min(a, b)
            aliases[key] = iid
    group_keys: dict[int, list[str]] = {}
    for key, iid in aliases.items():
        group_keys.setdefault(root(iid), []).append(key)
    return {iid: _hash(sorted(group_keys.get(root(iid), [f"item:{iid}"]))) for iid in parents}


def build_ranking_snapshot(
    *, run_id: str, digest_id: str, decision_at: datetime, decision_local_date: date,
    reading_mode: str, candidates, scored, eligible, labeled, features,
    section_budgets: dict, settings, profile, profile_matrix, catch_up: bool,
    ingest_stats, feedback_before: dict, feedback_after: dict, stages: dict,
    selection_facts: dict | None = None,
) -> dict:
    """Capture every local candidate, including rejected and unselected rows.

    Scores and eligibility are frozen inputs for selection replay. This does not
    promise a re-fetch or a full embedding rescore: candidate vectors and remote
    enrichment responses are not archived. Credentials/settings paths are
    excluded from captured settings; profiles and source diagnostics stay local.
    """
    candidates = list(candidates)
    if selection_facts is None:
        selection_facts = capture_selection_facts(candidates)
    families = _families(candidates)
    input_order = {int(row.id): pos for pos, (row, _) in enumerate(eligible)}
    scores = {int(row.id): float(features.get(int(row.id), {}).get('final_score', score))
              for row, score in scored}
    scores.update({int(row.id): float(score) for row, score in eligible})
    selected = {}
    positions: dict[str, int] = {}
    for row, _, label in labeled:
        section = row.section or ""
        selected[int(row.id)] = (positions.get(section, 0), label)
        positions[section] = positions.get(section, 0) + 1
    stage_ids = {name: {int(row.id) for row in rows} for name, rows in stages.items()}
    frozen = []
    for row in candidates:
        iid = int(row.id)
        metadata = {}
        for name in ("source", "external_id", "url", "title", "abstract", "authors",
                     "published_at", "fetched_at", "metadata_json"):
            value = getattr(row, name, None)
            metadata[name] = value.isoformat() if isinstance(value, datetime) else value
        feat = deepcopy(features.get(iid, {}))
        evidence = getattr(row, "enrichment_evidence", None)
        if isinstance(evidence, dict):
            feat["enrichment"] = deepcopy(evidence)
        first_failed = next((name for name, ids in stage_ids.items() if iid not in ids), None)
        if first_failed is None and iid not in input_order:
            first_failed = "post_score_eligibility"
        frozen.append({
            "item_id": iid, "family_id": families[iid], "section": row.section or "",
            "identity_keys": candidate_identity_keys(row),
            "eligible": iid in input_order, "input_position": input_order.get(iid),
            "ranking_score": scores.get(iid),
            "quality_score": feat.get("confidence_score", feat.get("quality_score")),
            "selected": iid in selected, "position": selected.get(iid, (None, None))[0],
            "label": selected.get(iid, (None, None))[1], "features": feat,
            "metadata": metadata, "text_signature": _hash([metadata["title"], metadata["abstract"]]),
            "first_rejected_stage": first_failed,
            "selection_facts": deepcopy(selection_facts[iid]),
        })
    profile_data = profile.model_dump(mode="json") if hasattr(profile, "model_dump") else vars(profile)
    ranking_settings = {name: getattr(settings, name) for name in RANKING_SETTINGS
                        if hasattr(settings, name)}
    snapshot = {
        "schema_version": 1, "run_id": run_id, "digest_id": digest_id,
        "decision_at": decision_at.isoformat(), "decision_local_date": decision_local_date.isoformat(),
        "reading_mode": reading_mode, "catch_up": catch_up,
        "exploration_slots": int(getattr(settings, "exploration_slots", 0) or 0),
        "complete_candidate_pool": True, "pool_scope": "all locally available candidates for this run",
        "replay_scope": "selection from frozen scores and eligibility; not retrieval or full rescoring",
        "section_budgets": section_budgets, "candidates": frozen,
        "selection_settings": {name: ranking_settings[name] for name in SELECTION_SETTINGS},
        "ranking_settings": ranking_settings, "settings_signature": _hash(ranking_settings),
        "profile": profile_data, "profile_signature": _hash(profile_data),
        "profile_matrix": profile_matrix.tolist(),
        "scorer_version": quality.RANKER_VERSION,
        "embedding_signature": active_embedding_signature(),
        "query_embedding_signature": active_embedding_signature(is_query=True),
        "embedding_provenance": active_embedding_provenance(),
        "feedback_cutoff": decision_at.isoformat(), "feedback_at_start": feedback_before,
        "feedback_consistent": feedback_before == feedback_after,
        "feedback_at_end": feedback_after,
        "stage_results": {name: sorted(ids) for name, ids in stage_ids.items()},
        "source_health": [stat.model_dump(mode="json") for stat in ingest_stats],
    }
    # JSON roundtrip freezes references and rejects non-finite evidence before
    # publication rather than persisting plausible but unreplayable records.
    return json.loads(json.dumps(snapshot, allow_nan=False))
