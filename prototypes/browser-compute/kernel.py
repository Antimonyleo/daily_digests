"""Prototype-only in-memory bridge to unchanged production ranking functions.

The bundle contains pure code, fixture settings and public source classifications.
It contains no persistence, ingestion, online summarization or user profile.
"""
from __future__ import annotations

import json
import sys
import types

import numpy as np


def _module(name, source):
    module = types.ModuleType(name)
    module.__package__ = name.rpartition(".")[0]
    sys.modules[name] = module
    exec(compile(source, name, "exec"), module.__dict__)
    return module


def install_bundle(bundle):
    """Install a separate package namespace; never shadow production modules."""
    for name in ("browser_digest", "browser_digest.rank"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    _module("browser_digest.models", "from types import SimpleNamespace\nItem = SourceSpec = SimpleNamespace\n")
    config = _module("browser_digest.config", "from types import SimpleNamespace\n")
    config.settings = types.SimpleNamespace(**bundle["settings"])
    config.get_settings = lambda: config.settings
    config.load_sources = lambda: [types.SimpleNamespace(**row) for row in bundle["sources"]]
    quality = _module("browser_digest.rank.source_quality", bundle["quality_source"])
    dedupe = _module("browser_digest.dedupe", bundle["dedupe_source"])
    ranker = _module("browser_digest.rank.ranker", bundle["ranker_source"])
    gates = _module("browser_digest.gates", bundle["gates_source"])
    return quality, dedupe, ranker, gates


def rank_fixture(fixture, vectors, modules):
    """Run the tested scoring/gate/picker subset; no model calls or I/O here.

    This first prototype deliberately uses undated fictional records, no learned
    model/facet-priority attribution, enrichment, history, or opportunity streams.
    Topic scores and ordering are computed afresh from supplied embeddings.
    """
    quality, dedupe, ranker, gates = modules
    rows = [types.SimpleNamespace(**{**record, "metadata_json": json.dumps(record["metadata"]),
                                    "section": "research", "published_at": None})
            for record in fixture["papers"]]
    matrix = np.asarray(vectors["documents"], dtype=np.float32)
    queries = np.asarray(vectors["queries"], dtype=np.float32)
    if matrix.ndim != 2 or queries.ndim != 2 or len(matrix) != len(rows) or len(queries) != len(fixture["queries"]):
        raise ValueError("Embeddings must align with every fixture document and query")
    if matrix.shape[1] == 0 or queries.shape[1] != matrix.shape[1] or not np.isfinite(matrix).all() or not np.isfinite(queries).all():
        raise ValueError("Embedding dimensions must agree and vectors must be finite")
    for array in (matrix, queries):
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if (norms < 1e-6).any():
            raise ValueError("Zero embeddings cannot produce a valid ranking")
        array /= norms
    profile = queries * np.array(fixture["query_weights"], dtype=np.float32).reshape(-1, 1)
    topics = ranker._cosine_sim(matrix, profile)
    scores, features = ranker._apply_quality_adjustments_with_features(
        rows, topics, [], None, learned_scores=np.zeros(len(rows), dtype=np.float32),
        hybrid_scores=topics, scoring_mode="cosine")
    scored = sorted(zip(rows, scores, strict=True), key=ranker._scored_sort_key)
    kept, dropped = gates._quality_gate(rows)
    quality_ids = {row.id for row in kept}
    after_quality = [(row, score) for row, score in scored if row.id in quality_ids]
    after_topic = gates._filter_off_topic(after_quality, features)
    # Serving deduplication operates after quality eligibility; retain exact policy.
    representatives = dedupe.dedupe_ranking_candidates([row for row, _ in after_topic])
    representative_ids = {row.id for row in representatives}
    eligible = [(row, score) for row, score in after_topic if row.id in representative_ids]
    selected = ranker.pick_top_per_section(eligible, {"research": fixture["cap"]})
    selected_ids = [row.id for row, _ in selected]
    topic_ids = {row.id for row, _ in after_topic}
    quality_reasons = {row["item_id"]: row["reason"] for row in dropped}
    result = []
    for row, score in scored:
        status = ("selected" if row.id in selected_ids else "quality" if row.id not in quality_ids
                  else "topic" if row.id not in topic_ids else "duplicate" if row.id not in representative_ids
                  else "budget_or_source_policy")
        result.append({"id": row.id, "topic_score": float(features[row.id]["topic_score"]),
                       "final_score": float(score), "source_bucket": quality.source_bucket(row),
                       "status": status, "reason": quality_reasons.get(row.id, "")})
    return {"selected_ids": selected_ids, "candidates": result,
            "counts": {"input": len(rows), "after_quality": len(after_quality),
                       "after_topic": len(after_topic), "after_identity": len(eligible),
                       "selected": len(selected_ids)}}
