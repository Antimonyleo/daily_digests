"""Quoted local passage relevance; no contribution or task/object inference."""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np


_BREAK = re.compile(r"(?<=[.!?;])\s+")
_GROUPS = ("core", "context", "off_interest")
_MEANING = "Raw local cross-encoder passage relevance logits; not probabilities and not proof of an original contribution or reader relevance."


def _ordered_queries(queries):
    records = []
    for query in queries:
        if not isinstance(query, dict):
            raise ValueError("Query records must be objects")
        label = query.get("label", query.get("kind"))
        facet, text = query.get("facet"), query.get("text")
        facet = "" if facet is None else facet
        if (label not in _GROUPS or ("kind" in query and query["kind"] != label)
                or not isinstance(text, str) or not text.strip()
                or not isinstance(facet, str) or (label == "core" and not facet.strip())
                or (label != "core" and facet)):
            raise ValueError("Invalid ordered interest query schema")
        records.append({"label": label, "facet": facet, "text": text})
    order = [_GROUPS.index(q["label"]) for q in records]
    if set(order) != {0, 1, 2} or order != sorted(order):
        raise ValueError("Queries must contain core, context, off_interest in that order")
    if len({(q["label"], q["facet"], q["text"]) for q in records}) != len(records):
        raise ValueError("Duplicate query records")
    return records


def _passages(abstract):
    spans, start = [], 0
    for boundary in [match.start() for match in _BREAK.finditer(abstract)] + [len(abstract)]:
        raw = abstract[start:boundary]
        quote = raw.strip()
        if quote:
            offset = start + len(raw) - len(raw.lstrip())
            spans.append({"start": offset, "end": offset + len(quote),
                          "quote": quote, "passage_index": len(spans)})
        start = boundary
    return spans


def evidence_for_article(title, abstract, queries, scorer, max_sentences=12):
    """Score evenly sampled clauses, retaining exact abstract quote offsets.

    Pairs are passage-major and query-minor. At budgets of at least three,
    samples cover the head, middle and tail. Titles provide model context but
    are never quoted as supporting evidence. The scorer is explicitly supplied;
    this function has no network, model-download, database or provider paths.
    """
    if not isinstance(title, str) or not isinstance(abstract, str):
        raise ValueError("Article title and abstract must be strings")
    if type(max_sentences) is not int or max_sentences < 1:
        raise ValueError("max_sentences must be a positive integer")
    records = _ordered_queries(queries)
    spans = _passages(abstract)
    count = min(len(spans), max_sentences)
    indices = [round(i * (len(spans) - 1) / (count - 1)) for i in range(count)] if count > 1 else [0][:count]
    selected = [spans[i] for i in indices]
    result = {
        "version": "contribution-passage-evidence-v1", "score_meaning": _MEANING,
        "query_signature": hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "sampling": "Evenly spaced clause indices; title is context only",
        "total_passages": len(spans), "scored_passages": count,
        "pair_count": count * len(records), "passages": selected,
        "facet_evidence": {}, "competitor_evidence": {},
    }
    if not selected:
        return result
    pairs = [(query["text"], ((title.strip() + ". ") if title.strip() else "") + passage["quote"])
             for passage in selected for query in records]
    scores = np.asarray(scorer.score_pairs(pairs))
    if (scores.shape != (len(pairs),) or scores.dtype.kind not in "fiu"
            or not np.isfinite(scores).all()):
        raise ValueError("Expected one finite scalar numeric logit per ordered pair")
    for passage, row in zip(selected, scores.reshape(count, len(records)), strict=True):
        for index, (query, score) in enumerate(zip(records, row, strict=True)):
            bucket = result["facet_evidence"] if query["label"] == "core" else result["competitor_evidence"]
            key = query["facet"] if query["label"] == "core" else query["label"]
            if key not in bucket or float(score) > bucket[key]["score"]:
                bucket[key] = {**passage, "score": float(score), "query_index": index,
                               "query": dict(query), "task_object_evidence": passage["quote"]}
    return result
