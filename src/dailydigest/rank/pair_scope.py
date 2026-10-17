"""Experimental cross-encoder scope decisions using raw query–article logits."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from types import SimpleNamespace

import numpy as np

from .scope import LABELS, anchor_records, classify_scope, fingerprint, scope_metrics


PAIR_SCOPE_VERSION = "pair-scope-v1"
LOGIT_GRID = (-8., -4., -2., 0., 2., 4., 6.)
MARGIN_GRID = (0., .5, 1., 2., 3.)


def pair_queries(definitions: dict) -> list[dict]:
    """One precise query per core facet; retain explicit non-core competitors.

    Query selection is fixed before evaluating articles. Broad aliases and
    second explanatory core anchors are excluded, not chosen per result.
    """
    records = anchor_records(definitions)
    seen = set()
    result = []
    for record in records:
        if record["label"] == "core":
            if record["facet"] in seen:
                continue
            seen.add(record["facet"])
        result.append(record)
    return result


def pair_evidence(logits, queries: list[dict]) -> list[dict]:
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(queries) or not np.isfinite(scores).all():
        raise ValueError("Pair logits must be a finite document-by-query matrix")
    if {query["label"] for query in queries} != set(LABELS[:-1]):
        raise ValueError("All three scope classes need query evidence")
    result = []
    for row in scores:
        classes, facets, matches = {}, {}, {}
        for query, score in zip(queries, row, strict=True):
            label = query["label"]
            if label not in classes or score > classes[label]:
                classes[label], matches[label] = float(score), query["text"]
            if label == "core":
                facets[query["facet"]] = max(facets.get(query["facet"], -np.inf), float(score))
        result.append({"scores": classes, "facet_scores": facets,
                       "matched_anchors": matches, "valid_vector": True})
    return result


@dataclass(frozen=True)
class PairScopeThresholds:
    min_score: float
    min_margin: float
    min_abstract_chars: int = 80
    facet_margin: float = .5

    def __post_init__(self):
        if not all(np.isfinite(v) for v in (self.min_score, self.min_margin, self.facet_margin)):
            raise ValueError("Pair thresholds must be finite")
        if self.min_margin < 0 or self.facet_margin < 0:
            raise ValueError("Pair margins cannot be negative")
        if type(self.min_abstract_chars) is not int or self.min_abstract_chars < 1:
            raise ValueError("Abstract evidence threshold must be a positive integer")


def classify_pair_scope(evidence: dict, abstract: str, thresholds: PairScopeThresholds) -> dict:
    # Reuse the exact class/facet/abstention decision logic while validating the
    # unbounded logit scale here instead of constructing cosine thresholds.
    controls = SimpleNamespace(min_similarity=thresholds.min_score, min_margin=thresholds.min_margin,
                               min_abstract_chars=thresholds.min_abstract_chars,
                               facet_margin=thresholds.facet_margin)
    decision = classify_scope(evidence, abstract, controls)
    decision["score_meaning"] = "raw cross-encoder relevance logits and class margin; not probabilities"
    if decision["reason"] == "weak_similarity":
        decision["reason"] = "weak_pair_score"
    return decision


def calibrate_pair_scope(evidence: list[dict], examples: list[dict]) -> dict:
    """Choose only on development data with predeclared precision/recall limits.

    Failure to meet constraints still returns the lowest-loss diagnostic arm,
    clearly ineligible for deployment. No evaluation-label input is accepted.
    """
    if not examples or len(evidence) != len(examples) or any(ex.get("split") != "calibration" for ex in examples):
        raise ValueError("Only aligned nonempty calibration examples may choose pair thresholds")
    if len({ex["family_id"] for ex in examples}) != len(examples):
        raise ValueError("Calibration requires one representative per paper family")
    trials = []
    for score, margin in product(LOGIT_GRID, MARGIN_GRID):
        thresholds = PairScopeThresholds(score, margin)
        decisions = [classify_pair_scope(ev, ex["abstract"], thresholds)
                     for ev, ex in zip(evidence, examples, strict=True)]
        metrics = scope_metrics(examples, decisions)
        loss = sum(0 if ex["label"] == d["label"] else 4 if d["label"] == "core"
                   else .25 if d["label"] == "uncertain" and ex["label"] != "core" else 1
                   for ex, d in zip(examples, decisions, strict=True))
        passed = ((metrics["core_precision"] or 0) >= .8 and (metrics["core_recall"] or 0) >= .75
                  and metrics["coverage"] >= .5)
        trials.append({"thresholds": asdict(thresholds), "loss": loss,
                       "development_constraints_met": passed, "metrics": metrics})
    acceptable = [trial for trial in trials if trial["development_constraints_met"]]
    best = min(acceptable or trials, key=lambda t: (t["loss"], t["metrics"]["false_core"],
                                                   -(t["metrics"]["core_recall"] or 0),
                                                   -t["thresholds"]["min_margin"],
                                                   -t["thresholds"]["min_score"]))
    return {**best, "promotion_eligible": False, "trials": len(trials),
            "grid": {"logit": list(LOGIT_GRID), "margin": list(MARGIN_GRID)},
            "objective": "false core=4; missed core/other confident error=1; noncore abstention=.25",
            "development_constraints": "core precision >= .8; core recall >= .75; coverage >= .5",
            "calibration_signature": fingerprint(examples), "calibration_ids": [ex["id"] for ex in examples]}
