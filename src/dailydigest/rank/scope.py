"""Experimental local scope classifier, separate from utility and venue quality.

Embeddings compare contribution-level core anchors with explicit contextual and
off-interest competitors. Scores are cosines, never probabilities. This module
has no provider, store, profile mutation, or serving-default side effects.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import html
import json
from itertools import product
import unicodedata

import numpy as np


LABELS = ("core", "context", "off_interest", "uncertain")
SCOPE_VERSION = "contrastive-scope-v1"
# A predeclared development grid; evaluation rows never choose thresholds.
SIMILARITY_GRID = (.4, .5, .6, .65, .7, .75, .8)
MARGIN_GRID = (0., .01, .02, .03, .05, .08)


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def scope_title_identity(title: str) -> str:
    """Conservative long-title identity for split isolation, independent of dates.

    Production dedupe remains unchanged. Different feed dates or abstract
    versions must not let an otherwise identical work leak into evaluation.
    """
    text = unicodedata.normalize("NFKC", html.unescape(title)).casefold()
    normalized = "".join(character for character in text if character.isalnum())
    return normalized if len(normalized) >= 40 else ""


def anchor_records(definitions: dict) -> list[dict]:
    """Keep specific anchors separate; broad aliases are not averaged into them."""
    if set(definitions) != {"core", "context", "off_interest"}:
        raise ValueError("Definitions require core, context and off_interest anchors")
    if not isinstance(definitions["core"], dict) or not definitions["core"]:
        raise ValueError("At least one named core facet is required")
    result = []
    for label in LABELS[:-1]:
        groups = definitions["core"] if label == "core" else {"": definitions[label]}
        for facet, texts in groups.items():
            if label == "core" and (not isinstance(facet, str) or not facet.strip()):
                raise ValueError("Core facets must have nonempty names")
            if not isinstance(texts, list) or not texts:
                raise ValueError("Every scope class/facet needs an anchor list")
            for text in texts:
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Scope anchors must be nonempty strings")
                result.append({"label": label, "facet": facet, "text": text.strip()})
    return result


def validate_benchmark(data: dict) -> None:
    if data.get("schema_version") != 1 or not data.get("provenance", {}).get("kind"):
        raise ValueError("Benchmark requires schema_version 1 and explicit provenance")
    anchor_records(data["definitions"])
    examples = data.get("examples", [])
    if not examples:
        raise ValueError("Benchmark examples cannot be empty")
    seen = set()
    families = {}
    texts = {}
    titles = {}
    for example in examples:
        for field in ("id", "family_id", "split", "title", "label", "rationale"):
            if not isinstance(example.get(field), str) or not example[field].strip():
                raise ValueError(f"Benchmark example requires {field}")
        if example["id"] in seen:
            raise ValueError("Benchmark IDs must be unique")
        seen.add(example["id"])
        if example["split"] not in {"calibration", "evaluation"} or example["label"] not in LABELS:
            raise ValueError("Invalid benchmark split or scope label")
        if not isinstance(example.get("abstract"), str):
            raise ValueError("Abstract must be a string, possibly empty")
        facet = example.get("facet", "")
        if example["label"] == "core" and facet not in data["definitions"]["core"]:
            raise ValueError("Core example needs a defined facet")
        if example["label"] != "core" and facet:
            raise ValueError("Non-core example cannot assert a core facet")
        family, split = example["family_id"], example["split"]
        if family in families and families[family] != split:
            raise ValueError("Paper families must not cross benchmark splits")
        families[family] = split
        title_key = scope_title_identity(example["title"])
        if title_key:
            if title_key in titles and titles[title_key] != split:
                raise ValueError("A normalized paper title must not cross benchmark splits")
            titles[title_key] = split
        text_key = " ".join((example["title"] + " " + example["abstract"]).casefold().split())
        if text_key in texts:
            raise ValueError("Duplicate benchmark text, even under a different family ID")
        texts[text_key] = split
    if set(families.values()) != {"calibration", "evaluation"}:
        raise ValueError("Benchmark needs separate calibration and evaluation splits")


def score_scope(item_vectors: np.ndarray, anchor_vectors: np.ndarray,
                records: list[dict]) -> list[dict]:
    """Compute strongest core and competing-class evidence without user votes."""
    items = np.asarray(item_vectors, dtype=np.float64)
    anchors = np.asarray(anchor_vectors, dtype=np.float64)
    if (items.ndim != 2 or anchors.ndim != 2 or not anchors.size
            or items.shape[1] != anchors.shape[1] or len(records) != len(anchors)
            or not np.isfinite(items).all() or not np.isfinite(anchors).all()):
        raise ValueError("Scope embeddings must be finite, aligned matrices")
    if set(record["label"] for record in records) != set(LABELS[:-1]):
        raise ValueError("Core, context and off_interest evidence are all required")
    norms = np.linalg.norm(anchors, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("A zero anchor vector has no scope meaning")
    item_norms = np.linalg.norm(items, axis=1, keepdims=True)
    similarities = np.clip((items / np.clip(item_norms, 1e-12, None)) @ (anchors / norms).T, -1, 1)
    result = []
    for row, norm in zip(similarities, item_norms[:, 0], strict=True):
        scores, matches, facets = {}, {}, {}
        for index, record in enumerate(records):
            label, value = record["label"], float(row[index])
            if label not in scores or value > scores[label]:
                scores[label], matches[label] = value, record["text"]
            if label == "core":
                facet = record["facet"]
                facets[facet] = max(facets.get(facet, -1.), value)
        result.append({"scores": scores, "facet_scores": facets,
                       "matched_anchors": matches, "valid_vector": bool(norm > 1e-12)})
    return result


@dataclass(frozen=True)
class ScopeThresholds:
    min_similarity: float
    min_margin: float
    min_abstract_chars: int = 80
    facet_margin: float = .03

    def __post_init__(self):
        values = (self.min_similarity, self.min_margin, self.facet_margin)
        if not all(np.isfinite(v) for v in values) or not -1 <= self.min_similarity <= 1:
            raise ValueError("Scope similarity threshold must be finite and in [-1, 1]")
        if not 0 <= self.min_margin <= 2 or not 0 <= self.facet_margin <= 2:
            raise ValueError("Scope margins must be finite and in [0, 2]")
        if type(self.min_abstract_chars) is not int or self.min_abstract_chars < 1:
            raise ValueError("Abstract evidence threshold must be a positive integer")


def classify_scope(evidence: dict, abstract: str, thresholds: ScopeThresholds) -> dict:
    """Abstain on thin text, weak matches or ambiguity between scope classes.

    Two core facets never compete for the class margin. Diagnostic anchor
    matches are similarity evidence, not quoted explanations from the article.
    """
    scores = evidence["scores"]
    ordered = sorted(LABELS[:-1], key=lambda label: -scores[label])
    winner = ordered[0]
    margin = float(scores[winner] - scores[ordered[1]])
    label, reason = winner, "contrastive_match"
    if len(" ".join(abstract.split())) < thresholds.min_abstract_chars:
        label, reason = "uncertain", "insufficient_abstract"
    elif not evidence["valid_vector"]:
        label, reason = "uncertain", "invalid_document_vector"
    elif scores[winner] < thresholds.min_similarity:
        label, reason = "uncertain", "weak_similarity"
    elif margin <= max(thresholds.min_margin, 1e-7):
        label, reason = "uncertain", "ambiguous_scope"
    facets = []
    if label == "core":
        competitor = max(scores["context"], scores["off_interest"])
        facets = [facet for facet, score in sorted(evidence["facet_scores"].items(), key=lambda kv: (-kv[1], kv[0]))
                  if score >= scores["core"] - thresholds.facet_margin
                  and score >= thresholds.min_similarity
                  and score - competitor > max(thresholds.min_margin, 1e-7)]
    return {"label": label, "facet": facets[0] if facets else "", "facets": facets,
            "reason": reason, "scores": dict(scores), "margin": margin,
            "matched_anchor": evidence["matched_anchors"][winner],
            "score_meaning": "cosine and cross-class margin; not probability or article evidence"}


def scope_metrics(examples: list[dict], decisions: list[dict]) -> dict:
    if len(examples) != len(decisions):
        raise ValueError("Examples and predictions must align")
    confusion = {label: dict.fromkeys(LABELS, 0) for label in LABELS}
    counts = Counter(decision["label"] for decision in decisions)
    facets = {}
    for example, decision in zip(examples, decisions, strict=True):
        truth, predicted = example["label"], decision["label"]
        confusion[truth][predicted] += 1
        if truth == "core":
            facet = facets.setdefault(example["facet"], {"available": 0, "retained_as_core": 0,
                                                         "correct_facet": 0})
            facet["available"] += 1
            facet["retained_as_core"] += predicted == "core"
            facet["correct_facet"] += predicted == "core" and example["facet"] in decision.get("facets", [])
    n = len(examples)
    true_core = confusion["core"]["core"]
    available_core = sum(confusion["core"].values())
    uncertain_core = confusion["uncertain"]["core"]
    known_false_core = confusion["context"]["core"] + confusion["off_interest"]["core"]
    return {"n": n, "counts": {label: counts[label] for label in LABELS},
            "label_counts": {label: sum(confusion[label].values()) for label in LABELS},
            "confusion": confusion,
            "accuracy_including_abstentions": sum(confusion[label][label] for label in LABELS) / n if n else None,
            "coverage": 1 - counts["uncertain"] / n if n else None,
            "core_precision": true_core / counts["core"] if counts["core"] else None,
            # Retain the original strict-label agreement above for frozen
            # calibration objectives. An uncertain annotation is not a proven
            # off-scope paper: expose that ambiguity separately in reports.
            "core_precision_meaning": "strict core-label agreement; uncertain annotations count as unconfirmed",
            "known_noncore_as_core": known_false_core,
            "uncertain_as_core": uncertain_core,
            "core_precision_known_labels": true_core / (true_core + known_false_core)
            if true_core + known_false_core else None,
            "core_precision_bounds": [true_core / counts["core"],
                                      (true_core + uncertain_core) / counts["core"]]
            if counts["core"] else None,
            "core_recall": true_core / available_core if available_core else None,
            "false_core": counts["core"] - true_core,
            "per_facet": facets}


def calibrate_scope(evidence: list[dict], examples: list[dict]) -> dict:
    """Select abstention thresholds on calibration data only, not probabilities.

    False core assignments cost 4; missed core costs 1; incorrect confident
    non-core decisions cost 1; abstaining on a known non-core costs .25.
    Viability requires >=75% core recall and >=50% overall coverage on this
    development split. These are explicit development choices, not guarantees.
    """
    if not examples or len(evidence) != len(examples) or any(e.get("split") != "calibration" for e in examples):
        raise ValueError("Only aligned nonempty calibration examples may choose thresholds")
    if len({e["family_id"] for e in examples}) != len(examples):
        raise ValueError("Calibration must contain one example per paper family")
    trials = []
    for similarity, margin in product(SIMILARITY_GRID, MARGIN_GRID):
        thresholds = ScopeThresholds(similarity, margin)
        decisions = [classify_scope(ev, ex["abstract"], thresholds) for ev, ex in zip(evidence, examples, strict=True)]
        metrics = scope_metrics(examples, decisions)
        loss = 0.
        for ex, prediction in zip(examples, decisions, strict=True):
            truth, label = ex["label"], prediction["label"]
            if truth == label:
                continue
            loss += 4 if label == "core" else .25 if label == "uncertain" and truth != "core" else 1
        viable = (metrics["core_recall"] or 0) >= .75 and metrics["coverage"] >= .5
        trials.append({"thresholds": asdict(thresholds), "loss": loss, "coverage_viable": viable, "metrics": metrics})
    viable = [trial for trial in trials if trial["coverage_viable"]]
    best = min(viable or trials, key=lambda t: (t["loss"], t["metrics"]["false_core"],
                                               -(t["metrics"]["core_recall"] or 0),
                                               -t["thresholds"]["min_margin"],
                                               -t["thresholds"]["min_similarity"]))
    return {**best, "promotion_eligible": False, "trials": len(trials),
            "grid": {"similarity": list(SIMILARITY_GRID), "margin": list(MARGIN_GRID)},
            "objective": "false core=4; missed core/other confident error=1; noncore abstention=.25",
            "coverage_constraint": "calibration core recall >= .75 and coverage >= .5; not a precision or quality gate",
            "calibration_ids": [ex["id"] for ex in examples],
            "calibration_signature": fingerprint(examples)}
