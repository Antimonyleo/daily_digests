"""Fixed local pairwise contribution reranker; never a scope eligibility gate.

Features are the 24 ordered raw cross-encoder query scores. Only definitive
calibration families train; evaluation labels never choose coefficients or scale.
The bounded score expresses relative ranking, not probability or usefulness.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .scope import fingerprint


CONTRIBUTION_RANKER_VERSION = "pairwise-contribution-v1"
MODEL_PARAMETERS = dict(C=1, fit_intercept=False, random_state=0, max_iter=2000)
FOLD_COUNT = 4
TIERS = ("core", "context", "off_interest")
TIER_PAIRS = (("core", "context"), ("core", "off_interest"), ("context", "off_interest"))


def _matrix(logits, width):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != width or not np.isfinite(values).all():
        raise ValueError("Features must be a finite document-by-query matrix")
    return values


def _validate_queries(queries):
    if not isinstance(queries, list) or len(queries) != 24:
        raise ValueError("Exactly 24 frozen ordered queries are required")
    for query in queries:
        if not isinstance(query, dict) or query.get("label") not in TIERS:
            raise ValueError("Query labels must be core, context or off_interest")
        if not isinstance(query.get("text"), str) or not query["text"].strip():
            raise ValueError("Every query requires nonempty text")
        facet = query.get("facet")
        if not isinstance(facet, str) or (not facet.strip() if query["label"] == "core" else bool(facet)):
            raise ValueError("Only core queries require a nonempty facet")
    if {query["label"] for query in queries} != set(TIERS):
        raise ValueError("All three tiers require query evidence")


def _pairs(values, labels):
    differences, pair_counts = [], {}
    for higher, lower in TIER_PAIRS:
        key = f"{higher}_{lower}"
        left, right = values[labels == higher], values[labels == lower]
        differences.append((left[:, None, :] - right[None, :, :]).reshape(-1, values.shape[1]))
        pair_counts[key] = len(left) * len(right)
    if not all(pair_counts.values()):
        raise ValueError("Pairwise training requires all three contribution tiers")
    count = sum(pair_counts.values())
    # Keep average weight 1 so class balancing does not silently rescale fixed C.
    weights = np.concatenate([
        np.full(n, count / (len(TIER_PAIRS) * n)) for n in pair_counts.values()
    ])
    diffs = np.vstack(differences)
    return (np.vstack([diffs, -diffs]), np.concatenate([np.ones(count), np.zeros(count)]),
            np.tile(weights, 2), {
                "unordered_pair_counts": pair_counts, "mirrored_rows": 2 * count,
                "total_weight_by_tier_pair": {key: 2 * count / len(TIER_PAIRS) for key in pair_counts},
                "weight_rule": "mirrored row weight = total unordered pairs / (3 * tier-pair count)",
            })


def _fit(values, labels):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)
    pairs, targets, weights, summary = _pairs(scaled, labels)
    if not np.isfinite(pairs).all():
        raise ValueError("Feature scaling or pair differences produced nonfinite values")
    classifier = LogisticRegression(**MODEL_PARAMETERS)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(pairs, targets, sample_weight=weights)
    except ConvergenceWarning as exc:
        raise ValueError("The fixed contribution ranker failed to converge") from exc
    state = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
             "coefficients": classifier.coef_[0].tolist()}
    return state, summary


def _project(values, mean, scale, coefficients):
    with np.errstate(over="ignore", invalid="ignore"):
        result = ((values - mean) / scale) @ coefficients
    if not np.isfinite(result).all():
        raise ValueError("Contribution projection produced nonfinite values")
    return result


def _pair_metrics(scores, labels):
    metrics = {}
    for higher, lower in TIER_PAIRS:
        differences = scores[labels == higher][:, None] - scores[labels == lower][None, :]
        metrics[f"{higher}_{lower}"] = {
            "pairs": int(differences.size),
            "correct": float(np.sum(differences > 0) + .5 * np.sum(differences == 0)),
        }
    return metrics


@dataclass
class ContributionRanker:
    """Numeric model state with explicit JSON persistence; no external side effects."""

    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    oof_mean: float
    oof_std: float
    queries: list[dict]
    provenance: dict

    def predict_scores(self, logits, *, queries=None) -> np.ndarray:
        """Return bounded ranking signals in input row order, never eligibility decisions.

        Pass queries when scoring a new feature cache to enforce its column order.
        Omitting queries assumes columns match the frozen artifact's ordered queries.
        """
        if queries is not None and fingerprint(queries) != self.provenance["query_hash"]:
            raise ValueError("Prediction queries differ from the frozen ordered queries")
        values = _matrix(logits, len(self.queries))
        projections = _project(values, self.mean, self.scale, self.coefficients)
        with np.errstate(over="ignore"):
            standardized = (projections - self.oof_mean) / self.oof_std
        return .5 * (np.tanh(standardized / 2) + 1)

    def to_artifact(self) -> dict:
        payload = {
            "version": CONTRIBUTION_RANKER_VERSION, "queries": deepcopy(self.queries),
            "state": {"mean": self.mean.tolist(), "scale": self.scale.tolist(),
                      "coefficients": self.coefficients.tolist(),
                      "oof_mean": self.oof_mean, "oof_std": self.oof_std},
            "provenance": deepcopy(self.provenance),
        }
        return {**payload, "fingerprint": fingerprint(payload)}

    @classmethod
    def from_artifact(cls, artifact: dict) -> ContributionRanker:
        """Validate format, integrity, ordered feature identity and finite numeric state."""
        if not isinstance(artifact, dict) or artifact.get("version") != CONTRIBUTION_RANKER_VERSION:
            raise ValueError("Unsupported contribution ranker version")
        payload = {key: value for key, value in artifact.items() if key != "fingerprint"}
        if artifact.get("fingerprint") != fingerprint(payload):
            raise ValueError("Contribution artifact fingerprint mismatch")
        queries = deepcopy(artifact.get("queries"))
        _validate_queries(queries)
        provenance = deepcopy(artifact.get("provenance"))
        if not isinstance(provenance, dict) or provenance.get("query_hash") != fingerprint(queries):
            raise ValueError("Contribution artifact query provenance mismatch")
        if provenance.get("model_parameters") != MODEL_PARAMETERS or provenance.get("promotion_eligible") is not False:
            raise ValueError("Contribution artifact training policy mismatch")
        state = artifact.get("state")
        if not isinstance(state, dict):
            raise ValueError("Contribution artifact requires numeric state")
        try:
            mean, scale, coefficients = (
                np.asarray(state[key], dtype=np.float64) for key in ("mean", "scale", "coefficients")
            )
            oof_mean, oof_std = float(state["oof_mean"]), float(state["oof_std"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid contribution artifact numeric state") from exc
        if any(row.shape != (24,) or not np.isfinite(row).all() for row in (mean, scale, coefficients)):
            raise ValueError("Contribution artifact numeric state must have 24 finite columns")
        if np.any(scale <= 0) or not np.isfinite([oof_mean, oof_std]).all() or oof_std <= 0:
            raise ValueError("Contribution artifact scales must be finite and positive")
        return cls(mean, scale, coefficients, oof_mean, oof_std, queries, provenance)

    def save(self, path: str | Path) -> None:
        """Write a new explicit output; refuse to overwrite any existing artifact."""
        serialized = json.dumps(self.to_artifact(), sort_keys=True, allow_nan=False, indent=2) + "\n"
        with Path(path).open("x", encoding="utf-8") as handle:
            handle.write(serialized)

    @classmethod
    def load(cls, path: str | Path) -> ContributionRanker:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_artifact(json.load(handle))


def fit_contribution_ranker(logits, queries: list[dict], examples: list[dict]) -> ContributionRanker:
    """Fit predeclared pairwise ranking and its scale using calibration-only OOF.

    The first definitive row represents a family. Uncertain rows and subsequent
    same-tier versions never train scalers or models. Conflicting family tiers fail.
    Four stratified folds are family-disjoint after this representative selection.
    """
    queries = deepcopy(queries)
    _validate_queries(queries)
    values = _matrix(logits, len(queries))
    if not examples or len(examples) != len(values):
        raise ValueError("Aligned nonempty calibration examples are required")
    seen_ids, families, indices, uncertain, versions = set(), {}, [], [], []
    for index, example in enumerate(examples):
        if not isinstance(example, dict) or example.get("split") != "calibration":
            raise ValueError("Only calibration examples may train the contribution ranker")
        for key in ("id", "family_id"):
            if not isinstance(example.get(key), str) or not example[key].strip():
                raise ValueError(f"Every example requires a nonempty {key}")
        if example["id"] in seen_ids:
            raise ValueError("Calibration example IDs must be unique")
        seen_ids.add(example["id"])
        label, family = example.get("label"), example["family_id"]
        if label not in (*TIERS, "uncertain"):
            raise ValueError("Unsupported contribution tier")
        if label == "uncertain":
            uncertain.append(example["id"])
        elif family in families:
            if families[family] != label:
                raise ValueError("A paper family cannot have conflicting definitive tiers")
            versions.append(example["id"])
        else:
            families[family] = label
            indices.append(index)
    training = [{key: examples[index][key] for key in ("id", "family_id", "label")} for index in indices]
    labels = np.asarray([example["label"] for example in training])
    if any(np.sum(labels == tier) < FOLD_COUNT for tier in TIERS):
        raise ValueError("Each contribution tier requires at least 4 independent calibration families")
    values = values[indices]
    oof = np.full(len(training), np.nan)
    folds, counts = [], {f"{higher}_{lower}": {"pairs": 0, "correct": 0.0} for higher, lower in TIER_PAIRS}
    splitter = StratifiedKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=0)
    for fold, (train, validation) in enumerate(splitter.split(values, labels)):
        state, pair_summary = _fit(values[train], labels[train])
        projections = _project(values[validation], **state)
        oof[validation] = projections
        metrics = _pair_metrics(projections, labels[validation])
        for key, metric in metrics.items():
            counts[key]["pairs"] += metric["pairs"]
            counts[key]["correct"] += metric["correct"]
        folds.append({
            "fold": fold, "train_ids": [training[i]["id"] for i in train],
            "validation_ids": [training[i]["id"] for i in validation],
            "train_families": [training[i]["family_id"] for i in train],
            "validation_families": [training[i]["family_id"] for i in validation],
            "state": state, "training_pairs": pair_summary,
            "validation_projections": projections.tolist(),
        })
    if not np.isfinite(oof).all():
        raise ValueError("OOF contribution projections are incomplete or nonfinite")
    oof_mean, raw_std = float(oof.mean()), float(oof.std())
    degenerate = raw_std < 1e-12
    oof_std = 1.0 if degenerate else raw_std
    state, pair_summary = _fit(values, labels)
    per_pair = {key: {**value, "accuracy": value["correct"] / value["pairs"]} for key, value in counts.items()}
    provenance = {
        "model_parameters": deepcopy(MODEL_PARAMETERS), "sklearn_version": sklearn.__version__,
        "query_hash": fingerprint(queries), "training_ids": [example["id"] for example in training],
        "training_examples": training, "training_features_hash": fingerprint(values.tolist()),
        "excluded_uncertain_ids": uncertain, "excluded_family_version_ids": versions,
        "training_pairs": pair_summary, "oof_folds": folds,
        "oof_metrics": {"tier_pairs": per_pair,
                        "balanced_pair_accuracy": float(np.mean([value["accuracy"] for value in per_pair.values()])),
                        "comparison_scope": "different tiers within each held-out fold; ties count one half"},
        "scale_source": "four-fold family-disjoint OOF projection mean and population standard deviation",
        "degenerate_oof_scale": degenerate, "oof_raw_std": raw_std,
        "score_meaning": "sigmoid of OOF-standardized pairwise projection; ranking signals, not probabilities or usefulness",
        "promotion_eligible": False,
    }
    return ContributionRanker(*(np.asarray(state[key]) for key in ("mean", "scale", "coefficients")),
                              oof_mean, oof_std, queries, provenance)
