"""Experimental local scope learning from frozen cross-encoder query features.

Only definitive calibration labels train this predeclared model. Thresholds use
family-disjoint out-of-fold predictions, never in-sample or evaluation labels.
Decision-function scores and query-relative facet gaps are not probabilities.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .pair_scope import calibrate_pair_scope, pair_evidence
from .scope import LABELS, fingerprint


LEARNED_SCOPE_VERSION = "learned-pair-scope-v1"
CLASS_LABELS = set(LABELS[:-1])
MODEL_PARAMETERS = dict(C=1, class_weight="balanced", max_iter=2000, random_state=0)
FOLD_COUNT = 4


def _matrix(logits, width: int) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != width or not np.isfinite(values).all():
        raise ValueError("Scope features must be a finite document-by-query matrix")
    return values


def _validate_queries(queries: list[dict]) -> None:
    if len(queries) != 24:
        raise ValueError("This predeclared model requires exactly 24 frozen queries")
    for query in queries:
        if query.get("label") not in CLASS_LABELS:
            raise ValueError("Query labels must be core, context or off_interest")
        if not isinstance(query.get("text"), str) or not query["text"].strip():
            raise ValueError("Every query requires nonempty text")
        facet = query.get("facet")
        if not isinstance(facet, str) or (not facet.strip() if query["label"] == "core" else bool(facet)):
            raise ValueError("Only core queries require a nonempty facet")
    if {query["label"] for query in queries} != CLASS_LABELS:
        raise ValueError("All three scope classes require query evidence")


def _evidence(values: np.ndarray, decisions: np.ndarray, classes, queries) -> list[dict]:
    result = pair_evidence(values, queries)
    for evidence, logits in zip(result, decisions, strict=True):
        scores = {str(label): float(value) for label, value in zip(classes, logits, strict=True)}
        strongest_core_query = max(evidence["facet_scores"].values())
        evidence["facet_scores"] = {
            facet: scores["core"] - (strongest_core_query - raw_score)
            for facet, raw_score in evidence["facet_scores"].items()
        }
        evidence["scores"] = scores
        evidence["score_meaning"] = (
            "multinomial logistic decision scores; facets retain raw query-relative gaps; "
            "not probabilities"
        )
    return result


def _fit(values, labels):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)
    classifier = LogisticRegression(**MODEL_PARAMETERS)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(scaled, labels)
    except ConvergenceWarning as exc:
        raise ValueError("The predeclared scope model failed to converge") from exc
    return scaler, classifier


@dataclass
class LearnedScopeModel:
    scaler: StandardScaler
    classifier: LogisticRegression
    queries: list[dict]
    provenance: dict

    def predict_evidence(self, logits) -> list[dict]:
        """Preserve row order and derive class evidence without label input."""
        values = _matrix(logits, len(self.queries))
        if not len(values):
            return []
        decisions = self.classifier.decision_function(self.scaler.transform(values))
        if not np.isfinite(decisions).all():
            raise ValueError("Scope model produced nonfinite decision scores")
        return _evidence(values, decisions, self.classifier.classes_, self.queries)


def fit_learned_scope(logits, queries: list[dict], examples: list[dict]) -> LearnedScopeModel:
    """Fit one fixed model and calibrate thresholds using grouped OOF evidence.

    Multiple versions may train the model within a single held-together family.
    Threshold calibration uses the first definitive representative per family,
    matching the existing calibrator's one-paper-one-observation contract.
    """
    queries = deepcopy(queries)
    _validate_queries(queries)
    values = _matrix(logits, len(queries))
    if not examples or len(examples) != len(values):
        raise ValueError("Aligned nonempty calibration examples are required")
    facets = {query["facet"] for query in queries if query["label"] == "core"}
    seen_ids, labels_by_family, training_indices, excluded = set(), {}, [], []
    for index, example in enumerate(examples):
        if example.get("split") != "calibration":
            raise ValueError("Only calibration examples may train learned scope")
        for key in ("id", "family_id"):
            if not isinstance(example.get(key), str) or not example[key].strip():
                raise ValueError(f"Every example requires a nonempty {key}")
        if example["id"] in seen_ids:
            raise ValueError("Calibration example IDs must be unique")
        seen_ids.add(example["id"])
        label, facet = example.get("label"), example.get("facet", "")
        if label not in LABELS:
            raise ValueError("Unsupported scope class")
        if not isinstance(example.get("abstract"), str):
            raise ValueError("Every example requires an abstract string")
        if (label == "core" and facet not in facets) or (label != "core" and facet):
            raise ValueError("Example facets must match their scope label and frozen queries")
        if label == "uncertain":
            excluded.append(example["id"])
            continue
        family = example["family_id"]
        if family in labels_by_family and labels_by_family[family] != label:
            raise ValueError("A paper family cannot have conflicting definitive scope labels")
        labels_by_family[family] = label
        training_indices.append(index)
    group_counts = {
        label: sum(value == label for value in labels_by_family.values())
        for label in CLASS_LABELS
    }
    if any(count < FOLD_COUNT for count in group_counts.values()):
        raise ValueError("Each of core, context and off_interest needs at least 4 calibration families")

    training = [deepcopy(examples[index]) for index in training_indices]
    train_values = values[training_indices]
    labels = np.asarray([example["label"] for example in training])
    groups = np.asarray([example["family_id"] for example in training])
    classes = np.unique(labels)
    oof_scores = np.full((len(training), len(classes)), np.nan)
    folds = []
    splitter = StratifiedGroupKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=0)
    for fold, (train, validation) in enumerate(splitter.split(train_values, labels, groups)):
        if set(labels[train]) != CLASS_LABELS:
            raise ValueError("Grouped fold lacks a training class; additional independent families are required")
        scaler, classifier = _fit(train_values[train], labels[train])
        if not np.array_equal(classifier.classes_, classes):
            raise ValueError("Grouped model class ordering is inconsistent")
        oof_scores[validation] = classifier.decision_function(scaler.transform(train_values[validation]))
        folds.append({
            "fold": fold,
            "train_ids": [training[index]["id"] for index in train],
            "validation_ids": [training[index]["id"] for index in validation],
            "train_families": sorted(set(groups[train])),
            "validation_families": sorted(set(groups[validation])),
        })
    if not np.isfinite(oof_scores).all():
        raise ValueError("Grouped OOF scores are incomplete or nonfinite")
    oof_evidence = _evidence(train_values, oof_scores, classes, queries)
    representatives, seen_families = [], set()
    for index, example in enumerate(training):
        if example["family_id"] not in seen_families:
            representatives.append(index)
            seen_families.add(example["family_id"])
    calibration = calibrate_pair_scope(
        [oof_evidence[index] for index in representatives],
        [training[index] for index in representatives],
    )
    scaler, classifier = _fit(train_values, labels)
    model_state = {
        "version": LEARNED_SCOPE_VERSION,
        "sklearn_version": sklearn.__version__,
        "query_hash": fingerprint(queries),
        "scaler_parameters": scaler.get_params(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "scaler_variance": scaler.var_.tolist(),
        "model_parameters": classifier.get_params(),
        "coefficients": classifier.coef_.tolist(),
        "intercepts": classifier.intercept_.tolist(),
        "classes": classifier.classes_.tolist(),
    }
    provenance = {
        "version": LEARNED_SCOPE_VERSION,
        "feature_ordering": deepcopy(queries),
        "query_hash": fingerprint(queries),
        "training_ids": [example["id"] for example in training],
        "training_signature": fingerprint(training),
        "excluded_uncertain_ids": excluded,
        "uncertain_policy": "Excluded from model fitting and threshold calibration; never negatives",
        "oof_folds": folds,
        "oof_evidence": [dict(id=example["id"], **evidence)
                         for example, evidence in zip(training, oof_evidence, strict=True)],
        "threshold_family_policy": "First definitive representative per family in input order",
        "thresholds": calibration["thresholds"],
        "calibration_report": calibration,
        "score_meaning": "Logistic decision scores with raw query-relative facet gaps; not probabilities",
        "model_state": model_state,
        "model_fingerprint": fingerprint(model_state),
        "promotion_eligible": False,
    }
    return LearnedScopeModel(scaler, classifier, queries, provenance)
