"""Section-specific displayed-order diagnostics using current latest judgments.

Unjudged items are unknown: observed precision is a lower bound, accompanied by
an upper bound and judged coverage. nDCG/AP are conditional on observed positives
in the displayed pool, never estimates of retrieval recall. Same-day rebrews
replace this legacy view; immutable captured runs use ``rank.replay`` instead.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass, field

from ..store import DigestItemRow, VoteRow, init_db, session_scope
from ..votes import _latest_vote_values

logger = logging.getLogger(__name__)


@dataclass
class DigestEval:
    digest_id: str
    n_items: int
    n_up: int
    n_down: int
    ndcg_at_k: float | None
    precision_at_k: float | None
    average_precision: float | None
    pairwise_accuracy: float | None
    n_unknown: int = 0
    judged_coverage_at_k: float | None = None
    precision_upper_at_k: float | None = None
    graded_utility_at_k: float = 0.0


@dataclass
class EvalReport:
    k: int
    n_digests_total: int
    n_digests_scored: int
    n_votes: int
    ndcg_at_k: float | None
    precision_at_k: float | None
    map_score: float | None
    pairwise_accuracy: float | None
    per_digest: list[DigestEval] = field(default_factory=list)

    section: str = "research"
    unknown_items: int = 0
    judged_coverage_at_k: float | None = None
    precision_upper_at_k: float | None = None
    graded_utility_at_k: float | None = None
    zero_positive_digests: int = 0
    ndcg_defined_digests: int = 0

    def as_dict(self) -> dict:
        return {
            "k": self.k,
            "section": self.section,
            "label_basis": "current latest signed votes; not historical as-of judgments",
            "precision_basis": "macro across all displayed section-days; observed positive lower bound, unknowns may be positive",
            "ndcg_basis": "observed positives in displayed section; excludes zero-positive digests",
            "graded_utility_basis": "observed gain sum at K: must-read=3, relevant=1, others=0",
            "unknown_items": self.unknown_items,
            "judged_coverage_at_k": _round(self.judged_coverage_at_k),
            "precision_upper_at_k": _round(self.precision_upper_at_k),
            "graded_utility_at_k": _round(self.graded_utility_at_k),
            "zero_positive_digests": self.zero_positive_digests,
            "ndcg_defined_digests": self.ndcg_defined_digests,
            "digests_total": self.n_digests_total,
            "digests_scored": self.n_digests_scored,
            "votes_used": self.n_votes,
            "ndcg_at_k": _round(self.ndcg_at_k),
            "precision_at_k": _round(self.precision_at_k),
            "map": _round(self.map_score),
            "pairwise_accuracy": _round(self.pairwise_accuracy),
            "per_digest": [asdict(d) for d in self.per_digest],
        }


def _round(value: float | None, ndigits: int = 4) -> float | None:
    return round(float(value), ndigits) if value is not None else None


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _ndcg_at_k(labels_in_rank_order: list[int], k: int) -> float | None:
    """nDCG@k where label is 1 for relevant (up-vote), else 0."""
    if not labels_in_rank_order:
        return None
    gains = [1.0 if lbl > 0 else 0.0 for lbl in labels_in_rank_order[:k]]
    ideal = sorted(
        (1.0 if lbl > 0 else 0.0 for lbl in labels_in_rank_order), reverse=True
    )[:k]
    idcg = _dcg(ideal)
    if idcg <= 0:
        return None  # no relevant items in this digest — nDCG undefined
    return _dcg(gains) / idcg


def _precision_at_k(labels_in_rank_order: list[int], k: int) -> float | None:
    if not labels_in_rank_order:
        return None
    top = labels_in_rank_order[: min(k, len(labels_in_rank_order))]
    if not top:
        return None
    return sum(1 for lbl in top if lbl > 0) / len(top)


def _average_precision(labels_in_rank_order: list[int]) -> float | None:
    n_rel = sum(1 for lbl in labels_in_rank_order if lbl > 0)
    if n_rel == 0:
        return None
    hits = 0
    score = 0.0
    for i, lbl in enumerate(labels_in_rank_order):
        if lbl > 0:
            hits += 1
            score += hits / (i + 1)
    return score / n_rel


def _pairwise_accuracy(ranked: list[tuple[int, int]]) -> float | None:
    """Fraction of (up, down) pairs the ranking ordered correctly.

    ``ranked`` is a list of ``(label, rank_index)`` already in score order
    (rank_index ascending = better). Only +1 / -1 labels participate.
    """
    ups = [idx for idx, (lbl, _) in enumerate(ranked) if lbl > 0]
    downs = [idx for idx, (lbl, _) in enumerate(ranked) if lbl < 0]
    if not ups or not downs:
        return None
    correct = 0
    total = 0
    for u in ups:
        for d in downs:
            total += 1
            if u < d:  # up-voted item appears earlier (better) than down-voted
                correct += 1
    return correct / total if total else None


def _load_digest_orderings(section: str = "research") -> dict[str, list[tuple[int, int]]]:
    """Load displayed section membership in numeric label order (R2 before R10)."""
    # Labels persist the section that was displayed, even if current item
    # metadata is subsequently reclassified. These are the persisted label ABI.
    prefixes = {"research": "R", "industry": "I", "ai": "A", "regulatory": "G",
                "world": "W", "opportunities": "F", "events": "E"}
    if section not in prefixes:
        raise ValueError(f"unknown section: {section}")
    pattern = re.compile(re.escape(prefixes[section]) + r"(\d+)$")
    orderings: dict[str, list[tuple[int, int]]] = {}
    init_db()
    with session_scope() as s:
        rows = s.query(DigestItemRow).all()
        for row in rows:
            match = pattern.fullmatch(row.item_label or "")
            if match is None:
                continue
            position = int(match.group(1))
            orderings.setdefault(row.digest_id, []).append((int(row.item_id), position))
    for items in orderings.values():
        items.sort(key=lambda item: (item[1], item[0]))
    return orderings


def evaluate_history(k: int = 10, section: str = "research") -> EvalReport:
    """Evaluate one section's displayed order against current latest signed votes."""
    if k <= 0:
        raise ValueError("k must be positive")
    orderings = _load_digest_orderings(section)
    all_item_ids = [item_id for items in orderings.values() for item_id, _ in items]
    votes = _latest_vote_values(all_item_ids) if all_item_ids else {}

    with session_scope() as s:
        grades = {int(row.item_id): row.grade for row in s.query(VoteRow).all()}
    per_digest: list[DigestEval] = []
    ndcgs: list[float] = []
    precisions: list[float] = []
    aps: list[float] = []
    pairwise: list[float] = []
    votes_used = 0

    for digest_id, items in sorted(orderings.items()):
        labels = [int(votes.get(item_id, 0)) for item_id, _ in items]
        n_up = sum(1 for lbl in labels if lbl > 0)
        n_down = sum(1 for lbl in labels if lbl < 0)
        votes_used += n_up + n_down
        top = labels[:k]
        unknown = sum(label == 0 for label in top)
        coverage = sum(label != 0 for label in top) / len(top)
        upper = (sum(label > 0 for label in top) + unknown) / len(top)
        utility = sum(3.0 if (grades.get(item_id) or 0) >= 100 else 1.0
                      for item_id, _ in items[:k] if votes.get(item_id, 0) > 0)
        ndcg = _ndcg_at_k(labels, k)
        prec = _precision_at_k(labels, k)
        ap = _average_precision(labels)
        pw = _pairwise_accuracy([(lbl, i) for i, lbl in enumerate(labels)])

        per_digest.append(
            DigestEval(
                digest_id=digest_id,
                n_items=len(items),
                n_up=n_up,
                n_down=n_down,
                ndcg_at_k=ndcg,
                precision_at_k=prec,
                average_precision=ap,
                pairwise_accuracy=pw,
                n_unknown=len(labels) - n_up - n_down,
                judged_coverage_at_k=coverage,
                precision_upper_at_k=upper,
                graded_utility_at_k=utility,
            )
        )
        if ndcg is not None:
            ndcgs.append(ndcg)
        if prec is not None:
            precisions.append(prec)
        if ap is not None:
            aps.append(ap)
        if pw is not None:
            pairwise.append(pw)

    def _mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    return EvalReport(
        k=k,
        n_digests_total=len(orderings),
        n_digests_scored=sum(d.n_up + d.n_down > 0 for d in per_digest),
        n_votes=votes_used,
        ndcg_at_k=_mean(ndcgs),
        precision_at_k=_mean(precisions),
        map_score=_mean(aps),
        pairwise_accuracy=_mean(pairwise),
        per_digest=per_digest,
        section=section,
        unknown_items=sum(votes.get(item_id, 0) == 0 for item_id in all_item_ids),
        judged_coverage_at_k=_mean([d.judged_coverage_at_k for d in per_digest]),
        precision_upper_at_k=_mean([d.precision_upper_at_k for d in per_digest]),
        graded_utility_at_k=_mean([d.graded_utility_at_k for d in per_digest]),
        zero_positive_digests=sum(d.n_up == 0 and d.n_down > 0 for d in per_digest),
        ndcg_defined_digests=len(ndcgs),
    )
