"""Offline ranking evaluation against accumulated thumbs feedback.

This is the measurement loop that turns ranker tuning from guesswork into A/B:
it replays the *persisted* per-digest ordering (the scores the ranker actually
assigned at send time) and scores it against the votes the user later cast.

Relevance labels come from votes: an up-vote (+1) is relevant, a down-vote (-1)
is non-relevant, and un-voted items are treated as non-relevant (standard for
implicit feedback). Because sparse feedback makes absolute nDCG hard to read, we
also report **pairwise accuracy** — over every (up-voted, down-voted) pair in a
digest, the fraction the ranker ordered correctly — which is robust to how many
items went un-judged.

Metrics are macro-averaged across digests (each digest counts once) so a single
heavily-voted day cannot dominate. To compare two ranker configurations, run
each over a span of digests (e.g. via ``backfill``) and diff the report.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from ..store import DigestItemFeatureRow, DigestItemRow, init_db, session_scope
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

    def as_dict(self) -> dict:
        return {
            "k": self.k,
            "digests_total": self.n_digests_total,
            "digests_scored": self.n_digests_scored,
            "votes_used": self.n_votes,
            "ndcg_at_k": _round(self.ndcg_at_k),
            "precision_at_k": _round(self.precision_at_k),
            "map": _round(self.map_score),
            "pairwise_accuracy": _round(self.pairwise_accuracy),
            "per_digest": [
                {
                    "digest_id": d.digest_id,
                    "n_items": d.n_items,
                    "n_up": d.n_up,
                    "n_down": d.n_down,
                    "ndcg_at_k": _round(d.ndcg_at_k),
                    "precision_at_k": _round(d.precision_at_k),
                    "average_precision": _round(d.average_precision),
                    "pairwise_accuracy": _round(d.pairwise_accuracy),
                }
                for d in self.per_digest
            ],
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


def _load_digest_orderings() -> dict[str, list[tuple[int, float]]]:
    """Return ``{digest_id: [(item_id, score), ...]}`` ordered by score desc.

    Prefers the richer feature table (persisted ``final_score``) and falls back
    to the digest_items table.
    """
    orderings: dict[str, list[tuple[int, float]]] = {}
    init_db()
    with session_scope() as s:
        rows = s.query(DigestItemFeatureRow).all()
        for r in rows:
            if r.item_id is None:
                continue
            score = r.final_score if r.final_score is not None else 0.0
            orderings.setdefault(r.digest_id, []).append((int(r.item_id), float(score)))
        if not orderings:
            drows = s.query(DigestItemRow).all()
            for r in drows:
                if r.item_id is None:
                    continue
                score = r.score if r.score is not None else 0.0
                orderings.setdefault(r.digest_id, []).append(
                    (int(r.item_id), float(score))
                )
    for digest_id in orderings:
        orderings[digest_id].sort(key=lambda t: t[1], reverse=True)
    return orderings


def evaluate_history(k: int = 10) -> EvalReport:
    """Replay persisted digests and score their order against later votes."""
    orderings = _load_digest_orderings()
    all_item_ids = [item_id for items in orderings.values() for item_id, _ in items]
    votes = _latest_vote_values(all_item_ids) if all_item_ids else {}

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
        if n_up == 0 and n_down == 0:
            continue  # no feedback on this digest — nothing to score against

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
        n_digests_scored=len(per_digest),
        n_votes=votes_used,
        ndcg_at_k=_mean(ndcgs),
        precision_at_k=_mean(precisions),
        map_score=_mean(aps),
        pairwise_accuracy=_mean(pairwise),
        per_digest=per_digest,
    )
