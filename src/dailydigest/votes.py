"""Vote ingestion + dataset assembly for the LR ranker.

Parses CLI vote strings like ``"+R3 R7 -I5 W1"``, resolves the labels
against the most recent (or specified) digest, and persists votes to the
``votes`` table. Also exposes :func:`vote_dataset` which materializes the
embedding-based training matrix used by :class:`rank.ranker.LRRanker`.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
from sqlalchemy import func, select

from .rank.embedding_cache import embed_item_rows
from .rank.ranker import LRRanker, reset_lr_cache
from .store import DigestRow, ItemRow, VoteRow, init_db, session_scope

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"^([+-]?)([A-Za-z]+)(\d+)$")

MIN_VOTES_FOR_LR = 30


def parse_vote_line(line: str) -> tuple[list[str], list[str]]:
    """Parse a vote line into (up_labels, down_labels).

    Accepts forms like ``"+R3 R7 -I5 W1"``, ``"+R3 +R7 -I5"``, ``"R3 R7"``.
    Tokens without an explicit sign default to ``+``. Tokens are case-folded
    to upper-case section prefixes (e.g. ``r3`` -> ``R3``). Malformed tokens
    are logged and skipped.
    """
    up: list[str] = []
    down: list[str] = []
    if not line:
        return up, down

    current_sign = "+"
    for raw in line.strip().split():
        tok = raw.strip()
        if not tok:
            continue

        # A bare sign (rare but tolerable) sets the default for what follows.
        if tok in {"+", "-"}:
            current_sign = tok
            continue

        m = _TOKEN_RE.match(tok)
        if not m:
            logger.warning("vote: skipping malformed token %r", tok)
            continue

        sign, prefix, number = m.groups()
        sign = sign or current_sign or "+"
        label = f"{prefix.upper()}{number}"
        if sign == "-":
            down.append(label)
        else:
            up.append(label)
        # Reset to '+' after each tagged token so a stray leading '-' does
        # not bleed into subsequent tokens.
        current_sign = "+"

    return up, down


def latest_digest_id() -> str | None:
    """Return the id of the most recently created digest, or ``None``."""
    init_db()
    with session_scope() as s:
        row = s.execute(
            select(DigestRow.id).order_by(DigestRow.created_at.desc()).limit(1)
        ).first()
        if row is None:
            return None
        return row[0]


def _resolve_labels(
    s, labels: Iterable[str], digest_id: str
) -> tuple[list[int], list[str]]:
    """Resolve labels to item ids within a digest. Returns (ids, missing)."""
    found_ids: list[int] = []
    missing: list[str] = []
    for label in labels:
        item_id = s.execute(
            select(ItemRow.id).where(
                ItemRow.digest_id == digest_id,
                ItemRow.item_label == label,
            )
        ).scalar_one_or_none()
        if item_id is None:
            missing.append(label)
        else:
            found_ids.append(int(item_id))
    return found_ids, missing


def record_votes(line: str, digest_id: str | None = None) -> dict[str, int]:
    """Persist votes parsed from ``line`` against the named (or latest) digest.

    Unknown labels are logged and skipped. Returns a count dict
    ``{"up": int, "down": int, "unknown": int}``.
    """
    init_db()
    up_labels, down_labels = parse_vote_line(line)
    counts = {"up": 0, "down": 0, "unknown": 0}

    target_digest = digest_id or latest_digest_id()
    if target_digest is None:
        logger.warning("vote: no digest available to resolve labels against")
        counts["unknown"] = len(up_labels) + len(down_labels)
        return counts

    with session_scope() as s:
        up_ids, up_missing = _resolve_labels(s, up_labels, target_digest)
        down_ids, down_missing = _resolve_labels(s, down_labels, target_digest)

        for label in up_missing + down_missing:
            logger.warning(
                "vote: unknown label %r in digest %s", label, target_digest
            )

        for item_id in up_ids:
            _upsert_vote(s, item_id, 1)
        for item_id in down_ids:
            _upsert_vote(s, item_id, -1)

        counts["up"] = len(up_ids)
        counts["down"] = len(down_ids)
        counts["unknown"] = len(up_missing) + len(down_missing)

    return counts


def record_vote_by_id(item_id: int, value: int) -> bool:
    """Persist a single vote keyed directly by item id (label-less path).

    Used by the local web UI where each item already has its DB id and we
    don't need to resolve a label like ``R3``. ``value`` semantics:

    - ``+1`` / ``-1``: upsert the vote via :func:`_upsert_vote` (logs flips).
    - ``0``: neutral — persist a reviewed-but-neutral signal for UI feedback.

    Returns True on success, False if ``item_id`` does not exist.
    """
    if value not in (-1, 0, 1):
        logger.warning("vote: rejecting out-of-range value %r for item %d", value, item_id)
        return False

    init_db()
    with session_scope() as s:
        if s.get(ItemRow, item_id) is None:
            return False

        _upsert_vote(s, item_id, value)
        return True


def get_vote_value(item_id: int) -> int | None:
    """Return the stored vote value (+1/0/-1) for ``item_id``, or None."""
    init_db()
    with session_scope() as s:
        return s.execute(
            select(VoteRow.value).where(VoteRow.item_id == item_id)
        ).scalar_one_or_none()


def vote_counts() -> dict[str, int]:
    """Return persisted vote counts split by Good, Bad, and Neutral."""
    init_db()
    counts = {"good": 0, "bad": 0, "neutral": 0, "signed": 0, "total": 0}
    with session_scope() as s:
        rows = s.execute(
            select(VoteRow.value, func.count()).group_by(VoteRow.value)
        ).all()

    for value, n in rows:
        count = int(n or 0)
        if int(value) == 1:
            counts["good"] = count
        elif int(value) == -1:
            counts["bad"] = count
        elif int(value) == 0:
            counts["neutral"] = count

    counts["signed"] = counts["good"] + counts["bad"]
    counts["total"] = counts["signed"] + counts["neutral"]
    return counts


def lr_training_status() -> dict[str, object]:
    """Return lightweight LR training/ranking status for web/API display."""
    counts = vote_counts()
    signed = counts["signed"]
    remaining = max(MIN_VOTES_FOR_LR - signed, 0)
    model_trained = LRRanker().load()
    can_train = remaining == 0
    ranking_status = "lr_active" if model_trained and can_train else "cosine_baseline"
    training_status = "ready" if can_train else "needs_votes"
    return {
        "vote_counts": counts,
        "min_votes_for_lr": MIN_VOTES_FOR_LR,
        "remaining_votes_for_lr": remaining,
        "can_train": can_train,
        "model_trained": model_trained,
        "training_status": training_status,
        "ranking_status": ranking_status,
    }


def train_lr_ranker() -> dict[str, object]:
    """Train the persisted LR ranker from signed votes when enough exist."""
    status = lr_training_status()
    if not status["can_train"]:
        return {
            "ok": False,
            "trained": False,
            "reason": "needs_votes",
            "message": (
                f"Need {status['remaining_votes_for_lr']} more signed votes "
                f"before LR training."
            ),
            "status": status,
        }

    dataset = vote_dataset()
    if dataset is None:
        status = lr_training_status()
        return {
            "ok": False,
            "trained": False,
            "reason": "needs_votes",
            "message": "Not enough signed votes to train LR ranking.",
            "status": status,
        }

    X, y = dataset
    ranker = LRRanker()
    try:
        ranker.fit(X, y)
    except ValueError as e:
        status = lr_training_status()
        return {
            "ok": False,
            "trained": False,
            "reason": "invalid_dataset",
            "message": str(e),
            "status": status,
        }

    reset_lr_cache()
    status = lr_training_status()
    return {
        "ok": True,
        "trained": True,
        "trained_votes": int(len(y)),
        "status": status,
    }


def _upsert_vote(s, item_id: int, value: int) -> None:
    """Insert or update the single vote row for ``item_id``.

    Enforces "latest vote wins" semantics paired with the UNIQUE
    constraint on ``votes.item_id``. When a prior vote of the opposite
    sign exists, logs an INFO line so flip patterns are visible in the
    feedback loop.
    """
    existing = s.execute(
        select(VoteRow).where(VoteRow.item_id == item_id)
    ).scalar_one_or_none()
    now = datetime.now(timezone.utc)
    if existing is None:
        s.add(VoteRow(item_id=item_id, value=value, created_at=now))
        return
    if existing.value != value:
        logger.info(
            "flipped vote on item %d: %+d -> %+d",
            item_id,
            existing.value,
            value,
        )
    existing.value = value
    existing.created_at = now


def vote_dataset() -> tuple[np.ndarray, np.ndarray] | None:
    """Build (X, y) for LR training from all stored votes.

    X: float32 embedding matrix of ``title + ". " + abstract`` for each
    voted item. y: ``+1`` / ``-1`` labels. Returns ``None`` if fewer than
    :data:`MIN_VOTES_FOR_LR` votes exist.
    """
    init_db()
    with session_scope() as s:
        rows = s.execute(
            select(VoteRow.value, ItemRow)
            .join(ItemRow, VoteRow.item_id == ItemRow.id)
            .where(VoteRow.value.in_((-1, 1)))
        ).all()
        for _value, row in rows:
            s.expunge(row)

    if len(rows) < MIN_VOTES_FOR_LR:
        return None

    voted_rows: list[ItemRow] = []
    ys: list[int] = []
    for value, row in rows:
        voted_rows.append(row)
        ys.append(1 if int(value) >= 0 else -1)

    X = embed_item_rows(voted_rows).astype(np.float32, copy=False)
    y = np.asarray(ys, dtype=np.float32)
    return X, y
