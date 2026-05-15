"""Vote ingestion + dataset assembly for the LR ranker.

Parses CLI vote strings like ``"+R3 R7 -I5 W1"``, resolves the labels
against the most recent (or specified) digest, and persists votes to the
``votes`` table. Also exposes :func:`vote_dataset` which materializes the
embedding-based training matrix used by :class:`rank.ranker.LRRanker`.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Iterable

import numpy as np
from sqlalchemy import select

from .rank.embedding_cache import embed_item_rows
from .rank.ranker import LRRanker, reset_lr_cache
from .store import DigestItemRow, DigestRow, ItemRow, VoteRow, init_db, session_scope

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"^([+-]?)([A-Za-z]+)(\d+)$")
ALLOWED_REASONS = {
    "off_topic",
    "low_impact",
    "promotional",
    "already_known",
    "too_technical",
    "not_urgent",
    "access_friction",
    "duplicate",
}
REASON_PENALTIES = {
    "off_topic": 0.12,
    "low_impact": 0.10,
    "promotional": 0.14,
    "already_known": 0.08,
    "too_technical": 0.04,
    "not_urgent": 0.07,
    "access_friction": 0.08,
    "duplicate": 0.10,
}

def _build_item_features(rows: list, profile_mat: np.ndarray) -> np.ndarray:
    """Build 9 engineered features per item for LR training/inference."""
    from .rank.embedding_cache import embed_item_rows
    from .rank.source_quality import (
        novelty_score as _nov,
        promotional_score as _promo,
        access_friction_score as _friction,
        infer_source_quality,
    )
    vecs = embed_item_rows(rows)
    if profile_mat.ndim == 1:
        cos = (vecs @ profile_mat.astype(np.float32, copy=False)).astype(np.float32)
    else:
        import math as _math
        sims = vecs @ profile_mat.T.astype(np.float32)
        n = sims.shape[1]
        k = max(1, min(5, int(round(_math.log2(n + 1)))))
        k = min(k, n)
        if k <= 1:
            cos = sims.max(axis=1).astype(np.float32)
        else:
            top_k = np.sort(sims, axis=1)[:, -k:]
            cos = top_k.mean(axis=1).astype(np.float32)
    features = []
    for i, row in enumerate(rows):
        src = str(getattr(row, "source", "") or "")
        sec = str(getattr(row, "section", "") or "")
        try:
            sq = infer_source_quality(src, sec)
            prestige = float(sq.prestige_score)
        except Exception:
            prestige = 0.5
        features.append([
            float(cos[i]),
            float(_nov(row)),
            float(_promo(row)),
            float(_friction(row)),
            prestige,
            float(sec == "research"),
            float(sec == "industry"),
            float(sec == "regulatory"),
            float(sec == "world"),
        ])
    return np.array(features, dtype=np.float32)


MIN_VOTES_FOR_LR = 30
_VOTE_REASONS_LOCK = Lock()


def _vote_reasons_path() -> Path:
    from .config import get_settings

    return Path(get_settings().db_path).parent / "vote_reasons.json"


def _load_vote_reasons() -> dict:
    path = _vote_reasons_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning("vote reasons unreadable, resetting: %s", e)
        return {}


def _write_vote_reasons(data: dict) -> None:
    path = _vote_reasons_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


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
            select(DigestItemRow.item_id).where(
                DigestItemRow.digest_id == digest_id,
                DigestItemRow.item_label == label,
            )
        ).scalar_one_or_none()
        if item_id is None:
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
            select(VoteRow.value)
            .where(VoteRow.item_id == item_id)
            .order_by(VoteRow.created_at.desc(), VoteRow.id.desc())
            .limit(1)
        ).scalar_one_or_none()


def record_vote_reason(item_id: int, reason: str) -> bool:
    """Persist an optional qualitative feedback reason for a voted item."""
    if reason not in ALLOWED_REASONS:
        return False
    init_db()
    with session_scope() as s:
        if s.get(ItemRow, item_id) is None:
            return False
    with _VOTE_REASONS_LOCK:
        data = _load_vote_reasons()
        key = str(int(item_id))
        reasons = list(data.get(key) or [])
        if reason not in reasons:
            reasons.append(reason)
        data[key] = reasons
        _write_vote_reasons(data)
    return True


def remove_vote_reason(item_id: int, reason: str) -> bool:
    """Remove one qualitative feedback reason for an item."""
    if reason not in ALLOWED_REASONS:
        return False
    init_db()
    with session_scope() as s:
        if s.get(ItemRow, item_id) is None:
            return False
    with _VOTE_REASONS_LOCK:
        data = _load_vote_reasons()
        key = str(int(item_id))
        reasons = [r for r in list(data.get(key) or []) if r != reason]
        if reasons:
            data[key] = reasons
        else:
            data.pop(key, None)
        _write_vote_reasons(data)
    return True


def get_vote_reasons(item_id: int) -> list[str]:
    with _VOTE_REASONS_LOCK:
        data = _load_vote_reasons()
    reasons = data.get(str(int(item_id))) or []
    return [r for r in reasons if r in ALLOWED_REASONS]


def reason_penalty_map() -> dict[str, float]:
    """Return item-id keyed penalties derived from qualitative reason chips."""
    with _VOTE_REASONS_LOCK:
        data = _load_vote_reasons()
    penalties: dict[str, float] = {}
    for item_id, reasons in data.items():
        total = 0.0
        for reason in list(reasons or []):
            total += REASON_PENALTIES.get(str(reason), 0.0)
        if total > 0:
            penalties[str(item_id)] = min(total, 0.30)
    return penalties


def vote_counts() -> dict[str, int]:
    """Return persisted vote counts split by Good, Bad, and Neutral.

    Older local databases may predate the UNIQUE(item_id) constraint and can
    contain multiple vote rows for one item. Treat the newest row as canonical
    so the UI and LR training agree with "latest vote wins" semantics.
    """
    init_db()
    counts = {"good": 0, "bad": 0, "neutral": 0, "signed": 0, "total": 0}
    with session_scope() as s:
        rows = s.execute(
            select(VoteRow.item_id, VoteRow.value)
            .order_by(VoteRow.item_id, VoteRow.created_at.desc(), VoteRow.id.desc())
        ).all()

    latest_by_item: dict[int, int] = {}
    for item_id, value in rows:
        iid = int(item_id)
        if iid in latest_by_item:
            continue
        latest_by_item[iid] = int(value)

    for value in latest_by_item.values():
        if int(value) == 1:
            counts["good"] += 1
        elif int(value) == -1:
            counts["bad"] += 1
        elif int(value) == 0:
            counts["neutral"] += 1

    counts["signed"] = counts["good"] + counts["bad"]
    counts["total"] = counts["signed"] + counts["neutral"]
    return counts


def signed_vote_count() -> int:
    """Count distinct items with a signed (+1/-1) latest vote."""
    init_db()
    try:
        with session_scope() as s:
            rows = s.execute(
                select(VoteRow.item_id, VoteRow.value)
                .where(VoteRow.value.in_((-1, 1)))
                .order_by(VoteRow.item_id, VoteRow.created_at.desc(), VoteRow.id.desc())
            ).all()
        seen: set[int] = set()
        count = 0
        for item_id, _value in rows:
            iid = int(item_id)
            if iid not in seen:
                seen.add(iid)
                count += 1
        return count
    except Exception as e:  # noqa: BLE001
        logger.warning("vote: failed to count votes: %s", e)
        return 0


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

    try:
        dataset = vote_dataset()
    except Exception as e:  # noqa: BLE001
        logger.exception("LR training dataset assembly failed")
        status = lr_training_status()
        return {
            "ok": False,
            "trained": False,
            "reason": "training_error",
            "message": f"Could not assemble training data: {type(e).__name__}: {e}",
            "status": status,
        }
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
    n_pos = int((y == 1).sum())
    n_neg = int((y == -1).sum())
    if n_pos < 10 or n_neg < 10:
        status = lr_training_status()
        return {
            "ok": False,
            "trained": False,
            "reason": "class_imbalance",
            "message": (
                f"Need at least 10 votes of each sign for reliable LR training "
                f"(have {n_pos} positive, {n_neg} negative). "
                f"Keep voting to balance the dataset."
            ),
            "status": status,
        }

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
    except Exception as e:  # noqa: BLE001
        logger.exception("LR training failed")
        status = lr_training_status()
        return {
            "ok": False,
            "trained": False,
            "reason": "training_error",
            "message": f"Could not train LR ranking: {type(e).__name__}: {e}",
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
    existing_rows = (
        s.execute(
            select(VoteRow)
            .where(VoteRow.item_id == item_id)
            .order_by(VoteRow.created_at.desc(), VoteRow.id.desc())
        )
        .scalars()
        .all()
    )
    existing = existing_rows[0] if existing_rows else None
    for duplicate in existing_rows[1:]:
        s.delete(duplicate)
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
    """Build (X, y) for LR training using engineered features.

    Returns None if fewer than MIN_VOTES_FOR_LR signed votes exist.
    """
    init_db()
    with session_scope() as s:
        raw_rows = s.execute(
            select(VoteRow.item_id, VoteRow.value, ItemRow)
            .join(ItemRow, VoteRow.item_id == ItemRow.id)
            .where(VoteRow.value.in_((-1, 1)))
            .order_by(VoteRow.item_id, VoteRow.created_at.desc(), VoteRow.id.desc())
        ).all()

        seen_ids: set[int] = set()
        voted_data: list[tuple] = []
        for item_id, value, row in raw_rows:
            iid = int(item_id)
            if iid in seen_ids:
                continue
            seen_ids.add(iid)
            voted_data.append((
                int(value),
                str(getattr(row, "title", "") or ""),
                str(getattr(row, "abstract", "") or ""),
                str(getattr(row, "source", "") or ""),
                str(getattr(row, "section", "") or ""),
            ))

    if len(voted_data) < MIN_VOTES_FOR_LR:
        return None

    from types import SimpleNamespace
    rows = [
        SimpleNamespace(title=t, abstract=a, source=src, section=sec)
        for _v, t, a, src, sec in voted_data
    ]
    ys = [1 if v > 0 else -1 for v, *_ in voted_data]

    try:
        from .rank.profile import build_profile_matrix
        from .config import load_settings
        import yaml
        from pathlib import Path
        settings = load_settings()
        profile_data = yaml.safe_load(Path(settings.profile_path).read_text(encoding="utf-8"))
        from .models import Profile
        profile = Profile(**profile_data)
        profile_mat = build_profile_matrix(profile)
    except Exception as e:  # noqa: BLE001
        logger.warning("vote_dataset: could not load profile: %s", e)
        profile_mat = np.zeros((1, 384), dtype=np.float32)

    X = _build_item_features(rows, profile_mat)
    y = np.asarray(ys, dtype=np.float32)
    return X, y
