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
_GENERALIZED_REASON_HALF_LIFE_DAYS = 60.0
_GOOD_VOTE_COUNTER_SIGNAL = 0.14
_SOURCE_GENERALIZATION_CAP = 0.12
_BUCKET_GENERALIZATION_CAP = 0.06
_CONTENT_GENERALIZATION_CAP = 0.045

LR_FEATURE_SCHEMA_VERSION = "lr_ranker_engineered_features_v3"
LR_FEATURE_NAMES = (
    "cosine_similarity",
    "novelty_score",
    "promotional_score",
    "access_friction_score",
    "prestige_score",
    "age_norm",
    "source_bucket_score",
    "cosine_x_bucket_score",
    "cosine_x_prestige",
    "cosine_x_freshness",
)
LR_FEATURE_DIM = len(LR_FEATURE_NAMES)
LR_DEFAULT_SOURCE_BUCKET_SCORE = 0.6
LR_SOURCE_BUCKET_SCORES = {
    "published_journal": 1.0,
    "published_database": 0.8,
    "other_research": 0.65,
    "aggregator": 0.5,
    "bio_med_preprint": 0.4,
    "arxiv_other": 0.35,
    "preprint_other": 0.3,
    "arxiv_cs": 0.25,
}


def _as_utc(value: object) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _first_timestamp(*values: object) -> datetime | None:
    for value in values:
        timestamp = _as_utc(value)
        if timestamp is not None:
            return timestamp
    return None


def _recency_decay(timestamp: datetime | None, now: datetime) -> float:
    if timestamp is None:
        return 1.0
    age_days = max(0.0, (now - timestamp).total_seconds() / 86400)
    return 0.5 ** (age_days / _GENERALIZED_REASON_HALF_LIFE_DAYS)


def _add_capped_signal(target: dict[str, float], key: str, delta: float, cap: float) -> None:
    if not key:
        return
    target[key] = max(0.0, min(cap, target.get(key, 0.0) + delta))


def _build_item_features(rows: list, profile_mat: np.ndarray) -> np.ndarray:
    """Build the current engineered features per item for LR training/inference.

    Feature vector (``LR_FEATURE_SCHEMA_VERSION`` / ``LR_FEATURE_DIM`` dims):
    0. cosine similarity to profile (top-k-mean or max)
    1. novelty score
    2. promotional score
    3. access friction score
    4. prestige score
    5. age_norm (days since published, normalized: 0=today, 1=14+ days; 0.5 if unknown)
    6. source_bucket_score (numerical: published_journal=1.0, published_database=0.8,
                             aggregator=0.5, bio_med_preprint=0.4, arxiv_other=0.35,
                             preprint_other=0.3, arxiv_cs=0.25, default=0.6)
    7. cosine × bucket_score  (interaction: high-cosine journal papers preferred)
    8. cosine × prestige       (interaction: high-cosine prestigious papers preferred)
    9. cosine × freshness      (interaction: high-cosine fresh papers preferred;
                                freshness = 1.0 - age_norm)
    """
    from .rank.embedding_cache import embed_item_rows
    from .rank.source_quality import (
        novelty_score as _nov,
        promotional_score as _promo,
        access_friction_score as _friction,
        infer_source_quality,
        source_bucket,
    )
    from datetime import datetime, timezone

    vecs = embed_item_rows(rows)
    if profile_mat.ndim == 1:
        cos = (vecs @ profile_mat.astype(np.float32, copy=False)).astype(np.float32)
    else:
        import math as _math
        sims = vecs @ profile_mat.T.astype(np.float32)
        n = sims.shape[1]
        if n <= 1:
            cos = sims.max(axis=1).astype(np.float32)
        else:
            top1 = sims.max(axis=1)
            k3 = min(3, n)
            top3 = np.sort(sims, axis=1)[:, -k3:].mean(axis=1)
            cos = (0.7 * top1 + 0.3 * top3).astype(np.float32)

    now = datetime.now(timezone.utc)
    features = []
    for i, row in enumerate(rows):
        src = str(getattr(row, "source", "") or "")
        sec = str(getattr(row, "section", "") or "")
        try:
            sq = infer_source_quality(src, sec)
            prestige = float(sq.prestige_score)
        except Exception:
            prestige = 0.5

        # Age normalization (0=today, 1=14+ days old, 0.5=unknown)
        published = getattr(row, "published_at", None)
        if isinstance(published, datetime):
            ref = published if published.tzinfo else published.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (now - ref).total_seconds() / 86400)
            age_norm = min(1.0, age_days / 14.0)
        else:
            age_norm = 0.5

        bucket = source_bucket(row)
        bucket_score = LR_SOURCE_BUCKET_SCORES.get(bucket, LR_DEFAULT_SOURCE_BUCKET_SCORE)

        cos_val = float(cos[i])
        features.append([
            cos_val,                                    # 0. cosine
            float(_nov(row)),                           # 1. novelty
            float(_promo(row)),                         # 2. promotional
            float(_friction(row)),                      # 3. access friction
            prestige,                                   # 4. prestige
            age_norm,                                   # 5. age_norm
            bucket_score,                               # 6. bucket_score
            cos_val * bucket_score,                     # 7. cosine × bucket_score
            cos_val * prestige,                         # 8. cosine × prestige
            cos_val * (1.0 - age_norm),                 # 9. cosine × freshness
        ])
    if not features:
        return np.zeros((0, LR_FEATURE_DIM), dtype=np.float32)
    return np.asarray(features, dtype=np.float32)


def _learned_profile_path() -> Path:
    from .config import get_settings
    return Path(get_settings().db_path).parent / "learned_profile.npz"


def _update_rocchio(item_id: int, vote_value: int) -> None:
    """Update the Rocchio-style learned profile vector after each vote.

    Uses alpha=0.08 for upvotes and beta=0.04 for downvotes to accumulate
    a learned direction in embedding space. Called outside DB session.
    """
    if vote_value not in (1, -1):
        return
    alpha, beta = 0.08, 0.04
    try:
        from .store import ItemRow, session_scope
        with session_scope() as s:
            item = s.get(ItemRow, item_id)
            if item is None:
                return
            s.expunge(item)

        from .rank.embedding_cache import embed_item_rows
        vecs = embed_item_rows([item])
        if vecs.size == 0 or vecs.shape[0] == 0:
            return
        vec = vecs[0].astype(np.float32)

        path = _learned_profile_path()
        if path.exists():
            try:
                data = np.load(path)
                learned = data["profile"].astype(np.float32)
                stored_count = int(data["vote_count"][0])
            except Exception:
                learned = np.zeros(vec.shape, dtype=np.float32)
                stored_count = 0
        else:
            learned = np.zeros(vec.shape, dtype=np.float32)
            stored_count = 0

        if vote_value == 1:
            learned = learned + alpha * vec
        else:
            learned = learned - beta * vec

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".npz.tmp")
        np.savez(tmp, profile=learned, vote_count=np.array([stored_count + 1], dtype=np.int32))
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Rocchio update failed: %s", exc)


MIN_VOTES_FOR_LR = 20
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

    # Update Rocchio learned profile outside DB session
    for item_id in up_ids:
        _update_rocchio(item_id, 1)
        _clear_vote_reasons(item_id)
    for item_id in down_ids:
        _update_rocchio(item_id, -1)

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

    if value in (1, -1):
        _update_rocchio(item_id, value)
    if value == 1:
        _clear_vote_reasons(item_id)
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


def _latest_vote_values(item_ids: Iterable[int] | None = None) -> dict[int, int]:
    """Return latest vote values keyed by item id.

    Older local databases can contain duplicate vote rows from before the
    UNIQUE(item_id) migration. Always order before deduping so neutral/latest
    votes are not hidden by older signed rows.
    """
    ids = [] if item_ids is None else [int(item_id) for item_id in item_ids]
    if item_ids is not None and not ids:
        return {}
    init_db()
    with session_scope() as s:
        stmt = select(VoteRow.item_id, VoteRow.value).order_by(
            VoteRow.item_id,
            VoteRow.created_at.desc(),
            VoteRow.id.desc(),
        )
        if ids:
            stmt = stmt.where(VoteRow.item_id.in_(ids))
        rows = s.execute(stmt).all()

    latest: dict[int, int] = {}
    for item_id, value in rows:
        iid = int(item_id)
        if iid in latest:
            continue
        latest[iid] = int(value)
    return latest


def _clear_vote_reasons(item_id: int) -> None:
    """Remove stale reason chips for an item that is now marked Relevant."""
    with _VOTE_REASONS_LOCK:
        data = _load_vote_reasons()
        if data.pop(str(int(item_id)), None) is not None:
            _write_vote_reasons(data)


def record_vote_reason(item_id: int, reason: str) -> bool:
    """Persist an optional qualitative feedback reason for a non-positive vote."""
    if reason not in ALLOWED_REASONS:
        return False
    init_db()
    with session_scope() as s:
        if s.get(ItemRow, item_id) is None:
            return False
    if _latest_vote_values([item_id]).get(int(item_id)) not in (-1, 0):
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


def reason_penalty_map(rows: Iterable[object] | None = None) -> dict[str, float]:
    """Return penalties derived from qualitative reason chips.

    Without ``rows`` this returns exact item-id penalties for compatibility.
    With current ranking candidates, it also applies soft generalized
    penalties by source, source bucket, and content type so feedback can affect
    future articles from similar feeds.
    """
    with _VOTE_REASONS_LOCK:
        data = _load_vote_reasons()
    reasoned_ids: list[int] = []
    for key in data:
        try:
            reasoned_ids.append(int(key))
        except (TypeError, ValueError):
            continue
    latest_votes = _latest_vote_values(reasoned_ids) if reasoned_ids else {}
    valid_data = {
        str(item_id): data.get(str(item_id)) or []
        for item_id in reasoned_ids
        if latest_votes.get(item_id) in (-1, 0)
    }
    penalties: dict[str, float] = {}
    for item_id, reasons in valid_data.items():
        total = 0.0
        for reason in list(reasons or []):
            total += REASON_PENALTIES.get(str(reason), 0.0)
        if total > 0:
            penalties[str(item_id)] = min(total, 0.30)
    if rows is None or not valid_data:
        return penalties

    try:
        from .rank.source_quality import content_type, source_bucket
    except Exception as e:  # noqa: BLE001
        logger.warning("vote: could not load source-quality helpers: %s", e)
        return penalties

    init_db()
    with session_scope() as s:
        raw_reasoned_rows = s.execute(
            select(ItemRow, VoteRow.created_at)
            .outerjoin(VoteRow, VoteRow.item_id == ItemRow.id)
            .where(ItemRow.id.in_([int(item_id) for item_id in valid_data]))
            .order_by(ItemRow.id, VoteRow.created_at.desc(), VoteRow.id.desc())
        ).all()
        voted_rows: list[tuple[ItemRow, datetime | None]] = []
        seen_reasoned: set[int] = set()
        for row, vote_created_at in raw_reasoned_rows:
            row_id = int(row.id)
            if row_id in seen_reasoned:
                continue
            seen_reasoned.add(row_id)
            s.expunge(row)
            timestamp = _first_timestamp(
                vote_created_at,
                getattr(row, "fetched_at", None),
                getattr(row, "published_at", None),
            )
            voted_rows.append((row, timestamp))

        raw_vote_rows = s.execute(
            select(VoteRow.item_id, VoteRow.value, VoteRow.created_at, ItemRow)
            .join(ItemRow, VoteRow.item_id == ItemRow.id)
            .order_by(VoteRow.item_id, VoteRow.created_at.desc(), VoteRow.id.desc())
        ).all()
        good_rows: list[tuple[ItemRow, datetime | None]] = []
        seen_votes: set[int] = set()
        for item_id, value, vote_created_at, row in raw_vote_rows:
            row_id = int(item_id)
            if row_id in seen_votes:
                continue
            seen_votes.add(row_id)
            if int(value) != 1:
                continue
            s.expunge(row)
            timestamp = _first_timestamp(
                vote_created_at,
                getattr(row, "fetched_at", None),
                getattr(row, "published_at", None),
            )
            good_rows.append((row, timestamp))

    source_penalties: dict[str, float] = {}
    bucket_penalties: dict[str, float] = {}
    content_penalties: dict[str, float] = {}
    generalizable = {
        "off_topic",
        "low_impact",
        "promotional",
        "already_known",
        "not_urgent",
        "access_friction",
        "duplicate",
    }
    now = datetime.now(timezone.utc)
    for row, timestamp in voted_rows:
        total = 0.0
        for reason in list(valid_data.get(str(int(row.id))) or []):
            if str(reason) in generalizable:
                total += REASON_PENALTIES.get(str(reason), 0.0)
        if total <= 0:
            continue
        signal = total * _recency_decay(timestamp, now)
        source = str(row.source or "").strip().lower()
        if source:
            _add_capped_signal(
                source_penalties,
                source,
                signal * 0.35,
                _SOURCE_GENERALIZATION_CAP,
            )
        bucket = source_bucket(row)
        _add_capped_signal(
            bucket_penalties,
            bucket,
            signal * 0.20,
            _BUCKET_GENERALIZATION_CAP,
        )
        ctype = content_type(row)
        _add_capped_signal(
            content_penalties,
            ctype,
            signal * 0.15,
            _CONTENT_GENERALIZATION_CAP,
        )

    for row, timestamp in good_rows:
        signal = _GOOD_VOTE_COUNTER_SIGNAL * _recency_decay(timestamp, now)
        source = str(row.source or "").strip().lower()
        if source:
            _add_capped_signal(
                source_penalties,
                source,
                -(signal * 0.35),
                _SOURCE_GENERALIZATION_CAP,
            )
        _add_capped_signal(
            bucket_penalties,
            source_bucket(row),
            -(signal * 0.20),
            _BUCKET_GENERALIZATION_CAP,
        )
        _add_capped_signal(
            content_penalties,
            content_type(row),
            -(signal * 0.15),
            _CONTENT_GENERALIZATION_CAP,
        )

    for row in rows:
        row_id = getattr(row, "id", None)
        keys: list[str] = []
        if isinstance(row_id, int):
            keys.append(str(row_id))
        external = getattr(row, "external_id", None)
        if isinstance(external, str) and external:
            keys.append(external)
        url = getattr(row, "url", None)
        if isinstance(url, str) and url:
            keys.append(url)
        if not keys:
            continue
        exact_key = str(row_id) if isinstance(row_id, int) else None
        if exact_key is not None and exact_key in penalties:
            continue
        source = str(getattr(row, "source", "") or "").strip().lower()
        generalized = 0.0
        if source:
            generalized += source_penalties.get(source, 0.0)
        generalized += bucket_penalties.get(source_bucket(row), 0.0)
        generalized += content_penalties.get(content_type(row), 0.0)
        if generalized <= 0:
            continue
        value = min(0.30, generalized)
        for key in keys:
            penalties[key] = min(0.30, penalties.get(key, 0.0) + value)
    return penalties


def vote_counts() -> dict[str, int]:
    """Return persisted vote counts split by Good, Bad, and Neutral.

    Older local databases may predate the UNIQUE(item_id) constraint and can
    contain multiple vote rows for one item. Treat the newest row as canonical
    so the UI and LR training agree with "latest vote wins" semantics.
    """
    init_db()
    counts = {"good": 0, "bad": 0, "neutral": 0, "signed": 0, "total": 0}
    for value in _latest_vote_values().values():
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
        return sum(1 for value in _latest_vote_values().values() if value in (-1, 1))
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
    if n_pos < 3 or n_neg < 3:
        status = lr_training_status()
        return {
            "ok": False,
            "trained": False,
            "reason": "class_imbalance",
            "message": (
                f"Need at least 3 votes of each sign for reliable LR training "
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
    Uses real ItemRow objects (not SimpleNamespace) so embeddings are identical
    to what inference uses — preventing train/infer feature distribution mismatch.
    """
    init_db()
    with session_scope() as s:
        raw_rows = s.execute(
            select(VoteRow.item_id, VoteRow.value, ItemRow)
            .join(ItemRow, VoteRow.item_id == ItemRow.id)
            .order_by(VoteRow.item_id, VoteRow.created_at.desc(), VoteRow.id.desc())
        ).all()

        seen_ids: set[int] = set()
        rows: list[ItemRow] = []
        ys: list[int] = []
        for item_id, value, row in raw_rows:
            iid = int(item_id)
            if iid in seen_ids:
                continue
            seen_ids.add(iid)
            if int(value) not in (-1, 1):
                continue
            s.expunge(row)
            rows.append(row)
            ys.append(1 if int(value) > 0 else -1)

    if len(rows) < MIN_VOTES_FOR_LR:
        return None

    try:
        from .config import load_settings
        import yaml
        from pathlib import Path
        settings = load_settings()
        profile_data = yaml.safe_load(Path(settings.profile_path).read_text(encoding="utf-8"))
        from .models import Profile
        profile = Profile(**profile_data)
        try:
            from .rank.profile import build_profile_matrix_with_rocchio
            vote_count = int(signed_vote_count())
            profile_mat = build_profile_matrix_with_rocchio(profile, vote_count)
        except Exception:
            from .rank.profile import build_profile_matrix
            profile_mat = build_profile_matrix(profile)
    except Exception as e:  # noqa: BLE001
        logger.warning("vote_dataset: could not load profile: %s", e)
        profile_mat = np.zeros((1, 384), dtype=np.float32)

    X = _build_item_features(rows, profile_mat)
    y = np.asarray(ys, dtype=np.float32)

    # Generate pairwise training examples (RankNet/Bradley-Terry approach).
    # For each (upvoted item, downvoted item) pair, the difference vector
    # encodes "what makes a good item better than a bad one."
    # At inference, the LR weights apply to individual feature vectors directly —
    # this is valid because the weight vector encodes gradient from bad to good.
    up_indices = [i for i, label in enumerate(ys) if label == 1]
    down_indices = [i for i, label in enumerate(ys) if label == -1]
    if up_indices and down_indices:
        pairs_X: list[np.ndarray] = []
        pairs_y: list[float] = []
        # Cap at 300 pairs to avoid memory explosion with many votes
        max_pairs = 300
        pair_count = 0
        for ui in up_indices:
            for di in down_indices:
                if pair_count >= max_pairs:
                    break
                diff = X[ui] - X[di]
                pairs_X.append(diff)
                pairs_y.append(1.0)
                # Add reverse for class balance
                pairs_X.append(-diff)
                pairs_y.append(-1.0)
                pair_count += 1
            if pair_count >= max_pairs:
                break
        if pairs_X:
            X_pairs = np.array(pairs_X, dtype=np.float32)
            y_pairs = np.array(pairs_y, dtype=np.float32)
            X = np.vstack([X, X_pairs])
            y = np.concatenate([y, y_pairs])

    return X, y
