from __future__ import annotations

import json
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from threading import Lock

from sqlalchemy import (
    CheckConstraint,
    LargeBinary,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    event,
    func,
    or_,
    select,
    text,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from .config import SETTINGS, ensure_data_dir
from .models import Item


class Base(DeclarativeBase):
    pass


class ItemRow(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True)
    source = Column(String, nullable=False, index=True)
    section = Column(String, nullable=False, index=True)
    external_id = Column(String, nullable=False)
    url = Column(String, nullable=False)
    title = Column(Text, nullable=False)
    abstract = Column(Text, default="")
    authors = Column(Text, default="")
    published_at = Column(DateTime(timezone=True), index=True)
    fetched_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)

    summary = Column(Text, default="")
    score = Column(Float)
    digest_id = Column(String, index=True)
    item_label = Column(String)  # e.g., "R3"

    __table_args__ = (UniqueConstraint("source", "external_id", name="uq_source_external"),)


class VoteRow(Base):
    __tablename__ = "votes"

    id = Column(Integer, primary_key=True)
    item_id = Column(Integer, ForeignKey("items.id", ondelete="CASCADE"), nullable=False, index=True)
    value = Column(Integer, nullable=False)  # +1 / 0 / -1
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    item = relationship("ItemRow")

    # NOTE: UNIQUE on item_id enforces "latest vote wins" semantics in
    # ``votes.record_votes`` (upsert). Migration note: SQLAlchemy's
    # ``create_all`` only creates *missing* tables; it will NOT add this
    # constraint to a pre-existing ``votes`` table. For the in-tree test
    # fixtures this is a non-issue (they use ``tmp_path`` -> fresh DB). For
    # the persistent ``data/digest.db``, the votes table is empty in
    # practice and the GH Actions DB is artifact-restored (often fresh),
    # so dropping/recreating votes is acceptable if duplicates already
    # exist. If a future deployment hits a duplicate-row violation, the
    # fix is to drop the votes table once and let ``init_db`` recreate it.
    __table_args__ = (
        UniqueConstraint("item_id", name="uq_votes_item_id"),
        CheckConstraint("value IN (-1, 0, 1)", name="ck_vote_value"),
    )


class ItemEmbeddingRow(Base):
    __tablename__ = "item_embeddings"

    id = Column(Integer, primary_key=True)
    item_id = Column(Integer, ForeignKey("items.id", ondelete="CASCADE"), nullable=False, index=True)
    model = Column(String, nullable=False)
    text_hash = Column(String, nullable=False)
    dim = Column(Integer, nullable=False)
    vector = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    item = relationship("ItemRow")

    __table_args__ = (UniqueConstraint("item_id", "model", name="uq_item_embedding_item_model"),)


class DigestRow(Base):
    __tablename__ = "digests"

    id = Column(String, primary_key=True)  # e.g., "2026-05-04"
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    item_count = Column(Integer, default=0)
    sent_at = Column(DateTime(timezone=True))


class DigestItemFeatureRow(Base):
    __tablename__ = "digest_item_features"

    id = Column(Integer, primary_key=True)
    digest_id = Column(String, ForeignKey("digests.id", ondelete="CASCADE"), nullable=False, index=True)
    item_id = Column(Integer, ForeignKey("items.id", ondelete="CASCADE"), nullable=False, index=True)
    item_label = Column(String)
    final_score = Column(Float)
    features_json = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    item = relationship("ItemRow")
    digest = relationship("DigestRow")

    __table_args__ = (UniqueConstraint("digest_id", "item_id", name="uq_digest_item_features"),)


class DigestAuditRow(Base):
    __tablename__ = "digest_audits"

    id = Column(Integer, primary_key=True)
    digest_id = Column(String, ForeignKey("digests.id", ondelete="CASCADE"), nullable=False, index=True)
    audit_type = Column(String, nullable=False, index=True)
    payload_json = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    digest = relationship("DigestRow")


class DigestItemRow(Base):
    __tablename__ = "digest_items"

    id = Column(Integer, primary_key=True)
    digest_id = Column(String, ForeignKey("digests.id", ondelete="CASCADE"), nullable=False, index=True)
    item_id = Column(Integer, ForeignKey("items.id", ondelete="CASCADE"), nullable=False, index=True)
    item_label = Column(String, nullable=False)
    score = Column(Float)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    item = relationship("ItemRow")
    digest = relationship("DigestRow")

    __table_args__ = (
        UniqueConstraint("digest_id", "item_id", name="uq_digest_items_item"),
        UniqueConstraint("digest_id", "item_label", name="uq_digest_items_label"),
    )


class RunRow(Base):
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True)
    started_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    finished_at = Column(DateTime(timezone=True))
    stage = Column(String)  # ingest, rank, send
    status = Column(String)  # ok, error
    detail = Column(Text)


_ENGINE = None
_ENGINE_LOCK = Lock()
_INITIALIZED: bool = False


def _engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE
        ensure_data_dir()
        from .config import get_settings
        _ENGINE = create_engine(
            f"sqlite:///{get_settings().db_path}",
            future=True,
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(_ENGINE, "connect")
        def _set_sqlite_pragma(dbapi_conn, _record):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=5000")
            cur.close()

    return _ENGINE


def init_db() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    eng = _engine()
    Base.metadata.create_all(eng)
    _migrate_sqlite_schema(eng)
    _INITIALIZED = True


def _reset_init_for_testing() -> None:
    global _INITIALIZED, _ENGINE, _SessionLocal
    _INITIALIZED = False
    _ENGINE = None
    _SessionLocal = None


def _migrate_sqlite_schema(eng) -> None:
    """Add columns introduced after early local DBs were created.

    SQLite ``create_all`` creates missing tables but does not alter existing
    ones. Keep this intentionally small and additive so old local databases can
    open the web UI without manual reset.
    """
    if eng.dialect.name != "sqlite":
        return

    required_columns = {
        "items": {
            "abstract": "TEXT DEFAULT ''",
            "authors": "TEXT DEFAULT ''",
            "published_at": "DATETIME",
            "fetched_at": "DATETIME",
            "summary": "TEXT DEFAULT ''",
            "score": "FLOAT",
            "digest_id": "VARCHAR",
            "item_label": "VARCHAR",
        },
        "digests": {
            "created_at": "DATETIME",
            "item_count": "INTEGER DEFAULT 0",
            "sent_at": "DATETIME",
        },
        "votes": {
            "created_at": "DATETIME",
        },
    }

    with eng.begin() as conn:
        for table_name, columns in required_columns.items():
            exists = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name=:name"),
                {"name": table_name},
            ).first()
            if exists is None:
                continue
            existing = {
                row[1]
                for row in conn.execute(text(f"PRAGMA table_info({table_name})")).all()
            }
            for column, ddl in columns.items():
                if column not in existing:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column} {ddl}"))


_SessionLocal = None


def session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=_engine(), expire_on_commit=False, future=True)
    return _SessionLocal


@contextmanager
def session_scope():
    s: Session = session_factory()()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


# ---------- repo helpers ----------

def upsert_items(items: Iterable[Item]) -> int:
    """Insert items, skipping duplicates on (source, external_id). Returns count inserted."""
    init_db()
    seen_keys: set[tuple[str, str]] = set()
    inserted = 0
    with session_scope() as s:
        for it in items:
            key = (it.source, it.external_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            stmt = (
                sqlite_insert(ItemRow)
                .values(
                    source=it.source,
                    section=it.section,
                    external_id=it.external_id,
                    url=it.url,
                    title=it.title,
                    abstract=it.abstract,
                    authors=it.authors,
                    published_at=it.published_at,
                )
                .on_conflict_do_nothing(index_elements=["source", "external_id"])
            )
            result = s.execute(stmt)
            if result.rowcount:
                inserted += 1
    return inserted


def recent_items(days: int = 2) -> list[ItemRow]:
    init_db()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with session_scope() as s:
        rows = (
            s.execute(
                select(ItemRow)
                .where(
                    or_(
                        ItemRow.published_at >= cutoff,
                        (ItemRow.published_at.is_(None) & (ItemRow.fetched_at >= cutoff)),
                    )
                )
                .order_by(func.coalesce(ItemRow.published_at, ItemRow.fetched_at).desc())
            )
            .scalars()
            .all()
        )
        # detach for use after session close
        for r in rows:
            s.expunge(r)
        return list(rows)


def exclude_reviewed_items(rows: list[ItemRow]) -> list[ItemRow]:
    """Drop rows the user has already reviewed (voted 0 or -1).

    Up-voted (+1) items are kept so they can reappear in backfill runs and
    continue training the LR without being silently excluded.  Neutral and
    down-voted items have been explicitly dismissed and should not resurface.
    """
    ids = [int(r.id) for r in rows if r.id is not None]
    if not ids:
        return rows
    reviewed = {
        item_id
        for item_id, value in _latest_vote_values(ids).items()
        if value != 1
    }
    if not reviewed:
        return rows
    return [r for r in rows if r.id is None or int(r.id) not in reviewed]


def _latest_vote_values(item_ids: list[int]) -> dict[int, int]:
    """Return latest vote values for ids, tolerating legacy duplicate rows."""
    if not item_ids:
        return {}
    with session_scope() as s:
        rows = s.execute(
            select(VoteRow.item_id, VoteRow.value)
            .where(VoteRow.item_id.in_(item_ids))
            .order_by(VoteRow.item_id, VoteRow.created_at.desc(), VoteRow.id.desc())
        ).all()
    latest: dict[int, int] = {}
    for item_id, value in rows:
        iid = int(item_id)
        if iid not in latest:
            latest[iid] = int(value)
    return latest


def exclude_previously_shown(rows: list[ItemRow], days_lookback: int = 7) -> list[ItemRow]:
    """Drop only explicitly dismissed rows shown in recent sent digests.

    A missing vote is ambiguous: the user may simply not have had time to read.
    Treat only Neutral/Bad as a hide signal. Good items are preserved for
    training and possible backfill resurfacing.
    """
    ids = [int(r.id) for r in rows if r.id is not None]
    if not ids:
        return rows
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_lookback)
    with session_scope() as s:
        shown_ids = {
            int(item_id)
            for item_id in s.execute(
                select(DigestItemRow.item_id)
                .join(DigestRow, DigestRow.id == DigestItemRow.digest_id)
                .where(
                    DigestRow.sent_at.isnot(None),
                    DigestRow.sent_at >= cutoff,
                    DigestItemRow.item_id.in_(ids),
                )
            ).scalars()
        }
    latest_votes = _latest_vote_values(list(shown_ids))
    dismissed_ids = {
        item_id
        for item_id, value in latest_votes.items()
        if value != 1
    }
    hidden = dismissed_ids
    if not hidden:
        return rows
    return [r for r in rows if r.id is None or int(r.id) not in hidden]


def prune(days: int) -> int:
    """Delete items older than ``days``, but preserve voted items.

    Voted items are kept indefinitely so the LR ranker always has training
    data even after 30-day rotation.  Without this guard, CASCADE deletion
    via ``ON DELETE CASCADE`` on ``votes`` would silently erase all training
    signal accumulated over months.
    """
    init_db()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with session_scope() as s:
        voted_subq = select(VoteRow.item_id).distinct().scalar_subquery()
        stmt = delete(ItemRow).where(
            ItemRow.fetched_at < cutoff,
            ItemRow.id.not_in(voted_subq),
        )
        result = s.execute(stmt)
        return result.rowcount or 0


def prune_old_digests(days: int) -> int:
    """Remove digest metadata rows older than ``days``.  Returns count deleted."""
    init_db()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with session_scope() as s:
        result = s.execute(delete(DigestRow).where(DigestRow.created_at < cutoff))
        return result.rowcount or 0


def write_digest(digest_id: str, labeled_items: list[tuple]) -> None:
    """Persist digest item assignments.

    ``labeled_items`` accepts ``(label, item_id)`` for backwards compatibility
    or ``(label, item_id, score)`` so ranked rows can carry their final score to
    the web/email renderers.
    """
    init_db()
    with session_scope() as s:
        digest = s.get(DigestRow, digest_id)
        if digest is None:
            s.add(DigestRow(id=digest_id, item_count=len(labeled_items)))
        else:
            # Preserve sent_at while allowing dry-run reruns to refresh preview rows.
            digest.item_count = len(labeled_items)
        previous = (
            s.execute(select(ItemRow).where(ItemRow.digest_id == digest_id))
            .scalars()
            .all()
        )
        for row in previous:
            row.digest_id = None
            row.item_label = None
        for item in labeled_items:
            label, item_id = item[0], item[1]
            score = item[2] if len(item) >= 3 else None
            row = s.get(ItemRow, item_id)
            if row is not None:
                row.digest_id = digest_id
                row.item_label = label
                if score is not None:
                    row.score = float(score)
        s.execute(delete(DigestItemRow).where(DigestItemRow.digest_id == digest_id))
        for item in labeled_items:
            label, item_id = item[0], item[1]
            score = item[2] if len(item) >= 3 else None
            if s.get(ItemRow, int(item_id)) is not None:
                s.add(
                    DigestItemRow(
                        digest_id=digest_id,
                        item_id=int(item_id),
                        item_label=str(label),
                        score=float(score) if score is not None else None,
                    )
                )


def write_digest_features(
    digest_id: str,
    feature_rows: list[tuple[str, int, float, dict]],
) -> None:
    """Persist rank-feature snapshots for the local web UI.

    Each tuple is ``(label, item_id, final_score, features)``. Features are
    stored as JSON so the ranker can evolve without requiring DB migrations.
    """
    if not feature_rows:
        return
    init_db()
    with session_scope() as s:
        if s.get(DigestRow, digest_id) is None:
            s.add(DigestRow(id=digest_id, item_count=len(feature_rows)))
        for label, item_id, final_score, features in feature_rows:
            payload = json.dumps(features or {}, sort_keys=True)
            stmt = (
                sqlite_insert(DigestItemFeatureRow)
                .values(
                    digest_id=digest_id,
                    item_id=int(item_id),
                    item_label=label,
                    final_score=float(final_score),
                    features_json=payload,
                    created_at=datetime.now(timezone.utc),
                )
                .on_conflict_do_update(
                    index_elements=["digest_id", "item_id"],
                    set_={
                        "item_label": label,
                        "final_score": float(final_score),
                        "features_json": payload,
                        "created_at": datetime.now(timezone.utc),
                    },
                )
            )
            s.execute(stmt)


def load_digest_features(digest_id: str) -> dict[int, dict]:
    """Return persisted rank-feature snapshots keyed by item id."""
    init_db()
    with session_scope() as s:
        rows = (
            s.execute(
                select(DigestItemFeatureRow.item_id, DigestItemFeatureRow.features_json)
                .where(DigestItemFeatureRow.digest_id == digest_id)
            )
            .all()
        )
    out: dict[int, dict] = {}
    for item_id, raw in rows:
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            out[int(item_id)] = payload
    return out


def write_digest_audit(digest_id: str, audit_type: str, payloads: list[dict]) -> None:
    """Replace one audit payload set for a digest."""
    init_db()
    with session_scope() as s:
        if s.get(DigestRow, digest_id) is None:
            s.add(DigestRow(id=digest_id, item_count=0))
        s.execute(
            delete(DigestAuditRow).where(
                DigestAuditRow.digest_id == digest_id,
                DigestAuditRow.audit_type == audit_type,
            )
        )
        for payload in payloads:
            s.add(
                DigestAuditRow(
                    digest_id=digest_id,
                    audit_type=audit_type,
                    payload_json=json.dumps(payload or {}, sort_keys=True),
                )
            )


def load_digest_audit(digest_id: str, audit_type: str) -> list[dict]:
    """Return persisted digest audit payloads."""
    init_db()
    with session_scope() as s:
        rows = (
            s.execute(
                select(DigestAuditRow.payload_json)
                .where(
                    DigestAuditRow.digest_id == digest_id,
                    DigestAuditRow.audit_type == audit_type,
                )
                .order_by(DigestAuditRow.id)
            )
            .scalars()
            .all()
        )
    out: list[dict] = []
    for raw in rows:
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            out.append(payload)
    return out


def write_summaries(summaries: dict[int, str]) -> None:
    """Persist per-item summaries for the local web UI."""
    if not summaries:
        return
    init_db()
    with session_scope() as s:
        for item_id, summary in summaries.items():
            row = s.get(ItemRow, int(item_id))
            if row is not None:
                row.summary = summary


def mark_sent(digest_id: str) -> None:
    init_db()
    with session_scope() as s:
        d = s.get(DigestRow, digest_id)
        if d is not None:
            d.sent_at = datetime.now(timezone.utc)


def days_since_last_sent(exclude_digest_id: str | None = None) -> int:
    """Return calendar days since the most-recently sent digest.

    Returns -1 when no digest has been sent yet (first run).
    ``exclude_digest_id`` is typically today's digest id so we don't count
    the current run as the "last sent" baseline.
    """
    init_db()
    with session_scope() as s:
        q = select(func.max(DigestRow.sent_at)).where(DigestRow.sent_at.isnot(None))
        if exclude_digest_id:
            q = q.where(DigestRow.id != exclude_digest_id)
        last_sent = s.execute(q).scalar_one_or_none()
    if last_sent is None:
        return -1
    if last_sent.tzinfo is None:
        last_sent = last_sent.replace(tzinfo=timezone.utc)
    return max(0, (datetime.now(timezone.utc) - last_sent).days)


def db_path_exists() -> bool:
    return Path(SETTINGS.db_path).exists()
