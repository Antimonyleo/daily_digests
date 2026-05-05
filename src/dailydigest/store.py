from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import (
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
    select,
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
    __table_args__ = (UniqueConstraint("item_id", name="uq_votes_item_id"),)


class DigestRow(Base):
    __tablename__ = "digests"

    id = Column(String, primary_key=True)  # e.g., "2026-05-04"
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    item_count = Column(Integer, default=0)
    sent_at = Column(DateTime(timezone=True))


class RunRow(Base):
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True)
    started_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    finished_at = Column(DateTime(timezone=True))
    stage = Column(String)  # ingest, rank, send
    status = Column(String)  # ok, error
    detail = Column(Text)


_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        ensure_data_dir()
        _ENGINE = create_engine(f"sqlite:///{SETTINGS.db_path}", future=True)

        @event.listens_for(_ENGINE, "connect")
        def _set_sqlite_pragma(dbapi_conn, _record):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return _ENGINE


def init_db() -> None:
    eng = _engine()
    Base.metadata.create_all(eng)


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
                select(ItemRow).where(ItemRow.fetched_at >= cutoff).order_by(ItemRow.fetched_at.desc())
            )
            .scalars()
            .all()
        )
        # detach for use after session close
        for r in rows:
            s.expunge(r)
        return list(rows)


def prune(days: int) -> int:
    init_db()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with session_scope() as s:
        result = s.execute(delete(ItemRow).where(ItemRow.fetched_at < cutoff))
        return result.rowcount or 0


def write_digest(digest_id: str, labeled_items: list[tuple[str, int]]) -> None:
    """labeled_items: list of (label, item_id)."""
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
        for label, item_id in labeled_items:
            row = s.get(ItemRow, item_id)
            if row is not None:
                row.digest_id = digest_id
                row.item_label = label


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


def db_path_exists() -> bool:
    return Path(SETTINGS.db_path).exists()
