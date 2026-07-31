from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    and_,
    create_engine,
    delete,
    event,
    func,
    or_,
    select,
    text,
    update,
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
    value = Column(Integer, nullable=False)  # sign of the preference: +1 / 0 / -1
    # Preference strength as a percentage (0-100) from the 4-level feedback:
    # must-read 100, relevant 70, hmmm 40, not-for-me 10. Nullable for legacy rows
    # (mapped from `value` when absent). `value` stays the coarse sign for the LR.
    grade = Column(Integer, nullable=True)
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


class ItemEnrichmentRow(Base):
    """Cached OpenAlex enrichment (citations + venue impact) per item."""

    __tablename__ = "item_enrichment"

    item_id = Column(
        Integer,
        ForeignKey("items.id", ondelete="CASCADE"),
        primary_key=True,
    )
    cited_by_count = Column(Integer)
    venue_impact = Column(Float)
    venue = Column(String)
    fetched_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    item = relationship("ItemRow")


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


class ImpressionRow(Base):
    """Immutable, append-only log of what was shown in each brew RUN.

    Unlike ``digest_items`` (which is REPLACED on every rebrew so it always
    reflects the latest slate), this table is the permanent record: each brew
    run INSERTS a fresh set of rows keyed by ``run_id`` and never deletes or
    overwrites prior runs. This is the raw data offline evaluation and CTR
    modeling accumulate from, so it must not be clobbered by a dry-run/rebrew.
    """

    __tablename__ = "impressions"

    id = Column(Integer, primary_key=True)
    # run_id groups all rows written by one brew; a rebrew of the same digest_id
    # produces a NEW run_id, so both runs' rows coexist.
    run_id = Column(String, nullable=False, index=True)
    digest_id = Column(String, ForeignKey("digests.id", ondelete="CASCADE"), nullable=False, index=True)
    item_id = Column(Integer, ForeignKey("items.id", ondelete="CASCADE"), nullable=False, index=True)
    section = Column(String, nullable=False)
    position = Column(Integer, nullable=False)  # 0-based rank within the section
    final_score = Column(Float)
    model_version = Column(String)  # ranker_version at brew time
    # True when the item made the final digest slate; False for scored-but-unpicked
    # candidates (e.g. the research candidate pool). This makes the log an A/B
    # substrate: alternative rankers can be compared against the identical pool.
    selected = Column(Boolean, nullable=False, default=False)
    # Set later by the UI when the reader actually views the item; defaults False.
    viewed = Column(Boolean, nullable=False, default=False)
    # Per-candidate facet attribution captured at brew time, so the coverage
    # harness can attribute the UNSELECTED research pool (which is absent from
    # digest_item_features, that only records the selected slate).
    #   primary_facet: the item's dominant matched core-interest facet ("" if none).
    #   topic_score:   the raw relevance cosine the topic gate used (nullable).
    primary_facet = Column(String, default="")
    primary_facet_score = Column(Float)
    topic_score = Column(Float)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    item = relationship("ItemRow")
    digest = relationship("DigestRow")


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
_SessionLocal = None
_SESSION_LOCK = Lock()


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
            "grade": "INTEGER",
        },
        "impressions": {
            "selected": "BOOLEAN NOT NULL DEFAULT 0",
            "viewed": "BOOLEAN NOT NULL DEFAULT 0",
            "primary_facet": "VARCHAR DEFAULT ''",
            "primary_facet_score": "FLOAT",
            "topic_score": "FLOAT",
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


def session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        with _SESSION_LOCK:
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
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    # Publication APIs occasionally report a next-day issue date because of
    # timezone differences. Allow that small skew, but reject malformed or
    # scheduled dates months/years ahead so they do not remain "recent" forever.
    future_cutoff = now + timedelta(days=1)
    with session_scope() as s:
        rows = (
            s.execute(
                select(ItemRow)
                .where(
                    or_(
                        and_(
                            ItemRow.published_at >= cutoff,
                            ItemRow.published_at <= future_cutoff,
                        ),
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


def exclude_previously_shown(
    rows: list[ItemRow],
    days_lookback: int = 30,
    exclude_digest_id: str | None = None,
) -> list[ItemRow]:
    """Drop any row that already appeared in an earlier digest.

    A daily digest should feel fresh: once an item has been surfaced to the
    reader in a prior brew, it should not resurface — regardless of whether the
    reader voted on it. The previous "only hide explicitly-dismissed items shown
    in *sent* digests" rule failed for local usage, where brews are previewed in
    the web UI and never marked ``sent_at``; nothing was ever suppressed, so
    yesterday's items reappeared today.

    We now suppress on membership in ANY digest within ``days_lookback`` (matching
    the 30-day item retention), excluding ``exclude_digest_id`` so re-brewing the
    *current* day does not hide the items it is about to (re)assign.
    """
    ids = [int(r.id) for r in rows if r.id is not None]
    if not ids:
        return rows
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_lookback)
    with session_scope() as s:
        stmt = (
            select(DigestItemRow.item_id)
            .join(DigestRow, DigestRow.id == DigestItemRow.digest_id)
            .where(
                DigestRow.created_at >= cutoff,
                DigestItemRow.item_id.in_(ids),
            )
        )
        if exclude_digest_id is not None:
            stmt = stmt.where(DigestItemRow.digest_id != exclude_digest_id)
        shown_ids = {int(item_id) for item_id in s.execute(stmt).scalars()}
    if not shown_ids:
        return rows
    return [r for r in rows if r.id is None or int(r.id) not in shown_ids]


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
    init_db()
    with session_scope() as s:
        if s.get(DigestRow, digest_id) is None:
            s.add(DigestRow(id=digest_id, item_count=len(feature_rows)))
        # A same-day rebrew replaces the slate: drop feature rows for items no
        # longer featured under this digest_id so the table does not accumulate the
        # UNION of every rebrew's candidates (which distorted offline evaluation).
        # This DELETE must run even when ``feature_rows`` is empty — an early
        # `return` before it left stale rows behind after a zero-result rebrew
        # (displayed=0 but features=1), so it runs regardless of the new slate size.
        new_ids = [int(item_id) for _, item_id, _, _ in feature_rows]
        s.execute(
            delete(DigestItemFeatureRow).where(
                DigestItemFeatureRow.digest_id == digest_id,
                DigestItemFeatureRow.item_id.notin_(new_ids),
            )
        )
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


def write_impressions(
    digest_id: str,
    impression_rows: list[tuple],
    model_version: str | None = None,
    run_id: str | None = None,
) -> str:
    """Append one brew RUN's impressions to the immutable impression log.

    Each tuple is ``(section, item_id, position, final_score)`` or, to log the
    scored candidate pool for A/B, ``(section, item_id, position, final_score,
    selected)`` where ``selected`` marks whether the item made the final slate.
    Two further optional elements carry per-candidate facet attribution so the
    coverage harness can measure the unselected pool:
    ``(section, item_id, position, final_score, selected, primary_facet,
    primary_facet_score, topic_score)`` — ``primary_facet`` (str, "" when
    absent) and its raw per-facet cosine are distinct from ``topic_score``, the
    overall profile relevance cosine used by the topic gate.
    When ``selected`` is omitted it defaults to True (a selected-slate row). A
    fresh ``run_id`` is minted per call (unless supplied) so a rebrew of the same
    ``digest_id`` ADDS a new run's rows rather than replacing prior runs — this
    is the append-only counterpart to ``write_digest`` (which is destructive).
    Returns the ``run_id`` used.
    """
    init_db()
    # uuid4 is fine in normal Python; callers may pass an explicit id if they
    # need a deterministic/timestamp-derived one in a restricted context.
    run_id = run_id or uuid.uuid4().hex
    with session_scope() as s:
        if s.get(DigestRow, digest_id) is None:
            s.add(DigestRow(id=digest_id, item_count=len(impression_rows)))
        now = datetime.now(timezone.utc)
        for row in impression_rows:
            section, item_id, position, final_score = row[0], row[1], row[2], row[3]
            selected = bool(row[4]) if len(row) >= 5 else True
            primary_facet = str(row[5]) if len(row) >= 6 else ""
            # Keep the prior 7-tuple contract: its final value was topic_score.
            # The new per-facet score is present only in the explicit 8-tuple.
            primary_facet_score = (
                float(row[6]) if len(row) >= 8 and row[6] is not None else None
            )
            topic_score = (
                float(row[7]) if len(row) >= 8 and row[7] is not None
                else float(row[6]) if len(row) == 7 and row[6] is not None
                else None
            )
            s.add(
                ImpressionRow(
                    run_id=run_id,
                    digest_id=digest_id,
                    item_id=int(item_id),
                    section=str(section or ""),
                    position=int(position),
                    final_score=float(final_score) if final_score is not None else None,
                    model_version=model_version,
                    selected=selected,
                    primary_facet=primary_facet,
                    primary_facet_score=primary_facet_score,
                    topic_score=topic_score,
                    created_at=now,
                )
            )
    return run_id


def mark_impressions_viewed(
    digest_id: str, visible_item_ids: Iterable[int] | None = None
) -> int:
    """Mark the LATEST run's SELECTED impression rows for ``digest_id`` as viewed.

    Called from the web digest view so the ``viewed`` flag reflects that the
    reader actually opened this digest. Only the most-recent run's rows are
    updated (older runs stay as recorded). Within that run, only ``selected``
    rows that are still visible under the reader's current section settings are
    marked viewed; hidden and unselected candidate-pool rows stay
    ``viewed=False``. Returns the number of rows updated. When
    ``visible_item_ids`` is omitted, all selected rows retain the legacy
    behavior.
    """
    visible_ids = (
        {int(item_id) for item_id in visible_item_ids}
        if visible_item_ids is not None
        else None
    )
    if visible_ids is not None and not visible_ids:
        return 0

    init_db()
    with session_scope() as s:
        latest_run = s.execute(
            select(ImpressionRow.run_id)
            .where(ImpressionRow.digest_id == digest_id)
            .order_by(ImpressionRow.created_at.desc(), ImpressionRow.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        if latest_run is None:
            return 0
        conditions = [
            ImpressionRow.digest_id == digest_id,
            ImpressionRow.run_id == latest_run,
            ImpressionRow.selected.is_(True),
        ]
        if visible_ids is not None:
            conditions.append(ImpressionRow.item_id.in_(visible_ids))
        result = s.execute(
            update(ImpressionRow).where(*conditions).values(viewed=True)
        )
        return result.rowcount or 0


def recent_viewed_facet_dates(
    *,
    before_digest_id: str,
    days: int = 7,
    min_primary_facet_score: float = 0.0,
) -> dict[str, datetime]:
    """Return the latest viewed selection date for each recent research facet.

    Only the newest brew run per sent digest and facets at or above
    ``min_primary_facet_score`` count. This prevents an earlier, viewed rebrew
    or a weak/broad attribution from suppressing coverage for a genuine niche
    match. The helper issues SELECTs only.
    """
    if days <= 0:
        return {}
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with session_scope() as s:
        latest = (
            select(
                ImpressionRow.digest_id.label("digest_id"),
                func.max(ImpressionRow.created_at).label("created_at"),
            )
            .group_by(ImpressionRow.digest_id)
            .subquery()
        )
        rows = s.execute(
            select(ImpressionRow.primary_facet, DigestRow.sent_at)
            .join(
                latest,
                (ImpressionRow.digest_id == latest.c.digest_id)
                & (ImpressionRow.created_at == latest.c.created_at),
            )
            .join(DigestRow, DigestRow.id == ImpressionRow.digest_id)
            .where(
                ImpressionRow.section == "research",
                ImpressionRow.selected.is_(True),
                ImpressionRow.viewed.is_(True),
                ImpressionRow.primary_facet.is_not(None),
                ImpressionRow.primary_facet != "",
                ImpressionRow.primary_facet_score.is_not(None),
                ImpressionRow.primary_facet_score >= float(min_primary_facet_score),
                DigestRow.id != before_digest_id,
                DigestRow.sent_at.is_not(None),
                DigestRow.sent_at >= cutoff,
            )
        ).all()
    latest_by_facet: dict[str, datetime] = {}
    for facet, sent_at in rows:
        key = str(facet or "").casefold()
        if not key or not isinstance(sent_at, datetime):
            continue
        # SQLite commonly round-trips timezone-aware timestamps as naive values.
        # Digest sent times are written in UTC, so restore that contract before
        # the pipeline computes a recency age against an aware UTC `now`.
        if sent_at.tzinfo is None:
            sent_at = sent_at.replace(tzinfo=timezone.utc)
        previous = latest_by_facet.get(key)
        if previous is None or sent_at > previous:
            latest_by_facet[key] = sent_at
    return latest_by_facet


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


def days_since_last_digest(exclude_digest_id: str | None = None) -> int:
    """Return calendar days since the most-recent digest of *any* kind.

    Unlike :func:`days_since_last_sent`, this counts brewed/rendered digests
    too (``sent_at`` may be null for local or dry-run use), so the catch-up
    window works whether the digest is emailed by the cron or previewed locally.
    Returns -1 when no prior digest exists (first ever run).
    """
    init_db()
    with session_scope() as s:
        q = select(func.max(DigestRow.created_at))
        if exclude_digest_id:
            q = q.where(DigestRow.id != exclude_digest_id)
        last = s.execute(q).scalar_one_or_none()
    if last is None:
        return -1
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return max(0, (datetime.now(timezone.utc) - last).days)


def db_path_exists() -> bool:
    return Path(SETTINGS.db_path).exists()
