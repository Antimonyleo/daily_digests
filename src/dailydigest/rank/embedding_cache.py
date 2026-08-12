"""SQLite-backed item embedding cache.

The expensive local sentence-transformer call is keyed by item id, model name,
and the normalized title+abstract text hash. Repeated ranking runs only embed
new or changed items.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select

from ..store import ItemEmbeddingRow, ItemRow, init_db, session_scope
from .embed import active_model_name, embed_texts


def item_text(row: ItemRow) -> str:
    title = (row.title or "").strip()
    abstract = (row.abstract or "").strip()
    if abstract:
        return f"{title}. {abstract}"
    return title


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _serialize(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _deserialize(raw: bytes, dim: int) -> np.ndarray | None:
    """Decode a cached float32 vector, returning None on any integrity problem.

    A truncated/corrupt blob (interrupted write, disk-full) or an out-of-range
    ``dim`` would otherwise make ``np.frombuffer(count=dim)`` raise and abort the
    whole ranking run; instead we validate and let the caller re-embed.
    """
    if dim <= 0 or dim > 8192 or len(raw) != dim * 4:
        return None
    return np.frombuffer(raw, dtype=np.float32, count=dim).copy()


def embed_item_rows(rows: list[ItemRow]) -> np.ndarray:
    """Return cached float32 embeddings for DB-backed item rows.

    Rows without an integer ``id`` are embedded directly and are not cached.
    """
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)

    texts = [item_text(row) for row in rows]
    ids: list[int | None] = []
    for row in rows:
        raw_id = getattr(row, "id", None)
        ids.append(int(raw_id) if isinstance(raw_id, int) else None)

    if any(item_id is None for item_id in ids):
        return embed_texts(texts)

    hashes = [_text_hash(text) for text in texts]
    vectors: list[np.ndarray | None] = [None] * len(rows)
    missing_indexes: list[int] = []

    model_name = active_model_name()
    init_db()
    with session_scope() as s:
        cached_rows = s.execute(
            select(ItemEmbeddingRow).where(
                ItemEmbeddingRow.item_id.in_([item_id for item_id in ids if item_id is not None]),
                ItemEmbeddingRow.model == model_name,
            )
        ).scalars().all()
        by_item_id = {int(row.item_id): row for row in cached_rows}

        for idx, item_id in enumerate(ids):
            assert item_id is not None
            cached = by_item_id.get(item_id)
            vec = (
                _deserialize(cached.vector, int(cached.dim))
                if cached is not None and cached.text_hash == hashes[idx]
                else None
            )
            if vec is not None:
                vectors[idx] = vec
            else:
                missing_indexes.append(idx)

        if missing_indexes:
            new_vecs = embed_texts([texts[idx] for idx in missing_indexes]).astype(np.float32, copy=False)
            now = datetime.now(timezone.utc)
            for vec_offset, idx in enumerate(missing_indexes):
                item_id = ids[idx]
                assert item_id is not None
                vec = np.asarray(new_vecs[vec_offset], dtype=np.float32)
                cached = by_item_id.get(item_id)
                if cached is None:
                    cached = ItemEmbeddingRow(
                        item_id=item_id,
                        model=model_name,
                        text_hash=hashes[idx],
                        dim=int(vec.shape[0]),
                        vector=_serialize(vec),
                        created_at=now,
                        updated_at=now,
                    )
                    s.add(cached)
                    # Register it so a repeated item id later in this batch
                    # updates this pending row instead of inserting a second one
                    # (which violates UNIQUE(item_id, model) on commit).
                    by_item_id[item_id] = cached
                else:
                    cached.text_hash = hashes[idx]
                    cached.dim = int(vec.shape[0])
                    cached.vector = _serialize(vec)
                    cached.updated_at = now
                vectors[idx] = vec

    ready = [vec for vec in vectors if vec is not None]
    if not ready:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(ready).astype(np.float32, copy=False)
