from __future__ import annotations

from datetime import datetime, timezone


def _reset_store_for_tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test_digest.db"))
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod

    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    return store_mod


def _insert_item(store_mod) -> int:
    store_mod.init_db()
    with store_mod.session_scope() as s:
        row = store_mod.ItemRow(
            source="Test",
            section="research",
            external_id="test-1",
            url="https://example.com/test-1",
            title="RNA nanotechnology paper",
            abstract="A useful abstract.",
            published_at=datetime.now(timezone.utc),
            digest_id="2026-05-05",
            item_label="R1",
        )
        s.add(row)
        s.flush()
        return int(row.id)


def test_neutral_vote_is_persisted_for_visible_feedback(tmp_path, monkeypatch):
    store_mod = _reset_store_for_tmp_db(monkeypatch, tmp_path)
    from dailydigest import votes as votes_mod

    item_id = _insert_item(store_mod)

    assert votes_mod.record_vote_by_id(item_id, 0) is True

    assert votes_mod.get_vote_value(item_id) == 0


def test_vote_dataset_ignores_neutral_votes(tmp_path, monkeypatch):
    store_mod = _reset_store_for_tmp_db(monkeypatch, tmp_path)
    from dailydigest import votes as votes_mod

    store_mod.init_db()
    with store_mod.session_scope() as s:
        for idx in range(31):
            item = store_mod.ItemRow(
                source="Test",
                section="research",
                external_id=f"neutral-{idx}",
                url=f"https://example.com/neutral-{idx}",
                title=f"Item {idx}",
                abstract="Neutral example.",
                published_at=datetime.now(timezone.utc),
            )
            s.add(item)
            s.flush()
            s.add(store_mod.VoteRow(item_id=item.id, value=0))

    def _should_not_embed(_texts):
        raise AssertionError("neutral-only votes should not be embedded")

    monkeypatch.setattr(votes_mod, "embed_texts", _should_not_embed)

    assert votes_mod.vote_dataset() is None
