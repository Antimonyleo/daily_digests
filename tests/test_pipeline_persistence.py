from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace


def _reset_store(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    monkeypatch.setenv("PROFILE_PATH", "config/profile.example.yaml")
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    store_mod.init_db()
    return store_mod


def test_quality_gate_protects_high_quality_journal_metadata_and_audits_drops():
    from dailydigest import pipeline as pipeline_mod
    from dailydigest import store as store_mod

    protected = store_mod.ItemRow(
        source="Nature",
        section="research",
        external_id="protected",
        url="https://example.com/protected",
        title="Short",
        abstract="",
    )
    thin_low_quality = store_mod.ItemRow(
        source="Minor Journal",
        section="research",
        external_id="thin",
        url="https://example.com/thin",
        title="Thin low-quality research metadata",
        abstract="Too short.",
    )
    fillers = [
        store_mod.ItemRow(
            source="Nature Biotechnology",
            section="research",
            external_id=f"filler-{idx}",
            url=f"https://example.com/filler-{idx}",
            title=f"Substantive protected research item {idx}",
            abstract="Primary research with methods, efficacy, and mechanism details.",
        )
        for idx in range(8)
    ]

    kept, drops = pipeline_mod._quality_gate([protected, thin_low_quality, *fillers])

    assert protected in kept
    assert thin_low_quality not in kept
    assert any(drop["reason"] == "thin abstract from non-protected source" for drop in drops)


def test_run_all_persists_summaries_for_web_view(monkeypatch, tmp_path):
    from dailydigest import config as config_mod
    from dailydigest import pipeline as pipeline_mod
    from dailydigest import store as store_mod

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    monkeypatch.setenv("PROFILE_PATH", "config/profile.example.yaml")
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False

    store_mod.init_db()
    with store_mod.session_scope() as s:
        row = store_mod.ItemRow(
            source="Test",
            section="research",
            external_id="summary-1",
            url="https://example.com/summary-1",
            title="Summary persistence",
            abstract="Original abstract.",
            published_at=datetime.now(timezone.utc),
        )
        s.add(row)
        s.flush()
        item_id = int(row.id)

    monkeypatch.setattr(pipeline_mod, "ingest_all", lambda progress_callback=None: 0)
    monkeypatch.setattr(pipeline_mod, "load_profile", lambda: SimpleNamespace(bio="", keywords=[], downweight=[]))
    monkeypatch.setattr(pipeline_mod, "build_profile_matrix", lambda _profile: __import__("numpy").zeros((1, 3)))
    monkeypatch.setattr(pipeline_mod, "recent_items", lambda days=2: [store_mod.session_factory()().get(store_mod.ItemRow, item_id)])
    monkeypatch.setattr(pipeline_mod, "score_items", lambda items, _pv, _downweight, reason_penalty_map=None: [(items[0], 0.9)])
    monkeypatch.setattr(pipeline_mod, "pick_top_per_section", lambda scored, _caps: scored)
    monkeypatch.setattr(pipeline_mod, "summarize_items", lambda rows: {rows[0].id: "Persisted summary."})
    monkeypatch.setattr(pipeline_mod, "send_digest", lambda html, subject, dry_run=False: True)

    pipeline_mod.run_all(dry_run=True)

    with store_mod.session_scope() as s:
        saved = s.get(store_mod.ItemRow, item_id)
        assert saved.summary == "Persisted summary."
    audit = store_mod.load_digest_audit(pipeline_mod._digest_id(), "candidate_funnel")
    assert audit
    assert audit[0]["after_cross_source_dedupe"] == 1


def test_non_dry_run_skips_when_digest_already_sent(monkeypatch, tmp_path):
    from dailydigest import pipeline as pipeline_mod

    store_mod = _reset_store(tmp_path, monkeypatch)
    digest_id = "2026-05-05"
    sent_at = datetime(2026, 5, 5, tzinfo=timezone.utc)

    with store_mod.session_scope() as s:
        s.add(store_mod.DigestRow(id=digest_id, item_count=1, sent_at=sent_at))

    monkeypatch.setattr(pipeline_mod, "_digest_id", lambda: digest_id)
    monkeypatch.setattr(
        pipeline_mod,
        "ingest_all",
        lambda progress_callback=None: (_ for _ in ()).throw(AssertionError("ingest called")),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "send_digest",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("send called")),
    )

    assert pipeline_mod.run_all(dry_run=False) == digest_id


def test_dry_run_after_sent_digest_refreshes_preview_and_preserves_sent_at(monkeypatch, tmp_path):
    from dailydigest import pipeline as pipeline_mod

    store_mod = _reset_store(tmp_path, monkeypatch)
    digest_id = "2026-05-05"
    sent_at = datetime(2026, 5, 5, tzinfo=timezone.utc)

    with store_mod.session_scope() as s:
        s.add(store_mod.DigestRow(id=digest_id, item_count=1, sent_at=sent_at))
        old = store_mod.ItemRow(
            source="Old",
            section="research",
            external_id="old",
            url="https://example.com/old",
            title="Old item",
            digest_id=digest_id,
            item_label="R1",
        )
        new = store_mod.ItemRow(
            source="New",
            section="research",
            external_id="new",
            url="https://example.com/new",
            title="New item",
        )
        s.add_all([old, new])
        s.flush()
        old_id = int(old.id)
        new_id = int(new.id)

    def recent_items(days=2):
        with store_mod.session_scope() as s:
            row = s.get(store_mod.ItemRow, new_id)
            s.expunge(row)
            return [row]

    monkeypatch.setattr(pipeline_mod, "_digest_id", lambda: digest_id)
    monkeypatch.setattr(pipeline_mod, "ingest_all", lambda progress_callback=None: 0)
    monkeypatch.setattr(pipeline_mod, "load_profile", lambda: SimpleNamespace(bio="", keywords=[], downweight=[]))
    monkeypatch.setattr(pipeline_mod, "build_profile_matrix", lambda _profile: __import__("numpy").zeros((1, 3)))
    monkeypatch.setattr(pipeline_mod, "recent_items", recent_items)
    monkeypatch.setattr(pipeline_mod, "score_items", lambda items, _pv, _downweight, reason_penalty_map=None: [(items[0], 0.9)])
    monkeypatch.setattr(pipeline_mod, "pick_top_per_section", lambda scored, _caps: scored)
    monkeypatch.setattr(pipeline_mod, "summarize_items", lambda rows: {rows[0].id: "New summary."})
    monkeypatch.setattr(pipeline_mod, "send_digest", lambda html, subject, dry_run=False: False)

    pipeline_mod.run_all(dry_run=True)

    with store_mod.session_scope() as s:
        digest = s.get(store_mod.DigestRow, digest_id)
        old = s.get(store_mod.ItemRow, old_id)
        new = s.get(store_mod.ItemRow, new_id)
        assert digest.sent_at == sent_at.replace(tzinfo=None)
        assert digest.item_count == 1
        assert old.digest_id is None
        assert old.item_label is None
        assert new.digest_id == digest_id
        assert new.item_label == "R1"


def test_run_all_does_not_mark_sent_when_send_digest_returns_false(monkeypatch, tmp_path):
    from dailydigest import pipeline as pipeline_mod

    store_mod = _reset_store(tmp_path, monkeypatch)
    digest_id = "2026-05-05"

    with store_mod.session_scope() as s:
        item = store_mod.ItemRow(
            source="Test",
            section="research",
            external_id="unsent",
            url="https://example.com/unsent",
            title="Unsent item",
        )
        s.add(item)
        s.flush()
        item_id = int(item.id)

    def recent_items(days=2):
        with store_mod.session_scope() as s:
            row = s.get(store_mod.ItemRow, item_id)
            s.expunge(row)
            return [row]

    monkeypatch.setattr(pipeline_mod, "_digest_id", lambda: digest_id)
    monkeypatch.setattr(pipeline_mod, "ingest_all", lambda progress_callback=None: 0)
    monkeypatch.setattr(pipeline_mod, "load_profile", lambda: SimpleNamespace(bio="", keywords=[], downweight=[]))
    monkeypatch.setattr(pipeline_mod, "build_profile_matrix", lambda _profile: __import__("numpy").zeros((1, 3)))
    monkeypatch.setattr(pipeline_mod, "recent_items", recent_items)
    monkeypatch.setattr(pipeline_mod, "score_items", lambda items, _pv, _downweight, reason_penalty_map=None: [(items[0], 0.9)])
    monkeypatch.setattr(pipeline_mod, "pick_top_per_section", lambda scored, _caps: scored)
    monkeypatch.setattr(pipeline_mod, "summarize_items", lambda rows: {rows[0].id: "Summary."})
    monkeypatch.setattr(pipeline_mod, "send_digest", lambda html, subject, dry_run=False: False)

    pipeline_mod.run_all(dry_run=False)

    with store_mod.session_scope() as s:
        digest = s.get(store_mod.DigestRow, digest_id)
        assert digest.sent_at is None


def test_run_all_auto_backfill_when_days_missed(monkeypatch, tmp_path):
    """When the last sent digest was several days ago, the window auto-widens."""
    from datetime import datetime, timedelta, timezone
    from dailydigest import pipeline as pipeline_mod

    store_mod = _reset_store(tmp_path, monkeypatch)
    today_id = "2026-05-15"
    old_sent_at = datetime(2026, 5, 10, tzinfo=timezone.utc)  # 5 days ago

    with store_mod.session_scope() as s:
        s.add(store_mod.DigestRow(id="2026-05-10", item_count=3, sent_at=old_sent_at))

    captured_days: list[int] = []

    def fake_recent_items(days=2):
        captured_days.append(days)
        return []

    monkeypatch.setattr(pipeline_mod, "_digest_id", lambda: today_id)
    monkeypatch.setattr(pipeline_mod, "ingest_all", lambda progress_callback=None: 0)
    monkeypatch.setattr(pipeline_mod, "load_profile", lambda: SimpleNamespace(bio="", keywords=[], downweight=[]))
    monkeypatch.setattr(pipeline_mod, "build_profile_matrix", lambda _profile: __import__("numpy").zeros((1, 3)))
    monkeypatch.setattr(pipeline_mod, "recent_items", fake_recent_items)
    monkeypatch.setattr(pipeline_mod, "score_items", lambda items, _pv, _downweight, reason_penalty_map=None: [])
    monkeypatch.setattr(pipeline_mod, "pick_top_per_section", lambda scored, _caps: [])
    monkeypatch.setattr(pipeline_mod, "summarize_items", lambda rows: {})
    monkeypatch.setattr(pipeline_mod, "send_digest", lambda html, subject, dry_run=False: False)

    pipeline_mod.run_all(dry_run=True)

    # Gap is at least 5 days; auto days should be > 2
    assert captured_days and captured_days[0] >= 5


def test_run_all_explicit_backfill_days_overrides_auto(monkeypatch, tmp_path):
    """An explicit backfill_days= value is used as-is, ignoring auto-detection."""
    from dailydigest import pipeline as pipeline_mod

    store_mod = _reset_store(tmp_path, monkeypatch)
    captured_days: list[int] = []

    def fake_recent_items(days=2):
        captured_days.append(days)
        return []

    monkeypatch.setattr(pipeline_mod, "_digest_id", lambda: "2026-05-15")
    monkeypatch.setattr(pipeline_mod, "ingest_all", lambda progress_callback=None: 0)
    monkeypatch.setattr(pipeline_mod, "load_profile", lambda: SimpleNamespace(bio="", keywords=[], downweight=[]))
    monkeypatch.setattr(pipeline_mod, "build_profile_matrix", lambda _profile: __import__("numpy").zeros((1, 3)))
    monkeypatch.setattr(pipeline_mod, "recent_items", fake_recent_items)
    monkeypatch.setattr(pipeline_mod, "score_items", lambda items, _pv, _downweight, reason_penalty_map=None: [])
    monkeypatch.setattr(pipeline_mod, "pick_top_per_section", lambda scored, _caps: [])
    monkeypatch.setattr(pipeline_mod, "summarize_items", lambda rows: {})
    monkeypatch.setattr(pipeline_mod, "send_digest", lambda html, subject, dry_run=False: False)

    pipeline_mod.run_all(dry_run=True, backfill_days=7)

    assert captured_days and captured_days[0] == 7


def test_run_all_empty_digest_emits_done_with_zero_items(monkeypatch, tmp_path):
    from dailydigest import pipeline as pipeline_mod

    store_mod = _reset_store(tmp_path, monkeypatch)
    digest_id = "2026-05-05"
    events = []

    monkeypatch.setattr(pipeline_mod, "_digest_id", lambda: digest_id)
    monkeypatch.setattr(pipeline_mod, "ingest_all", lambda progress_callback=None: 0)
    monkeypatch.setattr(pipeline_mod, "load_profile", lambda: SimpleNamespace(bio="", keywords=[], downweight=[]))
    monkeypatch.setattr(pipeline_mod, "build_profile_matrix", lambda _profile: __import__("numpy").zeros((1, 3)))
    monkeypatch.setattr(pipeline_mod, "recent_items", lambda days=2: [])
    monkeypatch.setattr(pipeline_mod, "score_items", lambda items, _pv, _downweight, reason_penalty_map=None: [])
    monkeypatch.setattr(pipeline_mod, "pick_top_per_section", lambda scored, _caps: [])
    monkeypatch.setattr(pipeline_mod, "summarize_items", lambda rows: {})
    monkeypatch.setattr(pipeline_mod, "send_digest", lambda html, subject, dry_run=False: False)

    pipeline_mod.run_all(dry_run=True, progress_callback=lambda stage, payload: events.append((stage, payload)))

    assert events[-1] == ("done", {"digest_id": digest_id, "total_items": 0, "dry_run": True})
    with store_mod.session_scope() as s:
        digest = s.get(store_mod.DigestRow, digest_id)
        assert digest.item_count == 0
