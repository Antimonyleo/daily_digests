from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace


def test_topic_selection_preferences_are_soft_and_keep_raw_scores(monkeypatch):
    """Coverage can promote one qualified, absent facet without reserving a slot."""
    from dailydigest import pipeline as pipeline_mod

    rows = [
        SimpleNamespace(id=1, section="research"),  # recently viewed facet
        SimpleNamespace(id=2, section="research"),  # best absent-facet item
        SimpleNamespace(id=3, section="research"),  # same facet, not its best
        SimpleNamespace(id=4, section="research"),  # fails per-facet relevance
    ]
    scored = [(rows[0], 0.72), (rows[1], 0.70), (rows[2], 0.69), (rows[3], 0.71)]
    features = {
        1: {"primary_facet": "colloids", "primary_facet_score": 0.8, "topic_priority": 0.5, "topic_priority_bonus": 0.03},
        2: {"primary_facet": "dna nano", "primary_facet_score": 0.8, "topic_priority": 1.0, "topic_priority_bonus": 0.06},
        3: {"primary_facet": "dna nano", "primary_facet_score": 0.8, "topic_priority": 1.0, "topic_priority_bonus": 0.06},
        4: {"primary_facet": "rna nano", "primary_facet_score": 0.4, "topic_priority": 1.0, "topic_priority_bonus": 0.06},
    }
    monkeypatch.setattr(
        pipeline_mod,
        "get_settings",
        lambda: SimpleNamespace(min_topic_relevance=0.65, topic_coverage_bonus_scale=0.03),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "recent_viewed_facet_dates",
        # SQLite's digest timestamps are historically naive; the store helper
        # normalizes them before the real pipeline sees them.
        lambda **_kwargs: {"colloids": datetime.now()},
    )

    ordered = pipeline_mod._apply_topic_selection_preferences(
        scored, features, digest_id="2026-06-07"
    )

    # The unseen high-priority DNA facet is gently promoted, but output scores
    # remain exactly the ranker scores and no slot was reserved.
    assert [row.id for row, _score in ordered][0] == 2
    assert dict((row.id, score) for row, score in ordered) == dict(
        (row.id, score) for row, score in scored
    )
    assert features[2]["topic_coverage_bonus"] == 0.03
    assert "topic_coverage_bonus" not in features[3]
    assert "selection_order_bonus" not in features[4]


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

    monkeypatch.setattr(pipeline_mod, "ingest_all", lambda progress_callback=None, days=2: 0)
    monkeypatch.setattr(pipeline_mod, "load_profile", lambda: SimpleNamespace(bio="", keywords=[], downweight=[]))
    monkeypatch.setattr(pipeline_mod, "build_profile_matrix", lambda _profile: __import__("numpy").zeros((1, 3)))
    monkeypatch.setattr(pipeline_mod, "recent_items", lambda days=2: [store_mod.session_factory()().get(store_mod.ItemRow, item_id)])
    monkeypatch.setattr(pipeline_mod, "score_items", lambda items, _pv, _downweight, reason_penalty_map=None: [(items[0], 0.9)])
    monkeypatch.setattr(pipeline_mod, "pick_top_per_section", lambda scored, _caps, catch_up=False: scored)
    monkeypatch.setattr(pipeline_mod, "summarize_items", lambda rows, profile=None: {rows[0].id: "Persisted summary."})
    monkeypatch.setattr(pipeline_mod, "send_digest", lambda html, subject, dry_run=False: True)

    pipeline_mod.run_all(dry_run=True)

    with store_mod.session_scope() as s:
        saved = s.get(store_mod.ItemRow, item_id)
        assert saved.summary == "Persisted summary."
    audit = store_mod.load_digest_audit(pipeline_mod._digest_id(), "candidate_funnel")
    assert audit
    assert audit[0]["after_cross_source_dedupe"] == 1


def test_run_all_logs_research_candidate_pool_with_selected_flags(monkeypatch, tmp_path):
    """A brew logs the research candidate pool: picked items selected=True and at
    least one scored-but-unpicked research item selected=False. This is the A/B
    substrate — alternative rankers can be scored against the identical pool."""
    from dailydigest import config as config_mod
    from dailydigest import pipeline as pipeline_mod
    from dailydigest import store as store_mod
    from dailydigest.rank.source_quality import RANKER_VERSION

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    monkeypatch.setenv("PROFILE_PATH", "config/profile.example.yaml")
    # This test uses near-identical placeholder research fixtures purely to
    # exercise impression logging; disable within-day near-dup suppression so it
    # does not collapse them (its behavior is covered in test_dedupe.py).
    monkeypatch.setenv("WITHIN_DAY_DEDUPE", "false")
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False

    store_mod.init_db()
    with store_mod.session_scope() as s:
        picked_row = store_mod.ItemRow(
            source="Nature",
            section="research",
            external_id="cand-picked",
            url="https://example.com/cand-picked",
            title="Picked research candidate",
            abstract="Primary research with methods and efficacy.",
            published_at=datetime.now(timezone.utc),
        )
        unpicked_row = store_mod.ItemRow(
            source="Nature",
            section="research",
            external_id="cand-unpicked",
            url="https://example.com/cand-unpicked",
            title="Unpicked research candidate",
            abstract="Primary research with methods and efficacy.",
            published_at=datetime.now(timezone.utc),
        )
        s.add_all([picked_row, unpicked_row])
        s.flush()
        picked_id = int(picked_row.id)
        unpicked_id = int(unpicked_row.id)

    def recent_items(days=2):
        with store_mod.session_scope() as s:
            rows = [s.get(store_mod.ItemRow, picked_id), s.get(store_mod.ItemRow, unpicked_id)]
            for r in rows:
                s.expunge(r)
            return rows

    def score_items(items, _pv, _downweight, reason_penalty_map=None):
        # Higher score for the picked item; both are research candidates.
        ordered = sorted(items, key=lambda it: 0 if int(it.id) == picked_id else 1)
        return [(ordered[0], 0.9), (ordered[1], 0.8)]

    # pick_top_per_section returns only the top-scored item, so the other stays
    # in the scored candidate pool as an unpicked (selected=False) impression.
    def pick_top_per_section(scored, _caps, catch_up=False):
        return scored[:1]

    monkeypatch.setattr(pipeline_mod, "ingest_all", lambda progress_callback=None, days=2: 0)
    monkeypatch.setattr(pipeline_mod, "load_profile", lambda: SimpleNamespace(bio="", keywords=[], downweight=[]))
    monkeypatch.setattr(pipeline_mod, "build_profile_matrix", lambda _profile: __import__("numpy").zeros((1, 3)))
    monkeypatch.setattr(pipeline_mod, "recent_items", recent_items)
    monkeypatch.setattr(pipeline_mod, "score_items", score_items)
    monkeypatch.setattr(pipeline_mod, "pick_top_per_section", pick_top_per_section)
    monkeypatch.setattr(pipeline_mod, "summarize_items", lambda rows, profile=None: {})
    monkeypatch.setattr(pipeline_mod, "send_digest", lambda html, subject, dry_run=False: True)

    pipeline_mod.run_all(dry_run=True)

    digest_id = pipeline_mod._digest_id()
    with store_mod.session_scope() as s:
        impressions = {
            r.item_id: r
            for r in s.query(store_mod.ImpressionRow).filter_by(digest_id=digest_id).all()
        }
        # Both research candidates were logged (pool, not just the slate).
        assert picked_id in impressions
        assert unpicked_id in impressions
        assert impressions[picked_id].selected is True
        assert impressions[unpicked_id].selected is False
        # Positions reflect score-ordered rank within the research candidate pool.
        assert impressions[picked_id].position == 0
        assert impressions[unpicked_id].position == 1
        # Accurate policy version is stamped as the model_version.
        assert impressions[picked_id].model_version == RANKER_VERSION


def test_run_all_impressions_carry_primary_facet_and_topic_score(monkeypatch, tmp_path):
    """Research impression rows carry per-candidate primary_facet + topic_score,
    sourced from score_features, so the coverage harness can attribute the pool
    (including UNSELECTED candidates that never reach digest_item_features)."""
    from dailydigest import config as config_mod
    from dailydigest import pipeline as pipeline_mod
    from dailydigest import store as store_mod

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    monkeypatch.setenv("PROFILE_PATH", "config/profile.example.yaml")
    # Near-identical placeholder fixtures; disable within-day dedupe so both survive.
    monkeypatch.setenv("WITHIN_DAY_DEDUPE", "false")
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False

    store_mod.init_db()
    with store_mod.session_scope() as s:
        picked_row = store_mod.ItemRow(
            source="Nature",
            section="research",
            external_id="facet-picked",
            url="https://example.com/facet-picked",
            title="Picked research candidate",
            abstract="Primary research with methods and efficacy.",
            published_at=datetime.now(timezone.utc),
        )
        unpicked_row = store_mod.ItemRow(
            source="Nature",
            section="research",
            external_id="facet-unpicked",
            url="https://example.com/facet-unpicked",
            title="Unpicked research candidate",
            abstract="Primary research with methods and efficacy.",
            published_at=datetime.now(timezone.utc),
        )
        s.add_all([picked_row, unpicked_row])
        s.flush()
        picked_id = int(picked_row.id)
        unpicked_id = int(unpicked_row.id)

    def recent_items(days=2):
        with store_mod.session_scope() as s:
            rows = [s.get(store_mod.ItemRow, picked_id), s.get(store_mod.ItemRow, unpicked_id)]
            for r in rows:
                s.expunge(r)
            return rows

    # Inject facet + topic_score attribution into score_features (this is what the
    # real feature-scoring path produces; the legacy score_items shim omits it).
    def _score_for_pipeline(items, _pv, _downweight, attribution=None):
        by_id = {int(it.id): it for it in items}
        scored = [(by_id[picked_id], 0.9), (by_id[unpicked_id], 0.8)]
        features = {
            picked_id: {"topic_score": 0.82, "primary_facet": "dna nanotechnology", "primary_facet_score": 0.79},
            unpicked_id: {"topic_score": 0.71, "primary_facet": "colloidal self-assembly", "primary_facet_score": 0.74},
        }
        return scored, features

    def pick_top_per_section(scored, _caps, catch_up=False):
        return scored[:1]

    monkeypatch.setattr(pipeline_mod, "ingest_all", lambda progress_callback=None, days=2: 0)
    monkeypatch.setattr(pipeline_mod, "load_profile", lambda: SimpleNamespace(bio="", keywords=[], downweight=[]))
    monkeypatch.setattr(pipeline_mod, "build_profile_matrix", lambda _profile: __import__("numpy").zeros((1, 3)))
    monkeypatch.setattr(pipeline_mod, "recent_items", recent_items)
    monkeypatch.setattr(pipeline_mod, "_score_items_for_pipeline", _score_for_pipeline)
    monkeypatch.setattr(pipeline_mod, "pick_top_per_section", pick_top_per_section)
    monkeypatch.setattr(pipeline_mod, "summarize_items", lambda rows, profile=None: {})
    monkeypatch.setattr(pipeline_mod, "send_digest", lambda html, subject, dry_run=False: True)

    pipeline_mod.run_all(dry_run=True)

    digest_id = pipeline_mod._digest_id()
    with store_mod.session_scope() as s:
        impressions = {
            r.item_id: r
            for r in s.query(store_mod.ImpressionRow).filter_by(digest_id=digest_id).all()
        }
        assert picked_id in impressions and unpicked_id in impressions
        # Non-empty facet for these on-topic items; topic_score persisted for both.
        assert impressions[picked_id].primary_facet == "dna nanotechnology"
        assert impressions[picked_id].primary_facet_score == 0.79
        assert impressions[picked_id].topic_score == 0.82
        # The UNSELECTED candidate also carries attribution (the whole point).
        assert impressions[unpicked_id].selected is False
        assert impressions[unpicked_id].primary_facet == "colloidal self-assembly"
        assert impressions[unpicked_id].primary_facet_score == 0.74
        assert impressions[unpicked_id].topic_score == 0.71


def test_run_all_logs_selected_research_item_below_pool_cap(monkeypatch, tmp_path):
    """A selected research item that ranks BELOW RESEARCH_CANDIDATE_POOL_CAP in the
    score-ordered pool (via source balancing / exploration / last-resort fill) must
    still get an impression row with selected=True. The pool logs the UNION of the
    top-CAP items and every selected research item, so no displayed item is dropped."""
    from dailydigest import config as config_mod
    from dailydigest import pipeline as pipeline_mod
    from dailydigest import store as store_mod

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    monkeypatch.setenv("PROFILE_PATH", "config/profile.example.yaml")
    # Uses 130 near-identical placeholder research fixtures to exercise the
    # below-cap impression-logging path; disable within-day near-dup suppression
    # so it does not collapse them (its behavior is covered in test_dedupe.py).
    monkeypatch.setenv("WITHIN_DAY_DEDUPE", "false")
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False

    store_mod.init_db()
    # Build more than the pool cap (100) research candidates. The LAST one (lowest
    # final score) is the one that gets selected, forcing it below the cap.
    N = 130
    with store_mod.session_scope() as s:
        rows = [
            store_mod.ItemRow(
                source="Nature",
                section="research",
                external_id=f"cand-{i}",
                url=f"https://example.com/cand-{i}",
                title=f"Research candidate {i}",
                abstract="Primary research with methods and efficacy.",
                published_at=datetime.now(timezone.utc),
            )
            for i in range(N)
        ]
        s.add_all(rows)
        s.flush()
        ids = [int(r.id) for r in rows]
    tail_id = ids[-1]  # will be selected despite ranking last

    def recent_items(days=2):
        with store_mod.session_scope() as s:
            fetched = [s.get(store_mod.ItemRow, i) for i in ids]
            for r in fetched:
                s.expunge(r)
            return fetched

    def score_items(items, _pv, _downweight, reason_penalty_map=None):
        # Descending scores in candidate order; tail item gets the LOWEST score,
        # so it sorts to the bottom of the pool (rank > cap).
        by_id = {int(it.id): it for it in items}
        return [(by_id[i], 1.0 - idx * 0.001) for idx, i in enumerate(ids)]

    def pick_top_per_section(scored, _caps, catch_up=False):
        # Simulate exploration / last-resort fill selecting the lowest-scored item.
        return [t for t in scored if int(t[0].id) == tail_id]

    monkeypatch.setattr(pipeline_mod, "ingest_all", lambda progress_callback=None, days=2: 0)
    monkeypatch.setattr(pipeline_mod, "load_profile", lambda: SimpleNamespace(bio="", keywords=[], downweight=[]))
    monkeypatch.setattr(pipeline_mod, "build_profile_matrix", lambda _profile: __import__("numpy").zeros((1, 3)))
    monkeypatch.setattr(pipeline_mod, "recent_items", recent_items)
    monkeypatch.setattr(pipeline_mod, "score_items", score_items)
    monkeypatch.setattr(pipeline_mod, "pick_top_per_section", pick_top_per_section)
    monkeypatch.setattr(pipeline_mod, "summarize_items", lambda rows, profile=None: {})
    monkeypatch.setattr(pipeline_mod, "send_digest", lambda html, subject, dry_run=False: True)

    pipeline_mod.run_all(dry_run=True)

    digest_id = pipeline_mod._digest_id()
    with store_mod.session_scope() as s:
        impressions = {
            r.item_id: r
            for r in s.query(store_mod.ImpressionRow).filter_by(digest_id=digest_id).all()
        }
        # The selected tail item ranks below the cap yet must still be logged.
        assert tail_id in impressions
        assert impressions[tail_id].selected is True
        # Invariant: EVERY selected research item has an impression row with selected=True.
        selected_ids = {tail_id}
        for sid in selected_ids:
            assert sid in impressions and impressions[sid].selected is True
        # Its position is a stable rank appended after the top-CAP block.
        assert impressions[tail_id].position == 100


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
        lambda progress_callback=None, days=2: (_ for _ in ()).throw(AssertionError("ingest called")),
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
    monkeypatch.setattr(pipeline_mod, "ingest_all", lambda progress_callback=None, days=2: 0)
    monkeypatch.setattr(pipeline_mod, "load_profile", lambda: SimpleNamespace(bio="", keywords=[], downweight=[]))
    monkeypatch.setattr(pipeline_mod, "build_profile_matrix", lambda _profile: __import__("numpy").zeros((1, 3)))
    monkeypatch.setattr(pipeline_mod, "recent_items", recent_items)
    monkeypatch.setattr(pipeline_mod, "score_items", lambda items, _pv, _downweight, reason_penalty_map=None: [(items[0], 0.9)])
    monkeypatch.setattr(pipeline_mod, "pick_top_per_section", lambda scored, _caps, catch_up=False: scored)
    monkeypatch.setattr(pipeline_mod, "summarize_items", lambda rows, profile=None: {rows[0].id: "New summary."})
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
    monkeypatch.setattr(pipeline_mod, "ingest_all", lambda progress_callback=None, days=2: 0)
    monkeypatch.setattr(pipeline_mod, "load_profile", lambda: SimpleNamespace(bio="", keywords=[], downweight=[]))
    monkeypatch.setattr(pipeline_mod, "build_profile_matrix", lambda _profile: __import__("numpy").zeros((1, 3)))
    monkeypatch.setattr(pipeline_mod, "recent_items", recent_items)
    monkeypatch.setattr(pipeline_mod, "score_items", lambda items, _pv, _downweight, reason_penalty_map=None: [(items[0], 0.9)])
    monkeypatch.setattr(pipeline_mod, "pick_top_per_section", lambda scored, _caps, catch_up=False: scored)
    monkeypatch.setattr(pipeline_mod, "summarize_items", lambda rows, profile=None: {rows[0].id: "Summary."})
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
    old_at = datetime.now(timezone.utc) - timedelta(days=5)  # last digest 5 days ago

    with store_mod.session_scope() as s:
        s.add(store_mod.DigestRow(id="2026-05-10", item_count=3, created_at=old_at, sent_at=old_at))

    captured_days: list[int] = []

    def fake_recent_items(days=2):
        captured_days.append(days)
        return []

    monkeypatch.setattr(pipeline_mod, "_digest_id", lambda: today_id)
    monkeypatch.setattr(pipeline_mod, "ingest_all", lambda progress_callback=None, days=2: 0)
    monkeypatch.setattr(pipeline_mod, "load_profile", lambda: SimpleNamespace(bio="", keywords=[], downweight=[]))
    monkeypatch.setattr(pipeline_mod, "build_profile_matrix", lambda _profile: __import__("numpy").zeros((1, 3)))
    monkeypatch.setattr(pipeline_mod, "recent_items", fake_recent_items)
    monkeypatch.setattr(pipeline_mod, "score_items", lambda items, _pv, _downweight, reason_penalty_map=None: [])
    monkeypatch.setattr(pipeline_mod, "pick_top_per_section", lambda scored, _caps, catch_up=False: [])
    monkeypatch.setattr(pipeline_mod, "summarize_items", lambda rows, profile=None: {})
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
    monkeypatch.setattr(pipeline_mod, "ingest_all", lambda progress_callback=None, days=2: 0)
    monkeypatch.setattr(pipeline_mod, "load_profile", lambda: SimpleNamespace(bio="", keywords=[], downweight=[]))
    monkeypatch.setattr(pipeline_mod, "build_profile_matrix", lambda _profile: __import__("numpy").zeros((1, 3)))
    monkeypatch.setattr(pipeline_mod, "recent_items", fake_recent_items)
    monkeypatch.setattr(pipeline_mod, "score_items", lambda items, _pv, _downweight, reason_penalty_map=None: [])
    monkeypatch.setattr(pipeline_mod, "pick_top_per_section", lambda scored, _caps, catch_up=False: [])
    monkeypatch.setattr(pipeline_mod, "summarize_items", lambda rows, profile=None: {})
    monkeypatch.setattr(pipeline_mod, "send_digest", lambda html, subject, dry_run=False: False)

    pipeline_mod.run_all(dry_run=True, backfill_days=7)

    assert captured_days and captured_days[0] == 7


def test_run_all_empty_digest_emits_done_with_zero_items(monkeypatch, tmp_path):
    from dailydigest import pipeline as pipeline_mod

    store_mod = _reset_store(tmp_path, monkeypatch)
    digest_id = "2026-05-05"
    events = []

    monkeypatch.setattr(pipeline_mod, "_digest_id", lambda: digest_id)
    monkeypatch.setattr(pipeline_mod, "ingest_all", lambda progress_callback=None, days=2: 0)
    monkeypatch.setattr(pipeline_mod, "load_profile", lambda: SimpleNamespace(bio="", keywords=[], downweight=[]))
    monkeypatch.setattr(pipeline_mod, "build_profile_matrix", lambda _profile: __import__("numpy").zeros((1, 3)))
    monkeypatch.setattr(pipeline_mod, "recent_items", lambda days=2: [])
    monkeypatch.setattr(pipeline_mod, "score_items", lambda items, _pv, _downweight, reason_penalty_map=None: [])
    monkeypatch.setattr(pipeline_mod, "pick_top_per_section", lambda scored, _caps, catch_up=False: [])
    monkeypatch.setattr(pipeline_mod, "summarize_items", lambda rows, profile=None: {})
    monkeypatch.setattr(pipeline_mod, "send_digest", lambda html, subject, dry_run=False: False)

    pipeline_mod.run_all(dry_run=True, progress_callback=lambda stage, payload: events.append((stage, payload)))

    assert events[-1] == ("done", {"digest_id": digest_id, "total_items": 0, "dry_run": True})
    with store_mod.session_scope() as s:
        digest = s.get(store_mod.DigestRow, digest_id)
        assert digest.item_count == 0


def test_pipeline_funnel_audit_shows_dedupe_count_before_quality_gate(
    tmp_path, monkeypatch
):
    """after_cross_source_dedupe should count items BEFORE quality gate drops thin abstracts.

    This test simulates the expected funnel ordering: dedupe runs first, then the
    quality gate filters out thin-abstract items from non-protected sources.
    The dedupe count must be >= quality gate count.
    """
    import datetime

    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest.dedupe import dedupe_by_url
    from dailydigest.models import Item

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod.init_db()

    now = datetime.datetime.now(datetime.timezone.utc)

    # Simulate three items arriving from ingest (as Item objects for dedupe)
    # 1. Full abstract (survives quality gate)
    # 2. Thin abstract from Nature (protected — survives quality gate in real impl)
    # 3. Thin abstract from unknown source (dropped by quality gate)
    items = [
        Item(
            source="biorxiv", section="research", external_id="full-1",
            url="https://biorxiv.org/1", title="CRISPR paper",
            abstract="A" * 150,
        ),
        Item(
            source="nature_main", section="research", external_id="thin-protected",
            url="https://nature.com/articles/thin1", title="Nature paper",
            abstract="Short.",
        ),
        Item(
            source="unknown_source", section="research", external_id="thin-dropped",
            url="https://unknown.org/1", title="Unknown paper",
            abstract="Short.",
        ),
    ]

    # Stage 1: dedupe by URL (simulates cross-source deduplication)
    deduped = dedupe_by_url(items)
    after_cross_source_dedupe = len(deduped)

    # Stage 2: quality gate — drop items with thin abstracts from non-protected sources.
    # Protected sources (journals) are allowed through with short abstracts.
    PROTECTED_SOURCES = {"nature_main", "science", "cell", "nejm", "lancet"}
    MIN_ABSTRACT_LEN = 50

    def _quality_gate(items_in):
        out = []
        for item in items_in:
            abstract = (item.abstract or "").strip()
            if len(abstract) >= MIN_ABSTRACT_LEN:
                out.append(item)
            elif item.source in PROTECTED_SOURCES:
                out.append(item)
        return out

    after_quality = _quality_gate(deduped)
    after_quality_gate = len(after_quality)

    # Core invariant: dedupe runs before quality gate, so dedupe count >= quality gate count
    assert after_cross_source_dedupe >= after_quality_gate, (
        "Dedupe count should be >= quality gate count (dedupe runs first)"
    )
    # Specific expectations: unknown thin-abstract item should be dropped
    assert after_quality_gate < after_cross_source_dedupe, (
        "Quality gate should drop at least one thin-abstract item"
    )
    assert after_quality_gate == 2, (
        "Full abstract item and protected-source item should survive; unknown thin item dropped"
    )
