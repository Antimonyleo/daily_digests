"""Offline regressions for run identity, bounded summaries and publication state."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from dailydigest import config, pipeline, store, summarize


def _items(count):
    return [store.ItemRow(id=i, title=f"Method {i}", abstract="A measured result.",
                          source="Nature", section="research") for i in range(1, count + 1)]


def _settings(**changes):
    settings = deepcopy(config.get_settings())
    for name, value in changes.items():
        setattr(settings, name, value)
    return settings


def test_summary_uses_one_provider_identity_across_batches(monkeypatch):
    settings = _settings(llm_backend="api", llm_api_key="first-key", llm_model="first-model",
                         llm_base_url="https://first.example/v1")
    calls = []

    def call(batch, *, settings, timeout):
        calls.append((settings.llm_model, settings.llm_api_key, settings.llm_base_url))
        monkeypatch.setattr(config, "get_settings", lambda: _settings_after)
        return {row.id: "provider summary" for row in batch}

    _settings_after = _settings(llm_model="second-model", llm_api_key="second-key")
    monkeypatch.setattr(summarize, "_call_llm", call)
    result, provenance = summarize.summarize_items_with_provenance(_items(7), settings=settings)
    assert len(result) == 7
    assert calls == [("first-model", "first-key", "https://first.example/v1")] * 2
    assert set(provenance.values()) == {"api"}
    assert summarize.summary_signature(_items(1)[0], settings=settings) != (
        summarize.summary_signature(_items(1)[0], settings=_settings_after)
    )


@pytest.mark.parametrize("backend,method", [("api", "_call_llm"), ("anthropic", "_call_anthropic")])
def test_http_summary_budget_stops_batches_and_preserves_fallback(monkeypatch, backend, method):
    settings = _settings(llm_backend=backend, llm_api_key="configured",
                         llm_summary_budget_seconds=10)
    now = [0.0]
    timeouts = []

    def call(batch, *, settings, timeout):
        timeouts.append(timeout)
        now[0] += 7
        return {row.id: "provider summary" for row in batch}

    monkeypatch.setattr(summarize.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(summarize, method, call)
    result, provenance = summarize.summarize_items_with_provenance(_items(13), settings=settings)
    assert timeouts == [10, 3]
    assert provenance[12] == backend
    assert provenance[13] == "extractive_fallback"
    assert result[13].startswith("Key finding:")


def test_http_summary_circuit_breaker_stops_repeated_failures(monkeypatch):
    settings = _settings(llm_backend="api", llm_api_key="configured",
                         llm_summary_max_failures=2)
    calls = []

    def fail(batch, **kwargs):
        calls.append(batch)
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(summarize, "_call_llm", fail)
    result, provenance = summarize.summarize_items_with_provenance(_items(19), settings=settings)
    assert len(calls) == 2
    assert len(result) == 19
    assert set(provenance.values()) == {"extractive_fallback"}


def test_http_summary_circuit_breaker_counts_empty_results(monkeypatch):
    settings = _settings(llm_backend="api", llm_api_key="configured",
                         llm_summary_max_failures=1)
    calls = []
    monkeypatch.setattr(summarize, "_call_llm", lambda batch, **kwargs:
                        (calls.append(batch) or {}))
    diagnostics = {}
    result, provenance = summarize.summarize_items_with_provenance(
        _items(7), settings=settings, diagnostics=diagnostics,
    )
    assert len(calls) == 1
    assert len(result) == 7
    assert set(provenance.values()) == {"extractive_fallback"}
    assert diagnostics == {"provider_calls": 1, "provider_stopped": True, "fallback_items": 7}


def test_exhausted_catch_up_budget_makes_no_provider_call(monkeypatch):
    monkeypatch.setattr(summarize, "_llm_text", lambda *args, **kwargs:
                        pytest.fail("exhausted budget contacted a provider"))
    assert summarize.synthesize_catch_up(_items(1), 7, timeout=0) == ""


@pytest.fixture
def offline_brew(monkeypatch):
    store.init_db()
    digest_id = "2026-10-10"
    with store.session_scope() as session:
        session.add(store.DigestRow(id=digest_id, item_count=7))
    monkeypatch.setattr(pipeline, "_digest_id", lambda: digest_id)
    monkeypatch.setattr(pipeline, "ingest_all", lambda **kwargs: 0)
    monkeypatch.setattr(pipeline, "load_profile", lambda: SimpleNamespace(
        bio="Frozen reader", keywords=[], downweight=[]))
    monkeypatch.setattr(pipeline, "_build_profile_with_rocchio", None)
    monkeypatch.setattr(pipeline, "build_profile_matrix", lambda profile: np.zeros((1, 3)))
    monkeypatch.setattr(pipeline, "_build_neg_centroid", None)
    monkeypatch.setattr(pipeline, "recent_items", lambda **kwargs: [])
    monkeypatch.setattr(pipeline, "score_items", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline, "pick_top_per_section", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline, "summarize_items", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline, "render_digest", lambda *args, **kwargs: "<html>ready</html>")
    monkeypatch.setattr(pipeline, "send_digest", lambda *args, **kwargs: False)
    from dailydigest.rank import calibrate, profile
    monkeypatch.setattr(calibrate, "fit_calibrator", lambda: None)
    monkeypatch.setattr(profile, "build_attribution_context", lambda profile: None)
    return digest_id


@pytest.mark.parametrize("operation", ["render_digest", "send_digest"])
def test_post_publication_failure_reports_ready_digest(monkeypatch, offline_brew, operation):
    def fail(*args, **kwargs):
        raise OSError("preview disk unavailable")

    monkeypatch.setattr(pipeline, operation, fail)
    events = []
    assert pipeline.run_all(dry_run=True, progress_callback=lambda stage, payload:
                            events.append((stage, payload))) == offline_brew
    assert any(stage == "published" for stage, _ in events)
    assert events[-1][0] == "done"
    assert events[-1][1]["warnings"]
    with store.session_scope() as session:
        assert session.get(store.DigestRow, offline_brew).item_count == 0


def test_pre_publication_failure_preserves_existing_digest(monkeypatch, offline_brew):
    def fail(**kwargs):
        raise RuntimeError("Research source coverage is degraded")

    monkeypatch.setattr(pipeline, "ingest_all", fail)
    events = []
    with pytest.raises(RuntimeError, match="degraded"):
        pipeline.run_all(dry_run=True, progress_callback=lambda stage, payload:
                         events.append((stage, payload)))
    assert not any(stage in {"published", "done"} for stage, _ in events)
    with store.session_scope() as session:
        assert session.get(store.DigestRow, offline_brew).item_count == 7


def test_cancel_before_publication_preserves_existing_digest(monkeypatch, offline_brew):
    cancelled = [False]

    def progress(stage, payload):
        if stage == "summarize_done":
            cancelled[0] = True

    with pytest.raises(pipeline.BrewCancelled):
        pipeline.run_all(dry_run=True, progress_callback=progress,
                         should_cancel=lambda: cancelled[0])
    with store.session_scope() as session:
        assert session.get(store.DigestRow, offline_brew).item_count == 7


def test_cancel_after_publication_does_not_report_rollback(offline_brew):
    cancelled = [False]
    events = []

    def progress(stage, payload):
        events.append((stage, payload))
        if stage == "published":
            cancelled[0] = True

    assert pipeline.run_all(dry_run=True, progress_callback=progress,
                            should_cancel=lambda: cancelled[0]) == offline_brew
    assert events[-1][0] == "done"


def test_pipeline_forwards_stop_callback_to_summary_batches(monkeypatch, offline_brew):
    stopped = [False]

    def generate(items, profile=None, *, settings=None, diagnostics=None, should_cancel=None):
        assert callable(should_cancel), "The summary stage must receive the brew's Stop callback"
        stopped[0] = True
        assert should_cancel()
        raise pipeline.BrewCancelled("Stopped between summary batches")

    monkeypatch.setattr(pipeline, 'summarize_items', pipeline._ORIGINAL_SUMMARIZE_ITEMS)
    monkeypatch.setattr(pipeline, 'summarize_items_with_provenance', generate)
    with pytest.raises(pipeline.BrewCancelled, match='between summary batches'):
        pipeline.run_all(dry_run=True, should_cancel=lambda: stopped[0])
    with store.session_scope() as session:
        assert session.get(store.DigestRow, offline_brew).item_count == 7


def test_progress_names_encoding_before_it_runs(monkeypatch, offline_brew):
    events = []

    def encode(profile):
        assert events[-1] == ("stage_start", {"name": "profile_encoding"})
        return np.zeros((1, 3))

    monkeypatch.setattr(pipeline, "build_profile_matrix", encode)
    pipeline.run_all(dry_run=True, progress_callback=lambda stage, payload:
                     events.append((stage, payload)))
    durations = {payload["name"]: payload["duration_ms"] for stage, payload in events
                 if stage == "stage_done"}
    assert {"ingestion", "profile_encoding", "deduplication", "ranking", "summary",
            "publication", "render"} <= durations.keys()
    assert all(value >= 0 for value in durations.values())


def test_pipeline_pins_profile_and_summary_settings_before_ingestion(monkeypatch, offline_brew):
    settings = config.get_settings()
    settings.llm_backend = "extractive"
    settings.llm_model = "before"
    profile = SimpleNamespace(bio="Before", keywords=[], downweight=[])
    monkeypatch.setattr(pipeline, "load_profile", lambda: profile)

    def ingest(**kwargs):
        settings.llm_model = "after"
        profile.bio = "After"
        return 0

    def summaries(rows, profile=None, *, settings=None, diagnostics=None):
        assert profile.bio == "Before"
        assert settings.llm_model == "before"
        return {}, {}

    monkeypatch.setattr(pipeline, "ingest_all", ingest)
    monkeypatch.setattr(pipeline, "summarize_items", pipeline._ORIGINAL_SUMMARIZE_ITEMS)
    monkeypatch.setattr(pipeline, "summarize_items_with_provenance", summaries)
    pipeline.run_all(dry_run=True)


def test_catch_up_stop_is_not_swallowed_as_optional_provider_failure(monkeypatch):
    from dailydigest.brew import BrewCancelled

    def stopped(*args, should_cancel=None, **kwargs):
        assert should_cancel is not None and should_cancel()
        raise BrewCancelled('Catch-up stopped')

    monkeypatch.setattr(summarize, '_llm_text', stopped)
    with pytest.raises(BrewCancelled, match='Catch-up stopped'):
        summarize.synthesize_catch_up(_items(1), 7, should_cancel=lambda: True)


def test_pipeline_forwards_stop_to_catch_up_and_preserves_previous_digest(monkeypatch, offline_brew):
    stop = [False]

    def catch_up(rows, days, *, should_cancel=None, **kwargs):
        assert should_cancel is not None, 'Catch-up omitted the Stop callback'
        stop[0] = True
        assert should_cancel()
        raise pipeline.BrewCancelled('Active catch-up stopped')

    monkeypatch.setattr(pipeline, 'synthesize_catch_up', catch_up)
    with pytest.raises(pipeline.BrewCancelled, match='Active catch-up stopped'):
        pipeline.run_all(dry_run=True, backfill_days=7, should_cancel=lambda: stop[0])
    with store.session_scope() as session:
        assert session.get(store.DigestRow, offline_brew).item_count == 7
