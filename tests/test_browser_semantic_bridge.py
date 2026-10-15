"""The browser topic matcher crosses JSPI without native downloads or partial results."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runtime():
    spec = importlib.util.spec_from_file_location('semantic_bridge_runtime', Path(__file__).parents[1] / 'browser/runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bridge_installs_lazily_and_preserves_window_metadata(runtime):
    calls = []
    records = [{'query': 'plant evolution', 'title': 'Genome evolution', 'abstract': 'Conserved synteny'}]
    results = [{'logit': 2.5, 'windows': 2, 'complete': False}]
    alignment = SimpleNamespace()
    def score(payload):
        calls.append(json.loads(payload))
        return json.dumps(results)
    runtime.install_semantic_backend(alignment, score, lambda result: result)
    backend = alignment._build_backend()
    assert calls == []
    assert backend.score_records(records) == results
    assert calls == [records]


def test_bridge_propagates_scoring_failure_without_fallback(runtime):
    alignment = SimpleNamespace()
    def failed(payload):
        raise RuntimeError('Missing local semantic model')
    runtime.install_semantic_backend(alignment, failed, lambda result: result)
    with pytest.raises(RuntimeError, match='Missing local semantic model'):
        alignment._build_backend().score_records([{'query': 'test'}])


@pytest.mark.parametrize('when', ['before', 'during', 'after'])
def test_bridge_translates_cancellation_even_after_the_final_result(runtime, monkeypatch, when):
    from dailydigest.pipeline import BrewCancelled

    cancelled = when == 'before'
    calls = []
    monkeypatch.setattr(runtime, 'browser_cancelled', lambda: cancelled)
    def score(payload):
        nonlocal cancelled
        calls.append(payload)
        cancelled = True
        if when == 'during':
            raise RuntimeError('Brew stopped during WASM')
        return '[]'
    alignment = SimpleNamespace()
    runtime.install_semantic_backend(alignment, score, lambda result: result)
    with pytest.raises(BrewCancelled):
        alignment._build_backend().score_records([])
    assert len(calls) == (0 if when == 'before' else 1)


def test_bridge_honors_native_cancellation_callback(runtime):
    alignment = SimpleNamespace()
    calls = []
    runtime.install_semantic_backend(alignment, lambda payload: calls.append(payload), lambda result: result)
    def cancelled():
        raise RuntimeError('native cancellation requested')
    with pytest.raises(RuntimeError, match='native cancellation requested'):
        alignment._build_backend().score_records([], check_cancel=cancelled)
    assert calls == []


def test_real_alignment_factory_signature_and_identity_reuse_cached_pairs(runtime, monkeypatch):
    from collections import OrderedDict
    from dailydigest.rank import semantic_alignment

    # Restore the native factory after this browser adapter test.
    monkeypatch.setattr(semantic_alignment, '_build_backend', semantic_alignment._build_backend)
    monkeypatch.setattr(semantic_alignment, '_PAIR_CACHE', OrderedDict())
    calls = []
    def score(payload):
        records = json.loads(payload)
        calls.append(records)
        return json.dumps([{'logit': 5., 'windows': 1, 'complete': True} for _ in records])
    runtime.install_semantic_backend(semantic_alignment, score, lambda result: result)
    profile = SimpleNamespace(keywords=['plant evolution'], canonical_facets={})
    article = SimpleNamespace(id=1, title='Plant genome evolution', section='research',
                              abstract='We compare genome assemblies and conserved synteny across flowering plants. ' * 2)
    backend = semantic_alignment._build_backend(None)
    assert semantic_alignment._build_backend('browser-provisioned-model') is backend
    for _ in range(2):
        features = {1: {'topic_score': .7}}
        result = semantic_alignment.apply_topic_alignment([(article, .6)], features, profile)
        assert result[0][1] == .6  # Topic verification preserves quality ordering.
        assert features[1]['topic_alignment']['logit'] == 5.
    assert len(calls) == 1
