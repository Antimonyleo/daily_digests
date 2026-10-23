"""Task-alignment accounting uses frozen metadata, not model downloads."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dailydigest.rank.cross_field_evaluation import evaluate_alignment, load_fixture

FIXTURE = Path(__file__).parent / "fixtures/cross_field_alignment.json"


def test_frozen_corpus_has_cross_field_provenance_and_no_exact_paper_split_leakage():
    corpus = load_fixture(FIXTURE)
    assert 80 <= len(corpus["cases"]) <= 120
    assert 6 <= len(corpus["profiles"]) <= 8
    assert {case["label"] for case in corpus["cases"]} == {"aligned", "off_task", "uncertain"}
    assert all(case["review_status"] == "provisional" for case in corpus["cases"])
    for profile in corpus["profiles"]:
        assert profile["intent"]
        assert profile["pool_provenance"]
        pool = {item["url"]: item for item in profile["pool"]}
        for case in corpus["cases"]:
            if case["profile_id"] == profile["id"]:
                assert pool[case["url"]]["title"]
                assert "abstract" in pool[case["url"]]
                assert "doi" in case


def example():
    corpus = load_fixture(FIXTURE)
    corpus["cases"] = [dict(corpus["cases"][0], id=str(index), url=str(index),
                            label=label, split="heldout")
                       for index, label in enumerate(["aligned", "aligned", "off_task", "uncertain"])]
    pid = corpus["cases"][0]["profile_id"]
    decisions = [{"profile_id": pid, "url": str(index), "eligible": index != 1,
                  "selected": index in {0, 2, 3}} for index in range(4)]
    return corpus, decisions


def test_retrieval_eligibility_selection_and_unknowns_remain_distinct():
    corpus, decisions = example()
    result = evaluate_alignment(corpus, decisions)["splits"]["heldout"]["all"]
    assert result["labels"] == {"aligned": 2, "off_task": 1, "uncertain": 1}
    assert result["stages"]["retrieved"]["retained_aligned_pool_fraction"] == 1
    selected = result["stages"]["selected"]
    assert selected["precision_on_decisive_labels"] == .5
    assert selected["retained_aligned_pool_fraction"] == .5
    assert selected["uncertain"] == 1
    assert "retrieval_recall" not in result


def test_missing_and_impossible_decisions_fail_instead_of_inventing_success():
    corpus, decisions = example()
    with pytest.raises(ValueError, match="Missing decision"):
        evaluate_alignment(corpus, decisions[:-1])
    with pytest.raises(ValueError, match="Duplicate decision"):
        evaluate_alignment(corpus, decisions + decisions[:1])
    decisions[1]["selected"] = True
    with pytest.raises(ValueError, match="must be eligible"):
        evaluate_alignment(corpus, decisions)


def test_empty_selected_denominator_is_unknown():
    corpus, decisions = example()
    for decision in decisions:
        decision["selected"] = False
    selected = evaluate_alignment(corpus, decisions)["splits"]["heldout"]["all"]["stages"]["selected"]
    assert selected["precision_on_decisive_labels"] is None
    assert selected["retained_aligned_pool_fraction"] == 0


def test_loader_rejects_paper_split_leakage(tmp_path):
    corpus = load_fixture(FIXTURE)
    duplicate = deepcopy(corpus["cases"][0])
    duplicate.update(id="different_profile_same_paper", split=(
        "dev" if duplicate["split"] == "heldout" else "heldout"))
    corpus["cases"].append(duplicate)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(corpus))
    with pytest.raises(ValueError, match="leaks across splits"):
        load_fixture(path)


def test_source_configuration_survives_replay_working_directory_change(monkeypatch, tmp_path):
    from dailydigest.rank.cross_field_evaluation import _source_configuration

    expected = _source_configuration()
    monkeypatch.chdir(tmp_path)
    assert _source_configuration() == expected
    assert expected.is_file()


def test_older_embedding_api_never_claims_observed_backend(monkeypatch):
    from dailydigest.rank import embed
    from dailydigest.rank.cross_field_evaluation import _embedding_provenance

    monkeypatch.delattr(embed, "active_embedding_provenance", raising=False)
    provenance = _embedding_provenance()
    assert provenance["actual_backend"] is None
    assert provenance["status"] == "requested_not_independently_verified"
    assert provenance["document_signature"]
    assert provenance["query_signature"]


def test_replay_selection_applies_actual_duplicate_and_dynamic_size_rules(monkeypatch):
    import numpy as np
    from dailydigest import pipeline
    from dailydigest.config import Settings
    from dailydigest.rank import embedding_cache
    from dailydigest.rank.cross_field_evaluation import _select_captured_research

    settings = Settings(_env_file=None, adaptive_section_sizes=True, top_research=12,
                        min_research=0, min_topic_relevance=.65, within_day_dedupe=True,
                        within_day_dedupe_threshold=.86)
    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    monkeypatch.setattr(pipeline, "recent_viewed_facet_dates", lambda **kwargs: {})
    rows = [SimpleNamespace(id=i, section="research", source="Nature", title=f"Paper {i}",
                            abstract="Measured evidence.", metadata_json="{}") for i in (1, 2, 3)]
    vectors = {1: [1., 0.], 2: [1., 0.], 3: [0., 1.]}
    monkeypatch.setattr(embedding_cache, "embed_item_rows", lambda items: np.asarray(
        [vectors[item.id] for item in items]))
    features = {i: {"topic_score": .9} for i in (1, 2, 3)}
    candidates = [(rows[0], .9), (rows[1], .8), (rows[2], .7)]
    eligible, picked, trace = _select_captured_research(
        candidates, features, settings=settings, digest_id="2026-10-15")
    assert [item.id for item, _ in eligible] == [1, 3]
    assert {item.id for item, _ in picked} == {1, 3}
    assert trace["section_budgets"]["research"] == 2
    assert trace["within_day_dropped_ids"] == [2]
    assert trace["reading_mode"] == "full"
    assert trace["effective_settings"]["top_research"] == 12
    assert "db_path" not in trace["effective_settings"]


def test_fixture_split_policy_acknowledges_reused_development_evidence():
    corpus = load_fixture(FIXTURE)
    assert "policy was revised" in corpus["split_policy"]
    assert "not a fresh" in corpus["split_policy"]
    assert "No tuning on either split" not in corpus["split_policy"]


@pytest.mark.parametrize("field,value", [("negative_interests", {"unwanted": 1.}),
                                         ("authors_of_interest", ["Watched Author"])])
def test_replay_rejects_unsupported_personalization_before_creating_state(tmp_path, field, value):
    from dailydigest.rank.cross_field_evaluation import replay_offline

    corpus = load_fixture(FIXTURE)
    corpus["profiles"][0]["profile"][field] = value
    state = tmp_path / "fresh-state"
    with pytest.raises(ValueError, match="without negative interests or watched authors"):
        replay_offline(corpus, state=state, model_cache=tmp_path)
    assert not state.exists()


def test_profile_database_does_not_reuse_previous_metadata_or_source_variants(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from dailydigest import config, store
    from dailydigest.models import Item
    from dailydigest.rank.cross_field_evaluation import _isolated_profile_store

    settings = config.Settings(_env_file=None, db_path=str(tmp_path / 'unused.db'))
    monkeypatch.setattr(config, 'get_settings', lambda: settings)
    original = Item(source='openalex', section='research', external_id='shared',
                    url='https://doi.org/10.1234/shared', title='Shared work',
                    abstract='Earlier, much richer metadata. ' * 15,
                    published_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    incoming = original.model_copy(update={
        'abstract': 'Current captured abstract.',
        'published_at': datetime(2026, 2, 1, tzinfo=timezone.utc)})
    with _isolated_profile_store(tmp_path, 0, settings):
        store.upsert_items([original, original.model_copy(update={'source': 'Nature'})])
        assert len(store.recent_items(days=36500)) == 2
    assert store._ENGINE is None and store._SessionLocal is None
    with _isolated_profile_store(tmp_path, 1, settings):
        assert store.recent_items(days=36500) == []
        store.upsert_items([incoming])
        rows = store.recent_items(days=36500)
        assert len(rows) == 1
        assert rows[0].source == incoming.source
        assert rows[0].abstract == incoming.abstract
        assert rows[0].published_at.date() == incoming.published_at.date()
    assert settings.db_path == str(tmp_path / 'unused.db')
    assert store._ENGINE is None and store._SessionLocal is None


def test_claims_representation_is_scoped_to_scoring_not_deduplication(monkeypatch):
    import numpy as np
    from dailydigest import pipeline
    from dailydigest.models import Profile
    from dailydigest.rank import embedding_cache, profile as profile_module
    from dailydigest.rank.cross_field_evaluation import _score_captured_research

    row = SimpleNamespace(title='Shared topic', abstract=(
        'Background describes related systems. We introduce a reusable analysis tool. '
        'Previous work cannot measure this quantity.'))
    original_text = embedding_cache.item_text(row)
    observed = []
    monkeypatch.setattr(profile_module, 'build_profile_matrix', lambda *a, **kw: np.ones((1, 2)))
    monkeypatch.setattr(profile_module, 'build_attribution_context', lambda *a: {})
    def score(rows, *args):
        observed.extend(embedding_cache.item_text(item) for item in rows)
        return [(row, .9)], {}
    monkeypatch.setattr(pipeline, '_score_items_for_pipeline', score)
    _score_captured_research([row], Profile(bio='', keywords=['analysis']),
                             profile_matrix='legacy', paper_text='claims')
    assert observed == ['Shared topic. We introduce a reusable analysis tool.']
    # The subsequent serving dedupe path calls this same function after scoring.
    assert embedding_cache.item_text(row) == original_text
