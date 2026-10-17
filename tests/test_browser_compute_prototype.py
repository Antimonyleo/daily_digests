"""A portable fixture experiment must retain the actual Python quality policy."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def prototype():
    path = Path(__file__).resolve().parents[1] / "prototypes/browser-compute/prepare.py"
    spec = importlib.util.spec_from_file_location("browser_prototype_prepare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def policy_fixture():
    paper = {"title": "Programmable DNA origami hinges establish molecular design rules",
             "abstract": "We introduce a transferable engineering method and measure designed molecular assembly geometry.",
             "authors": "Alex Example", "metadata": {}, "synthetic_scope": "core"}
    rows = [{**paper, "id": index, "source": source, "external_id": str(index),
             "url": f"https://example.invalid/{index}"}
            for index, source in enumerate(["Nature", "Nature Methods", "Science", "ACS Nano", "Unlisted Journal", "Nature", "Nature"], 1)]
    rows[4]["metadata"] = {"doi": "10.9999/prototype.low"}
    rows[5]["metadata"] = {"publication_types": ["Editorial"]}
    scores = [.86, .86, .86, .86, .86, .86, .2]
    vectors = {"documents": [[score, float(np.sqrt(1 - score**2))] for score in scores], "queries": [[1., 0.]]}
    return {"papers": rows, "queries": ["DNA nanotechnology"], "query_weights": [1.], "cap": 4}, vectors


def test_browser_bundle_reuses_quality_and_selection_with_native_parity(prototype):
    bundle = prototype.build_bundle()
    kernel = prototype._kernel()
    fixture, vectors = policy_fixture()
    native = prototype.native_rank(fixture, vectors, bundle)
    portable = kernel.rank_fixture(fixture, vectors, kernel.install_bundle(bundle))
    assert portable == native
    assert portable["selected_ids"] == [1, 2, 3, 4]
    by_id = {row["id"]: row for row in portable["candidates"]}
    assert by_id[5]["final_score"] < by_id[1]["final_score"]
    assert by_id[5]["status"] == "budget_or_source_policy"
    assert by_id[6]["status"] == "quality"
    assert by_id[7]["status"] == "topic"


def test_small_pool_allows_one_strong_unknown_match_and_rejects_editorial_items(prototype):
    fixture, vectors = policy_fixture()
    fixture["papers"] = fixture["papers"][4:]
    vectors["documents"] = vectors["documents"][4:]
    kernel = prototype._kernel()
    result = kernel.rank_fixture(fixture, vectors, kernel.install_bundle(prototype.build_bundle()))
    assert result["selected_ids"] == [5]
    by_id = {row['id']: row for row in result['candidates']}
    assert by_id[5]['source_bucket'] == 'unknown_journal'
    assert by_id[6]['status'] == 'quality'
    assert by_id[7]['status'] == 'topic'


def test_portable_version_identity_unicode_and_context_cases_match_native(prototype):
    bundle = prototype.build_bundle()
    kernel = prototype._kernel()
    modules = kernel.install_bundle(bundle)
    results = []
    for case in prototype.build_policy_cases():
        native = prototype.native_rank(case["fixture"], case["vectors"], bundle)
        portable = kernel.rank_fixture(case["fixture"], case["vectors"], modules)
        assert portable == native
        results.append(portable)
    assert results[0]["selected_ids"] == [1, 2, 3, 4]
    assert results[1]["selected_ids"] == [5]
    assert len({1, 2}.intersection(results[2]["selected_ids"])) == 1
    assert set(results[3]["selected_ids"]) == {1, 2}
    assert results[4]["selected_ids"] == [1]


def test_browser_fixture_contains_no_user_metadata_or_reference_labels_used_for_scoring(prototype):
    fixture = prototype.build_fixture()
    assert "Fictional" in fixture["provenance"]
    assert len(fixture["papers"]) == 76
    assert {"core", "context", "off_interest", "uncertain"} == {row["synthetic_scope"] for row in fixture["papers"]}
    assert len({row["id"] for row in fixture["papers"]}) == len(fixture["papers"])
    assert fixture["query_weights"][-2:] == [.45, .45]
    bundle = prototype.build_bundle()
    assert set(bundle["settings"]) == set(prototype.SETTINGS)
    assert not any("key" in key or "path" in key for key in bundle["settings"])


@pytest.mark.parametrize("bad", ["missing_document", "zero", "nan", "dimension"])
def test_invalid_embedding_results_cannot_produce_a_plausible_digest(prototype, bad):
    fixture, vectors = policy_fixture()
    if bad == "missing_document":
        vectors["documents"].pop()
    elif bad == "zero":
        vectors["documents"][0] = [0., 0.]
    elif bad == "nan":
        vectors["documents"][0] = [float("nan"), 0.]
    else:
        vectors["queries"] = [[1., 0., 0.]]
    kernel = prototype._kernel()
    with pytest.raises(ValueError):
        kernel.rank_fixture(fixture, vectors, kernel.install_bundle(prototype.build_bundle()))
