"""Validate the offline fixture contract without calling an embedding model.

These checks do not establish that synthetic labels reflect a reader's judgments.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path

import pytest


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "research_scope_benchmark.json"
FACETS = {
    "DNA nanotechnology",
    "RNA nanotechnology",
    "Colloidal self-assembly",
    "Protein engineering and design",
    "Protein structure and function prediction",
    "Machine learning for biomolecular design",
    "Gene and RNA therapeutic engineering",
    "Self-assembled functional materials",
}
LABELS = {"core", "context", "off_interest", "uncertain"}


@pytest.fixture(scope="module")
def benchmark():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_benchmark_declares_synthetic_provenance_and_complete_definitions(benchmark):
    assert benchmark["schema_version"] == 1
    assert benchmark["provenance"]["kind"] == "synthetic"
    description = benchmark["provenance"]["description"].lower()
    assert "fictional" in description
    assert "not actual papers" in description
    assert "user votes" in description
    assert "independently of classifier output" in description
    assert "personal relevance ground truth" in description

    definitions = benchmark["definitions"]
    assert set(definitions) == {"core", "context", "off_interest"}
    assert set(definitions["core"]) == FACETS
    anchor_groups = [
        *definitions["core"].values(),
        definitions["context"],
        definitions["off_interest"],
    ]
    anchors = []
    for group in anchor_groups:
        assert len(group) >= 2
        for anchor in group:
            assert isinstance(anchor, str)
            assert len(anchor.split()) >= 12
            assert anchor.endswith(".")
            anchors.append(anchor)
    assert len(anchors) == len(set(anchors))


def test_benchmark_examples_have_unambiguous_machine_readable_contract(benchmark):
    examples = benchmark["examples"]
    assert 60 <= len(examples) <= 80
    required = {
        "id", "family_id", "split", "title", "abstract", "label", "facet", "rationale"
    }
    identifiers = []
    titles = []
    for example in examples:
        assert set(example) == required
        assert all(isinstance(value, str) for value in example.values())
        assert example["id"].strip()
        assert example["family_id"].strip()
        assert example["split"] in {"calibration", "evaluation"}
        assert example["label"] in LABELS
        assert example["title"].strip()
        assert len(example["rationale"].split()) >= 6
        if example["label"] == "core":
            assert example["facet"] in FACETS
        else:
            assert example["facet"] == ""
        if example["label"] != "uncertain":
            assert len(example["abstract"].split()) >= 20
        identifiers.append(example["id"])
        titles.append(example["title"].casefold())
    assert len(set(identifiers)) == len(identifiers)
    assert len(set(titles)) == len(titles)


def test_benchmark_families_do_not_cross_calibration_and_evaluation(benchmark):
    splits_by_family = defaultdict(set)
    texts_by_split = defaultdict(set)
    for example in benchmark["examples"]:
        splits_by_family[example["family_id"]].add(example["split"])
        if example["abstract"]:
            texts_by_split[example["split"]].add(example["abstract"].casefold())
    assert all(len(splits) == 1 for splits in splits_by_family.values())
    assert texts_by_split["calibration"].isdisjoint(texts_by_split["evaluation"])


@pytest.mark.parametrize("split", ["calibration", "evaluation"])
def test_each_split_covers_all_facets_competitors_and_abstentions(benchmark, split):
    examples = [row for row in benchmark["examples"] if row["split"] == split]
    labels = Counter(row["label"] for row in examples)
    assert set(labels) == LABELS
    assert labels["context"] >= 8
    assert labels["off_interest"] >= 8
    assert labels["uncertain"] >= 4
    core_facets = Counter(row["facet"] for row in examples if row["label"] == "core")
    assert set(core_facets) == FACETS
    assert min(core_facets.values()) >= 2
    uncertain = [row for row in examples if row["label"] == "uncertain"]
    assert any(not row["abstract"] for row in uncertain)
    assert any(0 < len(row["abstract"].split()) < 20 for row in uncertain)
    assert any(len(row["abstract"].split()) >= 25 for row in uncertain)
