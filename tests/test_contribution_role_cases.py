"""Integrity checks for fictional role contrasts, not semantic accuracy tests."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pytest

from dailydigest.rank.contribution_ranker import fit_contribution_ranker
from dailydigest.rank.pair_scope import pair_queries


FIXTURE = Path(__file__).parent / "fixtures" / "contribution_role_cases.json"
SCOPE_FIXTURE = FIXTURE.with_name("research_scope_benchmark.json")
ROLES = {"introduced_method", "engineered_system", "application", "natural_mechanism", "review", "uncertain"}
LABELS = {"core", "context", "off_interest", "uncertain"}


@pytest.fixture(scope="module")
def cases():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_provenance_forbids_real_accuracy_and_training_claims(cases):
    assert cases["schema_version"] == 1
    provenance = cases["provenance"]
    assert provenance["kind"] == "synthetic_regression"
    assert provenance["training_use"] == "forbidden"
    description = provenance["description"].lower()
    for phrase in ("fictional", "not actual papers", "fresh biological findings", "reader judgments",
                   "scientific quality", "production accuracy", "personal relevance ground truth"):
        assert phrase in description
    assert "never silently merged" in provenance["comparison_contract"]
    assert "separate from scope" in provenance["validation_meaning"]
    assert "remains uncertain" in provenance["uncertainty_contract"]


def test_exact_contribution_quotes_and_complete_case_contract(cases):
    required = {"id", "family_id", "contrast_family_id", "split", "title", "abstract", "label",
                "expected_role", "contribution_quote", "contribution_object", "contribution_task",
                "validation_evidence", "rationale"}
    examples = cases["examples"]
    assert len(examples) == 26
    assert len({ex["id"] for ex in examples}) == len(examples)
    assert len({ex["family_id"] for ex in examples}) == len(examples)
    assert len({ex["title"] for ex in examples}) == len(examples)
    assert len({ex["abstract"] for ex in examples}) == len(examples)
    assert set(ex["label"] for ex in examples) == LABELS
    assert set(ex["expected_role"] for ex in examples) == ROLES
    for example in examples:
        assert set(example) == required
        assert all(isinstance(value, str) and value.strip() == value and value for value in example.values())
        assert example["id"].startswith("synthetic-role-")
        assert example["family_id"].startswith("synthetic-role:case:")
        assert example["contrast_family_id"].startswith("synthetic-role:contrast:")
        assert example["split"] == "regression"
        assert "fictional" in example["title"].lower()
        assert len(example["abstract"].split()) >= 20
        assert len(example["contribution_quote"].split()) >= 10
        start = example["abstract"].index(example["contribution_quote"])
        end = start + len(example["contribution_quote"])
        assert example["abstract"][start:end] == example["contribution_quote"]
        assert example["validation_evidence"] in {"strong", "weak", "not_assessed", "unknown"}


def test_matched_contrast_families_have_different_scope_cases(cases):
    grouped = defaultdict(list)
    for example in cases["examples"]:
        grouped[example["contrast_family_id"]].append(example)
    assert len(grouped) == 13
    assert Counter(len(group) for group in grouped.values()) == {2: 13}
    for group in grouped.values():
        assert len({example["label"] for example in group}) == 2
    assert {family.rsplit(":", 1)[-1] for family in grouped} == {
        "prediction-ownership", "assistant-versus-predictor", "rna-assay-versus-device",
        "condensate-ownership", "review-versus-original", "validation-versus-scope",
        "negation-attachment", "future-versus-present", "past-versus-own",
        "ambiguous-versus-explicit", "tool-use-versus-system", "planned-versus-delivered",
        "transferability-of-application",
    }


def test_role_alone_cannot_determine_scope_and_quality_is_separate(cases):
    by_id = {example["id"]: example for example in cases["examples"]}
    assistant, predictor = [by_id["synthetic-role-" + key] for key in ("literature-assistant", "molecular-predictor")]
    assert assistant["expected_role"] == predictor["expected_role"] == "introduced_method"
    assert (assistant["label"], predictor["label"]) == ("context", "core")
    transferable, isolated = [by_id["synthetic-role-" + key] for key in ("transferable-evaluation", "isolated-description")]
    assert transferable["expected_role"] == isolated["expected_role"] == "application"
    assert (transferable["label"], isolated["label"]) == ("core", "context")
    engineered = by_id["synthetic-role-existing-tool-engineering"]
    assert engineered["label"] == "core"
    assert engineered["expected_role"] == "engineered_system"
    assert "existing design tool" in engineered["contribution_quote"]
    strong, weak = [by_id["synthetic-role-" + key] for key in ("strong-off-interest", "weak-core-design")]
    assert (strong["validation_evidence"], weak["validation_evidence"]) == ("strong", "weak")
    assert (strong["label"], weak["label"]) == ("off_interest", "core")


def test_claim_attachment_and_ambiguous_roles_remain_explicit(cases):
    by_id = {example["id"]: example for example in cases["examples"]}
    negative = by_id["synthetic-role-negated-method"]
    positive = by_id["synthetic-role-negated-application"]
    assert "do not introduce" in negative["contribution_quote"]
    assert negative["label"] == "context"
    assert "do not merely apply" in positive["contribution_quote"]
    assert "; we introduce" in positive["contribution_quote"]
    assert positive["label"] == "core"
    prior = by_id["synthetic-role-own-method-after-prior"]
    assert "Previous approaches" in prior["contribution_quote"]
    assert "here we introduce" in prior["contribution_quote"]
    assert prior["label"] == "core"
    uncertain = [example for example in cases["examples"] if example["label"] == "uncertain"]
    assert {example["id"] for example in uncertain} == {
        "synthetic-role-ambiguous-workflow", "synthetic-role-future-only",
    }
    assert all(example["expected_role"] == "uncertain" and example["validation_evidence"] == "unknown"
               for example in uncertain)


def test_namespace_is_separate_and_regression_split_cannot_fit_ranker(cases):
    benchmark = json.loads(SCOPE_FIXTURE.read_text(encoding="utf-8"))
    existing = benchmark["examples"]
    assert {ex["id"] for ex in cases["examples"]}.isdisjoint(ex["id"] for ex in existing)
    assert {ex["family_id"] for ex in cases["examples"]}.isdisjoint(ex["family_id"] for ex in existing)
    assert all(ex["split"] not in {"calibration", "evaluation"} for ex in cases["examples"])
    with pytest.raises(ValueError, match="Only calibration"):
        fit_contribution_ranker(np.zeros((len(cases["examples"]), 24)), pair_queries(benchmark["definitions"]),
                                cases["examples"])
