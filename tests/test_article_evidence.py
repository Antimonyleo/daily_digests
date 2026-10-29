"""Article evidence quotes reported checks without assigning scientific quality."""

import pytest

from dailydigest.rank.article_evidence import assess_article_evidence


def assess(text, **kwargs):
    return assess_article_evidence("A research paper", text, **kwargs)


def test_independent_validation_is_author_reported_with_exact_offsets():
    text = "We developed a model.\n  We evaluated it on an independent held-out dataset containing diverse structures."
    result = assess(text)
    dimension = result["dimensions"]["validation"]
    assert dimension["status"] == "reported"
    claim = dimension["claims"][0]
    assert claim["kind"] == "independent_validation"
    assert claim["verification"] == "author_reported_unverified"
    assert text[claim["start"]:claim["end"]] == claim["quote"]
    assert claim["source"] == "abstract"
    assert "quality_score" not in result


def test_experimental_validation_and_benchmark_are_reported_separately():
    text = "We validated the designed structures using experimental measurements. We evaluated the algorithm on a public benchmark."
    claims = assess(text)["dimensions"]["validation"]["claims"]
    assert [claim["kind"] for claim in claims] == ["experimental_validation", "benchmark_validation"]


def test_reported_held_out_evaluations_are_quoted_without_verifying_results():
    text = "We report held-out evaluations for five molecular design targets, comparing random and model-guided baselines."
    dimension = assess(text)["dimensions"]["validation"]
    assert dimension["status"] == "reported"
    claim = dimension["claims"][0]
    assert claim["kind"] == "independent_validation"
    assert claim["quote"] == text
    assert claim["verification"] == "author_reported_unverified"


@pytest.mark.parametrize("text", [
    "Here we validate our ideas and describe a promising approach for future research.",
    "Future work will evaluate the model using an independent dataset and experimental measurements.",
    "We did not validate the model on an external cohort, and further independent validation is needed.",
    "Previous studies validated related models using independent datasets and experimental measurements.",
    "An independent dataset provides background context for our study; no validation was performed.",
    "No external validation was undertaken, and independent validation remains necessary for this model.",
    "Independent validation is fundamental to sound experimental research and motivates this study.",
    "We will report held-out evaluations for five design targets in future work.",
    "We did not report held-out evaluations for five design targets in this study.",
])
def test_generic_future_negated_and_background_validation_remain_unknown(text):
    dimension = assess(text)["dimensions"]["validation"]
    assert dimension == {"status": "unknown", "claims": []}


def test_data_code_assertions_and_actual_supplied_links_are_distinct():
    text = "Source code is available at https://github.com/example/research. Data are available upon reasonable request."
    dimension = assess(text)["dimensions"]["data_code"]
    assert dimension["status"] == "reported"
    code, data = dimension["claims"]
    assert code["links"] == ["https://github.com/example/research"]
    assert code["availability"] == "asserted_available"
    assert data["links"] == []
    assert data["availability"] == "upon_request"
    assert all(text[claim["start"]:claim["end"]] == claim["quote"] for claim in (code, data))


@pytest.mark.parametrize("text", [
    "The code will be released following publication; the data may become available in a future repository.",
    "Data and code are not available because the original study cannot release participant records.",
    "We used publicly available code to analyze the dataset, and describe the resulting structures.",
    "The introduction discusses https://github.com/example/research as related background material.",
])
def test_future_unavailable_or_used_external_resources_do_not_claim_reproducibility(text):
    assert assess(text)["dimensions"]["data_code"] == {"status": "unknown", "claims": []}


def test_limitations_are_reported_without_turning_them_into_a_penalty():
    text = "The study is limited to a small sample. Results may not generalize to other experimental settings."
    result = assess(text)
    dimension = result["dimensions"]["limitations"]
    assert dimension["status"] == "reported"
    assert len(dimension["claims"]) == 2
    assert all(text[claim["start"]:claim["end"]] == claim["quote"] for claim in dimension["claims"])
    assert "penalty" not in result and "priority" not in result


def test_provided_full_text_defines_quote_source_and_preserves_abstract_limit():
    abstract = "We introduce a method and describe its intended scientific applications."
    full_text = "Methods. We evaluated the model on an external cohort of independently collected measurements."
    result = assess(abstract, full_text=full_text)
    assert result["assessment_basis"] == "full_text"
    claim = result["dimensions"]["validation"]["claims"][0]
    assert claim["source"] == "full_text"
    assert full_text[claim["start"]:claim["end"]] == claim["quote"]


def test_full_text_metadata_header_does_not_hide_reported_validation():
    full_text = "DOI: 10.1234/paper.\nMethods. We evaluated the model on an independent dataset containing several structures."
    result = assess("", full_text=full_text)
    assert result["assessment_basis"] == "full_text"
    assert result["dimensions"]["validation"]["status"] == "reported"


@pytest.mark.parametrize("abstract", ["", "  ", "DOI: 10.1234/paper. Published online 2026-10-02."])
def test_missing_or_bibliographic_abstracts_have_unknown_dimensions(abstract):
    result = assess(abstract)
    assert result["assessment_basis"] == "insufficient"
    assert all(dimension == {"status": "unknown", "claims": []}
               for dimension in result["dimensions"].values())


def test_empty_full_text_falls_back_to_abstract_without_fetching():
    text = "We tested the method against an independent validation dataset covering several conditions."
    result = assess(text, full_text="  ")
    assert result["assessment_basis"] == "abstract"
    assert result["dimensions"]["validation"]["status"] == "reported"


def test_journal_prestige_or_promotional_title_cannot_change_article_assessment():
    abstract = "We describe a method and discuss its intended applications without reporting independent tests."
    high = assess_article_evidence("Nature: rigorously validated breakthrough", abstract)
    low = assess_article_evidence("A small journal reports preliminary research", abstract)
    assert high == low
    assert all(dimension["status"] == "unknown" for dimension in high["dimensions"].values())


@pytest.mark.parametrize("title,abstract,full_text", [(None, "text", None), ("title", None, None),
                                                   ("title", "text", 42)])
def test_invalid_supplied_text_is_rejected(title, abstract, full_text):
    with pytest.raises(ValueError):
        assess_article_evidence(title, abstract, full_text=full_text)


@pytest.mark.parametrize("text", [
    "No data are available for independent reproduction of the simulations.",
    "We provide no source code or datasets for independent reproduction of this model.",
    "Neither data nor code are available for independent reproduction of this study.",
])
def test_negated_resources_never_become_availability_claims(text):
    assert assess(text)["dimensions"]["data_code"] == {"status": "unknown", "claims": []}


def test_negated_independent_clause_cannot_upgrade_experimental_validation():
    text = "We evaluated the model on experimental data but not on an independent dataset."
    # The heuristic cannot safely assign scope within this mixed sentence.
    assert assess(text)["dimensions"]["validation"] == {"status": "unknown", "claims": []}


def test_negation_in_another_sentence_does_not_hide_positive_claims_or_change_offsets():
    text = ("No data are available for independent reproduction. "
            "We validated the method using experimental measurements. "
            "Source code is available at https://github.com/example/model.")
    dimensions = assess(text)["dimensions"]
    assert dimensions["validation"]["claims"][0]["kind"] == "experimental_validation"
    assert len(dimensions["data_code"]["claims"]) == 1
    for key in ("validation", "data_code"):
        claim = dimensions[key]["claims"][0]
        assert text[claim["start"]:claim["end"]] == claim["quote"]


def test_problems_the_work_solves_are_not_its_limitations():
    from dailydigest.rank.article_evidence import assess_article_evidence

    abstract = ("The LNPs address key limitations of current LNP systems, including immunogenicity. "
                "A limitation of this study is the small sample of donors.")
    claims = assess_article_evidence("T", abstract)["dimensions"]["limitations"]["claims"]
    assert [claim["quote"] for claim in claims] == ["A limitation of this study is the small sample of donors."]


def test_a_sentence_split_never_ends_inside_brackets():
    from dailydigest.rank.article_evidence import assess_article_evidence

    abstract = "Uptake rose in macrophages (66.5% pos. F4/80+ cells) in vivo. Data are publicly available."
    result = assess_article_evidence("T", abstract)
    assert result["dimensions"]["data_code"]["claims"][0]["quote"] == "Data are publicly available."
    assert all("(66.5% pos." not in c["quote"] or c["quote"].endswith("in vivo.")
               for d in result["dimensions"].values() for c in d["claims"])


def test_a_solved_limitation_does_not_hide_one_stated_in_the_same_sentence():
    abstract = ("Although our method overcomes limitations of conventional LNPs, it is limited to murine models. "
                "Delivery improved tenfold.")
    claims = assess_article_evidence("T", abstract)["dimensions"]["limitations"]["claims"]
    assert [claim["quote"] for claim in claims] == [
        "Although our method overcomes limitations of conventional LNPs, it is limited to murine models."]


def test_an_unclosed_bracket_does_not_merge_the_rest_of_the_abstract():
    abstract = ("We studied cells (see methods. Results were strong. "
                "Limitations: small sample size. We conclude.")
    claims = assess_article_evidence("T", abstract)["dimensions"]["limitations"]["claims"]
    assert [claim["quote"] for claim in claims] == ["Limitations: small sample size."]
