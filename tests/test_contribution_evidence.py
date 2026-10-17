"""Local passage scores quote evidence without inventing contribution claims."""

import numpy as np
import pytest

from dailydigest.rank.contribution_evidence import evidence_for_article


QUERIES = [
    {"label": "core", "facet": "Protein design", "text": "Engineer protein function"},
    {"label": "core", "facet": "DNA design", "text": "Design DNA origami"},
    {"label": "context", "facet": "", "text": "Natural protein mechanism"},
    {"label": "off_interest", "facet": "", "text": "Clinical outcomes"},
]


class Scorer:
    def __init__(self, score=lambda query, text: 0.):
        self.score = score
        self.pairs = []

    def score_pairs(self, pairs):
        self.pairs.extend(pairs)
        return np.array([self.score(query, text) for query, text in pairs])


def test_background_relevance_cannot_become_claimed_engineering():
    abstract = "Previous protein engineering methods used directed evolution. Here we investigate a native protein signaling mechanism."
    scorer = Scorer(lambda query, text: 9. if "Previous" in text else 1.)
    report = evidence_for_article("Study", abstract, QUERIES, scorer)
    match = report["facet_evidence"]["Protein design"]
    assert match["quote"] == "Previous protein engineering methods used directed evolution."
    assert match["task_object_evidence"] == match["quote"]
    assert "contribution" not in match and "priority" not in match
    assert "not proof" in report["score_meaning"]


def test_claimed_contribution_and_competitor_have_independent_passage_support():
    abstract = "Native signaling is a biological mechanism. Here we engineer protein variants with a designed catalytic function."
    scorer = Scorer(lambda query, text: 8. if (
        query == QUERIES[0]["text"] and "Here" in text
        or query == QUERIES[2]["text"] and "Native" in text
    ) else -2.)
    report = evidence_for_article("Study", abstract, QUERIES, scorer)
    assert report["facet_evidence"]["Protein design"]["quote"].startswith("Here")
    assert report["competitor_evidence"]["context"]["quote"].startswith("Native")
    assert [query for query, _text in scorer.pairs[:4]] == [q["text"] for q in QUERIES]


def test_long_abstract_sample_covers_head_middle_and_late_contribution():
    abstract = " ".join(f"Passage {i} describes observations." for i in range(40))
    scorer = Scorer(lambda query, text: 10. if "Passage 39" in text else 0.)
    report = evidence_for_article("Study", abstract, QUERIES, scorer, max_sentences=6)
    assert report["total_passages"] == 40
    assert report["scored_passages"] == 6
    assert report["passages"][0]["quote"].startswith("Passage 0")
    assert report["passages"][-1]["quote"].startswith("Passage 39")
    assert any(15 <= p["passage_index"] <= 24 for p in report["passages"])
    assert report["facet_evidence"]["Protein design"]["quote"].startswith("Passage 39")
    assert len(scorer.pairs) == 24


def test_quotes_and_offsets_are_exact_with_whitespace_and_semicolon_clauses():
    abstract = "  Native proteins act in cells;\n\tHere we engineer β-protein variants.  "
    scorer = Scorer()
    report = evidence_for_article("A title about proteins", abstract, QUERIES, scorer)
    assert len(report["passages"]) == 2
    for passage in report["passages"]:
        assert abstract[passage["start"]:passage["end"]] == passage["quote"]
        assert "A title" not in passage["quote"]
    assert all(text.startswith("A title about proteins. ") for _query, text in scorer.pairs)


def test_empty_abstract_has_no_evidence_or_inference_calls():
    scorer = Scorer()
    report = evidence_for_article("Promising design title", "  ", QUERIES, scorer)
    assert report["facet_evidence"] == {}
    assert report["competitor_evidence"] == {}
    assert report["scored_passages"] == 0
    assert not scorer.pairs


def test_short_abstract_is_scored_without_a_lexical_contribution_gate():
    scorer = Scorer()
    report = evidence_for_article("", "A new protein.", QUERIES, scorer)
    assert report["facet_evidence"]["Protein design"]["quote"] == "A new protein."
    assert report["pair_count"] == 4


def test_external_query_schema_matches_internal_order_and_fingerprint():
    external = [{"kind": q["label"], "facet": q["facet"] or None, "text": q["text"]}
                for q in QUERIES]
    first = evidence_for_article("", "Evidence.", QUERIES, Scorer())
    second = evidence_for_article("", "Evidence.", external, Scorer())
    assert first == second


def test_multiple_queries_for_a_facet_keep_winning_query_and_exact_passage():
    alternate = {**QUERIES[0], "text": "Design catalytic protein variants"}
    queries = [QUERIES[0], alternate, *QUERIES[1:]]
    scorer = Scorer(lambda query, text: 5. if query == alternate["text"] else -1.)
    report = evidence_for_article("", "Exact supporting passage.", queries, scorer)
    match = report["facet_evidence"]["Protein design"]
    assert match["query_index"] == 1
    assert match["query"]["text"] == alternate["text"]
    assert match["score"] == 5.
    assert match["quote"] == "Exact supporting passage."


@pytest.mark.parametrize("queries", [
    list(reversed(QUERIES)), QUERIES[:2],
    [{**QUERIES[0], "text": ""}, *QUERIES[1:]],
    [{**QUERIES[0], "kind": "context"}, *QUERIES[1:]],
    [{**QUERIES[0], "facet": ""}, *QUERIES[1:]],
    [*QUERIES[:3], {**QUERIES[3], "facet": False}],
    [QUERIES[0], *QUERIES],
])
def test_invalid_query_schema_or_order_fails_before_scoring(queries):
    scorer = Scorer()
    with pytest.raises(ValueError):
        evidence_for_article("", "Evidence.", queries, scorer)
    assert not scorer.pairs


@pytest.mark.parametrize("scores", [np.array([np.nan] * 4), np.ones((4, 1)),
                                    np.ones(3), np.array([True] * 4),
                                    np.array(["0"] * 4)])
def test_score_schema_and_nonfinite_logits_fail(scores):
    class InvalidScorer:
        def score_pairs(self, pairs):
            return scores

    with pytest.raises(ValueError, match="finite scalar"):
        evidence_for_article("", "Evidence.", QUERIES, InvalidScorer())


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_sentence_budget_fails(budget):
    with pytest.raises(ValueError):
        evidence_for_article("", "Evidence.", QUERIES, Scorer(), max_sentences=budget)
