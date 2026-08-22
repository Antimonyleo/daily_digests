"""Funding calls must obey the reader's negative interests, like research does."""

from __future__ import annotations

from types import SimpleNamespace


def _row(section: str, title: str):
    return SimpleNamespace(
        id=abs(hash(title)) % 100000,
        section=section,
        title=title,
        source="Grants.gov",
        url="https://example.com/x",
        abstract="",
        metadata_json="{}",
    )


def _run_filter(rows_with_features, monkeypatch):
    from dailydigest import pipeline as pipeline_mod

    monkeypatch.setattr(
        pipeline_mod,
        "get_settings",
        lambda: SimpleNamespace(
            adaptive_section_sizes=True,
            min_topic_relevance=0.65,
            min_news_quality=0.45,
            min_opportunity_relevance=0.58,
        ),
    )
    scored = [(row, score) for row, score, _feat in rows_with_features]
    features = {
        pipeline_mod._row_feature_key(row): feat for row, _score, feat in rows_with_features
    }
    return pipeline_mod._filter_off_topic(scored, features)


def test_a_clinical_call_above_the_floor_is_still_dropped(monkeypatch):
    """The asymmetry that let clinical grants through.

    Research gates on (topic + venue credit - negative penalty); opportunities
    gated on the BARE cosine, so explicit negative interests (clinical oncology,
    neurodegenerative pathology, trial outcomes) never applied to funding calls.
    Measured over 81 live opportunities the bare rule passed 16 and the
    symmetric rule passes 7, dropping ultra-rare cancers, a Type 1 Diabetes
    repository, GREGoRi clinical trials, childhood cancers and Duchenne.
    """
    clinical = _row("opportunities", "Novel approaches for therapeutic development in ultra-rare cancers")
    kept = _run_filter(
        [(clinical, 0.61, {"topic_score": 0.606, "negative_interest_penalty": 0.057})],
        monkeypatch,
    )
    assert kept == [], "a call the reader's negatives cover was still served"


def test_an_on_topic_call_with_no_penalty_survives(monkeypatch):
    on_topic = _row("opportunities", "RNomics Technology Enhancement Centers")
    kept = _run_filter(
        [(on_topic, 0.63, {"topic_score": 0.627, "negative_interest_penalty": 0.020})],
        monkeypatch,
    )
    assert len(kept) == 1


def test_events_are_gated_the_same_way(monkeypatch):
    event = _row("events", "Clinical trial methodology symposium")
    kept = _run_filter(
        [(event, 0.62, {"topic_score": 0.620, "negative_interest_penalty": 0.090})],
        monkeypatch,
    )
    assert kept == []


def test_research_gating_is_unchanged(monkeypatch):
    """The fix must not alter the research floor it was modelled on."""
    paper = _row("research", "DNA origami actuator")
    kept = _run_filter(
        [(paper, 0.80, {"topic_score": 0.70, "negative_interest_penalty": 0.01})],
        monkeypatch,
    )
    assert len(kept) == 1
