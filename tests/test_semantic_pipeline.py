"""The joint matcher supplements topic evidence without relaxing source policy."""
from types import SimpleNamespace

import pytest

from dailydigest import pipeline
from dailydigest.config import Settings


def row():
    return SimpleNamespace(id=1, section="research", source="Nature", url="https://example.org/1")


def configure(monkeypatch, **kwargs):
    kwargs.setdefault("semantic_matching", True)
    settings = Settings(_env_file=None, adaptive_section_sizes=True, **kwargs)
    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    monkeypatch.setattr(pipeline, "venue_relevance_credit", lambda _: 0.0)
    return settings


def test_model_disagreement_does_not_veto_a_related_paper(monkeypatch):
    configure(monkeypatch)
    candidate = row()
    features = {1: {"topic_score": .82, "topic_alignment": {"logit": -10.2, "complete": True}}}
    assert pipeline._filter_off_topic([(candidate, .8)], features) == [(candidate, .8)]


def test_model_disagreement_does_not_veto_without_adaptive_sizing(monkeypatch):
    settings = configure(monkeypatch)
    settings.adaptive_section_sizes = False
    candidate = row()
    feature = {"topic_score": .9, "topic_alignment": {"logit": -11., "complete": True}}
    assert pipeline._filter_off_topic([(candidate, .9)], {1: feature}) == [(candidate, .9)]


def test_complete_disagreement_is_a_bounded_selection_nudge(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(pipeline, "recent_viewed_facet_dates", lambda **kwargs: {})
    first, second = row(), row()
    second.id = 2
    features = {1: {"topic_score": .82, "topic_alignment": {"logit": -11., "complete": True}},
                2: {"topic_score": .81}}
    scored = [(first, .8), (second, .78)]
    ordered = pipeline._apply_topic_selection_preferences(scored, features, digest_id="2026-10-15")
    assert ordered == [(second, .78), (first, .8)]
    assert features[1]["selection_order_bonus"] == -.06
    assert features[1]["topic_score"] == .82


def test_partial_coverage_does_not_lower_selection_priority(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(pipeline, "recent_viewed_facet_dates", lambda **kwargs: {})
    candidate = row()
    features = {1: {"topic_alignment": {"logit": -11., "complete": False}}}
    assert pipeline._apply_topic_selection_preferences([(candidate,.8)], features, digest_id="2026-10-15") == [(candidate,.8)]
    assert not features[1].get("selection_order_bonus")


def test_unchecked_candidates_keep_original_score_scale(monkeypatch):
    configure(monkeypatch)
    candidate = row()
    features = {1: {"topic_score": .9, "topic_alignment_status": "outside_shortlist"}}
    assert pipeline._filter_off_topic([(candidate, .9)], features) == [(candidate, .9)]


def test_journal_prestige_cannot_establish_topic_fit(monkeypatch):
    configure(monkeypatch, semantic_matching=True)
    monkeypatch.setattr(pipeline, "venue_relevance_credit", lambda _: .05)
    candidate = row()
    assert pipeline._filter_off_topic([(candidate, .9)], {1: {"topic_score": .62}}) == []


def test_adaptive_size_counts_joint_rescues(monkeypatch):
    settings = configure(monkeypatch, min_research=0)
    candidate = row()
    features = {1: {"topic_score": .62, "topic_alignment": {"logit": 5., "complete": True}}}
    assert pipeline._dynamic_section_caps([(candidate, .8)], features, settings=settings)["research"] == 1


def test_unseen_middle_is_not_negative_evidence(monkeypatch):
    configure(monkeypatch)
    candidate = row()
    features = {1: {"topic_score": .82, "topic_alignment": {"logit": -11., "complete": False}}}
    assert pipeline._filter_off_topic([(candidate, .8)], features) == [(candidate, .8)]


@pytest.mark.parametrize("topic,negative,expected", [(.62, 0., True), (.5, 0., False), (.62, .01, False)])
def test_strong_joint_match_rescues_only_borderline_nonnegative_candidate(monkeypatch, topic, negative, expected):
    configure(monkeypatch)
    candidate = row()
    features = {1: {"topic_score": topic, "negative_interest_penalty": negative,
                    "topic_alignment": {"logit": 5., "complete": True}}}
    assert bool(pipeline._filter_off_topic([(candidate, .8)], features)) is expected


def test_alignment_disabled_does_not_load_model(monkeypatch):
    settings = configure(monkeypatch, semantic_matching=False)
    candidates = [(row(), .8)]
    assert pipeline._apply_topic_alignment(candidates, {}, object(), settings) is candidates


def test_pipeline_calls_joint_matcher_with_cancellation(monkeypatch):
    from dailydigest.rank import semantic_alignment

    settings = configure(monkeypatch, semantic_matching=True)
    candidates, features, profile = [(row(), .8)], {}, object()
    check = lambda: None
    calls = []

    def apply(scored, actual_features, actual_profile, **kwargs):
        calls.append((scored, actual_features, actual_profile, kwargs))
        return scored

    monkeypatch.setattr(semantic_alignment, "apply_topic_alignment", apply)
    assert pipeline._apply_topic_alignment(candidates, features, profile, settings, check_cancel=check) is candidates
    assert calls[0][:3] == (candidates, features, profile)
    assert calls[0][3]["check_cancel"] is check


def test_extra_model_is_off_by_default(monkeypatch):
    monkeypatch.delenv("SEMANTIC_MATCHING", raising=False)
    assert Settings(_env_file=None).semantic_matching is False


def test_joint_evidence_roundtrips_to_display():
    from dailydigest.rank.source_quality import breakdown_payload
    from dailydigest.web import _breakdown_payload

    candidate = SimpleNamespace(id=1, section="research", source="Nature", url="https://example.org/1",
                                title="Test", abstract="", score=.8, published=None)
    evidence = {"topic": "catalysis", "query": "asymmetric catalysis", "logit": 4.1,
                "windows": 1, "complete": True, "kind": "local_joint_match_v1"}
    persisted = breakdown_payload(candidate, .75, topic_alignment=evidence)
    assert persisted["topic_alignment"] == evidence
    assert _breakdown_payload(candidate, persisted)["topic_alignment"] == evidence
