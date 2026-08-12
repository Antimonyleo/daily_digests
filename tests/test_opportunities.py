from __future__ import annotations

from datetime import date

import pytest
import yaml

from dailydigest.models import Item


def _profile_payload() -> dict:
    return {
        "description": (
            "I am a postdoctoral researcher at a nonprofit university working "
            "on RNA nanotechnology and therapeutic delivery."
        ),
        "career_stage": "postdoctoral researcher",
        "institution_type": "nonprofit university",
        "country": "United States",
        "applicant_role": "fellow or co-investigator",
        "opportunity_types": ["fellowship", "travel_support"],
        "event_types": ["conference", "workshop"],
        "event_regions": ["North America", "online"],
        "event_formats": ["in_person", "online"],
        "requires_travel_support": True,
        "minimum_lead_days": 14,
    }


def test_opportunity_profile_round_trips_from_private_yaml(tmp_path):
    from dailydigest.opportunities import load_opportunity_profile

    path = tmp_path / "opportunities.yaml"
    path.write_text(yaml.safe_dump(_profile_payload(), sort_keys=False))

    profile = load_opportunity_profile(path)

    assert profile.career_stage == "postdoctoral researcher"
    assert profile.opportunity_types == ["fellowship", "travel_support"]
    assert profile.minimum_lead_days == 14


def test_opportunity_profile_accepts_structured_fields_without_legacy_description():
    from dailydigest.opportunities import OpportunityProfile

    payload = _profile_payload()
    payload.pop("description")

    profile = OpportunityProfile(**payload)

    assert profile.description == ""


def test_opportunity_metadata_updates_and_keeps_immutable_change_history():
    from dailydigest.store import (
        item_metadata,
        opportunity_history,
        recent_items,
        upsert_items,
    )

    first = Item(
        source="Grants.gov",
        section="opportunities",
        external_id="GRANT-1",
        url="https://grants.gov/opportunity/1",
        title="RNA delivery research grant",
        abstract="Supports research on RNA delivery systems.",
        metadata={
            "opportunity_type": "grant",
            "status": "open",
            "deadline": "2026-10-01",
            "amount_min": 100000,
            "amount_max": 500000,
            "currency": "USD",
            "official": True,
        },
    )
    assert upsert_items([first]) == 1
    row = next(row for row in recent_items(days=2) if row.external_id == "GRANT-1")
    assert item_metadata(row)["deadline"] == "2026-10-01"
    assert len(opportunity_history(int(row.id))) == 1

    # Re-verifying identical details is idempotent in immutable history.
    reverified = first.model_copy(
        update={"metadata": {**first.metadata, "verified_at": "2026-08-11T08:00:00Z"}}
    )
    assert upsert_items([reverified]) == 0
    assert len(opportunity_history(int(row.id))) == 1

    changed = first.model_copy(update={"metadata": {**first.metadata, "deadline": "2026-10-15"}})
    assert upsert_items([changed]) == 0

    updated = next(row for row in recent_items(days=2) if row.external_id == "GRANT-1")
    assert item_metadata(updated)["deadline"] == "2026-10-15"
    assert [entry["deadline"] for entry in opportunity_history(int(row.id))] == [
        "2026-10-01",
        "2026-10-15",
    ]


def test_material_change_can_resurface_a_previously_shown_opportunity():
    from dailydigest.store import (
        exclude_previously_shown,
        recent_items,
        upsert_items,
        write_digest,
    )

    item = Item(
        source="Grants.gov",
        section="opportunities",
        external_id="CHANGED-1",
        url="https://grants.gov/opportunity/changed",
        title="RNA research opportunity",
        abstract="Support for RNA research and technology development.",
        metadata={"status": "open", "deadline": "2026-10-01", "official": True},
    )
    upsert_items([item])
    row = next(r for r in recent_items(days=2) if r.external_id == "CHANGED-1")
    write_digest("2026-08-10", [("F1", int(row.id), 0.9)])
    assert exclude_previously_shown([row]) == []

    changed = item.model_copy(update={"metadata": {**item.metadata, "deadline": "2026-10-15"}})
    upsert_items([changed])
    refreshed = next(r for r in recent_items(days=2) if r.external_id == "CHANGED-1")
    assert exclude_previously_shown([refreshed]) == [refreshed]


def test_assessment_filters_ineligible_closed_and_too_soon_items():
    from dailydigest.opportunities import OpportunityProfile, assess_opportunity

    profile = OpportunityProfile(
        description=(
            "I am a postdoctoral researcher at a US university working on RNA "
            "nanotechnology and colloidal self-assembly."
        ),
        career_stage="postdoctoral researcher",
        institution_type="university",
        country="United States",
        applicant_role="principal investigator or co-investigator",
        opportunity_types=["grant", "fellowship"],
        minimum_lead_days=7,
    )
    base = {
        "status": "open",
        "opportunity_type": "grant",
        "deadline": "2026-09-15",
        "eligibility_tags": ["Public and State controlled institutions of higher education"],
    }

    likely = assess_opportunity(base, profile, today=date(2026, 8, 10))
    assert likely.actionable is True
    assert likely.eligibility == "likely"

    cooperative = assess_opportunity(
        {**base, "opportunity_type": "cooperative_agreement"},
        profile,
        today=date(2026, 8, 10),
    )
    assert cooperative.actionable is True

    unrestricted = assess_opportunity(
        {**base, "eligibility_tags": ["Unrestricted (open to any type of entity)"]},
        profile,
        today=date(2026, 8, 10),
    )
    assert unrestricted.actionable is True
    assert unrestricted.eligibility == "likely"

    non_us_profile = profile.model_copy(update={"country": "Germany"})
    domestic_only = assess_opportunity(
        {
            **base,
            "eligibility_description": "Applicants must be domestic only.",
        },
        non_us_profile,
        today=date(2026, 8, 10),
    )
    assert domestic_only.actionable is False
    assert "country" in domestic_only.reason

    wrong_institution = assess_opportunity(
        {**base, "eligibility_tags": ["Small businesses"]},
        profile,
        today=date(2026, 8, 10),
    )
    assert wrong_institution.actionable is False
    assert wrong_institution.eligibility == "unlikely"

    too_soon = assess_opportunity(
        {**base, "deadline": "2026-08-12"}, profile, today=date(2026, 8, 10)
    )
    assert too_soon.actionable is False
    assert "lead time" in too_soon.reason

    closed = assess_opportunity({**base, "status": "closed"}, profile, today=date(2026, 8, 10))
    assert closed.actionable is False

    event = {
        "record_type": "event",
        "status": "open",
        "event_type": "workshop",
        "format": "in_person",
        "location": "EMBL Heidelberg",
        "deadline": "2026-09-15",
    }
    profile_with_region = profile.model_copy(
        update={
            "event_types": ["workshop"],
            "event_formats": ["in_person"],
            "event_regions": ["North America"],
        }
    )
    outside_region = assess_opportunity(event, profile_with_region, today=date(2026, 8, 10))
    assert outside_region.actionable is False
    assert "location" in outside_region.reason


def test_structured_profile_affects_only_opportunity_ordering(monkeypatch):
    import numpy as np

    from dailydigest.opportunities import OpportunityProfile
    from dailydigest.pipeline import _apply_opportunity_profile_relevance
    from dailydigest.rank.ranker import _row_feature_key
    from dailydigest.store import ItemRow

    profile = OpportunityProfile(
        description=(
            "I am a postdoctoral researcher developing RNA nanotechnology at "
            "a university and seeking relevant research support."
        ),
        career_stage="postdoctoral researcher",
        institution_type="university",
        country="United States",
        applicant_role="fellow or co-investigator",
    )
    matching = ItemRow(
        id=1,
        source="Official",
        section="opportunities",
        external_id="match",
        url="https://example.org/match",
        title="RNA nanotechnology fellowship",
    )
    unrelated = ItemRow(
        id=2,
        source="Official",
        section="opportunities",
        external_id="other",
        url="https://example.org/other",
        title="Unrelated fellowship",
    )
    research = ItemRow(
        id=3,
        source="Journal",
        section="research",
        external_id="paper",
        url="https://example.org/paper",
        title="Research paper",
    )
    embedded_queries = []

    def fake_embed_texts(texts, is_query=True):
        embedded_queries.extend(texts)
        return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(
        "dailydigest.rank.embed.embed_texts",
        fake_embed_texts,
    )
    monkeypatch.setattr(
        "dailydigest.rank.embedding_cache.embed_item_rows",
        lambda rows: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    features = {}

    adjusted = _apply_opportunity_profile_relevance(
        [(matching, 0.5), (unrelated, 0.5), (research, 0.5)], features, profile
    )

    assert adjusted[0][1] == pytest.approx(0.6)
    assert adjusted[1][1] == pytest.approx(0.5)
    assert adjusted[2][1] == pytest.approx(0.5)
    assert features[_row_feature_key(matching)]["opportunity_profile_score"] == 1.0
    assert "postdoctoral researcher" in embedded_queries[0]
    assert "United States" in embedded_queries[0]
