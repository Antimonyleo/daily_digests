"""Opportunity/event profile and deterministic matching helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class OpportunityProfile(BaseModel):
    """Private reader information used only for opportunity eligibility."""

    # Kept for compatibility with profiles created before structured setup
    # replaced the duplicate free-form paragraph. New profiles leave this blank.
    description: str = Field(default="", max_length=4000)
    career_stage: str = Field(min_length=2, max_length=120)
    institution_type: str = Field(min_length=2, max_length=120)
    country: str = Field(min_length=2, max_length=120)
    applicant_role: str = Field(min_length=2, max_length=160)
    citizenship_or_residency: str = Field(default="", max_length=240)
    opportunity_types: list[str] = Field(default_factory=list)
    event_types: list[str] = Field(default_factory=list)
    event_regions: list[str] = Field(default_factory=list)
    event_formats: list[str] = Field(default_factory=list)
    requires_travel_support: bool = False
    minimum_lead_days: int = Field(default=7, ge=0, le=365)

    @field_validator(
        "description",
        "career_stage",
        "institution_type",
        "country",
        "applicant_role",
        "citizenship_or_residency",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator(
        "opportunity_types", "event_types", "event_regions", "event_formats", mode="before"
    )
    @classmethod
    def _clean_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = value.split(",")
        if not isinstance(value, (list, tuple, set)):
            return []
        return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def load_opportunity_profile(path: str | Path) -> OpportunityProfile:
    profile_path = Path(path)
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Opportunity profile not found: {profile_path}; complete opportunity setup first"
        )
    data = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Opportunity profile at {profile_path} must be a YAML mapping")
    try:
        return OpportunityProfile(**data)
    except Exception as exc:
        raise ValueError(f"Opportunity profile at {profile_path} is invalid: {exc}") from exc


@dataclass(frozen=True)
class OpportunityAssessment:
    actionable: bool
    eligibility: str
    reason: str


_INSTITUTION_MARKERS: dict[str, tuple[str, ...]] = {
    "university": ("higher education", "university", "college"),
    "nonprofit": ("nonprofit", "non-profit", "higher education"),
    "company": ("small business", "for profit", "for-profit", "business"),
    "government": ("government", "state", "county", "city", "township"),
    "individual": ("individual",),
}

_OPPORTUNITY_TYPE_FAMILIES: dict[str, tuple[str, ...]] = {
    "grant": ("grant", "cooperative agreement", "funding opportunity"),
    "fellowship": ("fellowship", "scholarship", "training", "career development"),
    "award": ("award", "prize"),
    "travel support": ("travel", "accessibility grant", "bursary"),
}

_REGION_MARKERS: dict[str, tuple[str, ...]] = {
    "north america": ("united states", "u.s.", "usa", "canada", "mexico"),
    "europe": (
        "europe",
        "germany",
        "heidelberg",
        "united kingdom",
        "france",
        "italy",
        "spain",
        "switzerland",
        "netherlands",
        "belgium",
        "austria",
        "sweden",
        "denmark",
        "norway",
        "finland",
    ),
    "asia": ("asia", "china", "japan", "korea", "india", "singapore", "taiwan"),
}


def _iso_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _selected_type_matches(selected: list[str], actual: object) -> bool:
    if not selected or not actual:
        return True
    normalized = str(actual).casefold().replace("_", " ")
    for choice in selected:
        wanted = choice.casefold().replace("_", " ")
        if wanted in normalized or normalized in wanted:
            return True
        for family, members in _OPPORTUNITY_TYPE_FAMILIES.items():
            if wanted == family and any(member in normalized for member in members):
                return True
    return False


def _region_matches(selected: list[str], location: str) -> bool:
    if not selected or not location:
        return True
    normalized = location.casefold()
    for choice in selected:
        wanted = choice.casefold().replace("_", " ")
        if wanted == "online":
            continue
        if wanted in normalized:
            return True
        if any(marker in normalized for marker in _REGION_MARKERS.get(wanted, ())):
            return True
    return False


def assess_opportunity(
    metadata: dict[str, Any],
    profile: OpportunityProfile,
    *,
    today: date | None = None,
) -> OpportunityAssessment:
    """Apply only explicit status, timing, type, and institution constraints.

    Unknown eligibility stays visible and is labelled for manual verification;
    it is never silently promoted to "eligible". Topic relevance is handled by
    the existing semantic ranker, not duplicated here.
    """
    now = today or date.today()
    status = str(metadata.get("status") or "unknown").casefold()
    if status in {"closed", "cancelled", "canceled", "archived"}:
        return OpportunityAssessment(False, "unknown", f"status is {status}")

    deadline = _iso_date(metadata.get("deadline"))
    if deadline is not None:
        days_left = (deadline - now).days
        if days_left < 0:
            return OpportunityAssessment(False, "unknown", "deadline has passed")
        if days_left < profile.minimum_lead_days:
            return OpportunityAssessment(
                False,
                "unknown",
                f"only {days_left} days remain; below preferred lead time",
            )

    section_type = str(metadata.get("record_type") or "opportunity")
    selected_types = profile.event_types if section_type == "event" else profile.opportunity_types
    actual_type = metadata.get("event_type" if section_type == "event" else "opportunity_type")
    if not _selected_type_matches(selected_types, actual_type):
        return OpportunityAssessment(False, "unknown", "type is outside preferences")

    if section_type == "event":
        if not _selected_type_matches(profile.event_formats, metadata.get("format")):
            return OpportunityAssessment(False, "unknown", "format is outside preferences")
        location = str(metadata.get("location") or "")
        if metadata.get("format") != "online" and not _region_matches(
            profile.event_regions, location
        ):
            return OpportunityAssessment(False, "unknown", "location is outside preferences")
        return OpportunityAssessment(True, "unknown", "verify event requirements")

    tags = [
        str(tag).casefold() for tag in (metadata.get("eligibility_tags") or []) if str(tag).strip()
    ]
    if not tags:
        return OpportunityAssessment(True, "unknown", "verify official eligibility")

    if any("unrestricted" in tag or "all applicant" in tag for tag in tags):
        return OpportunityAssessment(True, "likely", "call is open to broad applicant types")
    if all("other" in tag or "see text" in tag for tag in tags):
        return OpportunityAssessment(True, "unknown", "official eligibility text needs review")

    country = profile.country.casefold()
    is_us = country in {"us", "u.s.", "usa", "united states", "united states of america"}
    eligibility_text = str(metadata.get("eligibility_description") or "").casefold()
    if not is_us and eligibility_text:
        welcomes_foreign = any(
            marker in eligibility_text
            for marker in ("foreign institution", "non-domestic", "international applicant")
        )
        domestic_only = any(
            marker in eligibility_text
            for marker in ("domestic only", "u.s. only", "united states only")
        )
        if domestic_only and not welcomes_foreign:
            return OpportunityAssessment(False, "unlikely", "country is outside eligibility")

    institution = profile.institution_type.casefold()
    markers: tuple[str, ...] = ()
    for key, values in _INSTITUTION_MARKERS.items():
        if key in institution:
            markers = values
            break
    if not markers:
        return OpportunityAssessment(True, "unknown", "institution type needs review")
    if any(marker in tag for marker in markers for tag in tags):
        if not is_us and not eligibility_text:
            return OpportunityAssessment(True, "unknown", "country eligibility needs review")
        return OpportunityAssessment(True, "likely", "institution type appears eligible")
    if any("individual" in tag for tag in tags):
        return OpportunityAssessment(True, "unknown", "individual eligibility needs review")
    return OpportunityAssessment(False, "unlikely", "institution type is not listed")


def opportunity_display(
    metadata: dict[str, Any], profile: OpportunityProfile | None = None
) -> dict[str, str]:
    """Return compact, presentation-ready facts from structured metadata."""

    def money(value: object) -> str:
        if not isinstance(value, (int, float)):
            return ""
        currency = str(metadata.get("currency") or "").upper()
        prefix = "$" if currency == "USD" else f"{currency} " if currency else ""
        return f"{prefix}{value:,.0f}"

    amount_min = money(metadata.get("amount_min"))
    amount_max = money(metadata.get("amount_max"))
    if amount_min and amount_max:
        amount = amount_min if amount_min == amount_max else f"{amount_min}–{amount_max}"
    elif amount_max:
        amount = f"Up to {amount_max}"
    elif amount_min:
        amount = f"From {amount_min}"
    else:
        amount = "Amount not stated"

    start = str(metadata.get("event_start") or "")
    end = str(metadata.get("event_end") or "")
    event_dates = start if not end or end == start else f"{start}–{end}"
    assessment = assess_opportunity(metadata, profile) if profile is not None else None
    eligibility = (
        "Potentially eligible"
        if assessment and assessment.eligibility == "likely"
        else "Eligibility needs verification"
    )
    return {
        "status": str(metadata.get("status") or "unknown").replace("_", " ").title(),
        "deadline": str(metadata.get("deadline") or "Not stated"),
        "amount": amount,
        "eligibility": eligibility,
        "eligibility_reason": assessment.reason if assessment else "Check official requirements",
        "event_dates": event_dates,
        "location": str(metadata.get("location") or "Not stated"),
        "format": str(metadata.get("format") or "").replace("_", " ").title(),
        "official": "Verified official source" if metadata.get("official") else "Verify source",
    }
