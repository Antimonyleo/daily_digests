"""Opportunity/event profile and deterministic matching helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from .config import user_local_date


# Opportunity types listed as calls (in Events & Calls when it is on) rather
# than as research funding.
CALL_TYPES = frozenset({"fellowship", "award", "prize", "travel_support"})


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

_EU_MEMBERS = (
    "austria", "belgium", "bulgaria", "croatia", "cyprus", "czechia", "czech republic",
    "denmark", "estonia", "finland", "france", "germany", "greece", "hungary", "ireland",
    "italy", "latvia", "lithuania", "luxembourg", "malta", "netherlands", "poland",
    "portugal", "romania", "slovakia", "slovenia", "spain", "sweden",
)
# Countries associated to Horizon Europe take part on the same terms as members.
_HORIZON_ASSOCIATED = (
    "albania", "armenia", "bosnia and herzegovina", "canada", "faroe islands", "georgia",
    "iceland", "israel", "kosovo", "moldova", "montenegro", "new zealand", "north macedonia",
    "norway", "serbia", "south korea", "republic of korea", "switzerland", "tunisia",
    "turkey", "türkiye", "ukraine", "united kingdom", "uk", "great britain", "england",
    "scotland", "wales", "northern ireland",
)
_US_NAMES = ("us", "u.s.", "usa", "u.s.a.", "united states", "united states of america", "america")

_REGION_MARKERS: dict[str, tuple[str, ...]] = {
    "north america": ("united states", "u.s.", "usa", "canada", "mexico"),
    "europe": (
        "europe", "heidelberg", *_EU_MEMBERS, "united kingdom", "uk", "england", "scotland",
        "wales", "switzerland", "norway", "iceland", "serbia", "ukraine", "turkey", "türkiye",
        "albania", "bosnia", "montenegro", "north macedonia", "moldova", "armenia",
    ),
    "asia": (
        "asia", "china", "japan", "korea", "india", "singapore", "taiwan", "hong kong",
        "thailand", "vietnam", "malaysia", "indonesia", "philippines", "bangladesh",
        "pakistan", "sri lanka", "nepal",
    ),
    "middle east": ("israel", "saudi arabia", "united arab emirates", "qatar", "jordan", "lebanon", "iran"),
    "south america": ("brazil", "argentina", "chile", "colombia", "peru", "uruguay", "ecuador"),
    "africa": ("africa", "egypt", "morocco", "tunisia", "kenya", "nigeria", "ghana", "ethiopia", "rwanda"),
    "oceania": ("australia", "new zealand"),
}


# Names a reader may type for their current country, keyed to ISO 3166-1
# alpha-2 codes. Sources and calls list the codes (or "eu"/"international")
# whose readers they serve.
_COUNTRY_CODES: dict[str, str] = {
    **{name: "us" for name in _US_NAMES},
    "austria": "at", "belgium": "be", "bulgaria": "bg", "croatia": "hr", "cyprus": "cy",
    "czechia": "cz", "czech republic": "cz", "denmark": "dk", "estonia": "ee", "finland": "fi",
    "france": "fr", "germany": "de", "deutschland": "de", "greece": "gr", "hungary": "hu",
    "ireland": "ie", "italy": "it", "latvia": "lv", "lithuania": "lt", "luxembourg": "lu",
    "malta": "mt", "netherlands": "nl", "the netherlands": "nl", "holland": "nl", "poland": "pl",
    "portugal": "pt", "romania": "ro", "slovakia": "sk", "slovenia": "si", "spain": "es",
    "sweden": "se", "united kingdom": "gb", "uk": "gb", "u.k.": "gb", "great britain": "gb",
    "britain": "gb", "england": "gb", "scotland": "gb", "wales": "gb", "northern ireland": "gb",
    "switzerland": "ch", "norway": "no", "iceland": "is", "israel": "il", "turkey": "tr",
    "türkiye": "tr", "ukraine": "ua", "serbia": "rs", "albania": "al", "armenia": "am",
    "bosnia and herzegovina": "ba", "georgia": "ge", "kosovo": "xk", "moldova": "md",
    "montenegro": "me", "north macedonia": "mk", "faroe islands": "fo", "tunisia": "tn",
    "canada": "ca", "mexico": "mx", "brazil": "br", "argentina": "ar", "chile": "cl",
    "colombia": "co", "peru": "pe", "uruguay": "uy", "japan": "jp", "china": "cn",
    "people's republic of china": "cn", "prc": "cn", "hong kong": "hk", "taiwan": "tw",
    "south korea": "kr", "korea": "kr", "republic of korea": "kr", "singapore": "sg",
    "india": "in", "pakistan": "pk", "bangladesh": "bd", "thailand": "th", "vietnam": "vn",
    "viet nam": "vn", "malaysia": "my", "indonesia": "id", "philippines": "ph",
    "australia": "au", "new zealand": "nz", "south africa": "za", "egypt": "eg",
    "morocco": "ma", "kenya": "ke", "nigeria": "ng", "ghana": "gh", "ethiopia": "et",
    "rwanda": "rw", "saudi arabia": "sa", "united arab emirates": "ae", "uae": "ae",
    "qatar": "qa", "jordan": "jo", "lebanon": "lb", "iran": "ir", "russia": "ru",
}
_EU_CODES = {
    _COUNTRY_CODES[name] for name in (*_EU_MEMBERS, *_HORIZON_ASSOCIATED) if name in _COUNTRY_CODES
}


# Abbreviations are matched without their dots, so "U.S.", "U.S.A." and "U.K."
# find the same codes as "US", "USA" and "UK".
_COUNTRY_LOOKUP = {name.replace(".", ""): code for name, code in _COUNTRY_CODES.items()}


def country_code(country: str) -> str | None:
    name = re.sub(r"\s+", " ", str(country or "").casefold().replace(".", "").strip())
    if len(name) == 2 and name.isalpha() and name in set(_COUNTRY_CODES.values()):
        return name
    return _COUNTRY_LOOKUP.get(name)


def funding_regions(country: str) -> set[str] | None:
    """Applicant regions for a reader's current country, or None when unknown.

    A recognised country gets its own code, plus "eu" for EU members and
    countries associated to Horizon Europe. Text DailyDigest cannot place
    returns None, so that reader sees every source, labelled for review.
    """
    code = country_code(country)
    if code is None:
        return None
    return {code, "eu"} if code in _EU_CODES else {code}


def serves_reader(applicant_regions: object, reader_regions: set[str] | None) -> bool:
    """Whether a source or call listing these applicant regions suits the reader."""
    wanted = {str(region).casefold() for region in applicant_regions or [] if str(region).strip()}
    if not wanted or reader_regions is None or "international" in wanted:
        return True
    return bool(wanted & reader_regions)


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
        # Whole words only: "uk" must not match Fukuoka, nor "india" Indiana.
        if any(re.search(rf"(?<![\w.]){re.escape(marker)}(?!\w)", normalized)
               for marker in _REGION_MARKERS.get(wanted, ())):
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
    now = today or user_local_date()
    status = str(metadata.get("status") or "unknown").casefold()
    if status in {"closed", "cancelled", "canceled", "archived"}:
        return OpportunityAssessment(False, "unknown", f"status is {status}")

    section_type = str(metadata.get("record_type") or "opportunity").casefold()
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

    if section_type == "event":
        event_start = _iso_date(metadata.get("event_start"))
        if event_start is not None:
            days_until = (event_start - now).days
            if days_until < 0:
                return OpportunityAssessment(False, "unknown", "event has passed")
            if days_until < profile.minimum_lead_days:
                return OpportunityAssessment(
                    False,
                    "unknown",
                    f"only {days_until} days remain; below preferred lead time",
                )

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

    if not serves_reader(metadata.get("applicant_regions"), funding_regions(profile.country)):
        return OpportunityAssessment(False, "unlikely", "open to applicants in other countries")
    country = profile.country.casefold()
    is_us = country in {"us", "u.s.", "usa", "united states", "united states of america"}
    eligibility_text = str(metadata.get("eligibility_description") or "").casefold()
    # Only short affirmative clauses justify exclusion. Negated, qualified,
    # or more complex eligibility prose needs review, not substring inference.
    domestic_only = re.search(
        r"(?:^|[.;]\s*)"
        r"(?:(?:applicants|applications|institutions|organizations|eligibility) "
        r"(?:must be|are|is) )?"
        r"(?:domestic only|u\.s\. only|united states only)\s*(?:[.;]|$)",
        eligibility_text,
    )
    if not is_us and domestic_only:
        return OpportunityAssessment(False, "unlikely", "country is outside eligibility")

    tags = [
        str(tag).casefold() for tag in (metadata.get("eligibility_tags") or []) if str(tag).strip()
    ]
    if not tags:
        return OpportunityAssessment(True, "unknown", "verify official eligibility")

    official = metadata.get("official") is True
    if any("unrestricted" in tag or "all applicant" in tag for tag in tags):
        if not is_us:
            return OpportunityAssessment(True, "unknown", "country eligibility needs review")
        if official:
            return OpportunityAssessment(True, "likely", "call is open to broad applicant types")
        return OpportunityAssessment(True, "unknown", "verify eligibility on the official source")
    if all("other" in tag or "see text" in tag for tag in tags):
        return OpportunityAssessment(True, "unknown", "official eligibility text needs review")

    institution = profile.institution_type.casefold()
    markers: tuple[str, ...] = ()
    for key, values in _INSTITUTION_MARKERS.items():
        if key in institution:
            markers = values
            break
    if not markers:
        return OpportunityAssessment(True, "unknown", "institution type needs review")
    if any(marker in tag for marker in markers for tag in tags):
        # Applicant-type tags and free text do not establish country eligibility;
        # even a mention of foreign institutions may explicitly exclude them.
        if not is_us:
            return OpportunityAssessment(True, "unknown", "country eligibility needs review")
        if not official:
            return OpportunityAssessment(True, "unknown", "verify eligibility on the official source")
        return OpportunityAssessment(True, "likely", "institution type appears eligible")
    if any("individual" in tag for tag in tags):
        return OpportunityAssessment(True, "unknown", "individual eligibility needs review")
    return OpportunityAssessment(False, "unlikely", "institution type is not listed")


# Reasons that only say "check the official page": cards and the Saved page
# show the eligibility word alone rather than repeat them.
_GENERIC_REASONS = frozenset({
    "verify official eligibility", "verify event requirements", "Check official requirements",
    "verify eligibility on the official source",
})


def opportunity_display(
    metadata: dict[str, Any], profile: OpportunityProfile | None = None, section: str = ""
) -> dict[str, str]:
    """Return compact, presentation-ready facts from structured metadata.

    ``kind`` is "event", "call" or "funding"; a record in the events section
    without a record type is an event.
    """

    def money(value: object) -> str:
        if not isinstance(value, (int, float)):
            return ""
        currency = str(metadata.get("currency") or "").upper()
        symbol = {"USD": "$", "GBP": "£", "EUR": "€"}.get(currency)
        prefix = symbol or (f"{currency} " if currency else "")
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
    deadline = str(metadata.get("deadline") or "")
    if not deadline and metadata.get("expires"):
        # Announcements that accept several due dates state only when they expire.
        deadline = f"See the announcement for due dates (expires {metadata['expires']})"
    elif not deadline and str(metadata.get("status") or "") == "recurring":
        deadline = "Recurring call; see the official page for this round"
    record_type = str(metadata.get("record_type") or "")
    reason = assessment.reason if assessment else "Check official requirements"
    return {
        # Fellowship and award calls share Events & Calls with events but
        # show funding facts (amount, deadline, eligibility).
        "kind": (record_type if record_type in {"event", "call"}
                 else "event" if section == "events" else "funding"),
        "status": str(metadata.get("status") or "unknown").replace("_", " ").title(),
        "deadline": deadline or "Not stated",
        "amount": amount,
        "eligibility": eligibility,
        "eligibility_reason": reason,
        "eligibility_note": "" if reason in _GENERIC_REASONS else reason,
        "event_dates": event_dates,
        "location": str(metadata.get("location") or "Not stated"),
        "format": str(metadata.get("format") or "").replace("_", " ").title(),
        "official": "Verified official source" if metadata.get("official") else "Verify source",
    }
