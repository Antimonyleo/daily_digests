"""A stored or provider summary shows no feed header as a finding and no vague caveat.

Fixtures are public-style patterns, not copies of user data.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from dailydigest import summarize as sm
from dailydigest.store import ItemRow
from dailydigest.web import _display_summary, _summary_fields

HEADER = "Example Nanotechnology, Published online: 13 October 2026; doi:10.1038/s41565-026-01234-5"
FINDING = "Key finding: We show that DNA origami boxes release cargo at pH 5."
CONTEXT = "DNA origami boxes for endosomal cargo release and uptake mechanism"


def row(abstract=CONTEXT, title="DNA origami boxes", summary=""):
    return ItemRow(id=1, source="Example Journal", section="research", external_id="guard",
                   url="https://example.org/a", title=title, abstract=abstract,
                   metadata_json=json.dumps({}), summary=summary)


# --- F4: a finding that was only feed metadata ------------------------------

@pytest.mark.parametrize("stored", [
    f"Key finding: {HEADER}",
    f"Key finding: {HEADER}\nWhy read: For DNA origami: cargo release.",
    f"Key finding: {HEADER}\nCaveat: Only tested in HeLa cells.",
])
def test_header_only_finding_yields_no_summary(stored):
    assert sm.normalize_cached_summary(stored, "research") == ""


def test_header_followed_by_real_text_keeps_the_text():
    stored = f"Key finding: {HEADER} We show that origami boxes release cargo at pH 5."
    assert sm.normalize_cached_summary(stored, "research") == (
        "Key finding: We show that origami boxes release cargo at pH 5.")


def test_template_fallback_cannot_show_the_header():
    shown = _display_summary(f"Key finding: {HEADER}\nWhy read: For DNA origami.", "research", row())
    assert shown == "" and _summary_fields(shown) == {}


def test_unlabeled_text_is_still_returned_unchanged():
    assert sm.normalize_cached_summary("A plain stored note.", "research") == "A plain stored note."


# --- F5: only study-specific caveats ----------------------------------------

@pytest.mark.parametrize("caveat", [
    "Further work will help clarify the implications.",
    "Further studies are needed to understand the broader implications.",
    "More research is needed to confirm these findings.",
    "Future work could address remaining questions.",
    "Results should be interpreted with care.",
    "No abstract provided; cannot evaluate the work's scope, methods, or findings.",
    "This is a review article synthesizing prior work, not a primary empirical contribution.",
    "RSS feed contains only citation and title.",
    "Relevance to DNA origami in your target systems remains indirect.",
])
def test_vague_caveats_are_dropped(caveat):
    stored = f"{FINDING}\nCaveat: {caveat}"
    assert sm.normalize_cached_summary(stored, "research", context=CONTEXT) == FINDING
    assert sm._finalize_provider_summary(row(), stored) == FINDING


@pytest.mark.parametrize("caveat", [
    "Tested only in HeLa cells over 24 h.",
    "Cargo release in serum was not tested.",
    "The uptake mechanism remains unclear.",
    "Experiments were limited to in vitro conditions.",
    "Performance was evaluated in a cohort of 12 patients.",
    "Study is entirely computational (DFT-based).",
    "Generalizability of the origami boxes to primary cells has not been demonstrated.",
    "Model benchmarked primarily on defined origami boxes.",
])
def test_study_specific_caveats_are_kept(caveat):
    stored = f"{FINDING}\nCaveat: {caveat}"
    expected = f"{FINDING}\nCaveat: {caveat}"
    assert sm.normalize_cached_summary(stored, "research", context=CONTEXT) == expected
    assert sm._finalize_provider_summary(row(), stored) == expected


def test_real_limitation_clause_survives_a_neighbouring_abstract_note():
    stored = f"{FINDING}\nCaveat: Abstract text is incomplete; tested only in HeLa cells."
    assert sm.normalize_cached_summary(stored, "research", context=CONTEXT) == (
        f"{FINDING}\nCaveat: Tested only in HeLa cells.")


def test_caveat_naming_nothing_from_the_study_is_dropped():
    stored = f"{FINDING}\nCaveat: Long-term stability was not assessed."
    # "stability" appears nowhere in this item; only the cue word remains.
    assert sm.normalize_cached_summary(stored, "research", context="unrelated words only") == FINDING
    assert sm.normalize_cached_summary(
        stored, "research", context="We measured long-term stability of the boxes.") == stored


def test_web_display_checks_caveat_against_the_item_abstract():
    stored = f"{FINDING}\nCaveat: Endosomal escape was not quantified."
    assert "Caveat" not in _display_summary(stored, "research", row(abstract="Boxes."))
    shown = _display_summary(stored, "research", row(abstract="We track endosomal escape."))
    assert _summary_fields(shown)["Caveat"] == "Endosomal escape was not quantified."


def test_extractive_limitation_from_abstract_survives_display():
    abstract = ("We built DNA origami boxes that open at pH 5 and release siRNA cargo inside cells. "
                "In HeLa cells the boxes silenced the target gene by 80 percent within 24 hours. "
                "Our analysis was limited by the absence of long-term follow-up in animals.")
    item = row(abstract=abstract, title="pH-gated DNA origami boxes deliver siRNA")
    summary = sm._extractive(item)
    caveat = sm._summary_parts(summary)["Caveat"]
    assert caveat.startswith("Our analysis was limited")
    assert sm._summary_parts(_display_summary(summary, "research", item))["Caveat"] == caveat


@pytest.mark.parametrize("limitation", [
    "Further studies are needed to confirm these findings in patients.",
    "Larger prospective trials are warranted to validate the risk score.",
])
def test_extractor_and_display_agree_on_recommendation_style_limitations(limitation):
    abstract = ("We derived a risk score from routine blood tests in 412 adults. "
                "The score predicted sepsis with an AUC of 0.86 in the validation set. "
                f"This study is limited by its retrospective design. {limitation}")
    item = row(abstract=abstract, title="A blood-test risk score for sepsis")
    caveat = sm._summary_parts(sm._extractive(item)).get("Caveat")
    assert caveat
    assert sm._summary_parts(_display_summary(sm._extractive(item), "research", item)).get("Caveat") == caveat


def test_extractor_skips_a_limitation_the_display_would_hide():
    abstract = ("We built DNA origami boxes that open at pH 5 and release siRNA cargo inside cells. "
                "In HeLa cells the boxes silenced the target gene by 80 percent within 24 hours. "
                "However, the present work is limited, and further work will help clarify the implications.")
    item = row(abstract=abstract, title="pH-gated DNA origami boxes deliver siRNA")
    summary = sm._extractive(item)
    assert sm._summary_parts(summary).get("Caveat") == sm._summary_parts(
        _display_summary(summary, "research", item)).get("Caveat")


def test_details_abstract_and_highlight_drop_the_feed_header():
    from dailydigest import store, web

    store.init_db()
    with store.session_scope() as session:
        session.add(store.ItemRow(id=7, source="Example Nanotechnology", section="research",
                                  external_id="10.1038/s41565-026-01234-5", url="https://doi.org/10.1038/x",
                                  title="DNA origami boxes", abstract=f"{HEADER} We show that DNA origami boxes release cargo.",
                                  published_at=datetime(2026, 10, 13, tzinfo=timezone.utc),
                                  metadata_json="{}"))
        session.add(store.ItemRow(id=8, source="Example Nanotechnology", section="research",
                                  external_id="10.1038/s41565-026-09999-9", url="https://doi.org/10.1038/y",
                                  title="Header only listing", abstract=HEADER,
                                  published_at=datetime(2026, 10, 13, tzinfo=timezone.utc),
                                  metadata_json="{}"))
    store.write_digest("2026-10-13", [("R1", 7, .9), ("R2", 8, .8)])
    sections, _ = web._load_today("2026-10-13", include_disabled_sections=True)
    entries = {entry["id"]: entry for entry in next(s for s in sections if s["key"] == "research")["entries"]}
    assert entries[7]["abstract"] == "We show that DNA origami boxes release cargo."
    assert "".join(segment["text"] for segment in entries[7]["abstract_segments"]) == entries[7]["abstract"]
    assert entries[8]["abstract"] == ""  # shown as title-only rather than a header
