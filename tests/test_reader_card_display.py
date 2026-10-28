"""Reader card order, compact metadata and task-specific opportunity facts."""

import json
import re
from datetime import date, datetime, timezone

import pytest

from tests.test_web import _request, _text_payload


# --- display transforms -----------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("Robinson, N. E.; Gerber, B.; Zhang, W.; Wang, K.", "N. E. Robinson … K. Wang (4 authors)"),
    ("Linghao Zhang, Qinze Rong, Xueting Wang, Xin Su", "Linghao Zhang … Xin Su (4 authors)"),
    ("Dao TM, Ghent S, Uversky VN, Rahman T", "Dao TM … Rahman T (4 authors)"),
    ("Tinghui Zhang, \nZhendong Yang, \nZhijie Xiao", "Tinghui Zhang … Zhijie Xiao (3 authors)"),
    ("Smith, J., Doe, A. B.", "J. Smith and A. B. Doe"),
    ("David Blumenthal and James A. Morone", "David Blumenthal and James A. Morone"),
    ("Giovanni Maglia", "Giovanni Maglia"),
    ("National Institutes of Health", "National Institutes of Health"),
    # Ambiguous structure: no invented count.
    ("Smith, John, Doe, Jane", "Smith et al."),
    ("A Author, B Writer et al.", "A Author et al."),
    ("Ada Lovelace, Department of Chemistry, University of Basel, Basel 4056", "Ada Lovelace et al."),
    ("Department of Chemistry, University of Basel, Basel 4056, Switzerland", ""),
    ("", ""),
])
def test_author_line_is_short_only_when_parsing_is_reliable(raw, expected):
    from dailydigest import web

    shown = web._author_display(raw)
    assert shown["short"] == expected
    assert shown["full"] == " ".join(raw.split())
    if "authors)" not in expected:
        assert shown["count"] in (None, 1, 2)


def test_openalex_sized_author_lists_do_not_claim_a_count():
    from dailydigest import web

    raw = ", ".join(f"Author Number{chr(65 + i % 26)}{i // 26}" for i in range(100))
    raw = re.sub(r"\d", "", raw)  # names may not contain digits
    shown = web._author_display(raw)
    assert shown["count"] is None and shown["short"].endswith("et al.")


def test_friendly_dates_keep_calendar_day_and_machine_value():
    from dailydigest import web

    reference = date(2026, 10, 13)
    late = datetime(2026, 10, 12, 0, 0, tzinfo=timezone.utc)
    assert web._friendly_date(late, reference) == {"label": "Oct 12", "iso": "2026-10-12"}
    assert web._friendly_date("2027-01-05", reference) == {"label": "Jan 5, 2027", "iso": "2027-01-05"}
    assert web._friendly_date("Rolling", reference) == {"label": "Rolling", "iso": ""}
    assert web._friendly_date(None, reference) == {"label": "", "iso": ""}
    assert web._friendly_span("2026-11-03–2026-11-05", reference) == "Nov 3–5"
    # Impossible calendar dates from a feed are shown as stored, never a page error.
    assert web._friendly_span("2026-02-30–2026-03-02", reference) == "2026-02-30–2026-03-02"
    assert web._friendly_span("2026-02-30", reference) == "2026-02-30"


def test_source_and_type_labels_drop_internal_wording_only():
    from dailydigest import web

    assert web._display_source_name("bioRxiv (recent)") == "bioRxiv"
    assert web._display_source_name("Chem (Cell Press)") == "Chem (Cell Press)"
    assert web._display_publication_label("Publication") == ""
    for kept in ("Preprint", "Review", "Perspective", "Retracted", "Research article"):
        assert web._display_publication_label(kept) == kept
    assert web._selection_note("capped preprint slot") == "One of a few preprints shown each day."
    assert "capped" not in web._selection_note("capped arXiv CS slot").lower()


def test_highlight_respects_unicode_word_boundaries():
    from dailydigest import web

    segments = web._highlight_segments("Straße, STRASSEN and straßenbahn; straße.", "straße")
    assert [s["text"] for s in segments if s["match"]] == ["Straße", "straße"]
    assert "".join(s["text"] for s in segments) == "Straße, STRASSEN and straßenbahn; straße."
    assert web._highlight_segments("text", "") == [{"text": "text", "match": False}]


def test_unknown_evidence_collapses_but_real_quotes_remain():
    from dailydigest import web

    view = web._evidence_view({
        "contribution_claims": {"claims": [{"quote": "Q1"}, {"quote": "Q1"}]},
        "article_evidence": {"dimensions": {"validation": {"claims": []}, "data_code": {"claims": []},
                                            "limitations": {"claims": [{"quote": "Small sample."}]}}},
    })
    assert view["known"] == [{"label": "Contribution evidence", "quotes": ["Q1"]},
                             {"label": "Limitations", "quotes": ["Small sample."]}]
    assert view["unknown_text"] == "Validation evidence and Data/code availability"
    assert web._evidence_view({}) is None


# --- rendered pages ----------------------------------------------------------

DIGEST = "2026-10-13"


def _seed(monkeypatch, tmp_path):
    from dailydigest import config, store, web

    monkeypatch.setenv("INCLUDE_OPPORTUNITIES", "true")
    config.reload_settings()
    monkeypatch.setattr(web, "SETTINGS", config.SETTINGS)
    profile = tmp_path / "reader.yaml"
    profile.write_text("name: Ada\nkeywords: [DNA origami]\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile)
    monkeypatch.setattr(web, "_digest_id", lambda: DIGEST)
    store.init_db()
    with store.session_scope() as session:
        session.add(store.DigestRow(id=DIGEST, item_count=3))
        paper = store.ItemRow(
            source="bioRxiv (recent)", section="research", external_id="paper",
            url="https://example.invalid/paper", title="Fixture DNA origami paper",
            authors="Robinson, N. E.; Gerber, B.; Zhang, W.",
            abstract="We fold <b>DNA origami</b> tiles; DNA origamis differ.",
            summary=("Key finding: Tiles fold in one step.\n"
                     "Why read: Compare fold yield.\nCaveat: Only one tile design was tested."),
            published_at=datetime(2026, 10, 12, tzinfo=timezone.utc), digest_id=DIGEST, item_label="R1",
        )
        plain = store.ItemRow(
            source="Fixture Journal", section="research", external_id="plain",
            url="https://example.invalid/plain", title="Paper without a reason",
            abstract="An abstract.", summary="Key finding: A result.\nCaveat: A limit.",
            published_at=datetime(2026, 10, 12, tzinfo=timezone.utc), digest_id=DIGEST, item_label="R2",
        )
        grant = store.ItemRow(
            source="Grants.gov", section="opportunities", external_id="grant",
            url="https://example.invalid/grant", title="Fixture funding call",
            authors="National Institutes of Health",
            summary="Key finding: Supports consortium centres.\nCaveat: None stated.",
            metadata_json=json.dumps({"deadline": "2026-12-14", "amount_max": 800000, "currency": "USD",
                                      "status": "open", "official": True}),
            published_at=datetime(2026, 10, 9, tzinfo=timezone.utc), digest_id=DIGEST, item_label="O1",
        )
        session.add_all([paper, plain, grant])
        session.flush()
        ids = paper.id, plain.id, grant.id
    store.write_digest(DIGEST, [("R1", ids[0], .9), ("R2", ids[1], .8), ("O1", ids[2], .7)])
    store.write_digest_features(DIGEST, [
        ("R1", ids[0], .9, {"primary_facet": "DNA origami", "source_bucket": "bio_med_preprint",
                            "selection_reason": "capped preprint slot",
                            "article_evidence": {"dimensions": {k: {"claims": []} for k in
                                                                ("validation", "data_code", "limitations")}}}),
        ("R2", ids[1], .8, {"source_bucket": "published_journal"}),
        ("O1", ids[2], .7, {"source_bucket": "funding"}),
    ])
    return store, web, ids


def _card(html, item_id):
    start = html.index(f'data-item-id="{item_id}"')
    end = html.find('<div class="item" data-item-id', start + 1)
    return html[start:end if end != -1 else None]


def test_card_order_compact_meta_and_caveat_only_in_details(monkeypatch, tmp_path):
    _store, web, (paper_id, plain_id, _grant) = _seed(monkeypatch, tmp_path)
    html = _text_payload(web.index(_request("GET", "/")))
    card = _card(html, paper_id)
    meta = re.search(r'<div class="meta">(.*?)</div>', card, re.S).group(1)
    assert 'N. E. Robinson … W. Zhang (3 authors)' in meta
    assert '<span class="venue">bioRxiv</span>' in meta and "(recent)" not in meta
    assert 'Publication' not in meta
    assert '<time datetime="2026-10-12">Oct 12</time>' in meta
    # title → source line → finding → why read → details → actions
    order = [card.index(marker) for marker in (
        'class="title-link"', 'class="meta"', 'Tiles fold in one step.', 'data-summary-label="Why read"',
        '<details class="item-details">', 'data-summary-label="Caveat"', 'class="vote-panel"', 'class="bookmark-btn')]
    assert order == sorted(order)
    assert card.count("Only one tile design was tested.") == 1
    assert 'class="summary-field reading-context-field" data-summary-label="Caveat"' not in card
    # Full authors in Details; matched term highlighted with escaping, never raw HTML.
    assert "Robinson, N. E.; Gerber, B.; Zhang, W." in card
    assert '&lt;b&gt;<mark class="match-term">DNA origami</mark>&lt;/b&gt;' in card
    assert "DNA origamis" in card and '<mark class="match-term">DNA origamis' not in card
    assert 'class="match-quote"' not in card
    assert card.count("Unknown from supplied text") == 1
    assert "One of a few preprints shown each day." in card and "apped preprint slot" not in card
    # No Why read field when the summary has none.
    assert 'data-summary-label="Why read"' not in _card(html, plain_id)


def test_funding_card_leads_with_amount_deadline_eligibility_and_link(monkeypatch, tmp_path):
    _store, web, (_paper, _plain, grant_id) = _seed(monkeypatch, tmp_path)
    card = _card(_text_payload(web.index(_request("GET", "/"))), grant_id)
    facts = card.index('class="opportunity-facts"')
    assert facts < card.index("Supports consortium centres.")
    for text in ("Up to $800,000", '<time datetime="2026-12-14">Dec 14</time>', "Open",
                 "Eligibility needs verification", 'href="https://example.invalid/grant"',
                 "Official call", "Add to calendar", "Mark known", "Save for later"):
        assert text in card, text
    assert "Potentially eligible" not in card  # no profile: never upgraded


def test_contents_tabs_and_controls_survive(monkeypatch, tmp_path):
    _store, web, _ids = _seed(monkeypatch, tmp_path)
    html = _text_payload(web.index(_request("GET", "/")))
    assert 'class="contents-links"' in html and "data-tabs-scrolled" in html
    for control in ('data-grade="100"', 'data-grade="70"', 'data-grade="40"', 'data-grade="10"',
                    'class="btn clear-rating"', 'class="copy-summary btn"', 'class="bookmark-btn known-btn'):
        assert control in html, control


def test_saved_card_moves_caveat_into_details(monkeypatch, tmp_path):
    store, web, (paper_id, _plain, _grant) = _seed(monkeypatch, tmp_path)
    store.set_bookmark(paper_id, True)
    html = _text_payload(web.saved_items(_request("GET", "/saved")))
    assert html.count("Only one tile design was tested.") == 1
    assert html.index('<details class="item-details">') < html.index("Only one tile design was tested.")
    assert '<span class="venue">bioRxiv</span>' in html and "(recent)" not in html
    assert 'Published <time datetime="2026-10-12">Oct 12</time>' in html


@pytest.mark.parametrize("status, shown", [
    ("rebuilt", True), ("unavailable", True), ("ok", False), ("static", False), (None, False),
])
def test_learned_profile_audit_is_one_quiet_line_only_when_actionable(monkeypatch, tmp_path, status, shown):
    store, web, _ids = _seed(monkeypatch, tmp_path)
    message = "Your learned reading profile was rebuilt from 12 ratings."
    if status is not None:
        store.write_digest_audit(DIGEST, "learned_profile",
                                 [{"status": status, "reason": "fixture", "message": message}])
    html = _text_payload(web.index(_request("GET", "/")))
    assert (message in html) is shown
    assert html.count('class="learned-profile-note"') == (1 if shown else 0)
    if shown:
        assert html.index('class="about-brew"') < html.index(message)


def test_older_boilerplate_is_hidden_at_display_time_but_never_rewritten(monkeypatch, tmp_path):
    store, web, (_paper, plain_id, grant_id) = _seed(monkeypatch, tmp_path)
    stored = ("Key finding: Legacy tiles fold.\n"
              "Why read: the abstract's contribution match remains unclear for DNA origami; "
              "check its relevance to your interests.\n"
              "Caveat: Preprint — not yet peer-reviewed, so treat conclusions as provisional.")
    with store.session_scope() as session:
        session.get(store.ItemRow, plain_id).summary = stored
    store.set_bookmark(plain_id, True)
    digest = _text_payload(web.index(_request("GET", "/")))
    saved = _text_payload(web.saved_items(_request("GET", "/saved")))
    for html in (_card(digest, plain_id), saved):
        assert "Legacy tiles fold." in html
        assert "contribution match remains unclear" not in html
        assert "not yet peer-reviewed" not in html
    with store.session_scope() as session:
        assert session.get(store.ItemRow, plain_id).summary == stored
    # The lead sentence is named by what it is; storage keeps "Key finding".
    assert '<div class="summary-label">Purpose</div>' in _card(digest, grant_id)
    assert web._finding_label("research", "Review") == "Scope"
    assert web._finding_label("research", "Research article") == "Key finding"
