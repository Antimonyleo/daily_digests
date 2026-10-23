from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from dailydigest import store
from dailydigest.models import Item
from dailydigest.rank.source_quality import should_skip_item


def _paper():
    return Item(source="Publisher", section="research", external_id="paper",
                title="Original lattice experiment", url="https://doi.org/10.9999/lattice",
                metadata={"doi": "10.9999/lattice", "publication_types": ["Research Article"]})


def _commentary():
    return Item(source="Publisher", section="research", external_id="commentary",
                title="Why these lattices matter", url="https://doi.org/10.9999/views",
                metadata={"doi": "10.9999/views", "publication_types": ["News & Views"],
                          "companion_of_dois": ["10.9999/lattice"]})


def _rows():
    with store.session_scope() as session:
        rows = list(session.scalars(select(store.ItemRow).order_by(store.ItemRow.id)))
        session.expunge_all()
        return rows


def test_later_commentary_attaches_to_previous_digest_without_republishing_parent():
    store.upsert_items([_paper()])
    parent = _rows()[0]
    store.write_digest("2026-10-20", [("R1", parent.id, .9)])
    store.upsert_items([_commentary()])
    children = store.load_publication_companions([parent])[parent.id]
    assert [child["title"] for child in children] == ["Why these lattices matter"]
    with store.session_scope() as session:
        assert list(session.scalars(select(store.DigestItemRow.digest_id))) == ["2026-10-20"]
    assert should_skip_item(_rows()[1])


def test_linked_perspective_is_not_a_second_recommendation_in_any_section():
    item = _commentary()
    item.abstract = "We propose a framework for comparing lattice assembly methods."
    item.metadata["publication_types"] = ["Perspective"]
    assert should_skip_item(item)
    item.section = "industry"
    assert should_skip_item(item)
    item.section = "research"
    item.metadata.pop("companion_of_dois")
    assert not should_skip_item(item)


def test_generic_refresh_preserves_companion_link():
    store.upsert_items([_paper(), _commentary()])
    refreshed = _commentary().model_copy(update={"metadata": {"work_type": "article",
        "companion_of_dois": [], "companion_relation_evidence": []}})
    store.upsert_items([refreshed])
    parent, child = _rows()
    assert store.load_publication_companions([parent])[parent.id][0]["item_id"] == child.id
    assert should_skip_item(child)


def test_prune_keeps_companions_of_retained_papers_but_not_unrelated_old_rows():
    unrelated = _commentary().model_copy(update={"external_id": "unrelated",
        "url": "https://doi.org/10.9999/unrelated",
        "metadata": {"publication_types": ["News & Views"], "companion_of_dois": ["10.9999/other"]}})
    store.upsert_items([_paper(), _commentary(), unrelated])
    parent = _rows()[0]
    store.set_bookmark(parent.id, True)
    with store.session_scope() as session:
        for row in session.scalars(select(store.ItemRow)):
            row.fetched_at = datetime.now(timezone.utc) - timedelta(days=90)
    assert store.prune(30) == 1
    assert {row.external_id for row in _rows()} == {"paper", "commentary"}


def test_no_parent_no_attachment_and_no_identity_or_rating_merge():
    store.upsert_items([_commentary()])
    assert store.load_publication_companions([]) == {}
    store.upsert_items([_paper()])
    child, parent = _rows()
    assert parent.id != child.id
    assert store.item_metadata(parent)["publication_types"] == ["Research Article"]
    assert store.load_publication_companions([parent])[parent.id][0]["item_id"] == child.id


def test_email_shows_companion_only_under_research_article():
    from dailydigest.email_render import render_digest

    store.upsert_items([_paper(), _commentary()])
    parent, child = _rows()
    parent.item_label, child.item_label = "R1", "R2"
    html = render_digest("2026-10-22", {"research": [(parent, .9, "Result"), (child, .8, "Comment")]})
    assert html.count("Why these lattices matter") == 1
    assert "Research article" in html
    assert "News &amp; Views" in html
    assert ">R2</span>" not in html


def test_older_companion_suppresses_new_generic_copy_without_mutating_rows():
    store.upsert_items([_paper(), _commentary()])
    generic = _commentary().model_copy(update={"source": "OpenAlex",
        "external_id": "new-copy", "metadata": {"work_type": "article"}})
    store.upsert_items([generic])
    parent, child, copy = _rows()
    assert store.exclude_publication_companions([parent, copy]) == [parent]
    assert store.item_metadata(copy) == {"work_type": "article"}
    assert len(_rows()) == 3
