"""Explicit arXiv publication links survive ingestion and record refreshes."""
import json
from html import escape

import pytest
from sqlalchemy import select

from dailydigest import store
from dailydigest.dedupe import candidate_identity_keys, dedupe_by_url, dedupe_ranking_candidates
from dailydigest.ingest.arxiv import ArxivSource
from dailydigest.models import SourceSpec
from dailydigest.rank.enrich import derive_doi
from dailydigest.rank.source_quality import is_preprint_source, quality_adjusted_score


def fetch(monkeypatch, doi="10.1038/S41586-026-12345-6", extra=""):
    field = f"<arxiv:doi>{escape(doi)}</arxiv:doi>" if doi is not None else ""
    xml = f'''<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry><id>https://arxiv.org/abs/2610.12345v2</id>
      <title>Programmable molecular structures with transferable design rules</title>
      <summary>References include 10.9999/unrelated. We introduce a design method.</summary>
      <author><name>Alice Smith</name></author>
      <published>2026-07-01T12:00:00Z</published><updated>2026-10-12T12:00:00Z</updated>
      <link href="https://arxiv.org/abs/2610.12345v2" rel="alternate" type="text/html"/>
      {field}{extra}</entry></feed>'''
    monkeypatch.setattr("dailydigest.ingest.arxiv._http_get_text", lambda *args: xml)
    return ArxivSource().fetch(SourceSpec(name="arXiv q-bio", kind="arxiv", section="research"))[0]


@pytest.mark.parametrize("doi", ["10.1038/S41586-026-12345-6", "https://doi.org/10.1038/S41586-026-12345-6", "doi:10.1038/S41586-026-12345-6"])
def test_explicit_publication_doi_connects_renamed_journal_record(monkeypatch, doi):
    item = fetch(monkeypatch, doi, "<arxiv:journal_ref>Nature 123, 45 (2026)</arxiv:journal_ref>")
    assert item.metadata["published_doi"] == "10.1038/s41586-026-12345-6"
    assert item.metadata["journal_reference"] == "Nature 123, 45 (2026)"
    assert item.external_id == "2610.12345"
    assert "doi" not in item.metadata and "venue" not in item.metadata
    assert is_preprint_source(item)
    assert derive_doi(item) is None  # Do not enrich/promote a preprint as its journal version.
    assert quality_adjusted_score(item, .8) == quality_adjusted_score(item.model_copy(update={"metadata": {}}), .8)
    assert item.published_at.month == 7  # A journal link is not a new publication date.
    store.upsert_items([item])
    with store.session_scope() as session:
        preprint = session.scalar(select(store.ItemRow))
        journal = store.ItemRow(id=999, section="research", source="Nature",
                                external_id="10.1038/s41586-026-12345-6",
                                url="https://doi.org/10.1038/s41586-026-12345-6",
                                title="A completely renamed paper describing molecular design",
                                abstract="New design method.", metadata_json="{}")
        assert "doi:10.1038/s41586-026-12345-6" in candidate_identity_keys(preprint)
        assert len(dedupe_ranking_candidates([preprint, journal])) == 1


@pytest.mark.parametrize("doi", [None, "NA", "published", "https://example.org/10.9999/paper", "10.9999/a 10.9999/b"])
def test_missing_or_ambiguous_doi_never_infers_a_publication_link(monkeypatch, doi):
    item = fetch(monkeypatch, doi, '<arxiv:journal_ref>Journal mentions 10.9999/reference</arxiv:journal_ref><link rel="related" href="https://doi.org/10.9999/reference"/>')
    assert "published_doi" not in item.metadata
    assert not any(key.startswith("doi:") for key in candidate_identity_keys(item))


def test_record_refresh_without_optional_fields_retains_publication_link(monkeypatch):
    store.upsert_items([fetch(monkeypatch)])
    store.upsert_items([fetch(monkeypatch, None)])
    with store.session_scope() as session:
        metadata = json.loads(session.scalar(select(store.ItemRow)).metadata_json)
        assert metadata["published_doi"] == "10.1038/s41586-026-12345-6"


@pytest.mark.parametrize("dedupe", [dedupe_by_url, dedupe_ranking_candidates])
def test_richer_copy_cannot_discard_the_same_preprints_publication_link(monkeypatch, dedupe):
    linked = fetch(monkeypatch)
    richer = fetch(monkeypatch, None).model_copy(update={"abstract": "A richer abstract. " * 100})
    kept = dedupe([richer, linked])
    assert kept == [richer]
    assert richer.metadata["published_doi"] == linked.metadata["published_doi"]


def test_publication_links_do_not_transfer_between_different_preprints(monkeypatch):
    linked = fetch(monkeypatch)
    other = fetch(monkeypatch, None).model_copy(update={"external_id": "2610.99999", "url": "https://arxiv.org/abs/2610.99999"})
    dedupe_ranking_candidates([other, linked])
    assert "published_doi" not in other.metadata


def test_conflicting_links_do_not_fill_an_unlinked_copy(monkeypatch):
    first, second, unknown = fetch(monkeypatch), fetch(monkeypatch, "10.9999/other-publication"), fetch(monkeypatch, None)
    unknown.abstract = "Richer text. " * 100
    kept = dedupe_ranking_candidates([unknown, first, second])
    assert kept == [unknown]
    assert "published_doi" not in unknown.metadata
