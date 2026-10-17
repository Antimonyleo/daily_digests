"""Hard publication exclusions survive dedupe only across durable identities."""
from datetime import datetime, timezone
import json

import pytest

from dailydigest.dedupe import dedupe_by_url, dedupe_ranking_candidates
from dailydigest.models import Item
from dailydigest.rank.source_quality import content_type, should_skip_item
from dailydigest.store import ItemRow


def _item(source, url, metadata=None, rich=False):
    return Item(source=source, section="research", external_id=source,
                url=url, title="DNA nanostructure molecular design method",
                abstract=("A primary method abstract. " * (40 if rich else 1)),
                published_at=datetime(2026, 10, 2, 12, tzinfo=timezone.utc), metadata=metadata or {})


@pytest.mark.parametrize("dedupe,same_url", [(dedupe_by_url, True), (dedupe_ranking_candidates, False)])
@pytest.mark.parametrize("reverse", [False, True])
def test_retraction_evidence_survives_preferred_rss_representative(dedupe, same_url, reverse):
    rss = _item("Nature", "https://doi.org/10.9999/paper", {"reader_note": "retain"}, rich=True)
    db = _item("PubMed", rss.url if same_url else "https://pubmed.ncbi.nlm.nih.gov/123/",
               {"doi": "10.9999/paper", "is_retracted": True})
    rows = [rss, db]
    result = dedupe(rows[::-1] if reverse else rows)
    assert len(result) == 1
    assert should_skip_item(result[0])
    assert rss.metadata["reader_note"] == "retain"


def test_editorial_type_survives_ranking_dedupe_without_overwriting_other_metadata():
    rss = _item("Nature", "https://doi.org/10.9999/paper", {"work_type": "article"}, rich=True)
    db = _item("PubMed", "https://pubmed.ncbi.nlm.nih.gov/123/",
               {"doi": "10.9999/paper", "publication_types": ["Editorial"]})
    result = dedupe_ranking_candidates([rss, db])
    assert result[0] is rss
    assert rss.metadata["work_type"] == "article"
    assert should_skip_item(rss)


def test_title_only_match_cannot_transfer_hard_exclusion_to_distinct_doi():
    rss = _item("Nature", "https://doi.org/10.9999/real", rich=True)
    db = _item("PubMed", "https://doi.org/10.9999/other", {"is_retracted": True})
    result = dedupe_ranking_candidates([rss, db])
    assert result == [rss]
    assert not should_skip_item(rss)
    assert "is_retracted" not in rss.metadata


def test_strong_identity_transfer_is_transitive_without_using_title_bridge():
    rss = _item("Nature", "https://doi.org/10.9999/paper", rich=True)
    bridge = _item("PubMed", "https://pubmed.ncbi.nlm.nih.gov/123/", {"doi": "10.9999/paper"})
    evidence = _item("Other", bridge.url, {"is_retracted": True})
    result = dedupe_ranking_candidates([rss, bridge, evidence])
    assert len(result) == 1
    assert should_skip_item(result[0])
    other = _item("Nature", "https://doi.org/10.9999/distinct", rich=True)
    dedupe_ranking_candidates([other, rss, bridge, evidence])
    assert not should_skip_item(other)


def test_stored_rows_receive_exclusion_evidence_in_metadata_json():
    rss = ItemRow(id=1, source="Nature", section="research", external_id="1",
                  url="https://doi.org/10.9999/paper", title="DNA molecular design method",
                  abstract="Rich abstract. " * 50, metadata_json='{"note":"preserve"}')
    db = ItemRow(id=2, source="PubMed", section="research", external_id="123",
                 url="https://pubmed.ncbi.nlm.nih.gov/123/", title=rss.title,
                 abstract="Short method text.", metadata_json=json.dumps({"doi":"10.9999/paper", "is_retracted":True}))
    result = dedupe_ranking_candidates([rss, db])
    assert result == [rss]
    assert json.loads(rss.metadata_json)["note"] == "preserve"
    assert should_skip_item(rss)


def test_notice_field_provenance_follows_strong_identity_only():
    provenance = {"field": "dc_type", "value": "Editorial"}
    rss = _item("Nature", "https://doi.org/10.9999/paper", rich=True)
    notice = _item("Other", "https://example.org/editorial", {
        "doi": "10.9999/paper", "publication_types": ["Editorial"],
        "publication_type_evidence": [provenance, {"field": "dc_subject", "value": "Editorial"}],
    })
    unrelated = _item("Nature", "https://doi.org/10.9999/other", rich=True)
    dedupe_ranking_candidates([rss, notice, unrelated])
    assert rss.metadata["publication_type_evidence"] == [provenance]
    assert "publication_type_evidence" not in unrelated.metadata


@pytest.mark.parametrize("genre,expected", [("Review", "review"), ("Perspective", "editorial"), ("Primer", "review")])
def test_declared_genre_and_provenance_survive_richer_generic_representative(genre, expected):
    provenance = {"field": "dc_type", "value": genre}
    rss = _item("Nature", "https://doi.org/10.9999/paper", {"publication_types": ["Journal Article"]}, rich=True)
    typed = _item("Other", "https://example.org/paper", {
        "doi": "10.9999/paper", "publication_types": [genre], "publication_type_evidence": [provenance],
    })
    unrelated = _item("Nature", "https://doi.org/10.9999/distinct", rich=True)
    dedupe_ranking_candidates([rss, typed, unrelated])
    assert content_type(rss) == expected
    assert rss.metadata["publication_type_evidence"] == [provenance]
    assert "publication_type_evidence" not in unrelated.metadata
    assert content_type(unrelated) == "method"
