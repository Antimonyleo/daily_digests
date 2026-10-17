"""Routine reingest cannot clear affirmative publication exclusion evidence."""
import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from dailydigest.models import Item
from dailydigest.rank.source_quality import content_type, should_skip_item
from dailydigest import store


@pytest.mark.parametrize("old_metadata,new_metadata", [
    ({"is_retracted": True}, {"is_retracted": False, "work_type": "article"}),
    ({"publication_types": ["Retracted Publication"]}, {"publication_types": ["Journal Article"]}),
    ({"publication_types": ["Editorial"]}, {"publication_types": ["Journal Article"]}),
    ({"work_type": "retraction"}, {"work_type": "article"}),
    ({"is_retracted": True}, {}),
])
def test_reingest_preserves_known_exclusion_and_user_vote(old_metadata, new_metadata):
    item = Item(source="PubMed", section="research", external_id="123",
                url="https://pubmed.ncbi.nlm.nih.gov/123/", title="DNA nanostructure design method",
                abstract="Primary DNA self assembly method and mechanism.",
                metadata={"venue": "Nature", **old_metadata})
    assert store.upsert_items([item]) == 1
    with store.session_scope() as session:
        row = session.scalar(select(store.ItemRow))
        item_id = row.id
        session.add(store.VoteRow(item_id=item_id, value=1, grade=100))
    latest = item.model_copy(update={"metadata": {"venue": "Nature", "new_detail": "retain", **new_metadata}})
    assert store.upsert_items([latest]) == 0
    with store.session_scope() as session:
        row = session.get(store.ItemRow, item_id)
        assert should_skip_item(row)
        assert json.loads(row.metadata_json)["new_detail"] == "retain"
        vote = session.scalar(select(store.VoteRow))
        assert vote.item_id == item_id and vote.grade == 100


def test_non_exclusion_metadata_can_still_refresh():
    item = Item(source="PubMed", section="research", external_id="123",
                url="https://pubmed.ncbi.nlm.nih.gov/123/", title="DNA nanostructure design method",
                metadata={"venue": "Old Venue", "work_type": "article"})
    store.upsert_items([item])
    store.upsert_items([item.model_copy(update={"metadata": {"venue": "New Venue", "work_type": "review"}})])
    with store.session_scope() as session:
        row = session.scalar(select(store.ItemRow))
        assert json.loads(row.metadata_json) == {"venue": "New Venue", "work_type": "review"}
        assert not should_skip_item(row)


@pytest.mark.parametrize("genre", ["Review", "Perspective", "Primer"])
def test_refresh_preserves_declared_genre_and_rss_evidence(genre):
    evidence = {"field": "prism_genre", "value": genre}
    item = Item(source="Nature", section="research", external_id="genre-paper",
                url="https://publisher.test/genre", title="Engineering molecular systems",
                metadata={"publication_types": [genre], "publication_type_evidence": [evidence]})
    expected_skip = should_skip_item(item)
    store.upsert_items([item])
    store.upsert_items([item.model_copy(update={"metadata": {"publication_types": ["Journal Article"]}})])
    with store.session_scope() as session:
        row = session.scalar(select(store.ItemRow))
        metadata = json.loads(row.metadata_json)
        assert genre.casefold() in {value.casefold() for value in metadata["publication_types"]}
        assert metadata["publication_type_evidence"] == [evidence]
        assert content_type(row) == {"Review": "review", "Perspective": "editorial", "Primer": "review"}[genre]
        assert should_skip_item(row) == expected_skip


@pytest.mark.parametrize('old_source,new_source,expected_source', [
    ('published', 'updated', 'published'), ('updated', 'published', 'published'),
])
def test_reingest_keeps_publication_date_and_its_provenance_together(old_source, new_source, expected_source):
    old_date = datetime(2026, 9, 1, tzinfo=timezone.utc)
    new_date = datetime(2026, 10, 1, tzinfo=timezone.utc)
    original = Item(source='Nature', section='research', external_id='same-paper',
                    url='https://publisher.test/article', title='Measured molecular assembly',
                    published_at=old_date, metadata={'publication_date_source': old_source})
    store.upsert_items([original])
    latest = original.model_copy(update={'published_at': new_date,
                                        'metadata': {'publication_date_source': new_source, 'doi': '10.1234/retained'}})
    store.upsert_items([latest])
    with store.session_scope() as session:
        row = session.scalar(select(store.ItemRow))
        metadata = json.loads(row.metadata_json)
        assert metadata['publication_date_source'] == expected_source
        expected_date = old_date if old_source == 'published' else new_date
        assert row.published_at.replace(tzinfo=timezone.utc) == expected_date
        assert metadata['doi'] == '10.1234/retained'
