"""Final reading order must honor priorities without changing scores or eligibility."""
from types import SimpleNamespace

import pytest

from dailydigest import pipeline
from dailydigest.config import Settings
from dailydigest.rank.ranker import pick_top_per_section
from dailydigest.rank.source_quality import quality_adjusted_score


def row(i, *, section='research', genre=None):
    import json
    return SimpleNamespace(id=i, section=section, source='Nature',
        external_id=f'work-{i}', url=f'https://example.com/{i}',
        title=f'Independent experimental method {i}',
        abstract='We measure the structural response under controlled conditions.',
        metadata_json=json.dumps({'publication_types': [genre]} if genre else {}))


@pytest.mark.parametrize('mode', ['full', 'usual', 'minimal'])
def test_priority_survives_balancing_and_short_reading_modes(mode):
    candidates = [(row(i), .90 - i * .001) for i in range(20)]
    # The explicitly prioritized paper has lower raw score and originally lies
    # outside even the Usual list. No score or eligibility is edited.
    ordered = [candidates[-1], *candidates[:-1]]
    settings = Settings(_env_file=None, research_final_score_floor_frac=0)
    facts = {r.id: dict(source_bucket='published_journal', is_arxiv_cs=False,
        is_preprint=False, is_published=True, is_high_quality=True,
        is_low_impact=False, source_diversity_key=str(r.id)) for r, _ in ordered}
    picked = pick_top_per_section(ordered, {'research': 20}, settings=settings,
                                  selection_facts=facts)
    picked = pipeline._restore_research_order(picked, ordered)
    shown = pipeline.apply_reading_mode(picked, mode, preserve_research_order=True)
    assert shown[0] == candidates[-1]
    assert dict((r.id, s) for r, s in shown).items() <= dict((r.id, s) for r, s in candidates).items()
    assert len(shown) == {'full': 20, 'usual': 15, 'minimal': 5}[mode]


def test_restoring_research_order_keeps_news_order_and_never_adds_removed_rows():
    a, b, rejected = row(1), row(2), row(3)
    news, lesser_news = row(4, section='industry'), row(5, section='industry')
    chosen = [(a,.9), (news,.8), (b,.7), (lesser_news,.6)]
    expected = [(b,.7), (news,.8), (a,.9), (lesser_news,.6)]
    assert pipeline._restore_research_order(chosen, [(rejected,1), (b,.7), (a,.9)]) == expected


@pytest.mark.parametrize('genre', ['Review', 'Perspective', 'News & Views', 'Commentary', 'Editorial', 'Research Highlight', 'Meta-Analysis'])
def test_explicit_secondary_genre_cannot_borrow_positive_journal_bonus(monkeypatch, genre):
    from dailydigest.rank import source_quality as quality
    secondary = row(1, genre=genre)
    monkeypatch.setattr(quality, 'novelty_score', lambda _: 0)
    monkeypatch.setattr(quality, 'promotional_score', lambda _: 0)
    score = quality_adjusted_score(secondary, .82)
    assert score <= .82
    assert quality_adjusted_score(row(2, genre='Research Article'), .82) > .82
    assert quality_adjusted_score(row(3), .82) > .82


def test_meta_analysis_retains_explicit_review_label():
    from dailydigest.publications import publication_label
    assert publication_label(row(1, genre='Meta-Analysis')) == 'Review'


def test_priority_is_persisted_as_the_visible_digest_order(monkeypatch):
    """Run the actual publication path with fixed model evidence and no network."""
    import json
    from datetime import datetime, timezone
    import numpy as np
    from sqlalchemy import select
    from dailydigest import store
    from dailydigest.models import Profile

    store.init_db()
    with store.session_scope() as session:
        for i in range(2):
            session.add(store.ItemRow(source='Nature', section='research', external_id=f'priority-{i}',
                title=f'Distinct controlled assembly research experiment {i}',
                abstract='We experimentally evaluate and measure the assembly of molecular structures under controlled conditions.',
                url=f'https://example.com/priority-{i}', published_at=datetime.now(timezone.utc)))
    with store.session_scope() as session:
        rows = list(session.scalars(select(store.ItemRow).order_by(store.ItemRow.id)))
        session.expunge_all()
    first, preferred = rows
    features = {r.id: {'topic_score': .9, 'primary_facet': f'topic-{i}',
        'primary_facet_score': .9, 'topic_priority': i,
        'topic_priority_bonus': .06 * i, 'final_score': .8 - .01*i} for i,r in enumerate(rows)}
    monkeypatch.setattr(pipeline, '_digest_id', lambda: '2026-10-22')
    monkeypatch.setattr(pipeline, 'ingest_all', lambda **kwargs: 0)
    monkeypatch.setattr(pipeline, 'load_profile', lambda: Profile(bio='', keywords=['topic-0', 'topic-1']))
    monkeypatch.setattr(pipeline, '_build_profile_with_rocchio', None)
    monkeypatch.setattr(pipeline, '_build_neg_centroid', None)
    monkeypatch.setattr(pipeline, 'build_profile_matrix', lambda _: np.ones((1, 3)))
    monkeypatch.setattr('dailydigest.rank.profile.build_attribution_context', lambda _: None)
    monkeypatch.setattr(pipeline, 'recent_items', lambda **kwargs: rows)
    monkeypatch.setattr(pipeline, '_score_items_for_pipeline', lambda *args, **kwargs:
                        ([(first, .8), (preferred, .79)], features))
    monkeypatch.setattr(pipeline.get_settings(), 'within_day_dedupe', False)
    monkeypatch.setattr(pipeline, 'summarize_items', lambda items, **kwargs: {r.id: 'Measured result.' for r in items})
    monkeypatch.setattr(pipeline, 'render_digest', lambda *args, **kwargs: '<html>ready</html>')
    monkeypatch.setattr('dailydigest.rank.calibrate.fit_calibrator', lambda: None)
    digest_id = pipeline.run_all(dry_run=True, reading_mode='minimal')
    with store.session_scope() as session:
        slate = list(session.scalars(select(store.DigestItemRow).where(
            store.DigestItemRow.digest_id == digest_id).order_by(store.DigestItemRow.item_label)))
        assert [item.item_id for item in slate] == [preferred.id, first.id]
        assert [item.score for item in slate] == [.79, .8]
        snapshot = session.scalar(select(store.RankingSnapshotRow).where(store.RankingSnapshotRow.digest_id == digest_id))
        import zlib
        payload = json.loads(zlib.decompress(snapshot.payload_zlib))
        from dailydigest.rank.replay import frozen_selection
        assert frozen_selection(payload)['research'] == [preferred.id, first.id]
