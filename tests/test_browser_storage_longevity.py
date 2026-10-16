"""Accumulated browser choices survive pruning and a validated checkpoint."""
from datetime import datetime, timedelta, timezone
import json
import sqlite3

from tests.test_browser_state_differential import tracked_app, _assert_real_writer_matches  # noqa: F401


def test_ninety_days_of_choices_survive_retention_and_restoration(tracked_app):  # noqa: F811
    app = tracked_app
    now = datetime.now(timezone.utc)
    ids = []
    with app.store.session_scope() as session:
        for age in range(90, 0, -1):
            for index in range(3):
                item = app.store.ItemRow(source='Fixture', section='research',
                    external_id=f'{age}-{index}', url=f'https://fixture.invalid/{age}-{index}',
                    title=f'Fixture article {age}-{index}',
                    fetched_at=now-timedelta(days=age), published_at=now-timedelta(days=age))
                session.add(item)
                session.flush()
                ids.append(int(item.id))
    saved, known, voted = ids[:3], ids[3:6], ids[6:9]
    for item_id in saved:
        assert app.store.set_bookmark(item_id, True)
    for item_id in known:
        assert app.store.set_item_known(item_id, True)
    with app.store.session_scope() as session:
        for item_id in voted:
            session.add(app.store.VoteRow(item_id=item_id, value=1))
    before = app.runtime.state_generation()
    assert app.store.prune(30) > 100
    assert app.runtime.state_generation() > before
    with app.store.session_scope() as session:
        remaining = {row.id for row in session.query(app.store.ItemRow).all()}
    assert set(saved+known+voted) <= remaining
    assert not set(ids[9:90]) & remaining
    snapshot = app.runtime.export_snapshot(app.root)
    app.runtime.validate_snapshot_sqlite(app.runtime.decode_snapshot(snapshot), scratch_root=app.root.parent)
    _assert_real_writer_matches([{'label': 'ninety days and pruning',
        'generation': app.runtime.state_generation(), 'ready': True,
        'names': json.loads(app.runtime.snapshot_file_names(app.root)), 'snapshot': snapshot}])
    restored = app.root.parent/'restored-longevity'
    app.runtime.restore_snapshot(snapshot, root=restored)
    with sqlite3.connect(restored/'data/digest.db') as connection:
        for table, expected in [('bookmarks', saved), ('known_items', known), ('votes', voted)]:
            actual = {row[0] for row in connection.execute(f'SELECT item_id FROM {table}')}
            assert set(expected) <= actual
