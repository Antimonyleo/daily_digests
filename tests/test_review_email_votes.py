"""Email replies resolve the slate actually sent, never a later preview."""

from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

import pytest

from dailydigest import config, inbound, store, votes


DIGEST = '2026-10-16'
FIRST_RUN = '1' * 32
SECOND_RUN = '2' * 32


@pytest.fixture
def desk(monkeypatch, tmp_path):
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'digest.db'))
    config.reload_settings()
    store.SETTINGS = config.SETTINGS
    store._reset_init_for_testing()
    store.init_db()
    monkeypatch.setattr(votes, '_update_rocchio', lambda: None)
    with store.session_scope() as session:
        rows = [store.ItemRow(source='Fixture', section='research', external_id=name,
                              title=name, url=f'https://example.test/{name}')
                for name in ('emailed', 'preview')]
        session.add_all(rows)
        session.flush()
        ids = [int(row.id) for row in rows]
    yield ids
    store._reset_init_for_testing()
    config.get_settings.cache_clear()


def send_first(desk):
    store.write_digest(DIGEST, [('R3', desk[0])])
    store.mark_sent(DIGEST, run_id=FIRST_RUN, labeled_items=[('R3', desk[0])])


def test_email_reply_after_preview_rebrew_votes_for_original_item(desk):
    send_first(desk)
    store.write_digest(DIGEST, [('R3', desk[1])])

    assert votes.record_votes('+R3', DIGEST, run_id=FIRST_RUN, sent_only=True) == {
        'up': 1, 'down': 0, 'unknown': 0,
    }
    assert votes.get_vote_value(desk[0]) == 1
    assert votes.get_vote_value(desk[1]) is None


def test_cli_labels_still_resolve_current_preview(desk):
    send_first(desk)
    store.write_digest(DIGEST, [('R3', desk[1])])
    assert votes.record_votes('+R3', DIGEST)['up'] == 1
    assert votes.get_vote_value(desk[0]) is None
    assert votes.get_vote_value(desk[1]) == 1


def test_untagged_reply_cannot_guess_even_a_single_frozen_send(desk):
    send_first(desk)
    store.write_digest(DIGEST, [('R3', desk[1])])
    assert votes.record_votes('+R3', DIGEST, sent_only=True)['unknown'] == 1
    assert votes.get_vote_value(desk[0]) is None


def test_two_sent_runs_keep_distinct_labels_and_legacy_reply_is_unknown(desk):
    send_first(desk)
    store.write_digest(DIGEST, [('R3', desk[1])])
    store.mark_sent(DIGEST, run_id=SECOND_RUN, labeled_items=[('R3', desk[1])])

    assert votes.record_votes('+R3', DIGEST, sent_only=True) == {'up': 0, 'down': 0, 'unknown': 1}
    assert votes.record_votes('+R3', DIGEST, run_id=FIRST_RUN, sent_only=True)['up'] == 1
    assert votes.record_votes('-R3', DIGEST, run_id=SECOND_RUN, sent_only=True)['down'] == 1
    assert votes.get_vote_value(desk[0]) == 1
    assert votes.get_vote_value(desk[1]) == -1


@pytest.mark.parametrize('target_digest,target_run', [(DIGEST, '3' * 32), ('2026-10-15', FIRST_RUN)])
def test_unknown_or_mismatched_sent_run_cannot_fall_back_to_preview(desk, target_digest, target_run):
    send_first(desk)
    assert votes.record_votes('+R3', target_digest, run_id=target_run, sent_only=True)['unknown'] == 1
    assert votes.get_vote_value(desk[0]) is None


def test_preupgrade_email_without_snapshot_is_unresolved(desk, caplog):
    store.write_digest(DIGEST, [('R3', desk[0])])
    with store.session_scope() as session:
        session.get(store.DigestRow, DIGEST).sent_at = datetime.now(timezone.utc)
    store.write_digest(DIGEST, [('R3', desk[1])])
    assert votes.record_votes('+R3', DIGEST, sent_only=True)['unknown'] == 1
    assert 'sent email' in caplog.text.lower()
    assert votes.get_vote_value(desk[1]) is None


def test_tagged_resend_does_not_make_an_older_untagged_email_resolvable(desk):
    store.write_digest(DIGEST, [('R3', desk[0])])
    with store.session_scope() as session:
        session.get(store.DigestRow, DIGEST).sent_at = datetime.now(timezone.utc) - timedelta(hours=1)
    store.write_digest(DIGEST, [('R3', desk[1])])
    store.mark_sent(DIGEST, run_id=SECOND_RUN, labeled_items=[('R3', desk[1])])
    assert votes.record_votes('+R3', DIGEST, sent_only=True)['unknown'] == 1
    assert votes.get_vote_value(desk[1]) is None


def test_mark_sent_is_idempotent_and_cannot_remap_existing_run(desk):
    send_first(desk)
    store.mark_sent(DIGEST, run_id=FIRST_RUN, labeled_items=[('R3', desk[0])])
    with pytest.raises(ValueError, match='immutable'):
        store.mark_sent(DIGEST, run_id=FIRST_RUN, labeled_items=[('R3', desk[1])])
    assert votes.record_votes('+R3', DIGEST, run_id=FIRST_RUN, sent_only=True)['up'] == 1
    assert votes.get_vote_value(desk[0]) == 1


def test_mark_sent_captures_rendered_slate_even_if_current_preview_changed(desk):
    store.write_digest(DIGEST, [('R3', desk[1])])
    store.mark_sent(DIGEST, run_id=FIRST_RUN, labeled_items=[('R3', desk[0])])
    assert votes.record_votes('+R3', DIGEST, run_id=FIRST_RUN, sent_only=True)['up'] == 1
    assert votes.get_vote_value(desk[0]) == 1


def test_prune_preserves_items_referenced_by_sent_email(desk):
    send_first(desk)
    store.write_digest(DIGEST, [('R3', desk[1])])
    with store.session_scope() as session:
        session.get(store.ItemRow, desk[0]).fetched_at = datetime.now(timezone.utc) - timedelta(days=90)
    assert store.prune(30) == 0
    assert votes.record_votes('+R3', DIGEST, run_id=FIRST_RUN, sent_only=True)['up'] == 1


def test_sent_item_stays_suppressed_after_preview_replaces_membership(desk):
    send_first(desk)
    store.write_digest(DIGEST, [('R3', desk[1])])
    with store.session_scope() as session:
        emailed = session.get(store.ItemRow, desk[0])
        session.expunge(emailed)
    assert store.exclude_previously_shown([emailed], exclude_digest_id='2026-10-17') == []
    # Same-day preview upgrades remain allowed under the existing policy.
    assert store.exclude_previously_shown([emailed], exclude_digest_id=DIGEST) == [emailed]
    with store.session_scope() as session:
        for row in session.query(store.EmailDigestItemRow).all():
            row.sent_at = datetime.now(timezone.utc) - timedelta(days=31)
    assert store.exclude_previously_shown([emailed], exclude_digest_id='2026-10-17') == [emailed]


def test_inbound_extracts_run_identity_from_reply_subject(monkeypatch):
    message = EmailMessage()
    message['Subject'] = f'Re: DailyDigest {DIGEST} [run:{FIRST_RUN}]'
    message.set_content('+R3')

    class Mailbox:
        def login(self, *args):
            pass

        def select(self, *args):
            pass

        def search(self, *args):
            return 'OK', [b'1']

        def fetch(self, *args):
            return 'OK', [(b'1', message.as_bytes())]

        def close(self):
            pass

        def logout(self):
            pass

    monkeypatch.setattr(inbound.imaplib, 'IMAP4_SSL', lambda *args: Mailbox())
    replies = inbound.fetch_replies(inbound.InboundConfig(
        imap_host='imap.example.test', imap_user='fixture', imap_password='fixture'))
    assert replies[0]['run_id'] == FIRST_RUN


def test_inbound_passes_sent_snapshot_scope_and_leaves_unknown_reply_unread(monkeypatch):
    monkeypatch.setattr(inbound, 'load_inbound_config', lambda: inbound.InboundConfig(
        imap_user='reader@example.test', imap_password='fixture'))
    monkeypatch.setattr(inbound, 'get_settings', lambda: type('Settings', (), {'digest_to': 'reader@example.test'})())
    calls = []
    acknowledgements = []

    def record(line, digest_id, *, run_id, sent_only):
        calls.append((line, digest_id, run_id, sent_only))
        return {'up': 0, 'down': 0, 'unknown': 1}

    def fetch(cfg, *, on_reply):
        acknowledgements.append(on_reply({'from': 'reader@example.test', 'body': '+R3',
                                          'digest_id': DIGEST, 'run_id': FIRST_RUN}))

    monkeypatch.setattr(inbound.votes_mod, 'record_votes', record)
    monkeypatch.setattr(inbound, 'fetch_replies', fetch)
    result = inbound.process_replies()
    assert calls == [('+R3', DIGEST, FIRST_RUN, True)]
    assert acknowledgements == [False]
    assert result['votes_unknown'] == 1


def test_pipeline_freezes_labels_only_after_confirmed_send(desk, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from dailydigest import pipeline

    monkeypatch.setenv('RESEND_API_KEY', 'fixture-key')
    monkeypatch.setenv('DIGEST_TO', 'reader@example.test')
    config.reload_settings()
    with store.session_scope() as session:
        row = session.get(store.ItemRow, desk[0])
        row.source = 'Nature'
        row.title = 'Engineered molecular assembly with validated function'
        row.abstract = 'Methods and results describe a validated primary research study. ' * 10
        session.expunge(row)
    monkeypatch.setattr(pipeline, '_digest_id', lambda: DIGEST)
    monkeypatch.setattr(pipeline, 'ingest_all', lambda **kwargs: 0)
    monkeypatch.setattr(pipeline, 'load_profile', lambda: SimpleNamespace(bio='', keywords=[], downweight=[]))
    monkeypatch.setattr(pipeline, 'build_profile_matrix', lambda profile: np.zeros((1, 3)))
    monkeypatch.setattr(pipeline, 'recent_items', lambda **kwargs: [row])
    monkeypatch.setattr(pipeline, 'score_items', lambda items, *args, **kwargs: [(items[0], 0.9)])
    monkeypatch.setattr(pipeline, 'pick_top_per_section', lambda scored, *args, **kwargs: scored)
    monkeypatch.setattr(pipeline, 'summarize_items', lambda rows, **kwargs: {desk[0]: 'Fixture summary.'})
    subjects = []
    monkeypatch.setattr(pipeline, 'send_digest', lambda html, subject, **kwargs: subjects.append(subject) or True)

    assert pipeline.run_all(dry_run=False) == DIGEST
    assert len(subjects) == 1
    match = inbound._RUN_IN_SUBJECT_RE.search(subjects[0])
    assert match
    store.write_digest(DIGEST, [('R1', desk[1])])
    assert votes.record_votes('+R1', DIGEST, run_id=match.group(1), sent_only=True)['up'] == 1
    assert votes.get_vote_value(desk[0]) == 1
