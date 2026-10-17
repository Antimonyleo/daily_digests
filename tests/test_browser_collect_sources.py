"""Public source publication gates use fixture metadata only."""
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('collector_gate', Path(__file__).parents[1] / 'browser/collect_sources.py')
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)
NOW = datetime(2026, 10, 14, tzinfo=timezone.utc)


def report():
    return {'version': 1, 'public_only': True, 'generated_at': NOW.isoformat(), 'days': 21,
            'sources': {name: {'kind': kind, 'section': section, 'url': 'https://example.test/' + name,
                              'category': None, 'days': 21, 'status': 'ok', 'fetched_at': NOW.isoformat(),
                              'items': []}
                        for name, kind, section in [('a', 'rss', 'research'), ('b', 'rss', 'industry'),
                                                    ('c', 'arxiv', 'research'), ('d', 'rss', 'research')]}}


def fail(data, names):
    for name in names:
        data['sources'][name].update(status='error', items=[], fetched_at=None,
                                     attempted_at=NOW.isoformat(), error='Network failed')


@pytest.mark.parametrize('names', [('a', 'b', 'c', 'd'), ('a', 'b', 'd'), ('c',), ('b',)])
def test_degraded_collection_is_rejected_before_replacing_output(tmp_path, names):
    data = report()
    fail(data, names)
    output = tmp_path / 'public.json'
    output.write_text('previous good snapshot')
    with pytest.raises(ValueError, match='coverage'):
        collector.publish(data, output, now=NOW)
    assert output.read_text() == 'previous good snapshot'
    assert not output.with_suffix('.json.pending').exists()


def test_genuinely_empty_successful_feeds_are_healthy(tmp_path):
    data = report()
    collector.publish(data, tmp_path / 'public.json', now=NOW)
    assert collector.validate_report(data, now=NOW)['fresh_ok'] == 4


@pytest.mark.parametrize('mutation', [
    lambda row: row.update(fetched_at=NOW.replace(tzinfo=None).isoformat()),
    lambda row: row.update(fetched_at=(NOW + timedelta(hours=1)).isoformat()),
    lambda row: row.update(items=[{'source': 'wrong', 'section': 'research', 'external_id': '1',
                                 'url': 'https://example.test/item', 'title': 'Wrong source'}]),
    lambda row: row.update(items='not a list'),
    lambda row: row.update(days=0),
])
def test_malformed_ok_record_cannot_pass_publication(tmp_path, mutation):
    data = report()
    mutation(data['sources']['a'])
    with pytest.raises(ValueError):
        collector.publish(data, tmp_path / 'public.json', now=NOW)
    assert not (tmp_path / 'public.json').exists()


def test_partial_failure_carries_only_matching_recent_good_record():
    previous, current = report(), report()
    original = (NOW - timedelta(hours=30)).isoformat()
    previous['sources']['d']['fetched_at'] = original
    fail(current, ['d'])
    merged = collector.carry_forward(current, previous, now=NOW)
    row = merged['sources']['d']
    assert row['status'] == 'ok' and row['carried_forward'] is True
    assert row['fetched_at'] == original
    assert row['latest_attempt']['error'] == 'Network failed'
    assert current['sources']['d']['status'] == 'error'
    assert collector.validate_report(merged, now=NOW)['fresh_ok'] == 3


@pytest.mark.parametrize('change', ['expired', 'mismatch', 'malformed'])
def test_carryforward_rejects_expired_changed_or_malformed_records(change):
    previous, current = report(), report()
    fail(current, ['d'])
    if change == 'expired':
        previous['sources']['d']['fetched_at'] = (NOW - timedelta(hours=49)).isoformat()
    elif change == 'mismatch':
        previous['sources']['d']['url'] = 'https://changed.test/rss'
    else:
        previous['sources']['d']['items'] = [{}]
    assert collector.carry_forward(current, previous, now=NOW)['sources']['d']['status'] == 'error'


def test_carryforward_cannot_mask_total_current_collection_failure():
    current = report()
    fail(current, list(current['sources']))
    merged = collector.carry_forward(current, report(), now=NOW)
    with pytest.raises(ValueError, match='coverage'):
        collector.validate_report(merged, now=NOW)


def test_fixture_manifest_must_not_contain_empty_config():
    data = report()
    data['sources'] = {}
    with pytest.raises(ValueError, match='coverage'):
        collector.validate_report(data, now=NOW)


def test_every_actual_configured_public_source_section_can_publish(tmp_path):
    sources = collector.public_specs(Path(__file__).parents[1] / 'config/sources.yaml')
    data = {'version': 1, 'public_only': True, 'generated_at': NOW.isoformat(), 'days': 21,
            'sources': {spec.name: {'kind': spec.kind, 'section': spec.section,
                                   'url': spec.url, 'category': spec.category,
                                   'days': 21, 'status': 'ok', 'fetched_at': NOW.isoformat(),
                                   'items': []} for spec in sources}}
    summary = collector.publish(data, tmp_path / 'actual-source-shapes.json', now=NOW)
    assert summary['fresh_ok'] == len(sources)
    assert {row['section'] for row in data['sources'].values()} >= {'research', 'industry', 'ai', 'regulatory', 'world'}


def test_push_section_outage_reuses_previous_validated_snapshot(tmp_path):
    previous, current = report(), report()
    old_time = (NOW - timedelta(hours=30)).isoformat()
    for row in previous['sources'].values():
        row['fetched_at'] = old_time
    fail(current, ['b'])  # A single-source section is unavailable.
    selected = collector.prepare_publication(current, previous, event='push', now=NOW)
    assert selected['publication']['mode'] == 'push-carry-forward'
    assert 'previous' in selected['publication']['warning'].lower()
    assert all(row['fetched_at'] == old_time for row in selected['sources'].values())
    assert all(row['carried_forward'] for row in selected['sources'].values())
    assert selected['sources']['b']['latest_attempt']['status'] == 'error'
    assert selected['sources']['a']['latest_attempt']['status'] == 'ok'
    summary = collector.publish(selected, tmp_path / 'public.json', now=NOW,
                                allow_carry_forward=True)
    assert summary['ok'] == 4 and summary['fresh_ok'] == 0
    # Push fallback is not mistaken for a successful scheduled refresh.
    with pytest.raises(ValueError, match='coverage'):
        collector.validate_report(selected, now=NOW)


@pytest.mark.parametrize('event', ['schedule', 'workflow_dispatch'])
def test_refresh_section_outage_remains_strict(event):
    current = report()
    fail(current, ['b'])
    with pytest.raises(ValueError, match='coverage missing section'):
        collector.prepare_publication(current, report(), event=event, now=NOW)


@pytest.mark.parametrize('previous', [None, {}])
def test_push_fallback_requires_previous_snapshot(previous):
    current = report()
    fail(current, ['b'])
    with pytest.raises(ValueError, match='previous.*unavailable|previous.*invalid'):
        collector.prepare_publication(current, previous, event='push', now=NOW)


def test_push_fallback_refuses_previous_older_than_48_hours():
    current, previous = report(), report()
    fail(current, ['b'])
    previous['sources']['b']['fetched_at'] = (NOW - timedelta(hours=49)).isoformat()
    with pytest.raises(ValueError, match='previous.*expired|previous.*invalid'):
        collector.prepare_publication(current, previous, event='push', now=NOW)


@pytest.mark.parametrize('event', ['push', 'schedule', 'workflow_dispatch'])
def test_all_error_collection_never_reuses_previous_snapshot(event):
    current = report()
    fail(current, list(current['sources']))
    with pytest.raises(ValueError, match='no fresh successes'):
        collector.prepare_publication(current, report(), event=event, now=NOW)


def test_push_fallback_cannot_claim_new_or_changed_source_coverage():
    current, previous = report(), report()
    fail(current, ['b'])
    previous['sources']['b']['url'] = 'https://changed.test/feed'
    with pytest.raises(ValueError, match='previous.*configuration'):
        collector.prepare_publication(current, previous, event='push', now=NOW)


def test_push_fallback_can_reuse_a_still_valid_previous_fallback():
    current = report()
    fail(current, ['b'])
    first = collector.prepare_publication(current, report(), event='push', now=NOW)
    second = collector.prepare_publication(current, first, event='push', now=NOW + timedelta(hours=1))
    assert second['sources']['a']['fetched_at'] == NOW.isoformat()
    assert collector.validate_report(second, now=NOW + timedelta(hours=1),
                                     allow_carry_forward=True)['ok'] == 4


def test_carryforward_flag_does_not_allow_all_error_attempts():
    selected = report()
    selected['publication'] = {'mode': 'push-carry-forward'}
    for row in selected['sources'].values():
        row['carried_forward'] = True
        row['latest_attempt'] = {**row, 'status': 'error', 'items': []}
    with pytest.raises(ValueError, match='no fresh successes'):
        collector.validate_report(selected, now=NOW, allow_carry_forward=True)


def test_push_healthy_collection_needs_no_previous_snapshot():
    selected = collector.prepare_publication(report(), None, event='push', now=NOW)
    assert 'publication' not in selected
    assert collector.validate_report(selected, now=NOW)['fresh_ok'] == 4


def test_push_fallback_at_48_hour_boundary_is_still_valid():
    current, previous = report(), report()
    fail(current, ['b'])
    for row in previous['sources'].values():
        row['fetched_at'] = (NOW - timedelta(hours=48)).isoformat()
    selected = collector.prepare_publication(current, previous, event='push', now=NOW)
    assert collector.validate_report(selected, now=NOW, allow_carry_forward=True)['ok'] == 4


@pytest.mark.parametrize('event, expected', [('push', 0), ('schedule', 1), ('workflow_dispatch', 1)])
def test_cli_wires_event_policy_without_live_collection(tmp_path, monkeypatch, capsys, event, expected):
    from contextlib import nullcontext
    import json
    import sys

    current, previous = report(), report()
    # Main uses wall-clock time; make fixture age independent of the run date.
    now = datetime.now(timezone.utc).isoformat()
    current['generated_at'] = previous['generated_at'] = now
    for data in (current, previous):
        for row in data['sources'].values():
            row['fetched_at'] = now
    fail(current, ['b'])
    previous_path = tmp_path / 'previous.json'
    previous_path.write_text(json.dumps(previous))
    output = tmp_path / 'public.json'
    output.write_text('retained old output')
    monkeypatch.setattr(collector, 'collect', lambda *args: current)
    monkeypatch.setattr(collector, 'public_specs', lambda *args: [])
    monkeypatch.setattr(collector, 'isolated_settings', lambda *args: nullcontext())
    monkeypatch.setattr(collector.tempfile, 'mkdtemp', lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(sys, 'argv', ['collect_sources', '--output', str(output),
                                    '--previous', str(previous_path), '--event', event])
    assert collector.main() == expected
    logged = json.loads(capsys.readouterr().out)
    if event == 'push':
        assert logged['publication']['mode'] == 'push-carry-forward'
        assert json.loads(output.read_text())['sources']['b']['fetched_at'] == now
    else:
        assert logged['status'] == 'rejected'
        assert output.read_text() == 'retained old output'


def test_malformed_previous_publication_metadata_is_rejected_cleanly():
    current, previous = report(), report()
    fail(current, ['b'])
    previous['publication'] = 'not an object'
    with pytest.raises(ValueError, match='previous.*invalid'):
        collector.prepare_publication(current, previous, event='push', now=NOW)


@pytest.mark.parametrize('degraded', [True, False])
def test_cli_warns_below_eighty_percent_fresh_without_rejecting_valid_coverage(
    tmp_path, monkeypatch, capsys, degraded
):
    from contextlib import nullcontext
    import json
    import sys

    current = report()
    now = datetime.now(timezone.utc).isoformat()
    current['generated_at'] = now
    for row in current['sources'].values():
        row['fetched_at'] = now
    if degraded:
        fail(current, ['d'])  # Three of four fresh; both kinds and all sections remain.
    output = tmp_path / 'public.json'
    monkeypatch.setattr(collector, 'collect', lambda *_args: current)
    monkeypatch.setattr(collector, 'public_specs', lambda *_args: [])
    monkeypatch.setattr(collector, 'isolated_settings', lambda *_args: nullcontext())
    monkeypatch.setattr(collector.tempfile, 'mkdtemp', lambda **_kwargs: str(tmp_path))
    monkeypatch.setattr(sys, 'argv', ['collect_sources', '--output', str(output), '--event', 'schedule'])

    assert collector.main() == 0
    captured = capsys.readouterr()
    logged = json.loads(captured.out)
    assert logged['fresh_ok'] == (3 if degraded else 4)
    assert output.is_file()
    if degraded:
        assert '::warning::' in captured.err
        assert '3/4' in captured.err and '80%' in captured.err
        assert logged['freshness_warning']
    else:
        assert not captured.err
        assert logged['freshness_warning'] is None
