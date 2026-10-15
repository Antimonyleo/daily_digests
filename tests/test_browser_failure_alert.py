"""Operational alerts are opt-in, public-only and tested without networking."""
import importlib.util
from pathlib import Path
import json


spec = importlib.util.spec_from_file_location('failure_alert', Path(__file__).resolve().parents[1] / 'browser/report_failure.py')
alert = importlib.util.module_from_spec(spec)


def load():
    spec.loader.exec_module(alert)
    return alert


def test_missing_alert_configuration_does_not_contact_anything(monkeypatch, capsys):
    module = load()
    monkeypatch.delenv('DAILYDIGEST_ALERT_WEBHOOK', raising=False)
    monkeypatch.setattr(module, 'urlopen', lambda *a, **k: (_ for _ in ()).throw(AssertionError('Network')))
    assert module.main() == 0
    assert 'DAILYDIGEST_ALERT_WEBHOOK' in capsys.readouterr().out


def test_configured_alert_has_only_public_workflow_link(monkeypatch):
    module = load()
    monkeypatch.setenv('DAILYDIGEST_ALERT_WEBHOOK', 'https://alerts.example/private-token')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'example/digest')
    monkeypatch.setenv('GITHUB_RUN_ID', '123')
    calls = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(module, 'urlopen', lambda request, timeout: calls.append((request, timeout)) or Reply())
    assert module.main() == 0
    assert len(calls) == 1 and calls[0][1] == 10
    payload = json.loads(calls[0][0].data)
    assert set(payload) == {'text'}
    assert 'https://github.com/example/digest/actions/runs/123' in payload['text']
    assert 'private-token' not in payload['text']


def test_alert_failure_does_not_print_secret_or_replace_original_failure(monkeypatch, capsys):
    module = load()
    monkeypatch.setenv('DAILYDIGEST_ALERT_WEBHOOK', 'https://alerts.example/private-token')
    monkeypatch.setattr(module, 'urlopen', lambda *a, **k: (_ for _ in ()).throw(OSError('private-token')))
    assert module.main() == 1
    assert 'private-token' not in capsys.readouterr().out


def issue_environment(monkeypatch, result='failure'):
    monkeypatch.setenv('DAILYDIGEST_ALERT_MODE', 'github-issue')
    monkeypatch.setenv('DAILYDIGEST_REFRESH_RESULT', result)
    monkeypatch.setenv('GITHUB_TOKEN', 'fixture-private-token')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'example/digest')
    monkeypatch.setenv('GITHUB_RUN_ID', '123')


def issue_transport(module, monkeypatch, existing):
    calls = []

    class Reply:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return json.dumps(self.payload).encode()

    def request(req, timeout):
        calls.append(req)
        assert req.headers['Authorization'] == 'Bearer fixture-private-token'
        assert timeout == 10
        return Reply(existing if req.get_method() == 'GET' else {'number': 42})

    monkeypatch.setattr(module, 'urlopen', request)
    return calls


def test_scheduled_failure_creates_one_public_only_recurring_issue(monkeypatch):
    module = load()
    issue_environment(monkeypatch)
    calls = issue_transport(module, monkeypatch, [])
    assert module.main() == 0
    assert [req.get_method() for req in calls] == ['GET', 'POST']
    assert calls[-1].full_url == 'https://api.github.com/repos/example/digest/issues'
    payload = json.loads(calls[-1].data)
    assert 'actions/runs/123' in payload['body']
    assert '48 hours' in payload['body']
    assert 'fixture-private-token' not in json.dumps(payload)


def test_repeated_failure_reopens_the_existing_issue_without_duplicates(monkeypatch):
    module = load()
    issue_environment(monkeypatch)
    calls = issue_transport(module, monkeypatch, [{'number': 42, 'state': 'closed',
        'title': module.ISSUE_TITLE, 'body': module.ISSUE_MARKER,
        'user': {'login': 'github-actions[bot]', 'type': 'Bot'}}])
    assert module.main() == 0
    assert [req.get_method() for req in calls] == ['GET', 'PATCH']
    assert calls[-1].full_url.endswith('/issues/42')
    assert json.loads(calls[-1].data)['state'] == 'open'


def test_successful_strict_refresh_resolves_existing_issue_but_creates_none(monkeypatch):
    module = load()
    issue_environment(monkeypatch, 'success')
    calls = issue_transport(module, monkeypatch, [{'number': 42, 'state': 'open',
        'title': module.ISSUE_TITLE, 'body': module.ISSUE_MARKER,
        'user': {'login': 'github-actions[bot]', 'type': 'Bot'}}])
    assert module.main() == 0
    assert json.loads(calls[-1].data)['state'] == 'closed'
    calls = issue_transport(module, monkeypatch, [])
    assert module.main() == 0
    assert [req.get_method() for req in calls] == ['GET']


def test_issue_permission_error_is_visible_without_credentials(monkeypatch, capsys):
    module = load()
    issue_environment(monkeypatch)
    monkeypatch.setattr(module, 'urlopen', lambda *a, **k: (_ for _ in ()).throw(OSError('fixture-private-token')))
    assert module.main() == 1
    output = capsys.readouterr().out
    assert 'issue' in output.lower()
    assert 'fixture-private-token' not in output


def test_recurring_alert_ignores_human_spoofs_and_pull_requests(monkeypatch):
    module = load()
    issue_environment(monkeypatch)
    common = {'title': module.ISSUE_TITLE, 'body': module.ISSUE_MARKER, 'state': 'open'}
    calls = issue_transport(module, monkeypatch, [
        {**common, 'number': 17, 'user': {'login': 'untrusted-reader', 'type': 'User'}},
        {**common, 'number': 18, 'user': {'login': 'github-actions[bot]', 'type': 'Bot'}, 'pull_request': {}},
        {**common, 'number': 42, 'user': {'login': 'github-actions[bot]', 'type': 'Bot'}},
    ])
    assert module.main() == 0
    assert calls[-1].full_url.endswith('/issues/42')
