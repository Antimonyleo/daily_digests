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
