"""An already cached configuration must not bypass interrupted-save recovery."""
import json

import pytest

from dailydigest import config
from dailydigest.settings_transaction import SettingsRecoveryError


def test_cached_settings_reject_unverifiable_journal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('DD_APP_ROOT', str(tmp_path))
    config.get_settings.cache_clear()
    original = config.get_settings()
    assert config.get_settings() is original
    (tmp_path / '.settings-transaction.json').write_text(json.dumps({'state': 'damaged'}))
    with pytest.raises(SettingsRecoveryError):
        config.get_settings()


def test_cached_settings_are_refreshed_after_recovery(tmp_path, monkeypatch):
    from dailydigest import settings_transaction as transaction

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('DD_APP_ROOT', str(tmp_path))
    config.get_settings.cache_clear()
    original = config.get_settings()
    calls = []

    def recover(_root):
        calls.append(True)
        return len(calls) == 1

    monkeypatch.setattr(transaction, 'recover_settings_transaction', recover)
    recovered = config.get_settings()
    assert recovered is not original
    assert config.get_settings() is recovered
    assert len(calls) >= 3
