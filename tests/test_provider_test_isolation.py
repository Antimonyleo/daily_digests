"""Owned provider children inherit the test suite's no-network contract."""
import pytest

from dailydigest import summarize


@pytest.mark.parametrize('url', [
    'https://api.example.invalid/v1/messages',
    'https://localhost.example.invalid/v1/messages',
    'https://192.0.2.1/v1/messages',
])
def test_owned_provider_cannot_bypass_test_network_guard(url, monkeypatch):
    def must_not_spawn(*_args, **_kwargs):
        pytest.fail('Network guard must reject external providers before spawning')

    monkeypatch.setattr(summarize.subprocess, 'Popen', must_not_spawn)
    with pytest.raises(RuntimeError, match='Network call blocked'):
        summarize._http_post(url, headers={}, body={}, timeout=1, should_cancel=lambda: False)
