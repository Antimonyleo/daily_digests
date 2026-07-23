from __future__ import annotations

import pytest
import typer


def test_web_cli_binds_loopback_only():
    from dailydigest.cli import _is_loopback_bind

    assert _is_loopback_bind("127.0.0.1")
    assert _is_loopback_bind("localhost")
    assert _is_loopback_bind("::1")
    assert not _is_loopback_bind("0.0.0.0")
    assert not _is_loopback_bind("192.168.1.10")


def test_require_loopback_bind_rejects_non_loopback_by_default(monkeypatch):
    from dailydigest.cli import _require_loopback_bind

    monkeypatch.delenv("DD_ALLOW_REMOTE_BIND", raising=False)
    _require_loopback_bind("127.0.0.1")  # loopback always fine
    with pytest.raises(typer.BadParameter):
        _require_loopback_bind("0.0.0.0")


def test_require_loopback_bind_allows_remote_via_flag(monkeypatch):
    from dailydigest.cli import _require_loopback_bind

    monkeypatch.delenv("DD_ALLOW_REMOTE_BIND", raising=False)
    # Opt-in via flag: 0.0.0.0 accepted, no exception raised.
    _require_loopback_bind("0.0.0.0", allow_remote=True)


def test_require_loopback_bind_allows_remote_via_env(monkeypatch):
    from dailydigest.cli import _require_loopback_bind

    monkeypatch.setenv("DD_ALLOW_REMOTE_BIND", "1")
    _require_loopback_bind("0.0.0.0")
