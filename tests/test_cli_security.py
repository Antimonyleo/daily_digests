from __future__ import annotations


def test_web_cli_binds_loopback_only():
    from dailydigest.cli import _is_loopback_bind

    assert _is_loopback_bind("127.0.0.1")
    assert _is_loopback_bind("localhost")
    assert _is_loopback_bind("::1")
    assert not _is_loopback_bind("0.0.0.0")
    assert not _is_loopback_bind("192.168.1.10")
