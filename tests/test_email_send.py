from __future__ import annotations

import sys


def test_send_digest_returns_false_when_resend_raises(monkeypatch, tmp_path):
    from dailydigest import config as config_mod
    from dailydigest import email_send

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    monkeypatch.setenv("RESEND_API_KEY", "test-key")
    monkeypatch.setenv("DIGEST_TO", "hao@example.com")
    config_mod.reload_settings()
    email_send.SETTINGS = config_mod.SETTINGS

    class _Emails:
        @staticmethod
        def send(_payload):
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "resend", type("FakeResend", (), {"Emails": _Emails, "api_key": ""}))

    assert email_send.send_digest("<p>x</p>", "Subject", dry_run=False) is False
