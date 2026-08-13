from __future__ import annotations

import pytest
import typer
from typer.testing import CliRunner


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


def test_brew_defaults_to_safe_local_preview(monkeypatch):
    from dailydigest import cli

    calls = []
    monkeypatch.setattr(
        cli,
        "run_all",
        lambda dry_run, backfill_days: calls.append((dry_run, backfill_days))
        or "2026-08-10",
    )

    result = CliRunner().invoke(cli.app, ["brew"])

    assert result.exit_code == 0
    assert calls == [(True, None)]
    assert "brewed digest 2026-08-10 (local preview saved)" in result.stdout


def test_brew_send_and_backfill_are_explicit(monkeypatch):
    from dailydigest import cli

    calls = []
    monkeypatch.setattr(
        cli,
        "run_all",
        lambda dry_run, backfill_days: calls.append((dry_run, backfill_days))
        or "2026-08-10",
    )

    result = CliRunner().invoke(cli.app, ["brew", "--send", "--backfill", "5"])

    assert result.exit_code == 0
    assert calls == [(False, 5)]
    assert "email not sent; local preview saved" in result.stdout


def test_brew_rejects_negative_backfill():
    from dailydigest import cli

    result = CliRunner().invoke(cli.app, ["brew", "--backfill", "-1"])

    assert result.exit_code == 2
    assert "x>=0" in result.stderr


def test_vote_training_stays_inside_the_compute_job(monkeypatch):
    from dailydigest import cli

    inside_compute_job = False

    def run_compute_job(action):
        nonlocal inside_compute_job
        inside_compute_job = True
        try:
            return action()
        finally:
            inside_compute_job = False

    class FakeRanker:
        def fit(self, _features, _labels):
            assert inside_compute_job is True

    monkeypatch.setattr(cli, "_run_compute_job", run_compute_job)
    monkeypatch.setattr(cli.votes_mod, "vote_dataset", lambda: ([[1.0]], [1]))
    monkeypatch.setattr(cli, "LRRanker", FakeRanker)
    monkeypatch.setattr(cli, "reset_lr_cache", lambda: None)

    result = CliRunner().invoke(cli.app, ["vote", "--train"])

    assert result.exit_code == 0
    assert "trained on 1 votes" in result.stdout
    assert inside_compute_job is False
