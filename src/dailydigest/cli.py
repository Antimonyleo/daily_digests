"""Typer CLI entry point: dd ingest|rank|send|run-all|prune."""

from __future__ import annotations

import ipaddress
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import typer

from . import inbound as inbound_mod
from . import votes as votes_mod
from .config import SETTINGS, load_profile, section_enabled
from .dedupe import dedupe_ranking_candidates
from .pipeline import ingest_all, run_all
from .rank.profile import build_profile_matrix_with_rocchio
from .rank.ranker import LRRanker, reset_lr_cache, score_items
from .store import exclude_reviewed_items, recent_items
from .store import prune as store_prune

app = typer.Typer(add_completion=False, help="DailyDigest CLI")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _is_loopback_bind(host: str) -> bool:
    h = host.strip().lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _remote_bind_allowed(allow_remote: bool) -> bool:
    """True iff the opt-in escape hatch for non-loopback binds is enabled.

    Either the ``--allow-remote`` flag or ``DD_ALLOW_REMOTE_BIND=1`` (used by the
    Docker image, whose port is loopback-mapped by docker-compose) opts in.
    """
    if allow_remote:
        return True
    return os.environ.get("DD_ALLOW_REMOTE_BIND", "").strip() in {"1", "true", "yes"}


def _require_loopback_bind(host: str, allow_remote: bool = False) -> None:
    if _is_loopback_bind(host):
        return
    if _remote_bind_allowed(allow_remote):
        typer.echo(
            f"WARNING: binding {host} (non-loopback). DailyDigest has no auth; "
            "it must sit behind Docker port-mapping or a trusted network only.",
            err=True,
        )
        return
    raise typer.BadParameter(
        "DailyDigest has no authentication; bind to 127.0.0.1 or localhost "
        "(or set DD_ALLOW_REMOTE_BIND=1 / pass --allow-remote for containerized use)."
    )


def should_run_now() -> bool:
    """True iff the current hour in USER_TZ matches DIGEST_HOUR."""
    try:
        tz = ZoneInfo(SETTINGS.user_tz)
    except Exception:
        tz = ZoneInfo("UTC")
    now = datetime.now(tz)
    return now.hour == SETTINGS.digest_hour


@app.command()
def ingest() -> None:
    """Run ingest stage only."""
    n = ingest_all()
    typer.echo(f"upserted {n} new items")


@app.command()
def rank() -> None:
    """Re-rank recent items and print top 20."""
    profile = load_profile()
    # Use the SAME profile representation the pipeline ranks with (weighted
    # facet matrix + Rocchio blend), not the legacy single centroid — otherwise
    # this "inspect the current ranking" view disagrees with the real digest.
    pv = build_profile_matrix_with_rocchio(profile, votes_mod.signed_vote_count())
    items = [
        item
        for item in exclude_reviewed_items(recent_items(days=2))
        if section_enabled(SETTINGS, item.section or "")
    ]
    items = dedupe_ranking_candidates(items)
    scored = score_items(
        items,
        pv,
        profile.downweight,
        reason_penalty_map=votes_mod.reason_penalty_map(items),
    )
    for row, s in scored[:20]:
        title = (row.title or "").strip().replace("\n", " ")
        if len(title) > 90:
            title = title[:87] + "..."
        typer.echo(f"{s:+.4f}  [{(row.section or '')[:3]}] {(row.source or '')[:18]:<18}  {title}")


@app.command()
def send() -> None:
    """Run the full pipeline (ingest + rank + summarize + render + send)."""
    digest_id = run_all(dry_run=False)
    typer.echo(f"sent digest {digest_id}")


@app.command("run-all")
def run_all_cmd(
    dry_run: bool = typer.Option(False, "--dry-run", help="Render to disk; do not email."),
    gate: bool = typer.Option(False, "--gate", help="Only run if local hour matches DIGEST_HOUR."),
    backfill: int = typer.Option(
        0, "--backfill", help="Look back this many days when ranking (0 = use default of 2)."
    ),
) -> None:
    """Full pipeline with optional dry-run / time-gate / backfill."""
    if gate and not should_run_now():
        typer.echo("gate: not the digest hour; skipping.")
        raise typer.Exit(code=0)
    digest_id = run_all(dry_run=dry_run, backfill_days=backfill or None)
    typer.echo(f"digest {digest_id} {'(dry-run)' if dry_run else 'sent'}")


@app.command()
def brew(
    send: bool = typer.Option(
        False,
        "--send",
        help="Send by email when configured; otherwise create a local preview.",
    ),
    backfill: int = typer.Option(
        0,
        "--backfill",
        help="Look back this many days (0 = choose automatically).",
        min=0,
    ),
) -> None:
    """Brew today's digest from the command line."""
    digest_id = run_all(dry_run=not send, backfill_days=backfill or None)
    destination = "email requested; local fallback enabled" if send else "local preview"
    typer.echo(f"brewed digest {digest_id} ({destination})")


@app.command()
def vote(
    line: str = typer.Argument(
        "",
        help='Vote string, e.g. "+R3 R7 -I5". Unsigned tokens default to +.',
    ),
    train: bool = typer.Option(
        False,
        "--train",
        help="Retrain the LR ranker on all stored votes.",
    ),
    digest_id: str = typer.Option(
        "",
        "--digest-id",
        help="Resolve labels against this digest id (default: most recent).",
    ),
) -> None:
    """Record thumbs feedback or retrain the LR ranker."""
    if train:
        dataset = votes_mod.vote_dataset()
        if dataset is None:
            typer.echo(
                f"need at least {votes_mod.MIN_VOTES_FOR_LR} votes to train; skipping."
            )
            raise typer.Exit(code=0)
        X, y = dataset
        ranker = LRRanker()
        try:
            ranker.fit(X, y)
        except ValueError as e:
            typer.echo(f"train aborted: {e}")
            raise typer.Exit(code=1) from e
        reset_lr_cache()
        typer.echo(f"trained on {len(y)} votes")
        return

    if not line.strip():
        typer.echo("usage: dd vote \"+R3 R7 -I5\"  |  dd vote --train")
        raise typer.Exit(code=2)

    counts = votes_mod.record_votes(line, digest_id=digest_id or None)
    typer.echo(
        f"recorded: up={counts['up']} down={counts['down']} unknown={counts['unknown']}"
    )


@app.command()
def prune() -> None:
    """Prune items older than retention_days."""
    n = store_prune(SETTINGS.retention_days)
    typer.echo(f"pruned {n} items")


@app.command("eval")
def eval_ranking(
    k: int = typer.Option(10, "--k", help="Cutoff for nDCG@k / precision@k."),
    json_out: bool = typer.Option(False, "--json", help="Emit the full report as JSON."),
) -> None:
    """Score past digests against your votes (nDCG@k, P@k, MAP, pairwise acc).

    Replays the persisted ranking of each historical digest against the thumbs
    you later cast. Use it to A/B ranker changes: run digests under config A,
    then config B, and compare. Higher is better for all metrics.
    """
    from .rank.evaluate import evaluate_history

    report = evaluate_history(k=k)
    if json_out:
        import json

        typer.echo(json.dumps(report.as_dict(), indent=2))
        return

    def _fmt(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else "n/a"

    typer.echo(
        f"digests scored: {report.n_digests_scored}/{report.n_digests_total} "
        f"(votes used: {report.n_votes})"
    )
    if report.n_digests_scored == 0:
        typer.echo("No voted digests yet — vote on a few items, then re-run.")
        return
    typer.echo(f"nDCG@{k}:           {_fmt(report.ndcg_at_k)}")
    typer.echo(f"precision@{k}:      {_fmt(report.precision_at_k)}")
    typer.echo(f"MAP:               {_fmt(report.map_score)}")
    typer.echo(f"pairwise accuracy: {_fmt(report.pairwise_accuracy)}")


@app.command()
def calibrate() -> None:
    """Fit the score→probability calibrator from your vote history.

    Maps ranking scores to P(relevant) so the relevance floor self-tunes to your
    feedback. Needs a modest number of votes spanning both thumbs.
    """
    from .rank.calibrate import MIN_VOTES_FOR_CALIBRATION, fit_calibrator

    params = fit_calibrator()
    if params is None:
        typer.echo(
            f"Not enough feedback to calibrate yet "
            f"(need ~{MIN_VOTES_FOR_CALIBRATION} votes of both signs)."
        )
        return
    typer.echo(
        f"Calibrated on {params['n']} votes: P(relevant)=sigmoid("
        f"{params['a']:.3f}*score + {params['b']:.3f})."
    )


@app.command("ingest-replies")
def ingest_replies() -> None:
    """Poll IMAP for digest replies and record votes from their bodies."""
    summary = inbound_mod.process_replies()
    if summary.get("skipped"):
        typer.echo("imap not configured; set IMAP_USER and IMAP_PASSWORD in env.")
        return
    msgs = summary.get("messages", 0)
    up = summary.get("votes_up", 0)
    down = summary.get("votes_down", 0)
    unknown = summary.get("votes_unknown", 0)
    typer.echo(
        f"processed {msgs} messages: up={up} down={down} unknown={unknown}"
    )
    errors = summary.get("errors") or []
    if errors:
        typer.echo(f"errors: {len(errors)}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (loopback only by default)."),
    port: int = typer.Option(8765, "--port", help="Port to listen on."),
    allow_remote: bool = typer.Option(
        False,
        "--allow-remote",
        help="Permit non-loopback --host (also via DD_ALLOW_REMOTE_BIND=1). "
        "No auth; only safe behind Docker port-mapping / a trusted network.",
    ),
) -> None:
    """Run the local FastAPI web UI for browsing today's digest and voting."""
    import uvicorn

    _require_loopback_bind(host, allow_remote)
    typer.echo(f"Open http://{host}:{port} in a browser.")
    uvicorn.run("dailydigest.web:app", host=host, port=port, log_level="info")


@app.command()
def start(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (loopback only by default)."),
    port: int = typer.Option(8765, "--port", help="Port to listen on."),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Don't auto-open the browser."
    ),
    allow_remote: bool = typer.Option(
        False,
        "--allow-remote",
        help="Permit non-loopback --host (also via DD_ALLOW_REMOTE_BIND=1). "
        "No auth; only safe behind Docker port-mapping / a trusted network.",
    ),
) -> None:
    """Boot the FastAPI server and auto-open the browser (recommended entrypoint)."""
    import threading
    import webbrowser

    import uvicorn

    _require_loopback_bind(host, allow_remote)
    url = f"http://{host}:{port}"
    typer.echo(f"Starting DailyDigest at {url}")
    if no_browser:
        typer.echo("First run? Open the URL to use the setup wizard. Press Ctrl-C to stop.")
    else:
        typer.echo("First run? The browser will open the setup wizard. Press Ctrl-C to stop.")

    if not no_browser:
        # Defer the browser open ~1.5s so uvicorn has time to bind the port.
        # Timer threads are daemon-by-default off, but uvicorn's SIGINT handler
        # tears the process down anyway; the timer is harmless if it fires
        # post-shutdown.
        threading.Timer(1.5, lambda: webbrowser.open(f"{url}/")).start()

    uvicorn.run("dailydigest.web:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
