"""Typer CLI entry point: dd ingest|rank|send|run-all|prune."""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import typer

from . import inbound as inbound_mod
from . import votes as votes_mod
from .config import SETTINGS, load_profile
from .pipeline import ingest_all, run_all
from .rank.profile import build_profile_vector
from .rank.ranker import LRRanker, reset_lr_cache, score_items
from .store import prune as store_prune
from .store import recent_items

app = typer.Typer(add_completion=False, help="DailyDigest CLI")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
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
    pv = build_profile_vector(profile)
    items = recent_items(days=2)
    scored = score_items(items, pv, profile.downweight)
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
) -> None:
    """Run the local FastAPI web UI for browsing today's digest and voting."""
    import uvicorn

    typer.echo(f"Open http://{host}:{port} in a browser.")
    uvicorn.run("dailydigest.web:app", host=host, port=port, log_level="info")


@app.command()
def start(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (loopback only by default)."),
    port: int = typer.Option(8765, "--port", help="Port to listen on."),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Don't auto-open the browser."
    ),
) -> None:
    """Boot the FastAPI server and auto-open the browser (recommended entrypoint)."""
    import threading
    import webbrowser

    import uvicorn

    url = f"http://{host}:{port}"
    typer.echo(f"Starting DailyDigest at {url}")
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
