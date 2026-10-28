"""Collect public configured RSS/arXiv/bioRxiv metadata for a static browser deployment.

This command never loads a reader profile, ranks articles, or uses credentials.
Run with the locked native environment; publish the output beside static assets.
Failures are retained in the manifest rather than replaced with invented items.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import yaml

# Kinds whose configured, non-profile queries are reproduced from the snapshot.
# Profile-driven, author-driven and OpenAlex venue sources stay live: venue
# queries are profile-independent but routinely hit the 120-item cap, so the
# native selection under a different window cannot be reproduced exactly.
SNAPSHOT_KINDS = {"rss", "arxiv", "biorxiv"}
# Publication still requires these kinds; a preprint-server outage leaves its
# record failed (readers fetch it directly) instead of blocking RSS/arXiv.
REQUIRED_KINDS = {"rss", "arxiv"}
# bioRxiv/medRxiv posting-date window. Readers request min_window_days (4) on
# first and daily brews; 5 days covers that from a snapshot made the previous
# UTC day. Longer catch-up windows are detected and fetched directly.
PREPRINT_DAYS = 5
# Collect the whole window. Readers re-apply the native 2,000-row cap
# themselves (falling back when it would apply), so CI must not truncate.
PREPRINT_COLLECTION_ROWS = 8000
IDENTITY_FIELDS = ("kind", "section", "url", "category", "server", "days")


@contextmanager
def isolated_settings(scratch):
    """Import config outside the project with no ambient settings or secrets."""
    working_directory = Path.cwd()
    environment = dict(os.environ)
    try:
        os.chdir(scratch)
        os.environ.clear()
        # Keep temporary files in this task's scratch. No credential-bearing
        # environment values reach the legacy module-level Settings instance.
        os.environ["TMPDIR"] = str(scratch)
        from dailydigest import config

        previous_settings, previous_accessor = config.SETTINGS, config.get_settings
        settings = config.Settings.model_construct(
            profile_path=str(scratch / "unused-profile.yaml"),
            opportunity_profile_path=str(scratch / "unused-opportunities.yaml"),
            db_path=str(scratch / "unused.db"), llm_backend="extractive",
            llm_api_key="", resend_api_key="",
        )
        config.SETTINGS = settings
        config.get_settings = lambda: settings
        try:
            yield settings
        finally:
            config.SETTINGS, config.get_settings = previous_settings, previous_accessor
    finally:
        os.chdir(working_directory)
        os.environ.clear()
        os.environ.update(environment)


def public_specs(path):
    from dailydigest.models import SourceSpec

    config = yaml.safe_load(Path(path).read_text())
    return [SourceSpec.model_validate({**row, "section": section})
            for section, rows in config.items() for row in rows
            if row.get("kind") in SNAPSHOT_KINDS and not row.get("profile_driven")]


def _preprint_record(adapter, spec, days):
    """Every parsed revision plus the evidence readers need to replay ``fetch``.

    Items are the adapter's own parsed rows before ``current_versions``, so a
    reader's shorter window selects revisions and withdrawals natively. Raw
    entry counts per posting date let readers detect the native row cap.
    """
    adapter.max_rows = PREPRINT_COLLECTION_ROWS
    adapter.fetch(spec, days=days)
    diagnostic = adapter.last_fetch_diagnostics["queries"][0]
    window_from, window_to = diagnostic["query"].split(":", 1)[1].split("/")
    dates = {key: dict(value) for key, value in adapter.last_fetch_entry_dates.items() if key}
    undated = dict(adapter.last_fetch_entry_dates.get(None) or {"entries": 0, "withdrawn": 0, "invalid": 0})
    if any(not window_from <= key <= window_to for key in dates):
        # Readers select a shorter window by posting date; entries the API
        # placed outside its interval could not be replayed exactly.
        raise RuntimeError("Preprint API returned entries outside the requested posting window")
    return {
        "server": spec.server,
        "window": {"from": window_from, "to": window_to},
        "coverage": {key: diagnostic[key] for key in (
            "total_available", "fetched_count", "exhausted", "truncated", "stop_reason")}
        | {"row_cap": adapter.max_rows, "dates": dict(sorted(dates.items())), "undated": undated},
        "items": [item.model_dump(mode="json") for item in adapter.last_fetch_versions],
        "withdrawals": [{"date": date, "item": item.model_dump(mode="json")} for date, item in zip(
            adapter.last_fetch_withdrawal_dates, adapter.last_fetch_withdrawals, strict=True)],
    }


def collect(specs, days=21, adapters=None, now=None):
    if adapters is None:
        from dailydigest.ingest.arxiv import ArxivSource
        from dailydigest.ingest.biorxiv import BiorxivSource
        from dailydigest.ingest.rss import RSSSource

        adapters = {"rss": RSSSource, "arxiv": ArxivSource, "biorxiv": BiorxivSource}
    now = now or (lambda: datetime.now(timezone.utc))

    def fetch(spec):
        window = PREPRINT_DAYS if spec.kind == "biorxiv" else days
        record = {"kind": spec.kind, "section": spec.section, "url": spec.url,
                  "category": spec.category, "days": window, "fetched_at": None}
        if spec.kind == "biorxiv":
            record["server"] = spec.server
        try:
            adapter = adapters[spec.kind]()
            if spec.kind == "biorxiv":
                fields = _preprint_record(adapter, spec, window)
            else:
                fields = {"items": [item.model_dump(mode="json")
                                    for item in adapter.fetch(spec, days=window)]}
            record.update(status="ok", fetched_at=now().isoformat(), **fields)
        except Exception as exc:
            record.update(status="error", attempted_at=now().isoformat(), items=[],
                          error=f"{type(exc).__name__}: {exc}"[:500])
        return spec.name, record

    # Publishers are independent; arXiv requests remain sequential and polite.
    rss = [spec for spec in specs if spec.kind == "rss"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        sources = dict(pool.map(fetch, rss))
    arxiv = [spec for spec in specs if spec.kind == "arxiv"]
    for index, spec in enumerate(arxiv):
        if index:
            time.sleep(3)
        name, record = fetch(spec)
        sources[name] = record
    # Both preprint servers share one API host; keep them sequential.
    for spec in (spec for spec in specs if spec.kind == "biorxiv"):
        name, record = fetch(spec)
        sources[name] = record
    return {"version": 1, "generated_at": now().isoformat(), "days": days,
            "public_only": True, "sources": sources}


def _count(value):
    return type(value) is int and value >= 0


def _validate_preprint_record(name, record, items):
    """Reject preprint evidence that could not reproduce the native adapter."""
    from datetime import date
    from dailydigest.models import Item

    server, window, coverage = record.get("server"), record.get("window"), record.get("coverage")
    if not isinstance(server, str) or not server:
        raise ValueError(f"Invalid preprint server: {name}")
    try:
        start, end = date.fromisoformat(window["from"]), date.fromisoformat(window["to"])
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"Invalid preprint window: {name}") from exc
    if (end - start).days != record["days"]:
        raise ValueError(f"Preprint window does not match its days: {name}")
    if (not isinstance(coverage, dict) or type(coverage.get("exhausted")) is not bool
            or type(coverage.get("truncated")) is not bool
            or not _count(coverage.get("fetched_count")) or not _count(coverage.get("row_cap"))
            or not (coverage.get("total_available") is None or _count(coverage["total_available"]))
            or not isinstance(coverage.get("dates"), dict) or not isinstance(coverage.get("undated"), dict)):
        raise ValueError(f"Invalid preprint coverage: {name}")
    buckets = {**coverage["dates"], None: coverage["undated"]}
    valid_entries = 0
    for key, counts in buckets.items():
        if key is not None:
            try:
                day = date.fromisoformat(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid preprint coverage date: {name}") from exc
            if not start <= day <= end:
                raise ValueError(f"Preprint coverage outside its window: {name}")
        if (not isinstance(counts, dict) or set(counts) != {"entries", "withdrawn", "invalid"}
                or not all(_count(value) for value in counts.values())
                or counts["withdrawn"] + counts["invalid"] > counts["entries"]):
            raise ValueError(f"Invalid preprint coverage counts: {name}")
        valid_entries += counts["entries"] - counts["withdrawn"] - counts["invalid"]
    if sum(counts["entries"] for counts in buckets.values()) != coverage["fetched_count"]:
        raise ValueError(f"Preprint coverage counts disagree with fetched entries: {name}")
    if valid_entries != len(items):
        raise ValueError(f"Preprint items disagree with coverage counts: {name}")
    for item in items:
        if item.published_at is not None and not start <= item.published_at.date() <= end:
            raise ValueError(f"Preprint item outside its window: {name}")
    withdrawals = record.get("withdrawals")
    if not isinstance(withdrawals, list):
        raise ValueError(f"Invalid preprint withdrawals: {name}")
    for row in withdrawals:
        if not isinstance(row, dict) or not (row.get("date") is None or row.get("date") in coverage["dates"]):
            raise ValueError(f"Invalid preprint withdrawal date: {name}")
        notice = Item.model_validate(row.get("item"))
        if (notice.source != name or notice.section != record["section"]
                or notice.metadata.get("is_retracted") is not True):
            raise ValueError(f"Preprint withdrawal identity mismatch: {name}")


def _validate_record(name, record, *, now):
    from dailydigest.models import Item, _VALID_SECTIONS

    if not isinstance(record, dict) or record.get("kind") not in SNAPSHOT_KINDS:
        raise ValueError(f"Invalid source record: {name}")
    if record.get("section") not in _VALID_SECTIONS:
        raise ValueError(f"Invalid source section: {name}")
    days = record.get("days")
    if type(days) is not int or not 1 <= days <= 90:
        raise ValueError(f"Invalid source window: {name}")
    if record.get("status") == "error":
        if record.get("items") != []:
            raise ValueError(f"Failed source contains items: {name}")
        return False
    if record.get("status") != "ok" or not isinstance(record.get("items"), list):
        raise ValueError(f"Invalid source status/items: {name}")
    try:
        fetched = datetime.fromisoformat(record["fetched_at"])
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"Invalid source timestamp: {name}") from exc
    if fetched.tzinfo is None or not -timedelta(minutes=5) <= now - fetched <= timedelta(hours=48):
        raise ValueError(f"Source timestamp expired or invalid: {name}")
    items = [Item.model_validate(row) for row in record["items"]]
    for item in items:
        if item.source != name or item.section != record["section"]:
            raise ValueError(f"Source item identity mismatch: {name}")
    if record["kind"] == "biorxiv":
        _validate_preprint_record(name, record, items)
    return True


def validate_report(report, *, now=None, min_fresh_fraction=0.5, allow_carry_forward=False):
    """Validate records and fresh coverage, or an explicitly allowed push fallback.

    Empty but successfully fetched feeds are valid. Push fallback still requires
    current successes and complete prior coverage; it does not reset record age.
    """
    now = now or datetime.now(timezone.utc)
    if (not isinstance(report, dict) or report.get("version") != 1
            or report.get("public_only") is not True
            or not isinstance(report.get("sources"), dict) or not report["sources"]):
        raise ValueError("Public source coverage is empty or invalid")
    sources = report["sources"]
    healthy = {name for name, row in sources.items() if _validate_record(name, row, now=now)}
    fresh = {name for name in healthy if not sources[name].get("carried_forward")}
    publication = report.get("publication")
    if publication is not None and not isinstance(publication, dict):
        raise ValueError("Invalid public source publication metadata")
    fallback = allow_carry_forward and (publication or {}).get("mode") == "push-carry-forward"
    if fallback:
        # A push may preserve an earlier validated publication, but cannot turn
        # an entirely failed collection into a successful refresh. These are
        # the actual current attempt records, not refreshed old timestamps.
        attempted_ok = set()
        for name, row in sources.items():
            attempt = row.get("latest_attempt")
            if not row.get("carried_forward") or not isinstance(attempt, dict):
                raise ValueError("Push carry-forward requires current attempt records")
            if any(row.get(field) != attempt.get(field) for field in IDENTITY_FIELDS):
                raise ValueError("Push carry-forward attempt configuration mismatch")
            if _validate_record(name, attempt, now=now) and not attempt.get("carried_forward"):
                attempted_ok.add(name)
        if not attempted_ok:
            raise ValueError("Public source coverage has no fresh successes in the current collection")
    eligible = healthy if fallback else fresh
    if len(eligible) < max(1, math.ceil(len(sources) * min_fresh_fraction)):
        raise ValueError(f"Public source coverage too low: {len(eligible)}/{len(sources)} "
                         + ("retained successes" if fallback else "fresh successes"))
    for field in ("kind", "section"):
        # Optional snapshot kinds may fail without blocking publication; their
        # readers see the failed record and fetch those sources directly.
        configured = {row[field] for row in sources.values()
                      if field != "kind" or row[field] in REQUIRED_KINDS}
        successful = {sources[name][field] for name in eligible}
        if configured - successful:
            raise ValueError(f"Public source coverage missing {field}: {sorted(configured - successful)}")
    summary = {"sources": len(sources), "ok": len(healthy), "fresh_ok": len(fresh),
               "carried_forward": len(healthy - fresh)}
    if "language_decisions" in report:
        # Optional accelerator. Another rule version is valid but ignored by
        # readers; a malformed map indicates a collector bug and is rejected.
        from dailydigest.dedupe import language_decisions_status

        status = language_decisions_status(report["language_decisions"], max_entries=sum(
            len(row["items"]) for row in sources.values() if row.get("status") == "ok"))
        if status in {"absent", "malformed"}:  # an explicit null is not "absent"
            raise ValueError("Invalid public source language decisions")
        summary["language_decisions"] = status
    return summary


def with_language_decisions(report):
    """Attach current-rule language decisions for every published item.

    Decisions use exactly dedupe.filter_english's sample and rule, keyed by the
    sample's sha256 and rule version. Short and failed samples are omitted.
    """
    from dailydigest.dedupe import precompute_language_decisions
    from dailydigest.models import Item

    annotated = {key: value for key, value in report.items() if key != "language_decisions"}
    items = [Item.model_validate(row) for record in report.get("sources", {}).values()
             if record.get("status") == "ok" for row in record.get("items", [])]
    decisions = precompute_language_decisions(items)
    if decisions is not None:
        annotated["language_decisions"] = decisions
    return annotated


def carry_forward(report, previous, *, now=None):
    """Retain only compatible validated records without changing their age."""
    now = now or datetime.now(timezone.utc)
    merged = deepcopy(report)
    if (not isinstance(previous, dict) or previous.get("version") != 1
            or previous.get("public_only") is not True
            or not isinstance(previous.get("sources"), dict)):
        return merged
    for name, failed in merged["sources"].items():
        old = previous["sources"].get(name)
        if failed.get("status") != "error" or not isinstance(old, dict):
            continue
        if any(old.get(field) != failed.get(field) for field in IDENTITY_FIELDS):
            continue
        try:
            valid = _validate_record(name, old, now=now)
        except (ValueError, TypeError):
            continue
        if valid:
            merged["sources"][name] = {**deepcopy(old), "carried_forward": True,
                                       "latest_attempt": deepcopy(failed)}
    return merged


def prepare_publication(report, previous=None, *, event="schedule", now=None):
    """Keep refreshes strict; partial push outages may retain the previous site.

    The push escape hatch publishes only the previous validated records, with
    their original fetched_at. It never substitutes a partially refreshed slate
    or allows a collection with no genuine current successes.
    """
    now = now or datetime.now(timezone.utc)
    if event not in {"push", "schedule", "workflow_dispatch"}:
        raise ValueError(f"Unknown publication event: {event}")
    sources = report.get("sources", {}) if isinstance(report, dict) else {}
    fresh = {name for name, row in sources.items()
             if _validate_record(name, row, now=now) and not row.get("carried_forward")}
    if not fresh:
        raise ValueError("Public source coverage has no fresh successes in the current collection")
    merged = carry_forward(report, previous, now=now)
    try:
        validate_report(merged, now=now)
        return merged
    except ValueError:
        if event != "push":
            raise
    try:
        validate_report(previous, now=now, allow_carry_forward=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Push fallback previous snapshot unavailable, expired, or invalid: {exc}") from exc
    if set(previous["sources"]) != set(sources) or any(
        any(old.get(field) != sources[name].get(field)
            for field in IDENTITY_FIELDS)
        for name, old in previous["sources"].items()
    ):
        raise ValueError("Push fallback previous snapshot configuration does not match current sources")
    selected = deepcopy(previous)
    selected["publication"] = {
        "mode": "push-carry-forward",
        "attempted_at": report.get("generated_at"),
        "warning": "Partial collection outage: this code deployment retains the previous validated "
                   "source snapshot. Original source timestamps are unchanged; a refresh is still needed.",
    }
    for name, old in selected["sources"].items():
        old["carried_forward"] = True
        old["latest_attempt"] = deepcopy(sources[name])
    validate_report(selected, now=now, allow_carry_forward=True)
    return selected


def publish(report, output, *, now=None, allow_carry_forward=False):
    summary = validate_report(report, now=now, allow_carry_forward=allow_carry_forward)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".pending")
    temporary.write_text(json.dumps(report, ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="config/sources.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--days", type=int, default=21, choices=range(1, 91))
    parser.add_argument("--previous", type=Path, help="Previous public snapshot for validated partial carryforward")
    parser.add_argument("--event", choices=("push", "schedule", "workflow_dispatch"),
                        default="workflow_dispatch", help="Push deployments may retain a recent validated snapshot")
    args = parser.parse_args()
    sources_path, output = Path(args.sources).resolve(), Path(args.output).resolve()
    scratch = Path(tempfile.mkdtemp(prefix="dailydigest-public-sources-"))
    with isolated_settings(scratch):
        report = collect(public_specs(sources_path), args.days)
    try:
        previous = None
        if args.previous and args.previous.is_file():
            try:
                previous = json.loads(args.previous.read_text())
            except (ValueError, OSError):
                previous = None
        report = with_language_decisions(prepare_publication(report, previous, event=args.event))
        summary = publish(report, output, allow_carry_forward=args.event == "push")
    except ValueError as exc:
        print(json.dumps({"status": "rejected", "error": str(exc)}))
        return 1
    failures = {name: source["error"] for name, source in report["sources"].items()
                if source["status"] != "ok"}
    freshness_warning = None
    if summary["fresh_ok"] / summary["sources"] < 0.8:
        freshness_warning = (
            f'Fresh public source coverage is {summary["fresh_ok"]}/{summary["sources"]}, '
            'below 80%. Publication passed the minimum coverage gate; '
            'inspect failed sources and retained snapshot ages.'
        )
        print(f"::warning::{freshness_warning}", file=sys.stderr)
    print(json.dumps({"output": str(output), **summary, "failures": failures,
                      "freshness_warning": freshness_warning,
                      "publication": report.get("publication")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
