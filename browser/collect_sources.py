"""Collect public configured RSS/arXiv metadata for a static browser deployment.

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
            if row.get("kind") in {"rss", "arxiv"} and not row.get("profile_driven")]


def collect(specs, days=21, adapters=None, now=None):
    if adapters is None:
        from dailydigest.ingest.arxiv import ArxivSource
        from dailydigest.ingest.rss import RSSSource

        adapters = {"rss": RSSSource, "arxiv": ArxivSource}
    now = now or (lambda: datetime.now(timezone.utc))

    def fetch(spec):
        record = {"kind": spec.kind, "section": spec.section, "url": spec.url,
                  "category": spec.category, "days": days, "fetched_at": None}
        try:
            items = adapters[spec.kind]().fetch(spec, days=days)
            record.update(status="ok", fetched_at=now().isoformat(),
                          items=[item.model_dump(mode="json") for item in items])
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
    return {"version": 1, "generated_at": now().isoformat(), "days": days,
            "public_only": True, "sources": sources}


def _validate_record(name, record, *, now):
    from dailydigest.models import Item, _VALID_SECTIONS

    if not isinstance(record, dict) or record.get("kind") not in {"rss", "arxiv"}:
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
    for row in record["items"]:
        item = Item.model_validate(row)
        if item.source != name or item.section != record["section"]:
            raise ValueError(f"Source item identity mismatch: {name}")
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
            if any(row.get(field) != attempt.get(field)
                   for field in ("kind", "section", "url", "category", "days")):
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
        configured = {row[field] for row in sources.values()}
        successful = {sources[name][field] for name in eligible}
        if configured - successful:
            raise ValueError(f"Public source coverage missing {field}: {sorted(configured - successful)}")
    return {"sources": len(sources), "ok": len(healthy), "fresh_ok": len(fresh),
            "carried_forward": len(healthy - fresh)}


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
        if any(old.get(field) != failed.get(field) for field in ("kind", "section", "url", "category", "days")):
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
            for field in ("kind", "section", "url", "category", "days"))
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
        report = prepare_publication(report, previous, event=args.event)
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
