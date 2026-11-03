"""Judge a browser source audit: can the live site still reach every source?

Directly fetched sources need browser (CORS) access. Sources the daily public
snapshot carries are reachable while their published record is usable (the
same rules readers apply) or, failing that, while a direct read works. A
source with neither path fails the check. Run from the repo root after
``node browser/source-audit.mjs <origin> <report.json>``:

    uv run python browser/check_source_access.py <report.json> \
        [--snapshot public-sources.json] [--summary out.md]

Exit status 1 names the unreachable sources; the report holds evidence.
"""

import argparse
from datetime import datetime, timedelta, timezone
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from collect_sources import SNAPSHOT_KINDS  # noqa: E402

# Kinds with no network fetch at all.
OFFLINE_KINDS = {"curated_calls", "curated_events"}
# Readers ignore records older than this (browser/network.py).
SNAPSHOT_MAX_AGE = timedelta(hours=48)
# Two missed twice-daily refreshes: still served, but close to expiring.
SNAPSHOT_AGING = timedelta(hours=24)
IDENTITY_FIELDS = ("kind", "section", "url", "category", "server")
# The browser could read these answers, so CORS and transport work; the server
# was only busy (429) or briefly failing (5xx). For directly fetched sources
# they are reported, not failed. Timeouts and unreadable answers still fail,
# and so does a carried source with no usable snapshot record.
READABLE = "readable_http_success"


def upstream_trouble(row):
    return row.get("outcome") == "readable_rate_limit" or (
        row.get("outcome") == "readable_http_error" and (row.get("status") or 0) >= 500)


def website_sources(config_path):
    from dailydigest.config import load_sources

    return [spec for spec in load_sources(str(config_path))
            if spec.kind not in OFFLINE_KINDS and not spec.local_only]


def carried(spec):
    return spec.kind in SNAPSHOT_KINDS and not spec.profile_driven


def direct_sources(config_path):
    return {spec.name: spec for spec in website_sources(config_path) if not carried(spec)}


def snapshot_sources(config_path):
    return {spec.name: spec for spec in website_sources(config_path) if carried(spec)}


def snapshot_problem(snapshot, spec, now):
    """Why readers cannot use this source's snapshot record, or None."""
    if snapshot is None:
        return "snapshot unavailable"
    record = snapshot.get("sources", {}).get(spec.name)
    if not isinstance(record, dict):
        return "not in the snapshot"
    if record.get("status") != "ok":
        return "collection failed: " + str(record.get("error") or "unknown error")[:120]
    if any(record.get(field) != getattr(spec, field, None) for field in IDENTITY_FIELDS):
        return "snapshot record does not match the configuration"
    try:
        fetched = datetime.fromisoformat(record["fetched_at"])
        age = now - fetched
    except (TypeError, ValueError, KeyError):
        return "snapshot record has no valid timestamp"
    if age > SNAPSHOT_MAX_AGE or age < -timedelta(minutes=5):
        return f"snapshot record expired ({age.total_seconds() / 3600:.0f} h old)"
    return None


def snapshot_age(snapshot, spec, now):
    return now - datetime.fromisoformat(snapshot["sources"][spec.name]["fetched_at"])


def judge(report, direct, snapshot_specs=None, snapshot=None, now=None):
    """Return (failures, missing, aging, busy); failures are (name, outcome, status, error)."""
    now = now or datetime.now(timezone.utc)
    rows = {row["name"]: row for row in report.get("results", []) if row.get("section") != "supplementary"}
    failures, missing, aging, busy = [], [], [], []
    for name in sorted(direct):
        row = rows.get(name)
        if row is None:
            missing.append(name)
        elif upstream_trouble(row):
            busy.append((name, row.get("outcome"), row.get("status")))
        elif row.get("outcome") != READABLE:
            failures.append((name, row.get("outcome"), row.get("status"), (row.get("error") or "")[:120]))
    for name, spec in sorted((snapshot_specs or {}).items()):
        problem = snapshot_problem(snapshot, spec, now)
        row = rows.get(name) or {}
        if problem is None:
            if snapshot_age(snapshot, spec, now) > SNAPSHOT_AGING:
                aging.append(name)
        elif row.get("outcome") != READABLE:
            # Without a usable record a busy server is not enough: readers
            # would get the same refusal.
            failures.append((name, f"{problem}; direct read {row.get('outcome') or 'not probed'}",
                             None, ""))
    return failures, missing, aging, busy


def load_snapshot(path):
    try:
        snapshot = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return snapshot if isinstance(snapshot, dict) and isinstance(snapshot.get("sources"), dict) else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--config", type=Path, default=HERE.parent / "config/sources.yaml")
    parser.add_argument("--snapshot", type=Path, help="The deployed public-sources.json; "
                        "without it only directly fetched sources are judged")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text())
    direct = direct_sources(args.config)
    carried_specs = snapshot_sources(args.config) if args.snapshot else {}
    snapshot = load_snapshot(args.snapshot) if args.snapshot else None
    failures, missing, aging, busy = judge(report, direct, carried_specs, snapshot)
    lines = [f"Browser source access from {report.get('origin')} at {report.get('generatedAt')}",
             f"Direct sources checked: {len(direct)}; not probed: {len(missing)}"]
    if args.snapshot:
        lines.append(f"Snapshot sources checked: {len(carried_specs)}"
                     + ("" if snapshot is not None else " (the published snapshot could not be read)"))
    lines.append(f"Unreachable: {len(failures)}")
    lines += [f"- {name}: {outcome} {status or ''} {error}".rstrip() for name, outcome, status, error in failures]
    lines += [f"- {name}: not probed by the audit" for name in missing]
    if busy:
        lines.append("Readable but busy or briefly failing upstream (not counted as unreachable): "
                     + "; ".join(f"{name} ({outcome} {status or ''})".replace(" )", ")")
                                 for name, outcome, status in busy))
    if aging:
        lines.append(f"Snapshot records older than {SNAPSHOT_AGING.total_seconds() / 3600:.0f} hours "
                     "(still served; they expire at 48 hours): " + ", ".join(aging))
    text = "\n".join(lines)
    print(text)
    if args.summary:
        args.summary.write_text(text + "\n")
    return 1 if failures or missing else 0


if __name__ == "__main__":
    sys.exit(main())
