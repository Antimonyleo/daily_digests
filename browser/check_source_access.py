"""Judge a browser source audit: can the live site still read its direct sources?

Sources the daily public snapshot carries do not depend on browser (CORS)
access, so only directly fetched sources fail the check. Run from the repo
root after ``node browser/source-audit.mjs <origin> <report.json>``:

    uv run python browser/check_source_access.py <report.json> [--summary out.md]

Exit status 1 names the unreadable direct sources; the report holds evidence.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from collect_sources import SNAPSHOT_KINDS  # noqa: E402

# Kinds with no network fetch at all.
OFFLINE_KINDS = {"curated_calls", "curated_events"}


def direct_sources(config_path):
    from dailydigest.config import load_sources

    return {spec.name: spec for spec in load_sources(str(config_path))
            if spec.kind not in OFFLINE_KINDS and not spec.local_only
            and (spec.kind not in SNAPSHOT_KINDS or spec.profile_driven)}


def judge(report, direct):
    rows = {row["name"]: row for row in report.get("results", []) if row.get("section") != "supplementary"}
    failures, missing = [], []
    for name in sorted(direct):
        row = rows.get(name)
        if row is None:
            missing.append(name)
        elif row.get("outcome") != "readable_http_success":
            failures.append((name, row.get("outcome"), row.get("status"), (row.get("error") or "")[:120]))
    return failures, missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--config", type=Path, default=HERE.parent / "config/sources.yaml")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text())
    direct = direct_sources(args.config)
    failures, missing = judge(report, direct)
    lines = [f"Browser source access from {report.get('origin')} at {report.get('generatedAt')}",
             f"Direct sources checked: {len(direct)}; unreadable: {len(failures)}; not probed: {len(missing)}"]
    lines += [f"- {name}: {outcome} {status or ''} {error}".rstrip() for name, outcome, status, error in failures]
    lines += [f"- {name}: not probed by the audit" for name in missing]
    text = "\n".join(lines)
    print(text)
    if args.summary:
        args.summary.write_text(text + "\n")
    return 1 if failures or missing else 0


if __name__ == "__main__":
    sys.exit(main())
