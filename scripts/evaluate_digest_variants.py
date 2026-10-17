#!/usr/bin/env python
"""Evaluate frozen daily digests and prepare blinded, explicit judgment packets."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys


VERSION = "1"
FLAGS = "--snapshot FILE (repeat) or --database FILE, --days N (database only, default 7), --judgments JSONL, --judgment-manifest JSON, --challengers JSON, --outcome-as-of ISO, --recent-days N, --seed N, --output NEW.json, --packet NEW.jsonl, --rejected-per-run N, --inspected-families JSON, --help, --version"


class Parser(argparse.ArgumentParser):
    def error(self, message):
        print("error: " + json.dumps(message))
        print("help: " + json.dumps("Valid flags: " + FLAGS))
        raise SystemExit(2)


def _read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _new_paths(paths, database=None):
    from dailydigest.config import get_settings
    settings = get_settings()
    root = Path(__file__).resolve().parents[1]
    protected = {root / "data", Path(settings.db_path).resolve().parent,
                 Path(settings.profile_path).resolve().parent}
    if database is not None:
        protected.add(database.resolve().parent)
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Output report, packet and manifest need distinct paths")
    for path in resolved:
        if path.exists() or path.name == ".env" or any(path.is_relative_to(base) for base in protected):
            raise ValueError("Outputs must be new files outside production data/profile directories and .env")
    return resolved


def main(argv=None):
    args_in = list(sys.argv[1:] if argv is None else argv)
    if len(args_in) == 1 and args_in[0] in {"-v", "-V", "--version"}:
        print(VERSION)
        return 0
    if not args_in:
        print("bin: " + json.dumps(str(Path(__file__).resolve())))
        print('description: "Compare explicit frozen snapshots; create blinded judgment packets without running a brew."')
        print("snapshots: 0")
        print('help: "Use --snapshot FILE --outcome-as-of ISO --output NEW.json; add --judgments JSONL for reader outcomes."')
        return 0
    parser = Parser(description=__doc__, allow_abbrev=False,
                    epilog="Examples: --snapshot day1.json --snapshot day2.json --judgments judgments.jsonl --outcome-as-of 2026-10-03T00:00:00Z --output report.json; --snapshot day1.json --outcome-as-of 2026-10-03T00:00:00Z --output report.json --packet blind.jsonl")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", type=Path, action="append", help="Explicit frozen snapshot files (repeatable)")
    source.add_argument("--database", type=Path, help="Existing SQLite database; read-only, no brew or migration")
    parser.add_argument("--days", type=int, help="Latest observed local days from --database (default 7)")
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--judgment-manifest", type=Path)
    parser.add_argument("--challengers", type=Path)
    parser.add_argument("--outcome-as-of", required=True)
    parser.add_argument("--recent-days", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--rejected-per-run", type=int, default=5)
    parser.add_argument("--inspected-families", type=Path)
    parser.add_argument("-v", "-V", "--version", action="version", version=VERSION)
    args = parser.parse_args(args_in)
    if args.days is not None and not args.database:
        parser.error("--days requires --database")
    if args.judgment_manifest and not args.judgments:
        parser.error("--judgment-manifest requires --judgments")
    if args.database and args.judgment_manifest and args.days is not None:
        parser.error("--days cannot resample runs pinned by --judgment-manifest; omit --days")
    from dailydigest.rank.digest_evaluation import (
        build_judgment_packet, evaluate_digest_variants, expand_packet_judgments,
    )
    try:
        packet_manifest = args.packet.with_suffix(args.packet.suffix + ".manifest.json") if args.packet else None
        outputs = _new_paths([args.output] + ([args.packet, packet_manifest] if args.packet else []), args.database)
        manifest = _read(args.judgment_manifest) if args.judgment_manifest else None
        saved = None
        if args.database:
            from dailydigest.rank.saved_evaluation import load_saved_evaluation
            saved = load_saved_evaluation(args.database, outcome_as_of=args.outcome_as_of,
                                          days=args.days if args.days is not None else 7,
                                          run_ids=set(manifest["snapshot_signatures"]) if manifest is not None else None)
            snapshots = saved["snapshots"]
            if not snapshots:
                raise ValueError("No saved decisions before the cutoff; no report can be inferred")
        else:
            snapshots = [_read(path) for path in args.snapshot]
        judgments = _jsonl(args.judgments) if args.judgments else []
        if args.judgment_manifest:
            judgments = expand_packet_judgments(judgments, manifest, snapshots)
        if saved:
            judgments = saved["judgments"] + judgments
        challengers = _read(args.challengers) if args.challengers else None
        inspected = _read(args.inspected_families) if args.inspected_families else []
        if not isinstance(inspected, list) or any(not isinstance(value, str) for value in inspected):
            raise ValueError("Inspected families must be an explicit JSON list of family IDs")
        report = evaluate_digest_variants(snapshots, judgments, challengers=challengers,
                                          outcome_as_of=args.outcome_as_of, recent_days=args.recent_days, seed=args.seed)
        if saved:
            report["saved_input_coverage"] = saved["coverage"]
        packet = build_judgment_packet(snapshots, challengers=challengers,
                                        rejected_per_run=args.rejected_per_run, seed=args.seed,
                                        inspected_family_ids=inspected) if args.packet else None
        artifacts = [(outputs[0], json.dumps(report, indent=2, allow_nan=False) + "\n")]
        if packet:
            artifacts.extend([(outputs[1], "".join(json.dumps(row, allow_nan=False) + "\n" for row in packet["examples"])),
                              (outputs[2], json.dumps(packet["manifest"], indent=2, allow_nan=False) + "\n")])
        for path, content in artifacts:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as handle:
                handle.write(content)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("error: " + json.dumps(str(exc)))
        print('help: "Check explicit snapshots, judgment fields, timestamps and new output paths; use --help for the input flags."')
        return 1
    print('status: "offline_diagnostic"')
    print("runs: " + str(report["runs"]))
    print("days: " + str(report["days"]))
    print("judgments_used: " + str(report["judgments_used"]))
    print("report: " + json.dumps(str(outputs[0])))
    print("promotion_eligible: false")
    research = report["variants"]["baseline"]["aggregate"].get("research", {})
    for field in ("served", "judged", "useful_count", "unknown"):
        print("research_" + field + ": " + str(research.get(field, 0)))
    if packet:
        print("packet_count: " + str(len(packet["examples"])))
        print("packet: " + json.dumps(str(outputs[1])))
        print("private_manifest: " + json.dumps(str(outputs[2])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
