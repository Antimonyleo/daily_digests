"""Reader-feedback baseline; never initialize or migrate user data.

Existing feedback is development evidence, not a newly untouched test set.
Current votes, timestamped outcomes, and weak save signals stay separate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import uuid
import zlib

from .replay import _time, feedback_as_of

_CASE_OUTPUT_ROOT = Path(__file__).resolve().parents[3] / ".full-review/first-shot-next/feedback"
_SCORE_FIELDS = ("topic_score", "final_score", "learned_score", "confidence_score", "quality_score",
                 "negative_penalty", "topic_priority_bonus", "topic_coverage_bonus")
_STAGE_FIELDS = ("after_reviewed", "after_shown_and_known", "after_identity_dedupe",
                 "after_cross_day_dedupe", "after_quality_gate", "after_actionable_gate", "after_topic_gate")


def _counts(ids, labels):
    values = [labels.get(iid, {}).get("value", 0) for iid in ids]
    positive = sum(value == 1 for value in values)
    negative = sum(value == -1 for value in values)
    unknown = len(values) - positive - negative
    return {"occurrences": len(values), "positive": positive, "negative": negative,
            "unknown": unknown, "judged": positive + negative,
            "useful_fraction_bounds": [positive / len(values), (positive + unknown) / len(values)]
            if values else None}


def _read(database: str | Path):
    """Read all evidence in one SQLite snapshot without app imports or writes."""
    try:
        connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise ValueError("Cannot read existing feedback database") from exc
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        queries = {
            "votes": "SELECT item_id,value,grade FROM votes",
            "bookmarks": "SELECT item_id FROM bookmarks",
            "feedback_events": "SELECT id,item_id,value,grade,recorded_at FROM feedback_events ORDER BY id",
            "digest_items": "SELECT digest_id,item_id,item_label FROM digest_items",
            "ranking_snapshots": "SELECT run_id,created_at,payload_zlib FROM ranking_snapshots",
        }
        result = {table: [dict(row) for row in connection.execute(query)]
                  for table, query in queries.items() if table in tables}
        if not result:
            raise ValueError("No feedback or retained decision tables; database left unchanged")
        return result
    except sqlite3.Error as exc:
        raise ValueError("Cannot read feedback schema; database left unchanged") from exc
    finally:
        connection.close()


def _validate_labels(rows):
    for row in rows:
        if row["value"] not in (-1, 0, 1):
            raise ValueError("Invalid signed feedback")


def _current_evidence(raw):
    votes = raw.get("votes", [])
    _validate_labels(votes)
    labels = {row["item_id"]: row for row in votes}
    if len(labels) != len(votes):
        raise ValueError("Duplicate current votes; cannot determine latest without migration")
    saved = {row["item_id"] for row in raw.get("bookmarks", [])}
    days = defaultdict(list)
    for row in raw.get("digest_items", []):
        match = re.fullmatch(r"R([1-9]\d*)", row["item_label"] or "")
        if match:
            days[row["digest_id"]].append((int(match.group(1)), row["item_id"]))
    orders = [[iid for _, iid in sorted(rows)] for rows in days.values()]
    exposures = [iid for order in orders for iid in order]
    top = [iid for order in orders for iid in order[:5]]
    return {
        "basis": "Current state at read time, not labels as of the historical outcome cutoff; all retained research displays",
        "votes": _counts(labels, labels),
        "weak_saves": {"items": len(saved), "with_explicit_positive": sum(labels.get(iid, {}).get("value") == 1 for iid in saved),
                       "with_explicit_negative": sum(labels.get(iid, {}).get("value") == -1 for iid in saved),
                       "without_signed_rating": sum(labels.get(iid, {}).get("value", 0) == 0 for iid in saved),
                       "used_as_positive_labels": False},
        "retained_research": {
            "days": len(days), "displayed_occurrences": len(exposures), "unique_items": len(set(exposures)),
            "repeat_occurrences": len(exposures) - len(set(exposures)),
            "top_five": _counts(top, labels),
            "days_with_fewer_than_five": sum(len(order) < 5 for order in orders),
            "days_with_five_fully_judged": sum(len(order) >= 5 and all(labels.get(iid, {}).get("value") in (-1, 1)
                                                                       for iid in order[:5]) for order in orders),
        },
    }


def build_feedback_baseline(database: str | Path, *, outcome_as_of: str) -> dict:
    """Report population/label gaps without exporting private identities or text.

Historical outcomes use only immutable events strictly before the cutoff. They
can describe later preferences for an earlier slate, not necessarily preferences
at serving time. No personalized historical run proves first-use performance.
"""
    return _baseline_from_raw(_read(database), outcome_as_of=outcome_as_of)


def _retained_snapshots(raw, cutoff):
    latest = {}
    retained_before = 0
    fingerprints = []
    for row in raw.get("ranking_snapshots", []):
        if _time(row["created_at"]) >= cutoff:
            continue
        try:
            snapshot = json.loads(zlib.decompress(row["payload_zlib"]))
        except (ValueError, zlib.error, UnicodeError) as exc:
            raise ValueError("Unreadable retained snapshot; cannot certify feedback coverage") from exc
        if snapshot.get("schema_version") != 1:
            raise ValueError("Unsupported retained snapshot schema")
        if snapshot["run_id"] != row["run_id"]:
            raise ValueError("Stored snapshot identity differs from frozen evidence")
        when = _time(snapshot["decision_at"])
        if when >= cutoff:
            continue
        retained_before += 1
        fingerprints.append(hashlib.sha256(row["payload_zlib"]).hexdigest())
        day = snapshot["decision_local_date"]
        old = latest.get(day)
        if old is None or (when, snapshot["run_id"]) > (_time(old["decision_at"]), old["run_id"]):
            latest[day] = snapshot
    snapshots = sorted(latest.values(), key=lambda snapshot: (_time(snapshot["decision_at"]), snapshot["run_id"]))
    return snapshots, retained_before, fingerprints


def _baseline_from_raw(raw, *, outcome_as_of):
    cutoff = _time(outcome_as_of)
    events = raw.get("feedback_events", [])
    _validate_labels(events)
    labels = feedback_as_of(events, cutoff)
    past_events = [event for event in events if _time(event["recorded_at"]) < cutoff]
    snapshots, retained_before, fingerprints = _retained_snapshots(raw, cutoff)
    pool, eligible, selected, top, omitted, ineligible = [], [], [], [], [], []
    five_fully_judged = first_three_fully_judged = short_days = prefeedback_runs = 0
    signatures, versions, missing_signatures = [], set(), 0
    for snapshot in snapshots:
        candidates = [candidate for candidate in snapshot["candidates"] if candidate["section"] == "research"]
        ids = [candidate["item_id"] for candidate in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate candidate identity in retained research pool")
        chosen = sorted((candidate for candidate in candidates if candidate["selected"]), key=lambda candidate: candidate["position"])
        if any(not candidate["eligible"] for candidate in chosen):
            raise ValueError("Retained selected research candidate was ineligible")
        if [candidate["position"] for candidate in chosen] != list(range(len(chosen))):
            raise ValueError("Retained research display positions must be contiguous and zero-based")
        order = [candidate["item_id"] for candidate in chosen]
        pool.extend(ids)
        eligible.extend(candidate["item_id"] for candidate in candidates if candidate["eligible"])
        ineligible.extend(candidate["item_id"] for candidate in candidates if not candidate["eligible"])
        omitted.extend(candidate["item_id"] for candidate in candidates if candidate["eligible"] and not candidate["selected"])
        selected.extend(order)
        top.extend(order[:5])
        short_days += len(order) < 5
        five_fully_judged += len(order) >= 5 and all(iid in labels for iid in order[:5])
        first_three_fully_judged += len(order) >= 3 and all(iid in labels for iid in order[:3])
        prefeedback_runs += bool(feedback_as_of(events, snapshot["decision_at"]))
        signature = snapshot.get("profile_signature")
        if signature:
            signatures.append(signature)
        else:
            missing_signatures += 1
        if snapshot.get("scorer_version"):
            versions.add(snapshot["scorer_version"])
    # Hashes bind evidence without exporting per-reader, per-paper or per-run IDs.
    event_fingerprint = hashlib.sha256(json.dumps(past_events, sort_keys=True, default=str).encode()).hexdigest()
    pool_set = set(pool)
    return {
        "schema_version": 1, "outcome_as_of": cutoff.isoformat(),
        "access": "SQLite mode=ro, query_only=ON, one consistent read transaction; no initialization or migration",
        "privacy": "Aggregate only; no reader, topic, paper or run identities exported",
        "available_tables": sorted(raw),
        "current_state_development_only": _current_evidence(raw),
        "timestamped_development_only": {
            "basis": "Latest retained run per observed day; signed events strictly before outcome cutoff; no current-vote backfill",
            "retained_runs_before_cutoff": retained_before,
            "observed_days": len(snapshots), "timestamped_events_before_cutoff": len(past_events),
            "signed_items_at_cutoff": len(labels),
            "pool": _counts(pool, labels), "eligible_pool": _counts(eligible, labels),
            "selected": _counts(selected, labels), "top_five": _counts(top, labels),
            "eligible_omitted": _counts(omitted, labels), "ineligible": _counts(ineligible, labels),
            "unique_pool_items": len(pool_set),
            "unique_items": {"pool": _counts(pool_set, labels), "eligible": _counts(set(eligible), labels),
                             "selected": _counts(set(selected), labels), "eligible_omitted": _counts(set(omitted), labels),
                             "ineligible": _counts(set(ineligible), labels),
                             "eligible_omitted_never_selected_in_supplied_runs": _counts(set(omitted) - set(selected), labels)},
            "signed_items_outside_captured_research_pools": sum(iid not in pool_set for iid in labels),
            "days_with_fewer_than_five": short_days,
            "days_with_five_fully_judged": five_fully_judged,
            "days_with_first_three_fully_judged": first_three_fully_judged,
            "runs_with_observed_predecision_signed_feedback": prefeedback_runs,
            "runs_without_observed_predecision_signed_feedback": len(snapshots) - prefeedback_runs,
            "distinct_recorded_profile_signatures": len(set(signatures)),
            "runs_missing_profile_signature": missing_signatures,
            "distinct_recorded_scorer_versions": len(versions),
            "complete_captured_pool_runs": sum(bool(snap.get("complete_candidate_pool")) for snap in snapshots),
            "snapshot_evidence_sha256": hashlib.sha256(json.dumps(sorted(fingerprints)).encode()).hexdigest(),
            "event_evidence_sha256": event_fingerprint,
        },
        "limitations": [
            "Previously examined historical evidence is development data, never a fresh holdout.",
            "Current votes are undated latest preferences; current-state counts ignore the historical cutoff by design.",
            "Saving is a weak interest signal and never substitutes for an explicit relevance judgment.",
            "Repeated paper exposures and one reader's days are not independent readers or fields.",
            "Unique eligibility categories can overlap across days; an omitted or ineligible paper may have been selected on another day.",
            "Exposure-selected feedback is biased; unknown outcomes stay unknown, not negative.",
            "Profile signatures expose possible drift but do not assign an outcome to the reader goal active when it was rated.",
            "No observed prior signed event does not establish first use: legacy feedback or unseen history may exist.",
            "Captured pools cannot measure upstream retrieval recall; outside-pool ratings are review leads only.",
            "Recorded slate analysis does not establish replay parity, scientific validity, or current-model performance.",
        ],
        "first_shot_quality_certified": False,
    }


def build_feedback_cases(database: str | Path, *, outcome_as_of: str) -> dict:
    """Export pseudonymous numeric development cases, never a fabricated holdout.

    Random mappings exist only for this call: identities are consistent inside
    one export but cannot be joined to another export without private source data.
    """
    raw = _read(database)
    _baseline_from_raw(raw, outcome_as_of=outcome_as_of)  # Validate the retained evidence first.
    return _cases_from_raw(raw, outcome_as_of=outcome_as_of)


def _cases_from_raw(raw, *, outcome_as_of):
    cutoff = _time(outcome_as_of)
    events = raw.get("feedback_events", [])
    labels = feedback_as_of(events, cutoff)
    snapshots, _, _ = _retained_snapshots(raw, cutoff)
    tokens = {}

    def token(kind, identity):
        key = kind, identity
        if key not in tokens:
            tokens[key] = kind + "_" + uuid.uuid4().hex
        return tokens[key]

    def number(value):
        return value if type(value) in (int, float) and math.isfinite(value) else None

    cases = []
    for ordinal, snapshot in enumerate(snapshots, 1):
        prior = feedback_as_of(events, snapshot["decision_at"])
        stages = {name: set(snapshot.get("stage_results", {}).get(name, []))
                  for name in _STAGE_FIELDS if name in snapshot.get("stage_results", {})}
        signature = snapshot.get("profile_signature")
        for candidate in snapshot["candidates"]:
            if candidate["section"] != "research":
                continue
            iid = candidate["item_id"]
            label = labels.get(iid)
            family = candidate.get("family_id")
            # Keep only fixed schema fields; facet names, article metadata and
            # arbitrary feature/stage keys may contain private reader text.
            features = candidate.get("features", {})
            first_rejected = candidate.get("first_rejected_stage")
            if first_rejected not in (*_STAGE_FIELDS, "post_score_eligibility", None):
                first_rejected = "other_recorded_stage"
            cases.append({
                "split": "development", "paper_id": token("paper", iid),
                "goal_id": token("goal", signature) if signature else None,
                "goal_basis": "recorded_profile_signature" if signature else "unknown",
                "decision_id": token("decision", snapshot["run_id"]), "decision_ordinal": ordinal,
                "family_id": token("family", family) if family else None,
                "family_basis": "recorded_family_only" if family else "unavailable",
                "eligible": bool(candidate["eligible"]), "selected": bool(candidate["selected"]),
                "selected_position_zero_based": number(candidate.get("position")),
                "input_position_zero_based": number(candidate.get("input_position")),
                "scores": {"ranking_score": number(candidate.get("ranking_score")),
                           "snapshot_quality_score": number(candidate.get("quality_score")),
                           **{name: number(features.get(name)) for name in _SCORE_FIELDS}},
                "stages": {name: iid in passed for name, passed in stages.items()},
                "first_rejected_stage": first_rejected,
                "outcome_signed": label["value"] if label else None,
                "outcome_grade": number(label.get("grade")) if label else None,
                "outcome_basis": "timestamped_signed_event" if label else "unknown_or_cleared",
                "outcome_goal_alignment": "unknown",
                "item_had_predecision_signed_feedback": iid in prior,
                "run_had_observed_predecision_signed_feedback": bool(prior),
            })
    return {
        "schema_version": 1, "split": "development", "outcome_as_of": cutoff.isoformat(),
        "purpose": "Numeric re-analysis of recorded decisions; no training or untouched-holdout certification",
        "privacy": "Random within-export pseudonyms; mapping stays in memory and is not written; no article/profile/run identities or text",
        "case_count": len(cases), "cases": cases,
        "limitations": [
            "Current votes and saves never backfill missing timestamped outcomes; neutral clears are unknown.",
            "Cases without an explicit outcome are unknown, not negative training examples.",
            "Profile signature is only a recorded goal proxy; the goal at rating time is unknown, even when signatures match.",
            "Repeated papers and decision days are not independent readers; prior-feedback flags do not prove first use when false.",
            "Family grouping is only what snapshots recorded; absent family IDs mean alternate-version grouping unavailable.",
            "Recorded family IDs may change across snapshots; no cross-run version reconciliation is inferred.",
            "Pseudonyms are stable only within this export; historical data remain development evidence.",
            "Scores may contain effects of prior feedback; this is not a no-feedback reranking or upstream retrieval evaluation.",
        ],
        "release_approved": False,
    }


def forward_holdout_protocol(*, frozen_at: str, evidence_cutoff: str) -> dict:
    """Freeze an honest future evaluation contract, not imaginary human results."""
    frozen, cutoff = _time(frozen_at), _time(evidence_cutoff)
    if frozen < cutoff:
        raise ValueError("Holdout freeze must not precede inspected evidence cutoff")
    return {
        "schema_version": 1, "frozen_at": frozen.isoformat(), "evidence_cutoff": cutoff.isoformat(),
        "status": "protocol_only_not_enrolled_or_validated", "completed_human_evaluations": 0,
        "development": "All currently inspected snapshots, ratings, fixtures and their alternate paper versions; tuning allowed, no fresh-holdout claims",
        "forward_holdout": {
            "enrollment": "Only new decisions and independently supplied in-window paper targets after this freeze and after candidate/version choices are locked",
            "separation": "Group alternate paper versions, repeated exposures and reader-goal episodes; no same family in tuning and holdout",
            "goal_drift": "Freeze the reader goal with each decision; start a new episode after an explicit goal change and report episodes separately",
            "retrieval": "Specify independent relevant paper targets before comparing retrieved pools; record publication window, availability and identity matches",
            "judgments": "Blind readers to method and old labels; rate top-five union plus randomized omitted candidates; useful, not useful, or unsure, with optional reason",
            "weak_signals": "Saves remain separate diagnostics; unjudged and unsure are unknown; clears remove a signed outcome",
            "learning": "Do not tune on heldout outcomes; any inspected heldout set becomes development, then freeze a new future evaluation",
        },
        "reporting": [
            "Per reader-goal episode and field: served count, judged coverage, positive/negative/unknown, first-three errors and first-five utility bounds",
            "Matched pools: lost and gained known positives, retrieval targets missed, and eligible omitted known positives",
            "Separate no-feedback first sessions from personalized later sessions; do not pool their accuracy claims",
            "Separate Python cold start, source retrieval, ranking and summarization CPU timings; compare identical data and device configuration",
            "Cluster uncertainty by reader and goal episode; repeated days or paper versions are not independent samples",
        ],
        "promotion_gate": {
            "proposed_first_five": "At least four useful among five, all five judged; first three fully judged with no off-task paper",
            "per_field": "Report every field and reader separately; gains pooled across fields cannot conceal lost positives or added strong rejections",
            "quality": "Preserve quality/retraction/source-health checks; insufficient labels or missing retrieval targets cannot pass",
            "latency": "Require measured comparable CPU timings without reducing supplied content quality",
            "release_approved": False,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--outcome-as-of", required=True, help="Exclusive timestamp cutoff for immutable events and decisions")
    parser.add_argument("--output", type=Path, required=True, help="New aggregate JSON file; existing files are never overwritten")
    parser.add_argument("--cases-output", type=Path, help="Optional new pseudonymous development JSON under .full-review/first-shot-next/feedback/")
    args = parser.parse_args(argv)
    if args.cases_output is not None:
        cases_path = args.cases_output.resolve()
        if not cases_path.is_relative_to(_CASE_OUTPUT_ROOT.resolve()):
            raise ValueError("Development cases must stay under .full-review/first-shot-next/feedback/")
        if cases_path == args.output.resolve():
            raise ValueError("Aggregate and development cases require different output files")
        if cases_path.exists():
            raise FileExistsError("Development cases output already exists")
    raw = _read(args.database)
    report = _baseline_from_raw(raw, outcome_as_of=args.outcome_as_of)
    report["forward_holdout_protocol"] = forward_holdout_protocol(
        frozen_at=datetime.now(timezone.utc).isoformat(), evidence_cutoff=args.outcome_as_of)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if args.cases_output is not None:
        cases = _cases_from_raw(raw, outcome_as_of=args.outcome_as_of)
        args.cases_output.parent.mkdir(parents=True, exist_ok=True)
        with args.cases_output.open("x") as stream:
            stream.write(json.dumps(cases, indent=2, allow_nan=False) + "\n")
    print("Aggregate feedback baseline written; historical evidence remains development-only.")


if __name__ == "__main__":
    main()
