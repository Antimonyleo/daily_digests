"""Read-only daily evaluation inputs from retained decisions and feedback.

Never initialize/migrate the database or reconstruct decisions from current
items/votes. Reader outcomes describe preferences, not scientific validation.
"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import zlib

from .replay import _time


def load_saved_evaluation(database: str | Path, *, outcome_as_of, days: int = 7,
                          run_ids: set[str] | None = None) -> dict:
    """Select the latest decision per observed local day, strictly before cutoff.

    Latest events win, including clears. Missing event history stays unknown;
    legacy signed votes do not imply an importance grade. Feedback on IDs outside
    the supplied pools is a lead for review, not an upstream-recall measurement.
    """
    if type(days) is not int or days <= 0:
        raise ValueError("days must be a positive integer")
    if run_ids is not None and (not run_ids or any(not isinstance(run, str) or not run for run in run_ids)):
        raise ValueError("Explicit run IDs must be nonempty strings")
    cutoff = _time(outcome_as_of)
    path = Path(database).resolve()
    try:
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise ValueError("Cannot read the existing evaluation database") from exc
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.execute("BEGIN")  # Snapshots and events share one read transaction.
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "ranking_snapshots" not in tables:
            raise ValueError("No retained ranking snapshot table; do not infer old decisions")
        by_day, pinned = {}, {}
        for run_id, created_at, payload in connection.execute("SELECT run_id, created_at, payload_zlib FROM ranking_snapshots"):
            if run_ids is not None and run_id not in run_ids:
                continue
            if _time(created_at) >= cutoff:
                continue
            snapshot = json.loads(zlib.decompress(payload))
            when = _time(snapshot["decision_at"])
            if when >= cutoff:
                continue
            if snapshot["run_id"] != run_id:
                raise ValueError("Stored run ID differs from its frozen snapshot")
            if run_ids is not None:
                pinned[run_id] = snapshot
                continue
            day = snapshot["decision_local_date"]
            previous = by_day.get(day)
            if previous is None or (when, snapshot["run_id"]) > (_time(previous["decision_at"]), previous["run_id"]):
                by_day[day] = snapshot
        if run_ids is not None and set(pinned) != run_ids:
            raise ValueError("Pinned snapshots are missing or fall after the cutoff")
        snapshots = sorted((pinned if run_ids is not None else by_day).values(),
                           key=lambda snap: (snap["decision_local_date"], snap["decision_at"], snap["run_id"]))
        if run_ids is None:
            snapshots = snapshots[-days:]
        latest = {}
        if "feedback_events" in tables:
            for event_id, item_id, value, grade, recorded_at in connection.execute(
                "SELECT id, item_id, value, grade, recorded_at FROM feedback_events ORDER BY id"
            ):
                when = _time(recorded_at)
                if when >= cutoff:
                    continue
                old = latest.get(item_id)
                if old is None or (when, event_id) > (old["when"], old["id"]):
                    latest[item_id] = {"id": event_id, "when": when, "value": value, "grade": grade}
    except (sqlite3.Error, zlib.error) as exc:
        raise ValueError("Cannot read retained snapshots and feedback; check the database") from exc
    finally:
        connection.close()
    judgments, important_ineligible = [], []
    pool_ids = set()
    for snapshot in snapshots:
        for candidate in snapshot["candidates"]:
            iid = candidate["item_id"]
            pool_ids.add(iid)
            event = latest.get(iid)
            if event is None:
                continue
            value, grade = event["value"], event["grade"]
            if value not in {-1, 0, 1}:
                raise ValueError("Invalid signed feedback event")
            if value == 1 and grade == 100 and not candidate["eligible"]:
                important_ineligible.append({"run_id": snapshot["run_id"], "item_id": iid,
                                             "first_rejected_stage": candidate.get("first_rejected_stage")})
            judgments.append({"run_id": snapshot["run_id"], "item_id": iid, "rater": "reader",
                              "recorded_at": event["when"].isoformat(),
                              "useful": None if value == 0 else value == 1,
                              "important": grade == 100 if value != 0 and grade is not None else None,
                              "grade": grade if value != 0 else None})
    outside = sorted(iid for iid, event in latest.items()
                     if event["value"] == 1 and event["grade"] == 100 and iid not in pool_ids)
    return {"snapshots": snapshots, "judgments": judgments, "coverage": {
        "database_mode": "read_only", "outcome_as_of": cutoff.isoformat(),
        "day_selection": "Exact frozen runs pinned by judgment manifest" if run_ids is not None else "Latest retained decision per observed local day before cutoff",
        "requested_observed_days": None if run_ids is not None else days,
        "observed_days": len({snap["decision_local_date"] for snap in snapshots}),
        "feedback_history": "available" if "feedback_events" in tables else "missing",
        "signed_items_in_supplied_pools": sum(latest[iid]["value"] in {-1, 1} for iid in pool_ids if iid in latest),
        "important_outside_supplied_pools_item_ids": outside,
        "important_ineligible_candidates": important_ineligible,
        "limits": ["Unknown ratings remain unknown; no backfill from current votes.",
                   "Importance requires explicit grade 100; journal reputation is not an outcome.",
                   "Outside-pool item IDs can be older or alternate versions; they are review leads, not verified retrieval misses.",
                   "Important ineligible records need stage review: a cooldown or another selected publication version may explain them.",
                   "Uncollected papers and upstream recall cannot be measured from saved pools."]}}
