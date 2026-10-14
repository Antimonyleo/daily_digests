"""Compare frozen daily slates with explicit, separately attributed judgments.

This offline module never infers usefulness from scope, venue, or missing votes.
All comparative utility shares the original captured pool. A counterfactual
snapshot may verify changed selection scores but cannot rewrite article evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json

from .replay import evaluate_replay, feedback_as_of, paired_run_metrics, recorded_slate


VERSION = "digest-evaluation-v1"
SCOPES = {"core", "context", "off_interest", "uncertain"}


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _time(value) -> datetime:
    if not isinstance(value, (str, datetime)):
        raise ValueError("A valid explicit timestamp is required")
    result = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if result.tzinfo is None:
        raise ValueError("Timestamps must include a timezone")
    return result.astimezone(timezone.utc)


def _inventory(snapshots):
    runs = {}
    for snapshot in snapshots:
        run = snapshot.get("run_id")
        if not isinstance(run, str) or not run or run in runs:
            raise ValueError("Snapshots require unique nonempty run IDs")
        if snapshot.get("schema_version") != 1:
            raise ValueError("Only immutable snapshot schema 1 is supported")
        _time(snapshot["decision_at"])
        candidates = snapshot["candidates"]
        if (any(type(c.get("item_id")) is not int for c in candidates)
                or len({c["item_id"] for c in candidates}) != len(candidates)):
            raise ValueError("Snapshot candidate IDs must be unique integers")
        _hash(snapshot)  # Reject nonfinite evidence before any comparison.
        runs[run] = snapshot
    if not runs:
        raise ValueError("At least one explicit frozen snapshot is required")
    return runs


def _families(runs):
    """Join frozen families across runs using item IDs and strong identity keys."""
    parents, aliases, keys_by_ref = {}, {}, {}
    def root(ref):
        while parents[ref] != ref:
            parents[ref] = parents[parents[ref]]
            ref = parents[ref]
        return ref
    for run, snapshot in runs.items():
        for candidate in snapshot["candidates"]:
            ref = (run, candidate["item_id"])
            parents[ref] = ref
            keys = [f"item:{candidate['item_id']}"]
            if candidate.get("family_id"):
                keys.append("family:" + str(candidate["family_id"]))
            keys.extend(key for key in candidate.get("identity_keys", [])
                        if isinstance(key, str) and key.startswith(("doi:", "pmid:", "arxiv:", "url:")))
            keys_by_ref[ref] = keys
            for key in keys:
                if key in aliases:
                    a, b = root(ref), root(aliases[key])
                    parents[max(a, b)] = min(a, b)
                aliases[key] = ref
    groups = defaultdict(set)
    for ref, keys in keys_by_ref.items():
        groups[root(ref)].update(keys)
    return {ref: _hash(sorted(groups[root(ref)])) for ref in parents}


def _slates(runs, challengers):
    variants = {"baseline": {run: {"slate": recorded_slate(snapshot), "snapshot": snapshot}
                             for run, snapshot in runs.items()}}
    for name, declarations in (challengers or {}).items():
        if not isinstance(name, str) or not name or name == "baseline":
            raise ValueError("Challenger names must be nonempty and cannot be baseline")
        if set(declarations) != set(runs):
            raise ValueError("Every challenger must declare exactly the same snapshot runs")
        variants[name] = {}
        for run, value in declarations.items():
            if not isinstance(value, dict):
                raise ValueError("Each challenger needs a section slate or slate/snapshot declaration")
            declaration = value if "slate" in value else {"slate": value}
            original = runs[run]
            counterfactual = declaration.get("snapshot", original)
            if counterfactual is not original:
                immutable = set(original) - {"candidates"}
                if any(counterfactual.get(key) != original.get(key) for key in immutable):
                    raise ValueError("Counterfactual cannot change captured run inputs, source policy or budgets")
                _hash(counterfactual)
                old = {c["item_id"]: c for c in original["candidates"]}
                new = {c["item_id"]: c for c in counterfactual["candidates"]}
                if set(old) != set(new) or len(new) != len(counterfactual["candidates"]):
                    raise ValueError("Counterfactual must preserve all captured candidate IDs")
                for iid, candidate in new.items():
                    if any(candidate.get(key) != old[iid].get(key) for key in
                           ("metadata", "family_id", "section", "identity_keys", "text_signature", "selection_facts")):
                        raise ValueError("Counterfactual cannot rewrite frozen article or source evidence")
                    if candidate["eligible"] and not old[iid]["eligible"]:
                        raise ValueError("Counterfactual cannot re-enable an originally ineligible candidate")
            slate = declaration["slate"]
            if not isinstance(slate, dict) or any(
                not isinstance(ids, list) or any(type(iid) is not int for iid in ids)
                for ids in slate.values()
            ):
                raise ValueError("Slates must map sections to ordered integer ID lists")
            variants[name][run] = {"slate": slate, "snapshot": counterfactual}
    return variants


def _judgments(rows, runs, cutoff):
    indexes = {run: {c["item_id"]: c for c in snapshot["candidates"]} for run, snapshot in runs.items()}
    events, annotations = defaultdict(list), {}
    seen = set()
    accepted = []
    for row in rows:
        run, iid, rater = row.get("run_id"), row.get("item_id"), row.get("rater")
        if run not in runs or type(iid) is not int or iid not in indexes[run]:
            raise ValueError("Judgment must reference an item in an explicit snapshot run")
        if rater not in {"reader", "analyst"}:
            raise ValueError("Judgment rater must explicitly be reader or analyst")
        recorded = _time(row.get("recorded_at"))
        key = (run, iid, rater, recorded)
        if key in seen:
            raise ValueError("Duplicate judgment timestamp for the same item and rater")
        seen.add(key)
        useful, important, grade, scope = (row.get(k) for k in ("useful", "important", "grade", "scope"))
        if any(value is not None and type(value) is not bool for value in (useful, important)):
            raise ValueError("Useful and important judgments must be explicit booleans or null")
        if grade is not None and (type(grade) not in (int, float) or not 0 <= grade <= 100):
            raise ValueError("Reader grade must be finite and between 0 and 100")
        if rater != "reader" and any(value is not None for value in (useful, important, grade)):
            raise ValueError("Analyst scope labels cannot supply reader usefulness, importance or grades")
        if grade is not None and (useful is None or (grade > 50) != useful or grade == 50):
            raise ValueError("Signed reader grades must agree with explicit usefulness")
        if important is True and (useful is not True or (grade is not None and grade != 100)):
            raise ValueError("Important papers require useful=true and grade 100 if supplied")
        if important is False and grade == 100:
            raise ValueError("Grade 100 contradicts important=false")
        if scope is not None and scope not in SCOPES:
            raise ValueError("Scope must be core, context, off_interest, uncertain or null")
        facets = row.get("facets", [])
        if (not isinstance(facets, list) or any(not isinstance(f, str) or not f.strip() for f in facets)
                or len(set(facets)) != len(facets) or (scope != "core" and facets)):
            raise ValueError("Only core scope annotations can assert nonempty distinct facets")
        for field in ("contribution", "rejection_reason"):
            if field in row and not isinstance(row[field], str):
                raise ValueError(f"{field} must be annotation text")
        evidence = row.get("evidence", [])
        if not isinstance(evidence, list) or any(not isinstance(text, str) for text in evidence):
            raise ValueError("Evidence must be a list of text annotations")
        if row.get("family_id") is not None and str(row["family_id"]) != str(indexes[run][iid].get("family_id")):
            raise ValueError("Judgment family differs from the frozen candidate")
        if recorded >= cutoff:
            continue
        accepted.append(deepcopy(row))
        if rater == "reader" and "useful" in row:
            events[run].append({"id": len(events[run]) + 1, "item_id": iid,
                                "recorded_at": recorded.isoformat(),
                                "value": 0 if useful is None else 1 if useful else -1,
                                "grade": 100 if important else grade})
        if scope is not None:
            ref = (run, iid)
            previous = annotations.get(ref)
            if previous is None or recorded > _time(previous["recorded_at"]):
                annotations[ref] = deepcopy(row)
            elif recorded == _time(previous["recorded_at"]):
                raise ValueError("Simultaneous scope annotations need a resolved later judgment")
    return events, annotations, accepted


def _scope_metrics(ids, run, annotations):
    labels = [annotations.get((run, iid)) for iid in ids]
    counts = Counter(row["scope"] for row in labels if row is not None)
    unknown = sum(row is None for row in labels)
    uncertain = counts.get("uncertain", 0)
    n = len(ids)
    return {"counts": dict(counts), "unknown": unknown, "annotated": n - unknown,
            "off_interest_fraction_bounds": [counts["off_interest"] / n,
                                              (counts["off_interest"] + unknown + uncertain) / n] if n else [None, None],
            "rater_counts": dict(Counter(row["rater"] for row in labels if row)),
            "basis": "Explicit scope annotations only; scope is not usefulness or scientific quality"}


def _diagnostics(snapshot, slate, annotations, families, earlier, recent_days):
    run = snapshot["run_id"]
    candidates = {c["item_id"]: c for c in snapshot["candidates"]}
    now = _time(snapshot["decision_at"])
    recent = {family for when, group in earlier if now - timedelta(days=recent_days) <= when < now for family in group}
    result = {}
    for section in snapshot["section_budgets"]:
        ids = slate.get(section, [])
        if any(iid not in candidates for iid in ids):
            raise ValueError("A slate contains an unknown candidate")
        selected_families = [families[(run, iid)] for iid in ids]
        topic = Counter(candidates[iid].get("features", {}).get("primary_facet") or "unassigned" for iid in ids)
        judged_topics = Counter(facet for iid in ids for facet in annotations.get((run, iid), {}).get("facets", []))
        sources = Counter(candidates[iid].get("metadata", {}).get("source") or "unknown" for iid in ids)
        repetition = [iid for iid in ids if families[(run, iid)] in recent]
        result[section] = {
            "scope": _scope_metrics(ids, run, annotations),
            "scope_at_5": _scope_metrics(ids[:5], run, annotations),
            "duplicate_family_count": len(selected_families) - len(set(selected_families)),
            "recent_repeated_ids": repetition,
            "recent_repeated_count": len(repetition),
            "repetition_window_days": recent_days,
            "repetition_basis": "Earlier supplied snapshot slates only; absent days are not observed",
            "source_counts": dict(sources), "source_coverage": len(sources),
            "assigned_topic_counts": dict(topic),
            "judged_core_facet_counts": dict(judged_topics),
            "topic_basis": "Assigned topics are model output; judged facets are separate annotations",
            "annotations": [{"item_id": iid, **annotations[(run, iid)]} for iid in ids if (run, iid) in annotations],
        }
    return result


def _aggregate(reports):
    sections = {}
    names = sorted({name for report in reports for name in report["sections"]})
    for name in names:
        rows = [report["sections"][name] for report in reports if name in report["sections"]]
        selected = sum(row["served_k"] for row in rows)
        judged = sum(row["judged"] for row in rows)
        positive = sum(row["positive_count"] for row in rows)
        top = [row["at_5"] for row in rows]
        top_n, top_judged, top_positive = (sum(row[key] for row in top) for key in ("served_k", "judged", "positive_count"))
        sections[name] = {
            "served": selected, "judged": judged, "unknown": selected - judged,
            "useful_count": positive,
            "useful_fraction_bounds": [positive / selected, (positive + selected - judged) / selected] if selected else [None, None],
            "at_5": {"served": top_n, "judged": top_judged, "unknown": top_n - top_judged,
                     "useful_count": top_positive,
                     "useful_fraction_bounds": [top_positive / top_n, (top_positive + top_n - top_judged) / top_n] if top_n else [None, None]},
            "observed_important_available": sum(row["must_read_available"] for row in rows),
            "observed_important_recovered": sum(row["must_read_recovered"] for row in rows),
            "aggregation_basis": "Sum of daily exposures; repeated families across days are counted per exposure",
        }
    return sections


def evaluate_digest_variants(snapshots, judgments, *, challengers=None, outcome_as_of,
                             recent_days: int = 7, seed: int = 0) -> dict:
    """Evaluate actual captured runs; no synthetic historical candidate pools."""
    if type(recent_days) is not int or recent_days < 0:
        raise ValueError("Repetition window must be a nonnegative integer")
    snapshots, judgments = list(snapshots), list(judgments)
    runs = _inventory(snapshots)
    cutoff = _time(outcome_as_of)
    if any(cutoff <= _time(snapshot["decision_at"]) for snapshot in snapshots):
        raise ValueError("Outcome cutoff must be after every captured decision")
    variants = _slates(runs, challengers)
    families = _families(runs)
    events, annotations, accepted = _judgments(judgments, runs, cutoff)
    result = {"schema_version": 1, "version": VERSION, "promotion_eligible": False,
              "outcome_as_of": cutoff.isoformat(), "runs": len(runs),
              "days": len({snapshot["digest_id"] for snapshot in snapshots}),
              "snapshot_signatures": {run: _hash(snapshot) for run, snapshot in runs.items()},
              "judgment_signature": _hash(accepted), "judgments_used": len(accepted),
              "variants": {}, "comparisons": {},
              "limitations": ["Analyst scope labels never substitute for reader usefulness.",
                              "Unknown judgments are not negative; bounds do not remove exposure bias.",
                              "Important-paper recall is conditional on the captured eligible pool and observed reader labels.",
                              "Topic assignments and source diversity do not establish relevance or scientific quality.",
                              "No absent historical candidate pools or unobserved reading history are reconstructed.",
                              "Comparisons are diagnostic; none can promote a variant."]}
    ordered = sorted(runs.values(), key=lambda snap: (_time(snap["decision_at"]), snap["run_id"]))
    for name, declarations in variants.items():
        reports, daily, earlier = [], [], []
        for snapshot in ordered:
            run = snapshot["run_id"]
            declaration = declarations[run]
            slate, counterfactual = declaration["slate"], declaration["snapshot"]
            diagnostics = _diagnostics(snapshot, slate, annotations, families, earlier, recent_days)
            selected_refs = [families[(run, iid)] for ids in slate.values() for iid in ids]
            invalid = len(selected_refs) != len(set(selected_refs))
            known = feedback_as_of(events[run], cutoff)
            selected_families = set(selected_refs)
            for section, diagnostic in diagnostics.items():
                important = [candidate for candidate in snapshot["candidates"]
                             if candidate["section"] == section and candidate["eligible"]
                             and known.get(candidate["item_id"], {}).get("value") == 1
                             and (known[candidate["item_id"]].get("grade") or 0) >= 100]
                missed = [candidate["item_id"] for candidate in important
                          if families[(run, candidate["item_id"])] not in selected_families]
                diagnostic["observed_important_missed_ids"] = sorted(missed)
                diagnostic["observed_important_missed_families"] = sorted({families[(run, iid)] for iid in missed})
            if invalid:
                report = None
                validation = {"status": "invalid", "reason": "duplicate item or paper family in slate"}
            else:
                report = evaluate_replay(snapshot, events[run], outcome_as_of=cutoff, slate=slate)
                if counterfactual is not snapshot:
                    policy = evaluate_replay(counterfactual, events[run], outcome_as_of=cutoff, slate=slate)
                    report["policy_validation"] = {**policy["policy_validation"],
                                                   "declared_counterfactual_signature": _hash(counterfactual)}
                validation = report["policy_validation"]
                reports.append(report)
            current = {families[(run, iid)] for ids in slate.values() for iid in ids}
            earlier.append((_time(snapshot["decision_at"]), current))
            daily.append({"run_id": run, "digest_id": snapshot["digest_id"], "slate": deepcopy(slate),
                          "policy_validation": validation, "reader_metrics": report,
                          "diagnostics": diagnostics})
        totals = {}
        for section in sorted({section for day in daily for section in day["diagnostics"]}):
            rows = [day["diagnostics"][section] for day in daily if section in day["diagnostics"]]
            totals[section] = {key: dict(sum((Counter(row[key]) for row in rows), Counter()))
                               for key in ("source_counts", "assigned_topic_counts", "judged_core_facet_counts")}
            totals[section].update({
                "duplicate_family_count": sum(row["duplicate_family_count"] for row in rows),
                "recent_repeated_count": sum(row["recent_repeated_count"] for row in rows),
                "scope_counts": dict(sum((Counter(row["scope"]["counts"]) for row in rows), Counter())),
                "scope_unknown": sum(row["scope"]["unknown"] for row in rows),
            })
        aggregate = _aggregate(reports)
        served = sum(row["served"] for row in aggregate.values())
        judged = sum(row["judged"] for row in aggregate.values())
        useful = sum(row["useful_count"] for row in aggregate.values())
        whole = {"served": served, "judged": judged, "unknown": served - judged, "useful_count": useful,
                 "useful_fraction_bounds": [useful / served, (useful + served - judged) / served] if served else [None, None]}
        result["variants"][name] = {"per_day": daily, "aggregate": aggregate,
                                    "whole_slate": whole, "diagnostic_totals": totals,
                                    "valid_runs": len(reports), "invalid_runs": len(runs) - len(reports),
                                    "replay_reports": reports}
    for name in variants:
        if name == "baseline":
            continue
        base, other = result["variants"]["baseline"], result["variants"][name]
        if base["invalid_runs"] or other["invalid_runs"]:
            result["comparisons"][name] = {"status": "invalid", "reason": "Invalid slate prevents a complete paired comparison"}
            continue
        result["comparisons"][name] = {
            section: paired_run_metrics(base["replay_reports"], other["replay_reports"], section=section, seed=seed)
            for section in sorted(set.intersection(*(set(snap["section_budgets"]) for snap in snapshots)))
        }
    return result


def build_judgment_packet(snapshots, *, challengers=None, rejected_per_run: int = 5,
                          seed: int = 0, inspected_family_ids=()) -> dict:
    """Blind union of selected papers plus a deterministic rejected-family sample.

    The private manifest records arm membership and snapshot references. Neither
    random partitioning nor an absent inspected marker establishes a fresh holdout.
    """
    if type(rejected_per_run) is not int or rejected_per_run < 0:
        raise ValueError("Rejected sample count must be a nonnegative integer")
    runs = _inventory(list(snapshots))
    variants = _slates(runs, challengers)
    families, inspected = _families(runs), set(inspected_family_ids)
    entries = {}
    for run, snapshot in sorted(runs.items()):
        candidates = {c["item_id"]: c for c in snapshot["candidates"]}
        origins = defaultdict(set)
        for name, declarations in variants.items():
            for ids in declarations[run]["slate"].values():
                for iid in ids:
                    if iid not in candidates:
                        raise ValueError("A packet slate contains an unknown candidate")
                    origins[iid].add(name)
        selected_families = {families[(run, iid)] for iid in origins}
        rejected = sorted((iid for iid in candidates if families[(run, iid)] not in selected_families),
                          key=lambda iid: _hash([seed, run, candidates[iid].get("family_id"), iid]))
        sampled = set()
        for iid in rejected:
            family = families[(run, iid)]
            if len(sampled) >= rejected_per_run:
                break
            if family not in sampled:
                origins[iid].add("rejected_sample")
                sampled.add(family)
        for iid, arms in origins.items():
            candidate, family = candidates[iid], families[(run, iid)]
            metadata = candidate.get("metadata", {})
            title, abstract = metadata.get("title") or "", metadata.get("abstract") or ""
            key = _hash([family, title, abstract])
            entry = entries.setdefault(key, {"family_id": family, "title": title, "abstract": abstract,
                                             "references": [], "inspected": False})
            entry["inspected"] |= family in inspected or str(candidate.get("family_id")) in inspected
            entry["references"].append({"run_id": run, "item_id": iid, "family_id": candidate.get("family_id"),
                                         "origins": sorted(arms)})
    inspected_canonical = {entry["family_id"] for entry in entries.values() if entry["inspected"]}
    examples, mapping = [], {}
    for ident in sorted(entries, key=lambda key: _hash([seed, key])):
        entry = entries[ident]
        was_inspected = entry["family_id"] in inspected_canonical
        split = "inspected" if was_inspected else "evaluation" if int(_hash([seed, entry["family_id"]])[:8], 16) % 5 == 0 else "calibration"
        examples.append({"packet_id": ident, "family_id": entry["family_id"], "split": split,
                         "inspection_status": "inspected" if was_inspected else "unknown",
                         "title": entry["title"], "abstract": entry["abstract"],
                         "rater": None, "recorded_at": None, "useful": None, "important": None,
                         "grade": None, "scope": None, "facets": [], "contribution": "",
                         "evidence": [], "rejection_reason": ""})
        mapping[ident] = entry["references"]
    manifest = {"schema_version": 1, "version": VERSION, "seed": seed,
                "snapshot_signatures": {run: _hash(snapshot) for run, snapshot in runs.items()},
                "references": mapping, "packet_signature": _hash(examples),
                "sampling": "All baseline/challenger selections plus hash-ordered rejected families per run",
                "split_basis": "Family-separated deterministic partition; inspection status is not independently verified",
                "blinding_limits": "Structured sources, arm membership, ranks and scores omitted; supplied text may contain identifying boilerplate"}
    manifest["manifest_signature"] = _hash(manifest)
    return {"examples": examples, "manifest": manifest}


def expand_packet_judgments(rows, manifest, snapshots):
    """Bind completed packet judgments to exact frozen runs without arm exposure."""
    if manifest.get("manifest_signature") != _hash({k: v for k, v in manifest.items() if k != "manifest_signature"}):
        raise ValueError("Judgment packet manifest signature mismatch")
    runs = _inventory(list(snapshots))
    if manifest.get("snapshot_signatures") != {run: _hash(snapshot) for run, snapshot in runs.items()}:
        raise ValueError("Judgment packet was built from different snapshots")
    result = []
    fields = ("rater", "recorded_at", "useful", "important", "grade", "scope", "facets",
              "contribution", "evidence", "rejection_reason", "split", "inspection_status")
    for row in rows:
        ident = row.get("packet_id")
        if ident not in manifest["references"]:
            raise ValueError("Unknown judgment packet ID")
        if ident != _hash([row.get("family_id"), row.get("title"), row.get("abstract")]):
            raise ValueError("Judgment packet article text or family has changed")
        for reference in manifest["references"][ident]:
            result.append({**{key: reference[key] for key in ("run_id", "item_id", "family_id")},
                           **{key: row[key] for key in fields if key in row}})
    return result
