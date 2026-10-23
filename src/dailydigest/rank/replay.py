"""Pure evaluation of immutable captured pools; never reads current user state.

A recorded slate is evidence of what was served, not proof of replay parity.
Only an independent selector supplied by the caller can validate replay. Outcome
judgments may follow a decision; selector feedback is strictly pre-decision.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable


def _time(value: str | datetime) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def feedback_as_of(events: Iterable[dict], cutoff: str | datetime) -> dict[int, dict]:
    """Latest immutable signed event strictly before cutoff; neutral clears it.

    Absence of old events remains unknown. Never backfill from today's votes.
    """
    before = _time(cutoff)
    latest: dict[int, dict] = {}
    ordered = sorted(events, key=lambda event: (_time(event["recorded_at"]), int(event.get("id", 0))))
    for event in ordered:
        if _time(event["recorded_at"]) < before:
            latest[int(event["item_id"])] = dict(event)
    return {iid: event for iid, event in latest.items() if event["value"] in (-1, 1)}


def recorded_slate(snapshot: dict) -> dict[str, list[int]]:
    """Extract captured display order without re-running selection."""
    result = {section: [] for section in snapshot["section_budgets"]}
    selected = sorted((c for c in snapshot["candidates"] if c["selected"]),
                      key=lambda c: (c["section"], c["position"], c["item_id"]))
    for candidate in selected:
        result.setdefault(candidate["section"], []).append(int(candidate["item_id"]))
    return result


def _validate_slate(snapshot: dict, slate: dict[str, list[int]]) -> dict[int, dict]:
    from ..pipeline import (
        DEADLINE_KEEP, DEADLINE_SECTIONS, MINIMAL_RESEARCH_LIMIT, READING_MODE_LIMITS,
    )

    if snapshot.get("schema_version") != 1:
        raise ValueError("unsupported snapshot schema")
    mode = snapshot.get("reading_mode")
    if mode not in READING_MODE_LIMITS:
        raise ValueError("unsupported reading mode")
    reading_count = sum(len(ids) for section, ids in slate.items() if section not in DEADLINE_SECTIONS)
    deadline_count = sum(len(ids) for section, ids in slate.items() if section in DEADLINE_SECTIONS)
    reading_limit = READING_MODE_LIMITS[mode]
    if reading_limit is not None and reading_count > reading_limit:
        raise ValueError("slate exceeds reading mode budget")
    deadline_limit = DEADLINE_KEEP[mode]
    if deadline_limit is not None and deadline_count > deadline_limit:
        raise ValueError("slate exceeds deadline budget")
    if mode == "minimal":
        if len(slate.get("research", [])) > MINIMAL_RESEARCH_LIMIT:
            raise ValueError("slate exceeds minimal research limit")
        if any(len(ids) > 1 for section, ids in slate.items()
               if section != "research" and section not in DEADLINE_SECTIONS):
            raise ValueError("slate exceeds minimal news section limit")
    candidates = {int(c["item_id"]): c for c in snapshot["candidates"]}
    if len(candidates) != len(snapshot["candidates"]):
        raise ValueError("duplicate candidate item IDs")
    seen: set[int] = set()
    families: set[str] = set()
    for section, ids in slate.items():
        if section not in snapshot["section_budgets"]:
            raise ValueError("unknown section")
        if len(ids) > snapshot["section_budgets"][section]:
            raise ValueError("slate exceeds section budget")
        for iid in ids:
            candidate = candidates.get(iid)
            if candidate is None or not candidate["eligible"] or candidate["section"] != section:
                raise ValueError("unknown or ineligible candidate in section")
            family = str(candidate.get("family_id") or f"item:{iid}")
            if iid in seen or family in families:
                raise ValueError("duplicate item or family in slate")
            seen.add(iid)
            families.add(family)
    return candidates


def validate_replay(snapshot: dict, selector: Callable | None = None,
                    events: Iterable[dict] = ()) -> dict:
    """Compare an independent frozen-input selector with the recorded slate."""
    if selector is None or not snapshot.get("complete_candidate_pool"):
        return {"status": "incomplete", "reason": "independent selector and complete pool required"}
    expected = recorded_slate(snapshot)
    try:
        actual = selector(deepcopy(snapshot), feedback_as_of(events, snapshot["decision_at"]))
    except IncompleteReplay as exc:
        return {"status": "incomplete", "reason": str(exc)}
    _validate_slate(snapshot, actual)
    return {"status": "matched" if actual == expected else "mismatch",
            "recorded": expected, "replayed": actual}


def _metrics(ids: list[int], pool: list[int], labels: dict[int, dict], families: dict[int, str]) -> dict:
    top_labels = [labels.get(iid) for iid in ids]
    judged = sum(label is not None for label in top_labels)
    positive = sum(label is not None and label["value"] > 0 for label in top_labels)
    def gain(iid: int) -> float:
        label = labels.get(iid)
        if not label or label["value"] <= 0:
            return 0.0
        return 3.0 if (label.get("grade") or 0) >= 100 else 1.0
    gains = [gain(iid) for iid in ids]
    family_gains: dict[str, float] = {}
    for iid in pool:
        family_gains[families[iid]] = max(family_gains.get(families[iid], 0), gain(iid))
    ideal = sorted(family_gains.values(), reverse=True)[:len(ids)]
    dcg = lambda values: sum(value / math.log2(rank + 2) for rank, value in enumerate(values))
    idcg = dcg(ideal)
    must = {families[iid] for iid in pool if gain(iid) == 3}
    recovered = len(must.intersection(families[iid] for iid in ids))
    n = len(ids)
    return {"served_k": n, "judged": judged, "unknown": n - judged,
            "positive_count": positive,
            "judged_coverage": judged / n if n else None,
            "precision_lower": positive / n if n else None,
            "precision_upper": (positive + n - judged) / n if n else None,
            "graded_utility": sum(gains),
            "graded_ndcg_observed_pool": dcg(gains) / idcg if idcg else None,
            "zero_observed_positive_pool": not any(gain(iid) for iid in pool),
            "pool_judged": sum(iid in labels for iid in pool), "pool_size": len(pool),
            "must_read_recovered": recovered, "must_read_available": len(must),
            "must_read_recall_observed_pool": recovered / len(must) if must else None,
            "strong_rejection_count": sum(label is not None and label["value"] < 0
                                          and label.get("grade") is not None
                                          and label["grade"] <= 10 for label in top_labels)}


def evaluate_replay(snapshot: dict, events: Iterable[dict], *, outcome_as_of: str | datetime,
                    slate: dict[str, list[int]] | None = None) -> dict:
    """Score a recorded/challenger slate on one pool using explicit outcome cutoff.

    nDCG and must-read recall use the observed eligible pool only. Unknowns have
    zero observed gain but remain explicit in coverage and precision bounds.
    A supplied slate is diagnostic unless the real selector reproduces it from
    these frozen inputs. Structural validation alone does not verify the source
    quality gates: candidates are captured before final selection applies them.
    """
    slate = recorded_slate(snapshot) if slate is None else slate
    candidates = _validate_slate(snapshot, slate)
    policy_validation = _selection_policy_validation(snapshot, slate)
    labels = feedback_as_of(events, outcome_as_of)
    families = {iid: str(c.get("family_id") or f"item:{iid}") for iid, c in candidates.items()}
    sections = {}
    for section, budget in snapshot["section_budgets"].items():
        ids = slate.get(section, [])
        pool = [iid for iid, candidate in candidates.items()
                if candidate["section"] == section and candidate["eligible"]]
        metrics = _metrics(ids, pool, labels, families)
        metrics["budget"] = budget
        metrics["at_5"] = _metrics(ids[:5], pool, labels, families)
        sections[section] = metrics
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    label_hash = hashlib.sha256(json.dumps(labels, sort_keys=True, default=str).encode()).hexdigest()
    return {"run_id": snapshot["run_id"], "digest_id": snapshot["digest_id"],
            "reading_mode": snapshot["reading_mode"], "catch_up": bool(snapshot.get("catch_up", False)),
            "decision_at": snapshot["decision_at"],
            "complete_candidate_pool": bool(snapshot.get("complete_candidate_pool")),
            "policy_validation": policy_validation,
            "pool_signature": hashlib.sha256(encoded).hexdigest(), "label_signature": label_hash,
            "outcome_as_of": _time(outcome_as_of).isoformat(), "sections": sections,
            "metric_basis": "observed labels in captured eligible pool; unknown feedback is not negative"}


def paired_run_metrics(baselines: list[dict], challengers: list[dict], *,
                       section: str = "research", seed: int = 0) -> dict:
    """Paired observed-positive changes, clustered by digest day for uncertainty.

    Precision bounds and judgments must be inspected before interpreting a gain;
    an interval over exposure-selected outcomes does not remove selection bias.
    Unverified slates can be compared diagnostically, but cannot qualify as a
    comparison preserving the serving policy. Policy eligibility is necessary,
    not sufficient, evidence for promotion.
    """
    base = {r["run_id"]: r for r in baselines}
    other = {r["run_id"]: r for r in challengers}
    if base.keys() != other.keys() or len(base) != len(baselines) or len(other) != len(challengers):
        raise ValueError("comparisons require the same runs exactly once")
    pairs = []
    unverified_policy_runs = []
    days: dict[str, list[float]] = defaultdict(list)
    for run_id in sorted(base):
        a, b = base[run_id], other[run_id]
        if any(a[key] != b[key] for key in ("pool_signature", "label_signature", "outcome_as_of")):
            raise ValueError("paired comparison requires identical pool and outcomes")
        am, bm = a["sections"][section], b["sections"][section]
        if any(report.get("policy_validation", {}).get("status") != "verified"
               for report in (a, b)):
            unverified_policy_runs.append(run_id)
        delta = bm["positive_count"] - am["positive_count"]
        days[a["digest_id"]].append(delta)
        pairs.append({"run_id": run_id, "positive_count_delta": delta,
                      "graded_utility_delta": bm["graded_utility"] - am["graded_utility"],
                      "baseline_served_k": am["served_k"], "challenger_served_k": bm["served_k"],
                      "baseline_precision_bounds": [am["precision_lower"], am["precision_upper"]],
                      "challenger_precision_bounds": [bm["precision_lower"], bm["precision_upper"]],
                      "strong_rejection_count_delta": bm["strong_rejection_count"] - am["strong_rejection_count"],
                      "must_read_recovered_delta": bm["must_read_recovered"] - am["must_read_recovered"],
                      "must_read_available": am["must_read_available"],
                      "baseline_unknown": am["unknown"], "challenger_unknown": bm["unknown"],
                      "reading_mode": a["reading_mode"], "catch_up": a["catch_up"]})
    values = [sum(v) / len(v) for _, v in sorted(days.items())]
    mean = sum(values) / len(values) if values else None
    interval = None
    if len(values) >= 2:
        rng = random.Random(seed)
        draws = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(2000))
        interval = [draws[49], draws[1949]]
    return {"n_runs": len(pairs), "n_days": len(days), "per_run": pairs,
            "policy_comparison_eligible": bool(pairs) and not unverified_policy_runs,
            "unverified_policy_runs": unverified_policy_runs,
            "mean_positive_count_delta": mean, "day_bootstrap_interval": interval,
            "uncertainty_basis": "exploratory paired day bootstrap; small day counts are unstable"}


class IncompleteReplay(ValueError):
    """The snapshot lacks a frozen input required to reproduce selection."""


def _selection_policy_validation(snapshot: dict, slate: dict[str, list[int]]) -> dict:
    """Verify using the serving implementation, never a parallel copy of its gates."""
    try:
        expected = frozen_selection(deepcopy(snapshot))
    except IncompleteReplay as exc:
        return {"status": "unverified", "reason": str(exc)}
    # Omitted empty sections are immaterial; order within a section is not.
    normalized = {section: slate.get(section, []) for section in snapshot["section_budgets"]}
    if normalized != expected:
        return {
            "status": "unverified",
            "reason": "slate is not reproduced by the serving policy on declared frozen scores and inputs",
        }
    return {"status": "verified", "scope": "selection from frozen scores and eligibility only"}


_SELECTION_SETTINGS = (
    "max_preprint_research_frac", "max_low_impact_research_frac",
    "low_impact_relevance_floor", "research_final_score_floor_frac",
    "research_final_score_min_keep", "adaptive_relevance_floor",
)
_SELECTION_FACTS = (
    "source_bucket", "is_arxiv_cs", "is_preprint", "is_published",
    "is_high_quality", "is_low_impact",
)


def frozen_selection(snapshot: dict, feedback: dict | None = None) -> dict[str, list[int]]:
    """Run real serving selection on captured eligibility, scores and source facts.

    This replays source caps, raw-topic floors, final-score cutoff, exploration
    and reading mode. It does not re-embed, retrain, rerun upstream eligibility,
    or claim feedback changed the already-captured scores. Input positions freeze
    the upstream topic-preference ordering used by the actual picker.
    """
    from datetime import date
    from types import SimpleNamespace

    from .ranker import apply_exploration, pick_top_per_section
    from .source_quality import RANKER_VERSION
    from ..pipeline import apply_reading_mode, _restore_research_order

    recorded_version = snapshot.get("scorer_version")
    # v15 fixes upstream journal identity; historical source facts and scores
    # remain frozen. This compatibility proves selection only, not rescoring
    # with the corrected identities. Revisit when selection policy changes.
    compatible_versions = {
        RANKER_VERSION, "2026-10-22-short-reader-text-v20",
        "2026-10-15-local-topic-verification-v14",
        "2026-10-14-cross-field-venues-v13", "2026-10-02-quality-guards-v10",
    }
    if recorded_version is not None and recorded_version not in compatible_versions:
        raise IncompleteReplay("captured scorer version is incompatible with current selection policy")
    required = ("selection_settings", "catch_up", "exploration_slots", "decision_local_date")
    if not snapshot.get("complete_candidate_pool") or any(key not in snapshot for key in required):
        raise IncompleteReplay("full pool and frozen selection settings/date are required")
    frozen_settings = snapshot["selection_settings"]
    if any(key not in frozen_settings for key in _SELECTION_SETTINGS):
        raise IncompleteReplay("selection settings are incomplete")
    rows = []
    features = {}
    facts = {}
    positions = set()
    for candidate in snapshot["candidates"]:
        if not candidate["eligible"]:
            continue
        if any(key not in candidate for key in ("input_position", "ranking_score", "features", "metadata")):
            raise IncompleteReplay("candidate selection inputs are incomplete")
        position = candidate["input_position"]
        if not isinstance(position, int) or position in positions:
            raise IncompleteReplay("candidate input positions must be unique integers")
        positions.add(position)
        iid = int(candidate["item_id"])
        metadata = dict(candidate["metadata"])
        if "source" not in metadata:
            raise IncompleteReplay("candidate frozen source is missing")
        row = SimpleNamespace(**{**metadata, "id": iid, "section": candidate["section"]})
        feature = candidate["features"]
        if row.section == "research":
            fact = candidate.get("selection_facts", {})
            if any(key not in fact for key in _SELECTION_FACTS) or "topic_score" not in feature:
                raise IncompleteReplay("research source facts or raw topic score missing")
            if "source_diversity_key" in fact and not isinstance(fact["source_diversity_key"], str):
                raise IncompleteReplay("research source diversity key must be a string")
            facts[iid] = fact
        features[iid] = feature
        score = float(candidate["ranking_score"])
        if not math.isfinite(score):
            raise IncompleteReplay("candidate ranking score is non-finite")
        rows.append((position, row, score))
    rows.sort(key=lambda value: value[0])
    scored = [(row, score) for _, row, score in rows]
    picked = pick_top_per_section(scored, snapshot["section_budgets"],
                                  catch_up=bool(snapshot["catch_up"]), score_features=features,
                                  settings=SimpleNamespace(**frozen_settings), selection_facts=facts)
    slots = int(snapshot["exploration_slots"])
    if slots > 0:
        uncertainty = {iid: 1.0 - 2.0 * abs(float(feature.get("learned_score", 0.0)) - 0.5)
                       for iid, feature in features.items()
                       if feature.get("scoring_mode") in ("hybrid_lr", "hybrid_knn")}
        picked = apply_exploration(picked, scored, uncertainty, slots=slots,
                                   eligible=lambda row: facts[row.id]["is_high_quality"],
                                   caps=snapshot["section_budgets"],
                                   catch_up=bool(snapshot["catch_up"]), score_features=features,
                                   settings=SimpleNamespace(**frozen_settings), selection_facts=facts)
    order_policy = snapshot.get("reading_order_policy", "score")
    if order_policy not in {"score", "selection_priority"}:
        raise IncompleteReplay("unknown reading order policy")
    if order_policy == "selection_priority":
        picked = _restore_research_order(picked, scored)
    picked = apply_reading_mode(picked, snapshot["reading_mode"],
                               as_of_date=date.fromisoformat(snapshot["decision_local_date"]),
                               preserve_research_order=order_policy == "selection_priority")
    slate = {section: [] for section in snapshot["section_budgets"]}
    for row, _score in picked:
        slate[row.section].append(row.id)
    _validate_slate(snapshot, slate)
    return slate
