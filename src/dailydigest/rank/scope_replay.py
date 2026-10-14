"""Offline scope interventions on immutable candidate pools.

Scope labels are hypotheses, not reader judgments. This module changes only
research eligibility and delegates both selections to the serving implementation.
It neither publishes a digest nor certifies semantic relevance.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json

from .replay import frozen_selection, recorded_slate


_LABELS = ("core", "context", "off_interest", "uncertain")


def _signature(value, name: str) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must contain finite JSON-serializable evidence") from exc
    return hashlib.sha256(encoded.encode()).hexdigest()


def _has_evidence(value) -> bool:
    """Reject nominal provenance such as {'version': ''} or {'inputs': []}."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_has_evidence(entry) for entry in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_evidence(entry) for entry in value)
    return value is not None


def compare_scope_slates(snapshot: dict, decisions: dict[int, dict], *, policy: dict) -> dict:
    """Compare baseline and core-only challenger through real frozen selection.

    Require exactly one decision for each originally eligible research candidate.
    Every decision has a label, string facet, and nonempty reason; core requires
    a named facet. Additional diagnostics must be finite JSON values. Missing
    decisions, extra decisions and unsupported replay evidence fail closed.

    ``policy`` records caller-supplied classifier/anchor/calibration provenance;
    its presence is validated, not its scientific adequacy. The output is an
    offline diagnostic even when both slates preserve selection constraints.
    Recorded selected/position fields remain baseline evidence; no challenger
    positions are written into the input or the copied snapshot.
    """
    if not isinstance(policy, dict) or not policy:
        raise ValueError("policy provenance must be a nonempty evidence dictionary")
    policy_signature = _signature(policy, "policy")
    if not _has_evidence(policy):
        raise ValueError("policy provenance must be a nonempty evidence dictionary")
    if not isinstance(decisions, dict) or any(type(iid) is not int for iid in decisions):
        raise ValueError("decisions must be keyed by integer item IDs")
    eligible_ids = {int(candidate["item_id"]) for candidate in snapshot["candidates"]
                    if candidate["eligible"] and candidate["section"] == "research"}
    if set(decisions) != eligible_ids:
        raise ValueError("decisions must cover exactly the originally eligible research IDs")
    for iid, decision in decisions.items():
        if not isinstance(decision, dict) or decision.get("label") not in _LABELS:
            raise ValueError(f"unsupported scope label for item {iid}")
        facet, reason = decision.get("facet"), decision.get("reason")
        if not isinstance(facet, str) or (decision["label"] == "core" and not facet.strip()):
            raise ValueError(f"invalid scope facet for item {iid}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"missing scope reason for item {iid}")
    decision_signature = _signature(decisions, "decisions")
    snapshot_signature = _signature(snapshot, "snapshot")

    baseline_snapshot = deepcopy(snapshot)
    challenger_snapshot = deepcopy(snapshot)
    for candidate in challenger_snapshot["candidates"]:
        iid = int(candidate["item_id"])
        if iid in eligible_ids and decisions[iid]["label"] != "core":
            candidate["eligible"] = False

    # Do not truncate the baseline slate or independently reimplement its gates.
    baseline = frozen_selection(baseline_snapshot)
    challenger = frozen_selection(challenger_snapshot)
    recorded = recorded_slate(snapshot)
    before_ids = baseline.get("research", [])
    after_ids = challenger.get("research", [])
    before_set, after_set = set(before_ids), set(after_ids)
    counts = Counter(decision["label"] for decision in decisions.values())
    explanations = []
    for candidate in snapshot["candidates"]:
        if candidate["section"] != "research":
            continue
        iid = int(candidate["item_id"])
        decision = decisions.get(iid)
        if not candidate["eligible"]:
            status = "originally_ineligible"
        elif decision["label"] != "core":
            status = "excluded_by_scope"
        elif iid in after_set:
            status = "selected"
        else:
            status = "not_selected_by_frozen_policy"
        explanations.append({
            "item_id": iid, "scope_decision": deepcopy(decision),
            "originally_eligible": bool(candidate["eligible"]),
            "baseline_selected": iid in before_set, "challenger_selected": iid in after_set,
            "selection_status": status,
        })
    research_budget = snapshot["section_budgets"].get("research", 0)
    return {
        "run_id": snapshot.get("run_id"), "status": "offline_diagnostic",
        "publication_allowed": False, "relevance_validation": "not_evaluated",
        "baseline_slate": baseline, "challenger_slate": challenger,
        "baseline_recorded_parity": {
            "status": "matched" if recorded == baseline else "mismatch", "recorded": recorded,
        },
        "removed_research_ids": [iid for iid in before_ids if iid not in after_set],
        "added_research_ids": [iid for iid in after_ids if iid not in before_set],
        "decision_counts": {label: counts[label] for label in _LABELS},
        "research_explanations": explanations,
        "coverage": {
            "originally_eligible_research_count": len(eligible_ids),
            "decided_research_count": len(decisions),
            "retained_core_candidate_count": counts["core"],
            "uncertain_fraction": counts["uncertain"] / len(decisions) if decisions else None,
            "research_section_budget": research_budget,
            "baseline_research_count": len(before_ids), "challenger_research_count": len(after_ids),
            "section_budget_shortfall": max(0, research_budget - len(after_ids)),
            "shorter_than_baseline": len(after_ids) < len(before_ids),
            "empty_research_slate": not after_ids,
        },
        "policy_provenance": deepcopy(policy), "policy_signature": policy_signature,
        "decision_signature": decision_signature, "snapshot_signature": snapshot_signature,
        "intervention_scope": "core-only research eligibility filter before frozen serving selection",
        "selection_validation": {
            "status": "verified", "scope": "both slates generated by actual frozen_selection",
            "baseline_inputs": "original captured eligibility and scores",
            "challenger_inputs": "original captured inputs with non-core research made ineligible",
        },
        "limitations": [
            "Scope labels and supplied provenance are not verified reader judgments or calibrated relevance.",
            "Only originally eligible research is assessed; upstream retrieval and eligibility are unchanged.",
            "No rescoring, retraining, embedding, provider calls, or publication occurs.",
            "Recorded selected/position fields describe the baseline; challenger slate is returned separately.",
            "Global reading budgets may change nonresearch selection despite unchanged nonresearch inputs.",
            "A short or empty challenger is coverage loss to inspect, not evidence of improved relevance.",
            "Remaining core candidates can still fail source caps, floors, exploration, or reading budgets; "
            "individual nonselection causes are not inferred from the final slate.",
        ],
    }
