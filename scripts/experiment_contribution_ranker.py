#!/usr/bin/env python
"""Replay a fixed pairwise contribution reranker from local cached logits only."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from dailydigest.rank.contribution_ranker import fit_contribution_ranker
from dailydigest.rank.pair_scope import pair_queries
from dailydigest.rank.replay import frozen_selection, recorded_slate
from dailydigest.rank.scope import fingerprint, scope_title_identity, validate_benchmark
try:
    from scripts.experiment_pair_scope import article_text, validate_scores
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from experiment_pair_scope import article_text, validate_scores


ROOT = Path(__file__).resolve().parents[1]
LABELS = ("core", "context", "off_interest", "uncertain")
WEIGHT = .7


def _ranking_metrics(examples, scores):
    """One definitive representative per family; unknown/uncertain stay explicit."""
    families, representatives = {}, []
    for example in examples:
        family = str(example.get("family_id") or example["id"])
        label = example["label"]
        if label == "uncertain":
            continue
        if family in families:
            if families[family] != label:
                raise ValueError("Evaluation family has conflicting definitive labels")
            continue
        families[family] = label
        representatives.append(example)
    metrics = {}
    for higher, lower in (("core", "context"), ("core", "off_interest"), ("context", "off_interest")):
        differences = [scores[left["id"]] - scores[right["id"]]
                       for left in representatives if left["label"] == higher
                       for right in representatives if right["label"] == lower]
        metrics[f"{higher}_{lower}"] = {
            "pairs": len(differences),
            "accuracy": sum(float(d > 0) + .5 * float(d == 0) for d in differences) / len(differences) if differences else None,
        }
    available = [entry["accuracy"] for entry in metrics.values() if entry["accuracy"] is not None]
    return {
        "example_count": len(examples), "definitive_family_count": len(representatives),
        "uncertain_count": sum(ex["label"] == "uncertain" for ex in examples),
        "tier_pairs": metrics, "balanced_pair_accuracy": float(np.mean(available)) if available else None,
        "mean_rank_signal_by_tier": {
            label: float(np.mean([scores[ex["id"]] for ex in representatives if ex["label"] == label]))
            if any(ex["label"] == label for ex in representatives) else None for label in LABELS[:-1]
        },
    }


def _normalize_fresh(fresh_labels):
    if not isinstance(fresh_labels, list):
        return fresh_labels
    if not fresh_labels or any(not isinstance(ex, dict) or not isinstance(ex.get("judgment_provenance"), str)
                               or not ex["judgment_provenance"].strip() for ex in fresh_labels):
        raise ValueError("List-form fresh labels require per-example judgment provenance")
    return {"examples": fresh_labels, "provenance": {
        "kind": "per_example_analyst_labels", "source_shape": "list",
        "judgment_provenance": sorted({ex["judgment_provenance"] for ex in fresh_labels}),
    }}


def _fresh_examples(fresh_labels, benchmark, candidates, full_pool):
    if fresh_labels is None:
        return []
    if not isinstance(fresh_labels, dict) or not isinstance(fresh_labels.get("examples"), list):
        raise ValueError("Fresh labels require an examples list and explicit provenance")
    if not isinstance(fresh_labels.get("provenance"), dict) or not fresh_labels["provenance"]:
        raise ValueError("Fresh label provenance is required")
    inspected_ids = {str(ex["id"]) for ex in benchmark["examples"]}
    inspected_families = {str(ex["family_id"]) for ex in benchmark["examples"]}
    inspected_titles = {scope_title_identity(ex["title"]) for ex in benchmark["examples"]} - {""}
    inspected_families.update(str(c["family_id"]) for c in full_pool
                              if str(c["item_id"]) in inspected_ids and c.get("family_id"))
    pool = {str(candidate["item_id"]): candidate for candidate in candidates}
    result, seen = [], set()
    for supplied in fresh_labels["examples"]:
        example = deepcopy(supplied)
        iid = str(example.get("id", example.get("item_id", "")))
        if iid not in pool or iid in seen or example.get("label") not in LABELS:
            raise ValueError("Fresh labels need unique eligible research IDs and valid scope labels")
        seen.add(iid)
        candidate = pool[iid]
        metadata = candidate["metadata"]
        for key in ("title", "abstract"):
            if key in example and example[key] != metadata.get(key):
                raise ValueError("Fresh labeled text differs from the captured article")
        family = str(candidate.get("family_id") or example.get("family_id") or iid)
        if example.get("family_id") and str(example["family_id"]) != family:
            raise ValueError("Fresh label family differs from the captured article family")
        if (iid in inspected_ids or family in inspected_families
                or scope_title_identity(metadata.get("title") or "") in inspected_titles):
            raise ValueError("Fresh labels overlap inspected benchmark examples")
        example.update(id=iid, family_id=family, split="fresh", title=metadata.get("title"), abstract=metadata.get("abstract"))
        result.append(example)
    # Validate conflicting labels even if no metric later uses one of the tiers.
    _ranking_metrics(result, {example["id"]: .5 for example in result})
    return result


def _scope_counts(ids, known):
    counts = Counter(known.get(str(iid), "unknown") for iid in ids)
    return {label: counts[label] for label in (*LABELS, "unknown")}


def _validate_feature_inputs(cached, manifest):
    if "feature_protocol" not in manifest:
        return None
    from dailydigest.rank.contribution_claims import FEATURE_PROTOCOL, contribution_text

    protocol = manifest["feature_protocol"]
    if fingerprint(protocol) != fingerprint(FEATURE_PROTOCOL):
        raise ValueError("Unsupported contribution feature protocol")
    for row in cached.values():
        expected = contribution_text(row.get("title") or "", row.get("abstract") or "")
        if (row.get("input_text") != expected
                or row.get("input_text_sha256") != hashlib.sha256(expected.encode()).hexdigest()):
            raise ValueError("Contribution feature input text or input hash differs from the declared protocol")
    return deepcopy(protocol)


def _final_score_metrics(examples, original_scores, final_scores):
    compared = [ex for ex in examples if ex["id"] in original_scores and ex["id"] in final_scores]
    return {
        "evaluation_example_count": len(examples), "compared_example_count": len(compared),
        "excluded_missing_original_scores_count": len(examples) - len(compared),
        "excluded_ids": [ex["id"] for ex in examples if ex not in compared],
        "baseline": _ranking_metrics(compared, original_scores),
        "challenger": _ranking_metrics(compared, final_scores),
    }


def run_experiment(benchmark, score_rows, manifest, *, snapshot, fresh_labels=None):
    """Fit on definitive calibration families and replay the actual frozen selector."""
    from dailydigest.rank.semantic_rerank import apply_contribution_scores, has_abstract, interest_signature

    validate_benchmark(benchmark)
    fresh_signature = fingerprint(fresh_labels) if fresh_labels is not None else None
    fresh_labels = _normalize_fresh(fresh_labels)
    queries, examples = pair_queries(benchmark["definitions"]), benchmark["examples"]
    profile = snapshot.get("profile", {})
    if (snapshot.get("schema_version") != 1
            or snapshot.get("profile_signature") != fingerprint(profile)
            or set(profile.get("canonical_facets", {})) != set(benchmark["definitions"]["core"])):
        raise ValueError("Snapshot schema/profile/facets differ from frozen evidence")
    candidates, seen = [], set()
    required = {ex["id"]: article_text(ex) for ex in examples}
    for candidate in snapshot["candidates"]:
        iid, metadata = int(candidate["item_id"]), candidate["metadata"]
        if iid in seen:
            raise ValueError("Snapshot candidate IDs must be unique")
        seen.add(iid)
        if candidate.get("text_signature") != fingerprint([metadata.get("title"), metadata.get("abstract")]):
            raise ValueError("Snapshot text fingerprint mismatch")
        if candidate["eligible"] and candidate["section"] == "research":
            key, text = str(iid), article_text(metadata)
            if key in required and required[key] != text:
                raise ValueError("Benchmark/snapshot article text conflict")
            required[key] = text
            candidates.append(candidate)
    cached = validate_scores(score_rows, manifest, queries, required)
    feature_protocol = _validate_feature_inputs(cached, manifest)
    fresh = _fresh_examples(fresh_labels, benchmark, candidates, snapshot["candidates"])
    baseline = frozen_selection(deepcopy(snapshot))
    recorded = recorded_slate(snapshot)
    if baseline != recorded:
        raise ValueError("Baseline frozen selection does not match the captured slate")
    training = [ex for ex in examples if ex["split"] == "calibration"]
    values = np.asarray([cached[ex["id"]]["raw_logits"] for ex in training])
    ranker = fit_contribution_ranker(values, queries, training)
    all_ids = list(required)
    all_values = np.asarray([cached[iid]["raw_logits"] for iid in all_ids]).reshape(-1, 24)
    predictions = ranker.predict_scores(all_values, queries=queries)
    signals = dict(zip(all_ids, map(float, predictions), strict=True))
    artifact = {
        "kind": "local_contribution_reranker", "schema_version": 1,
        "definitions": deepcopy(benchmark["definitions"]), "queries": queries,
        "encoder_provenance": deepcopy(manifest["model"]),
        "profile_interest_signature": interest_signature(profile),
        "ranker": ranker.to_artifact(),
    }
    if feature_protocol is not None:
        artifact["feature_protocol"] = feature_protocol
    artifact["fingerprint"] = fingerprint(artifact)
    challenger_snapshot = deepcopy(snapshot)
    eligible = [candidate for candidate in challenger_snapshot["candidates"] if candidate["eligible"]]
    scored, features = [], {}
    by_id = {int(candidate["item_id"]): candidate for candidate in eligible}
    for candidate in eligible:
        iid = int(candidate["item_id"])
        row = SimpleNamespace(**{**candidate["metadata"], "id": iid, "section": candidate["section"]})
        scored.append((row, float(candidate["ranking_score"])))
        features[iid] = candidate["features"]
    # Preserve the captured tie order when handing unchanged scores to the helper.
    scored.sort(key=lambda pair: by_id[pair[0].id]["input_position"])
    adjusted = apply_contribution_scores(scored, features, {int(c["item_id"]): signals[str(c["item_id"])] for c in candidates},
                                         weight=WEIGHT, model_fingerprint=artifact["fingerprint"])
    bonuses = {row.id: float(features[row.id].get("selection_order_bonus", 0)) for row, _ in adjusted}
    if not all(np.isfinite(bonus) for bonus in bonuses.values()):
        raise ValueError("Captured selection order bonuses must be finite")
    adjusted.sort(key=lambda pair: pair[1] + bonuses[pair[0].id], reverse=True)
    for position, (row, score) in enumerate(adjusted):
        by_id[row.id].update(ranking_score=float(score), input_position=position, features=features[row.id])
    challenger = frozen_selection(challenger_snapshot)
    known = {str(ex["id"]): ex["label"] for ex in examples}
    known.update({ex["id"]: ex["label"] for ex in fresh})
    evaluation = [ex for ex in examples if ex["split"] == "evaluation"]
    original_scores = {str(c["item_id"]): float(c["ranking_score"]) for c in candidates}
    final_scores = {str(iid): float(candidate["ranking_score"]) for iid, candidate in by_id.items()}
    evaluation_final_scores = _final_score_metrics(evaluation, original_scores, final_scores)
    evaluation_final_scores.update(
        comparison_meaning="Same eligible snapshot-present evaluation examples; historical or ineligible examples lacking original comparison scores are excluded, never assigned synthetic scores",
        groups={group: _final_score_metrics(
            [ex for ex in evaluation if ex.get("evaluation_group", "evaluation") == group],
            original_scores, final_scores,
        ) for group in sorted({ex.get("evaluation_group", "evaluation") for ex in evaluation})},
    )
    baseline_ids, challenger_ids = baseline.get("research", []), challenger.get("research", [])
    matched_k = min(11, len(baseline_ids), len(challenger_ids))
    report = {
        "schema_version": 1, "status": "offline_experiment", "promotion_eligible": False,
        "benchmark_signature": fingerprint(benchmark), "snapshot_signature": fingerprint(snapshot),
        "cache_protocol_sha256": manifest["protocol_sha256"], "query_sha256": manifest["query_sha256"],
        "cached_document_count": len(cached), "artifact": artifact,
        "fixed_intervention": {"learned_weight": WEIGHT, "original_weight": 1 - WEIGHT,
                               "minimum_abstract_characters": 40, "insufficient_abstract": "original score retained"},
        "baseline_recorded_parity": {"status": "matched", "recorded": recorded},
        "baseline_slate": baseline, "challenger_slate": challenger,
        "matched_budget_prefixes": {"original_research_count": len(baseline_ids), "requested_k": 11,
                                    "matched_k": matched_k, "baseline": baseline_ids[:matched_k],
                                    "challenger": challenger_ids[:matched_k]},
        "scope_counts": {"baseline": _scope_counts(baseline_ids, known),
                         "challenger": _scope_counts(challenger_ids, known),
                         "matched_baseline": _scope_counts(baseline_ids[:matched_k], known),
                         "matched_challenger": _scope_counts(challenger_ids[:matched_k], known)},
        "evaluation": _ranking_metrics(evaluation, signals),
        "evaluation_final_score_metrics": evaluation_final_scores,
        "evaluation_groups": {
            group: _ranking_metrics([ex for ex in evaluation if ex.get("evaluation_group", "evaluation") == group], signals)
            for group in sorted({ex.get("evaluation_group", "evaluation") for ex in evaluation})
        },
        "fresh_evaluation": {"status": "available" if fresh_labels is not None else "not_supplied",
                             "provenance": deepcopy(fresh_labels.get("provenance")) if fresh_labels is not None else None,
                             "signature": fresh_signature,
                             "metrics": _ranking_metrics(fresh, signals),
                             "baseline_final_score_metrics": _ranking_metrics(fresh, original_scores),
                             "challenger_final_score_metrics": _ranking_metrics(fresh, final_scores),
                             "comparison_meaning": "Same held-out fresh families; original scores versus the frozen 0.7/0.3 mixture"},
        "candidate_scores": [{"item_id": int(c["item_id"]), "title": c["metadata"].get("title"),
                              "original_score": float(c["ranking_score"]), "rank_signal": signals[str(c["item_id"])],
                              "challenger_score": by_id[int(c["item_id"])]["ranking_score"],
                              "scope_label": known.get(str(c["item_id"]), "unknown"),
                              "abstract_usable": has_abstract(SimpleNamespace(**c["metadata"])),
                              "baseline_selected": int(c["item_id"]) in baseline_ids,
                              "challenger_selected": int(c["item_id"]) in challenger_ids} for c in candidates],
        "limitations": [
            "One captured candidate pool; no causal or multi-day usefulness claim.",
            "Analyst scope labels are not reader usefulness or scientific quality labels.",
            "Previous benchmark evaluation is inspected regression; supplied fresh labels are reported separately.",
            "Unknown slate scope remains unknown; cache consistency does not independently verify model assets.",
            "Eligibility, topic gates and source facts remain captured; global budgets may change other selections.",
            "Captured topic/coverage selection bonuses remain frozen; coverage champions are not recomputed from mutable history.",
        ],
    }
    return report


class Parser(argparse.ArgumentParser):
    def error(self, message):
        print("error: " + json.dumps(message))
        print('help: "Use --benchmark --scores --manifest --snapshot --output NEW.json --artifact NEW.json [--fresh-labels FILE]"')
        raise SystemExit(2)


def main(argv=None):
    parser = Parser(description=__doc__, allow_abbrev=False)
    for flag in ("benchmark", "scores", "manifest", "snapshot", "output", "artifact"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--fresh-labels", type=Path)
    parser.add_argument("-v", "-V", "--version", action="version", version="1")
    args = parser.parse_args(argv)
    from dailydigest.config import get_settings
    settings = get_settings()
    protected = {ROOT / "data", Path(settings.db_path).resolve().parent, Path(settings.profile_path).resolve().parent}
    outputs = [args.output.resolve(), args.artifact.resolve()]
    if outputs[0] == outputs[1] or any(path.exists() or path == ROOT / ".env" or any(path.is_relative_to(p) for p in protected) for path in outputs):
        parser.error("Outputs must be distinct, new and outside production data/profile directories and .env")
    try:
        read = lambda path: json.loads(path.read_text(encoding="utf-8"))
        report = run_experiment(read(args.benchmark),
                                [json.loads(line) for line in args.scores.read_text(encoding="utf-8").splitlines() if line.strip()],
                                read(args.manifest), snapshot=read(args.snapshot),
                                fresh_labels=read(args.fresh_labels) if args.fresh_labels else None)
        payloads = [report, report["artifact"]]
        serialized = [json.dumps(payload, indent=2, allow_nan=False) + "\n" for payload in payloads]
        for output, text in zip(outputs, serialized, strict=True):
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                handle.write(text)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print("error: " + json.dumps(str(exc)))
        print('help: "Check frozen inputs, cache provenance and baseline parity; production data was not written."')
        return 1
    print('status: "offline_experiment"')
    print("report: " + json.dumps(str(outputs[0])))
    print("artifact: " + json.dumps(str(outputs[1])))
    print("promotion_eligible: false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
