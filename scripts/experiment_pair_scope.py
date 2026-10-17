#!/usr/bin/env python
"""Evaluate two frozen local pair-scope arms from cached raw logits, without inference.

Generate scores separately with LocalCrossEncoder(explicit_local_model_directory).
This runner never downloads models, opens a database, or changes production policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from dailydigest.rank.learned_scope import fit_learned_scope
from dailydigest.rank.pair_scope import (
    PairScopeThresholds, calibrate_pair_scope, classify_pair_scope, pair_evidence, pair_queries,
)
from dailydigest.rank.scope import fingerprint, scope_metrics, validate_benchmark
from dailydigest.rank.scope_replay import compare_scope_slates

ROOT = Path(__file__).resolve().parents[1]


def compact_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def article_text(row):
    return (row.get("title") or "") + ". " + (row.get("abstract") or "")


def validate_scores(rows, manifest, queries, required):
    """Validate all cached rows, including unused superset rows, before fitting."""
    expected_queries = [{"kind": q["label"], "facet": q["facet"] or None,
                         "text": q["text"]} for q in queries]
    if manifest.get("queries") != expected_queries or len(queries) != 24:
        raise ValueError("Cached query order/text differs from the frozen definitions")
    if manifest.get("query_sha256") != compact_hash(expected_queries):
        raise ValueError("Cached query fingerprint mismatch")
    model = manifest["model"]
    assets = model.get("assets_sha256", {})
    if (not {"onnx/model.onnx", "tokenizer.json", "config.json"} <= set(assets)
            or any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
                   for v in assets.values())
            or model.get("score_kind") != "raw_logit"
            or model.get("truncation") != "only_second"
            or model.get("providers") != ["CPUExecutionProvider"]
            or any(not model.get(key) for key in ("onnxruntime_version", "tokenizers_version", "padding"))
            or any(type(model.get(key)) is not int or model[key] <= 0
                   for key in ("max_length", "batch_size", "threads"))):
        raise ValueError("Cached model needs complete local raw-logit asset/runtime provenance")
    if model.get("fingerprint") != compact_hash({k: v for k, v in model.items() if k != "fingerprint"}):
        raise ValueError("Cached model fingerprint mismatch")
    protocol = manifest.get("protocol_sha256")
    if not isinstance(protocol, str) or len(protocol) != 64 or any(c not in "0123456789abcdef" for c in protocol):
        raise ValueError("A protocol fingerprint is required")
    cached = {}
    for row in rows:
        iid = row["id"]
        if not isinstance(iid, str) or not iid or iid in cached:
            raise ValueError("Cached score IDs must be unique nonempty strings")
        text = article_text(row)
        if row.get("text") != text or row.get("text_sha256") != hashlib.sha256(text.encode()).hexdigest():
            raise ValueError("Cached article text fingerprint mismatch")
        if (row.get("query_sha256") != manifest["query_sha256"]
                or row.get("protocol_sha256") != protocol
                or row.get("model_fingerprint") != model["fingerprint"]):
            raise ValueError("Cached row provenance mismatch")
        values = np.asarray(row["raw_logits"], dtype=float)
        if values.shape != (24,) or not np.isfinite(values).all():
            raise ValueError("Cached logits must contain 24 finite scores")
        cached[iid] = row
    missing = set(required) - set(cached)
    if missing:
        raise ValueError(f"Cached scores missing {len(missing)} required documents")
    if any(cached[iid]["text"] != text for iid, text in required.items()):
        raise ValueError("Cached article differs from benchmark/snapshot text")
    return cached


def run_experiment(benchmark, score_rows, manifest, *, snapshot=None):
    validate_benchmark(benchmark)
    queries, examples = pair_queries(benchmark["definitions"]), benchmark["examples"]
    required = {ex["id"]: article_text(ex) for ex in examples}
    candidates = []
    if snapshot is not None:
        profile = snapshot.get("profile", {})
        if set(profile.get("canonical_facets", {})) != set(benchmark["definitions"]["core"]):
            raise ValueError("Snapshot canonical facets differ from definitions")
        if snapshot.get("profile_signature") != fingerprint(profile):
            raise ValueError("Snapshot profile fingerprint mismatch")
        for candidate in snapshot["candidates"]:
            metadata = candidate["metadata"]
            if candidate.get("text_signature") != fingerprint([metadata.get("title"), metadata.get("abstract")]):
                raise ValueError("Snapshot text fingerprint mismatch")
            if candidate["eligible"] and candidate["section"] == "research":
                iid, text = str(candidate["item_id"]), article_text(metadata)
                if iid in required and required[iid] != text:
                    raise ValueError("Benchmark/snapshot article text conflict")
                required[iid] = text
                candidates.append(candidate)
    cached = validate_scores(score_rows, manifest, queries, required)
    values = np.asarray([cached[ex["id"]]["raw_logits"] for ex in examples])
    indexes = [i for i, ex in enumerate(examples) if ex["split"] == "calibration"]
    training = [examples[i] for i in indexes]
    raw_evidence = pair_evidence(values, queries)
    representatives, families = [], set()
    for i in indexes:
        if examples[i]["family_id"] not in families:
            representatives.append(i)
            families.add(examples[i]["family_id"])
    zero_calibration = calibrate_pair_scope([raw_evidence[i] for i in representatives],
                                            [examples[i] for i in representatives])
    learned = fit_learned_scope(values[indexes], queries, training)
    result = {"schema_version": 1, "status": "offline_experiment", "promotion_eligible": False,
              "benchmark_signature": fingerprint(benchmark), "provenance": benchmark["provenance"],
              "model": manifest["model"], "queries": queries,
              "query_sha256": manifest["query_sha256"], "protocol_sha256": manifest["protocol_sha256"],
              "cached_document_count": len(cached), "arms": {},
              "provenance_validation": "Internal digest consistency; no independent verification of model assets or protocol contents",
              "limitations": ["Analyst scope labels are not reader usefulness or scientific-quality judgments.",
                              "Both arms are experimental and cannot promote themselves.",
                              "Replay changes eligibility only; original scores and upstream retrieval remain frozen."]}
    for name, evidence, calibration, provenance in (
        ("zeroshot", raw_evidence, zero_calibration, {"version": "pair-scope-v1"}),
        ("learned", learned.predict_evidence(values), learned.provenance["calibration_report"], learned.provenance),
    ):
        thresholds = PairScopeThresholds(**calibration["thresholds"])
        def decide(ev, abstract, thresholds=thresholds, name=name, provenance=provenance):
            decision = classify_pair_scope(ev, abstract, thresholds)
            if name == "learned":
                decision["score_meaning"] = provenance["score_meaning"]
            return decision
        decisions = [decide(ev, ex["abstract"]) for ev, ex in zip(evidence, examples, strict=True)]
        arm = {"calibration": calibration, "provenance": provenance,
               "thresholds": calibration["thresholds"], "promotion_eligible": False,
               "benchmark_metrics": {}, "evaluation_groups": {},
               "calibration_metrics_meaning": ("in-sample full-fit diagnostic; threshold selection uses calibration OOF metrics"
                                               if name == "learned" else "threshold-fitting development diagnostic"),
               "benchmark_predictions": [{"id": ex["id"], "split": ex["split"],
                                            "expected": ex["label"], "expected_facet": ex.get("facet", ""),
                                            "decision": decision}
                                           for ex, decision in zip(examples, decisions, strict=True)]}
        for split in ("calibration", "evaluation"):
            selected = [i for i, ex in enumerate(examples) if ex["split"] == split]
            arm["benchmark_metrics"][split] = scope_metrics([examples[i] for i in selected], [decisions[i] for i in selected])
        for group in sorted({ex.get("evaluation_group", "evaluation") for ex in examples if ex["split"] == "evaluation"}):
            selected = [i for i, ex in enumerate(examples) if ex["split"] == "evaluation" and ex.get("evaluation_group", "evaluation") == group]
            arm["evaluation_groups"][group] = scope_metrics([examples[i] for i in selected], [decisions[i] for i in selected])
        if snapshot is not None:
            pool = np.asarray([cached[str(c["item_id"])]["raw_logits"] for c in candidates]).reshape(-1, len(queries))
            pool_evidence = pair_evidence(pool, queries) if name == "zeroshot" else learned.predict_evidence(pool)
            scope_decisions = {c["item_id"]: decide(ev, c["metadata"].get("abstract") or "")
                               for c, ev in zip(candidates, pool_evidence, strict=True)}
            policy = {"arm": name, "model": manifest["model"], "query_sha256": manifest["query_sha256"],
                      "calibration": calibration, "provenance": provenance, "promotion_eligible": False}
            arm["replay"] = compare_scope_slates(snapshot, scope_decisions, policy=policy)
        result["arms"][name] = arm
    return result


class Parser(argparse.ArgumentParser):
    def error(self, message):
        print("error: " + json.dumps(message))
        print('help: "Use --benchmark FILE --scores JSONL --manifest JSON [--snapshot FILE] --output NEW.json"')
        raise SystemExit(2)


def main(argv=None):
    parser = Parser(description=__doc__, allow_abbrev=False)
    for flag in ("benchmark", "scores", "manifest", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("-v", "-V", "--version", action="version", version="1")
    args = parser.parse_args(argv)
    from dailydigest.config import get_settings
    settings = get_settings()
    output = args.output.resolve()
    protected = {ROOT / "data", Path(settings.db_path).resolve().parent, Path(settings.profile_path).resolve().parent}
    if output.exists() or output == ROOT / ".env" or any(output.is_relative_to(path) for path in protected):
        parser.error("--output must be new and outside configured production data/profile directories and .env")
    try:
        read = lambda p: json.loads(p.read_text(encoding="utf-8"))
        result = run_experiment(read(args.benchmark),
                                [json.loads(line) for line in args.scores.read_text().splitlines()],
                                read(args.manifest), snapshot=read(args.snapshot) if args.snapshot else None)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print("error: " + json.dumps(str(exc)))
        print('help: "Check frozen benchmark, score cache, manifest and snapshot fingerprints; no production data was written."')
        return 1
    print('status: "offline_experiment"')
    print("report: " + json.dumps(str(output)))
    print("promotion_eligible: false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
