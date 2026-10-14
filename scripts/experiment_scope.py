#!/usr/bin/env python
"""Offline contrastive scope benchmark and optional full-pool selection replay.

The bundled corpus is synthetic development evidence. No production defaults,
profiles, votes or caches are written, and no model downloads are permitted.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys


VERSION = "1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "tests/fixtures/research_scope_benchmark.json"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_text(candidate):
    metadata = candidate["metadata"]
    title, abstract = (metadata.get("title") or "").strip(), (metadata.get("abstract") or "").strip()
    return f"{title}. {abstract}" if abstract else title


def snapshot_vectors(candidates, signature, cache_db, embed):
    """Read compatible cached vectors, re-embed misses locally, never update DB."""
    import numpy as np

    texts = [_candidate_text(candidate) for candidate in candidates]
    vectors = [None] * len(candidates)
    if cache_db is not None:
        uri = Path(cache_db).resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            for start in range(0, len(candidates), 500):
                batch = candidates[start:start + 500]
                ids = [candidate["item_id"] for candidate in batch]
                marks = ",".join("?" for _ in ids)
                rows = connection.execute(
                    f"SELECT item_id, text_hash, dim, vector FROM item_embeddings WHERE model = ? AND item_id IN ({marks})",
                    [signature, *ids],
                ).fetchall()
                cached = {row[0]: row[1:] for row in rows}
                for offset, candidate in enumerate(batch, start):
                    row = cached.get(candidate["item_id"])
                    if row is None:
                        continue
                    text_hash, dim, blob = row
                    if (text_hash != hashlib.sha256(texts[offset].encode()).hexdigest()
                            or not isinstance(dim, int) or not 0 < dim <= 8192
                            or len(blob) != dim * 4):
                        continue
                    vec = np.frombuffer(blob, dtype=np.float32).copy()
                    if np.isfinite(vec).all() and np.linalg.norm(vec) > 1e-12:
                        vectors[offset] = vec
    missing = [i for i, vector in enumerate(vectors) if vector is None]
    if missing:
        fresh = embed([texts[i] for i in missing])
        for i, vector in zip(missing, fresh, strict=True):
            vectors[i] = vector
    return np.asarray(vectors, dtype=np.float32), {"cached": len(candidates) - len(missing), "embedded": len(missing)}


def run_experiment(benchmark, *, snapshot=None, cache_db=None, embed=None,
                   document_signature=None, query_signature=None):
    """Injected encoder/signatures make tests independent of models/services."""
    from dailydigest.rank.scope import (
        SCOPE_VERSION, ScopeThresholds, anchor_records, calibrate_scope, classify_scope,
        fingerprint, score_scope, scope_metrics, validate_benchmark,
    )

    validate_benchmark(benchmark)
    if embed is None:
        from dailydigest.rank.embed import active_embedding_signature, embed_texts
        embed = embed_texts
        document_signature = active_embedding_signature()
        query_signature = active_embedding_signature(is_query=True)
    if not document_signature or not query_signature:
        raise ValueError("Both document and query embedding signatures are required")
    if snapshot is not None:
        if (snapshot.get("embedding_signature") != document_signature
                or snapshot.get("query_embedding_signature") != query_signature):
            raise ValueError("Snapshot embedding signatures differ; use the captured encoder configuration")
        profile = snapshot.get("profile", {})
        if set(profile.get("canonical_facets", {})) != set(benchmark["definitions"]["core"]):
            raise ValueError("Scope definitions must cover exactly the snapshot's canonical facets")
        if snapshot.get("profile_signature") != fingerprint(profile):
            raise ValueError("Snapshot profile fingerprint does not match its contents")
        for candidate in snapshot["candidates"]:
            metadata = candidate["metadata"]
            if candidate.get("text_signature") != fingerprint([metadata.get("title"), metadata.get("abstract")]):
                raise ValueError("Snapshot text fingerprint does not match its contents")
    records = anchor_records(benchmark["definitions"])
    anchors = embed([record["text"] for record in records], is_query=True)
    examples = benchmark["examples"]
    texts = [f"{example['title'].strip()}. {example['abstract'].strip()}" if example["abstract"].strip()
             else example["title"].strip() for example in examples]
    evidence = score_scope(embed(texts), anchors, records)
    calibration_indexes = [i for i, example in enumerate(examples) if example["split"] == "calibration"]
    fitted = calibrate_scope([evidence[i] for i in calibration_indexes],
                            [examples[i] for i in calibration_indexes])
    calibrated_facets = {examples[i]["facet"] for i in calibration_indexes if examples[i]["label"] == "core"}
    fitted["unsupported_core_facets"] = sorted(set(benchmark["definitions"]["core"]) - calibrated_facets)
    fitted["evidence_warnings"] = []
    if fitted["metrics"]["label_counts"]["core"] < 10:
        fitted["evidence_warnings"].append("Fewer than ten labeled core calibration examples; recall/precision estimates are fragile")
    if fitted["unsupported_core_facets"]:
        fitted["evidence_warnings"].append("Some core facets have no positive calibration examples")
    thresholds = ScopeThresholds(**fitted["thresholds"])
    decisions = [classify_scope(ev, example["abstract"], thresholds)
                 for ev, example in zip(evidence, examples, strict=True)]
    splits = {}
    for split in ("calibration", "evaluation"):
        indexes = [i for i, example in enumerate(examples) if example["split"] == split]
        splits[split] = scope_metrics([examples[i] for i in indexes], [decisions[i] for i in indexes])
    policy = {"version": SCOPE_VERSION, "definition_signature": fingerprint(benchmark["definitions"]),
              "benchmark_signature": fingerprint(benchmark), "provenance": benchmark["provenance"],
              "document_embedding_signature": document_signature, "query_embedding_signature": query_signature,
              "anchor_vector_signature": hashlib.sha256(anchors.tobytes()).hexdigest(),
              "calibration": fitted, "thresholds": asdict(thresholds), "promotion_eligible": False}
    result = {"schema_version": 1, "status": "offline_experiment", "policy": policy,
              "definitions": benchmark["definitions"], "benchmark_metrics": splits,
              "benchmark_predictions": [{"id": example["id"], "split": example["split"],
                                          "expected": example["label"], "expected_facet": example["facet"],
                                          "decision": decision}
                                         for example, decision in zip(examples, decisions, strict=True)],
              "limitations": ["Benchmark scope labels are development judgments, not reader usefulness or scientific-quality ratings; see provenance.",
                              "Thresholds are selected on calibration examples only; no production promotion is implied.",
                              "The abstract-length evidence rule is a heuristic, not a quality judgment.",
                              "Existing upstream eligibility and rank scores are frozen in the optional replay."]}
    groups = sorted({example.get("evaluation_group", "evaluation") for example in examples
                     if example["split"] == "evaluation"})
    result["evaluation_groups"] = {}
    for group in groups:
        indexes = [i for i, example in enumerate(examples)
                   if example["split"] == "evaluation" and example.get("evaluation_group", "evaluation") == group]
        result["evaluation_groups"][group] = scope_metrics(
            [examples[i] for i in indexes], [decisions[i] for i in indexes])
    if snapshot is not None:
        from dailydigest.rank.scope_replay import compare_scope_slates
        candidates = [candidate for candidate in snapshot["candidates"]
                      if candidate["eligible"] and candidate["section"] == "research"]
        if candidates:
            vectors, cache = snapshot_vectors(candidates, document_signature, cache_db, embed)
            observed = score_scope(vectors, anchors, records)
            scope_decisions = {
                candidate["item_id"]: classify_scope(ev, candidate["metadata"].get("abstract") or "", thresholds)
                for candidate, ev in zip(candidates, observed, strict=True)
            }
        else:
            cache, scope_decisions = {"cached": 0, "embedded": 0}, {}
        result["replay"] = compare_scope_slates(snapshot, scope_decisions, policy=policy)
        result["replay"]["embedding_cache"] = cache
        result["replay"]["profile_signature"] = snapshot["profile_signature"]
        selected_ids = set(result["replay"]["baseline_slate"].get("research", [])) | set(result["replay"]["challenger_slate"].get("research", []))
        result["replay"]["selected_metadata"] = {candidate["item_id"]: {
            "title": candidate["metadata"]["title"], "source": candidate["metadata"]["source"],
            "text_signature": candidate["text_signature"], "decision": scope_decisions[candidate["item_id"]],
        } for candidate in candidates if candidate["item_id"] in selected_ids}
    return result


class Parser(argparse.ArgumentParser):
    def error(self, message):
        print("error: " + json.dumps(message))
        print('help: "Use --benchmark FILE [--snapshot FILE --cache-db COPY.db] --output NEW.json; --help lists all options."')
        raise SystemExit(2)


def main(argv=None):
    parser = Parser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK, help="Labeled definitions/examples JSON")
    parser.add_argument("--snapshot", type=Path, help="Exported complete local ranking snapshot JSON")
    parser.add_argument("--cache-db", type=Path, help="Optional copied SQLite embedding cache; opened read-only")
    parser.add_argument("--output", type=Path, required=True, help="New report JSON outside production data; never overwritten")
    parser.add_argument("-v", "-V", "--version", action="version", version=VERSION)
    args = parser.parse_args(argv)
    if args.cache_db and not args.snapshot:
        parser.error("--cache-db requires --snapshot")
    output = args.output.resolve()
    from dailydigest.config import get_settings
    settings = get_settings()
    protected = {ROOT / "data", Path(settings.db_path).resolve().parent,
                 Path(settings.profile_path).resolve().parent}
    if output.exists() or output == ROOT / ".env" or any(output.is_relative_to(path) for path in protected):
        parser.error("--output must be a new report file outside configured production data/profile directories and .env")
    for path in (args.benchmark, args.snapshot, args.cache_db):
        if path is not None and not path.is_file():
            parser.error(f"Input file does not exist: {path}")
    # Offline flags prevent fallback downloads; socket guard catches other routes.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import socket
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def no_network(*_args, **_kwargs):
        raise RuntimeError("This experiment requires a locally cached model; network access is disabled")

    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network
    try:
        benchmark = _read_json(args.benchmark)
        snapshot = _read_json(args.snapshot) if args.snapshot else None
        result = run_experiment(benchmark, snapshot=snapshot, cache_db=args.cache_db)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also protects inputs if another process creates this path.
        with output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, sqlite3.Error) as exc:
        print("error: " + json.dumps(str(exc)))
        print('help: "Check input schemas, captured encoder settings and locally cached model; no production data was written."')
        return 1
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
    print('status: "offline_experiment"')
    print("report: " + json.dumps(str(output)))
    print("evaluation_core_recall: " + json.dumps(result["benchmark_metrics"]["evaluation"]["core_recall"]))
    print("evaluation_false_core: " + str(result["benchmark_metrics"]["evaluation"]["false_core"]))
    print("promotion_eligible: false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
