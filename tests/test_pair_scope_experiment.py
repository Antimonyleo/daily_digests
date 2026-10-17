"""Frozen score cache integrity and calibration/evaluation separation."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from dailydigest.rank.scope import fingerprint

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/experiment_pair_scope.py"
spec = importlib.util.spec_from_file_location("experiment_pair_scope", SCRIPT)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


@pytest.fixture
def data():
    benchmark = json.loads((Path(__file__).parent / "fixtures/research_scope_benchmark.json").read_text())
    queries = experiment.pair_queries(benchmark["definitions"])
    external = [{"kind": q["label"], "facet": q["facet"] or None, "text": q["text"]} for q in queries]
    model = {"assets_sha256": {key: "a" * 64 for key in ("onnx/model.onnx", "tokenizer.json", "config.json")},
             "threads": 2, "max_length": 512, "batch_size": 16, "score_kind": "raw_logit",
             "truncation": "only_second", "providers": ["CPUExecutionProvider"],
             "onnxruntime_version": "test", "tokenizers_version": "test", "padding": "longest_in_batch"}
    model["fingerprint"] = experiment.compact_hash(model)
    manifest = {"queries": external, "query_sha256": experiment.compact_hash(external),
                "model": model, "protocol_sha256": "b" * 64}
    rows = []
    for ex in benchmark["examples"]:
        scores = [5. if q["label"] == ex["label"] and (q["label"] != "core" or q["facet"] == ex["facet"]) else -5. for q in queries]
        text = experiment.article_text(ex)
        rows.append({"id": ex["id"], "title": ex["title"], "abstract": ex["abstract"], "text": text,
                     "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "raw_logits": scores,
                     "query_sha256": manifest["query_sha256"], "protocol_sha256": manifest["protocol_sha256"],
                     "model_fingerprint": model["fingerprint"]})
    return benchmark, rows, manifest


def test_both_arms_fit_only_calibration_and_keep_groups_separate(data):
    benchmark, rows, manifest = data
    first = experiment.run_experiment(benchmark, rows, manifest)
    changed = deepcopy(benchmark)
    for ex in changed["examples"]:
        if ex["split"] == "evaluation":
            ex.update(label="off_interest", facet="", evaluation_group="fresh")
    second = experiment.run_experiment(changed, rows, manifest)
    for name in ("zeroshot", "learned"):
        a, b = first["arms"][name], second["arms"][name]
        assert a["calibration"] == b["calibration"]
        assert a["provenance"] == b["provenance"]
        assert a["benchmark_metrics"]["evaluation"] != b["benchmark_metrics"]["evaluation"]
        assert set(b["evaluation_groups"]) == {"fresh"}
        assert not a["promotion_eligible"]
    learned = first["arms"]["learned"]
    assert learned["calibration"] == learned["provenance"]["calibration_report"]
    assert "in-sample" in learned["calibration_metrics_meaning"]
    assert "OOF" in learned["calibration_metrics_meaning"]
    assert "Logistic decision scores" in learned["benchmark_predictions"][0]["decision"]["score_meaning"]


@pytest.mark.parametrize("bad", ["missing", "nan", "shape", "duplicate", "text", "query", "model", "protocol", "order"])
def test_invalid_cache_rejected_before_fitting(data, monkeypatch, bad):
    benchmark, rows, manifest = data
    def no_fit(*args, **kwargs):
        pytest.fail("Invalid cache reached fitting")
    monkeypatch.setattr(experiment, "fit_learned_scope", no_fit)
    monkeypatch.setattr(experiment, "calibrate_pair_scope", no_fit)
    if bad == "missing": rows.pop()
    elif bad == "nan": rows[0]["raw_logits"][0] = np.nan
    elif bad == "shape": rows[0]["raw_logits"].pop()
    elif bad == "duplicate": rows.append(rows[0])
    elif bad == "text": rows[0]["title"] = "tampered"
    elif bad == "query": rows[0]["query_sha256"] = "wrong"
    elif bad == "model": manifest["model"]["threads"] = 3
    elif bad == "protocol": rows[0]["protocol_sha256"] = "c" * 64
    else: manifest["queries"].reverse()
    with pytest.raises(ValueError): experiment.run_experiment(benchmark, rows, manifest)


@pytest.mark.parametrize("bad", ["profile", "facets", "text", "missing_pool"])
def test_snapshot_validation_precedes_fitting(data, bad, monkeypatch):
    benchmark, rows, manifest = data
    profile = {"canonical_facets": benchmark["definitions"]["core"]}
    metadata = {"title": "Uncached document", "abstract": "Long abstract " * 10}
    snapshot = {"profile": profile, "profile_signature": fingerprint(profile),
                "candidates": [{"item_id": 999, "eligible": True, "section": "research", "metadata": metadata,
                                "text_signature": fingerprint([metadata["title"], metadata["abstract"]])}]}
    if bad == "profile": snapshot["profile_signature"] = "wrong"
    elif bad == "facets": profile["canonical_facets"] = {}
    elif bad == "text": metadata["abstract"] += "altered"
    def no_fit(*args, **kwargs): pytest.fail("Invalid snapshot reached fitting")
    monkeypatch.setattr(experiment, "calibrate_pair_scope", no_fit)
    with pytest.raises(ValueError): experiment.run_experiment(benchmark, rows, manifest, snapshot=snapshot)


def test_cli_exclusive_output_and_offline_success(data, tmp_path, monkeypatch, capsys):
    import socket
    from types import SimpleNamespace
    import dailydigest.config
    monkeypatch.setattr(dailydigest.config, "get_settings", lambda: SimpleNamespace(
        db_path=tmp_path / "private/live.db", profile_path=tmp_path / "private/profile.yaml"))
    benchmark, rows, manifest = data
    def no_network(*args, **kwargs): pytest.fail("No network should be attempted")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    for name, value in (("benchmark", benchmark), ("manifest", manifest)):
        (tmp_path / (name + ".json")).write_text(json.dumps(value))
    (tmp_path / "scores.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    output = tmp_path / "report.json"
    args = ["--benchmark", str(tmp_path / "benchmark.json"), "--manifest", str(tmp_path / "manifest.json"),
            "--scores", str(tmp_path / "scores.jsonl"), "--output", str(output)]
    assert experiment.main(args) == 0
    assert json.loads(output.read_text())["status"] == "offline_experiment"
    before = output.read_bytes()
    with pytest.raises(SystemExit): experiment.main(args)
    assert output.read_bytes() == before
    assert "promotion_eligible: false" in capsys.readouterr().out


def test_cli_protects_configured_data_directory(data, tmp_path, monkeypatch):
    from dailydigest.config import get_settings
    monkeypatch.setenv("DB_PATH", str(tmp_path / "private/live.db"))
    get_settings.cache_clear()
    try:
        with pytest.raises(SystemExit):
            experiment.main(["--benchmark", "none", "--scores", "none", "--manifest", "none",
                             "--output", str(tmp_path / "private/new.json")])
        assert not (tmp_path / "private/new.json").exists()
    finally:
        get_settings.cache_clear()


def test_nominal_model_provenance_rejected_even_with_consistent_hashes(data):
    benchmark, rows, manifest = data
    manifest["model"] = {"fingerprint": experiment.compact_hash({})}
    for row in rows:
        row["model_fingerprint"] = manifest["model"]["fingerprint"]
    with pytest.raises(ValueError, match="asset/runtime provenance"):
        experiment.run_experiment(benchmark, rows, manifest)


def test_actual_selector_replay_refills_and_preserves_snapshot(data):
    benchmark, rows, manifest = data
    core = next(i for i, ex in enumerate(benchmark["examples"]) if ex["label"] == "core")
    off = next(i for i, ex in enumerate(benchmark["examples"]) if ex["label"] == "off_interest")
    profile = {"canonical_facets": benchmark["definitions"]["core"]}
    candidates = []
    for iid, source_index in enumerate((off, core, core), 1):
        row = deepcopy(rows[source_index]); row["id"] = str(iid); rows.append(row)
        metadata = {"title": row["title"], "abstract": row["abstract"], "source": f"Journal {iid}", "metadata_json": "{}"}
        candidates.append({"item_id": iid, "family_id": str(iid), "section": "research", "eligible": True,
                           "selected": iid <= 2, "position": iid - 1 if iid <= 2 else None, "input_position": iid - 1,
                           "ranking_score": 1 - iid / 100,
                           "features": {"topic_score": .9, "scoring_mode": "hybrid_knn", "learned_score": .8},
                           "metadata": metadata, "text_signature": fingerprint([row["title"], row["abstract"]]),
                           "selection_facts": {"source_bucket": "published_journal", "is_arxiv_cs": False,
                                               "is_preprint": False, "is_published": True, "is_high_quality": True,
                                               "is_low_impact": False}})
    snapshot = {"schema_version": 1, "run_id": "pair-test", "digest_id": "2026-10-01",
                "decision_at": "2026-10-01T12:00:00Z", "reading_mode": "full", "complete_candidate_pool": True,
                "section_budgets": {"research": 2}, "selection_settings": {"adaptive_relevance_floor": False,
                    "max_preprint_research_frac": .55, "max_low_impact_research_frac": .25,
                    "low_impact_relevance_floor": .7, "research_final_score_floor_frac": 0,
                    "research_final_score_min_keep": 0},
                "decision_local_date": "2026-10-01", "catch_up": False, "exploration_slots": 0,
                "profile": profile, "profile_signature": fingerprint(profile), "candidates": candidates}
    before = deepcopy(snapshot)
    result = experiment.run_experiment(benchmark, rows, manifest, snapshot=snapshot)
    for arm in result["arms"].values():
        assert arm["replay"]["baseline_slate"]["research"] == [1, 2]
        assert arm["replay"]["challenger_slate"]["research"] == [2, 3]
        assert arm["replay"]["selection_validation"]["status"] == "verified"
    assert snapshot == before
