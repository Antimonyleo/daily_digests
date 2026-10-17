"""Offline reranking must alter real selection without changing gates or training."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from dailydigest.rank.scope import fingerprint
from scripts.experiment_pair_scope import article_text, compact_hash, pair_queries


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/experiment_contribution_ranker.py"
spec = importlib.util.spec_from_file_location("contribution_experiment", SCRIPT)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


@pytest.fixture
def data():
    benchmark = json.loads((Path(__file__).parent / "fixtures/research_scope_benchmark.json").read_text())
    queries = pair_queries(benchmark["definitions"])
    external = [{"kind": q["label"], "facet": q["facet"] or None, "text": q["text"]} for q in queries]
    model = {"assets_sha256": {key: "a" * 64 for key in ("onnx/model.onnx", "tokenizer.json", "config.json")},
             "threads": 2, "max_length": 512, "batch_size": 16, "score_kind": "raw_logit",
             "truncation": "only_second", "providers": ["CPUExecutionProvider"],
             "onnxruntime_version": "test", "tokenizers_version": "test", "padding": "longest_in_batch"}
    model["fingerprint"] = compact_hash(model)
    manifest = {"queries": external, "query_sha256": compact_hash(external), "model": model,
                "protocol_sha256": "b" * 64}

    def cache(iid, title, abstract, label, facet=""):
        text = article_text({"title": title, "abstract": abstract})
        return {"id": str(iid), "title": title, "abstract": abstract, "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "raw_logits": [5. if q["label"] == label and (label != "core" or q["facet"] == facet) else -5. for q in queries],
                "query_sha256": manifest["query_sha256"], "protocol_sha256": manifest["protocol_sha256"],
                "model_fingerprint": model["fingerprint"]}

    rows = [cache(ex["id"], ex["title"], ex["abstract"], ex["label"], ex.get("facet", "")) for ex in benchmark["examples"]]
    candidates = []
    for iid in range(1, 16):
        label = "off_interest" if iid <= 5 else "context" if iid <= 10 else "core"
        title = f"Independent captured contribution research study number {iid} with a distinct family"
        abstract = "This supplied abstract describes the contribution and includes enough experimental evidence for classification."
        if iid == 1:
            abstract = "   "
        rows.append(cache(iid, title, abstract, label, next(iter(benchmark["definitions"]["core"]))))
        metadata = {"title": title, "abstract": abstract, "source": f"Journal {iid}", "metadata_json": "{}"}
        candidates.append({"item_id": iid, "family_id": f"captured-{iid}", "section": "research", "eligible": True,
                           "selected": iid <= 11, "position": iid - 1 if iid <= 11 else None, "input_position": iid - 1,
                           "ranking_score": 1 - iid * .015,
                           "features": {"topic_score": .9, "final_score": 1 - iid * .015, "other_feature": .4},
                           "metadata": metadata, "text_signature": fingerprint([title, abstract]),
                           "selection_facts": {"source_bucket": "published_journal", "is_arxiv_cs": False,
                                               "is_preprint": False, "is_published": True,
                                               "is_high_quality": True, "is_low_impact": False}})
    profile = {"bio": "Engineering and transferable prediction methods",
               "canonical_facets": benchmark["definitions"]["core"]}
    snapshot = {"schema_version": 1, "run_id": "contribution-test", "digest_id": "2026-10-01",
                "decision_at": "2026-10-01T12:00:00Z", "reading_mode": "full", "complete_candidate_pool": True,
                "section_budgets": {"research": 11}, "selection_settings": {"adaptive_relevance_floor": False,
                    "max_preprint_research_frac": .55, "max_low_impact_research_frac": .25,
                    "low_impact_relevance_floor": .7, "research_final_score_floor_frac": 0,
                    "research_final_score_min_keep": 0},
                "decision_local_date": "2026-10-01", "catch_up": False, "exploration_slots": 0,
                "profile": profile, "profile_signature": fingerprint(profile), "candidates": candidates}
    return benchmark, rows, manifest, snapshot


def test_actual_selector_changes_order_and_retains_missing_abstract(data, monkeypatch):
    benchmark, rows, manifest, snapshot = data
    before = deepcopy(snapshot)
    observed = []
    original = experiment.frozen_selection

    def inspect_frozen(captured):
        observed.append(deepcopy(captured))
        return original(captured)

    monkeypatch.setattr(experiment, "frozen_selection", inspect_frozen)
    report = experiment.run_experiment(benchmark, rows, manifest, snapshot=snapshot)
    assert snapshot == before
    assert report["baseline_slate"]["research"] == list(range(1, 12))
    assert report["challenger_slate"]["research"][1:6] == list(range(11, 16))
    assert report["baseline_slate"] != report["challenger_slate"]
    assert report["matched_budget_prefixes"]["matched_k"] == 11
    assert report["scope_counts"]["challenger"]["unknown"] == 11
    assert report["fresh_evaluation"]["status"] == "not_supplied"
    assert len(observed) == 2
    for original_row, adjusted in zip(before["candidates"], observed[-1]["candidates"], strict=True):
        assert adjusted["eligible"] == original_row["eligible"]
        assert adjusted["selection_facts"] == original_row["selection_facts"]
        assert adjusted["features"]["topic_score"] == original_row["features"]["topic_score"]
        assert adjusted["features"]["final_score"] == adjusted["ranking_score"]
        if original_row["item_id"] == 1:
            assert adjusted["ranking_score"] == original_row["ranking_score"]
    by_id = {row["item_id"]: row for row in report["candidate_scores"]}
    assert by_id[1]["abstract_usable"] is False
    assert by_id[2]["challenger_score"] == pytest.approx(.7 * by_id[2]["rank_signal"] + .3 * by_id[2]["original_score"])
    envelope = report["artifact"]
    assert envelope["kind"] == "local_contribution_reranker"
    assert envelope["fingerprint"] == fingerprint({k: v for k, v in envelope.items() if k != "fingerprint"})


def test_evaluation_and_fresh_labels_never_choose_model_or_scale(data):
    benchmark, rows, manifest, snapshot = data
    first = experiment.run_experiment(benchmark, rows, manifest, snapshot=snapshot)
    changed = deepcopy(benchmark)
    for ex in changed["examples"]:
        if ex["split"] == "evaluation":
            ex.update(label="off_interest", facet="", evaluation_group="changed")
    fresh = {"provenance": {"kind": "independent_analyst"}, "examples": [
        {"id": str(i), "family_id": f"captured-{i}", "label": label}
        for i, label in ((2, "off_interest"), (6, "context"), (12, "core"))
    ]}
    second = experiment.run_experiment(changed, rows, manifest, snapshot=snapshot, fresh_labels=fresh)
    assert first["artifact"] == second["artifact"]
    assert first["challenger_slate"] == second["challenger_slate"]
    assert first["evaluation"] != second["evaluation"]
    assert set(second["evaluation_groups"]) == {"changed"}
    assert second["fresh_evaluation"]["metrics"]["balanced_pair_accuracy"] == 1
    assert second["fresh_evaluation"]["metrics"]["definitive_family_count"] == 3
    assert second["fresh_evaluation"]["baseline_final_score_metrics"]["balanced_pair_accuracy"] == 0
    assert second["fresh_evaluation"]["challenger_final_score_metrics"]["balanced_pair_accuracy"] == 1
    assert second["scope_counts"]["baseline"]["unknown"] == 9


def test_list_form_fresh_labels_preserves_supplied_judgment_provenance(data):
    benchmark, rows, manifest, snapshot = data
    fresh = [{"id": "12", "family_id": "captured-12", "label": "core",
              "judgment_provenance": "Independent analyst; not reader usefulness"}]
    report = experiment.run_experiment(benchmark, rows, manifest, snapshot=snapshot, fresh_labels=fresh)
    assert report["fresh_evaluation"]["signature"] == fingerprint(fresh)
    assert report["fresh_evaluation"]["metrics"]["definitive_family_count"] == 1
    assert report["fresh_evaluation"]["provenance"]["source_shape"] == "list"


def test_captured_topic_priority_bonus_reorders_updated_scores(data, monkeypatch):
    benchmark, rows, manifest, snapshot = data
    snapshot["candidates"][1]["features"]["selection_order_bonus"] = .8
    captured = []
    original = experiment.frozen_selection
    def inspect_frozen(snapshot):
        captured.append(deepcopy(snapshot))
        return original(snapshot)
    monkeypatch.setattr(experiment, "frozen_selection", inspect_frozen)
    report = experiment.run_experiment(benchmark, rows, manifest, snapshot=snapshot)
    assert report["baseline_slate"]["research"] == list(range(1, 12))
    assert captured[-1]["candidates"][1]["input_position"] == 0
    assert 2 in report["challenger_slate"]["research"]
    assert any("bonuses remain frozen" in note for note in report["limitations"])


@pytest.mark.parametrize("bad", ["profile", "text", "parity", "cache", "query", "model"])
def test_invalid_inputs_rejected_before_fit(data, monkeypatch, bad):
    benchmark, rows, manifest, snapshot = data
    if bad == "profile": snapshot["profile_signature"] = "wrong"
    elif bad == "text": snapshot["candidates"][0]["metadata"]["title"] = "changed"
    elif bad == "parity": snapshot["candidates"][0]["position"] = 12
    elif bad == "cache": rows.pop()
    elif bad == "query": manifest["queries"].reverse()
    else: manifest["model"]["threads"] += 1
    def no_fit(*args, **kwargs):
        pytest.fail("Invalid evidence reached fitting")
    monkeypatch.setattr(experiment, "fit_contribution_ranker", no_fit)
    with pytest.raises(ValueError):
        experiment.run_experiment(benchmark, rows, manifest, snapshot=snapshot)


@pytest.mark.parametrize("bad", ["overlap", "text", "duplicate", "missing", "unknown_label", "conflict"])
def test_fresh_labels_are_separate_valid_captured_families(data, bad):
    benchmark, rows, manifest, snapshot = data
    fresh = {"provenance": {"kind": "analyst"}, "examples": [{"id": "2", "label": "context", "family_id": "captured-2"}]}
    if bad == "overlap": fresh["examples"][0]["family_id"] = benchmark["examples"][0]["family_id"]
    elif bad == "text": fresh["examples"][0]["title"] = "wrong title"
    elif bad == "duplicate": fresh["examples"].append(deepcopy(fresh["examples"][0]))
    elif bad == "missing": fresh["examples"][0]["id"] = "999999"
    elif bad == "unknown_label": fresh["examples"][0]["label"] = "unknown"
    else: fresh["examples"].append({"id": "3", "family_id": "captured-2", "label": "core"})
    with pytest.raises(ValueError):
        experiment.run_experiment(benchmark, rows, manifest, snapshot=snapshot, fresh_labels=fresh)


def test_cli_offline_outputs_new_only_and_protected(data, tmp_path, monkeypatch, capsys):
    import socket
    from types import SimpleNamespace
    import dailydigest.config
    monkeypatch.setattr(dailydigest.config, "get_settings", lambda: SimpleNamespace(
        db_path=tmp_path / "private/live.db", profile_path=tmp_path / "private/profile.yaml"))
    def no_network(*args, **kwargs):
        pytest.fail("Offline cache replay attempted networking")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    benchmark, rows, manifest, snapshot = data
    for name, payload in (("benchmark", benchmark), ("manifest", manifest), ("snapshot", snapshot)):
        (tmp_path / (name + ".json")).write_text(json.dumps(payload))
    (tmp_path / "scores.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    report, artifact = tmp_path / "report.json", tmp_path / "artifact.json"
    args = [argument for name in ("benchmark", "manifest", "snapshot") for argument in ("--" + name, str(tmp_path / (name + ".json")))]
    args += ["--scores", str(tmp_path / "scores.jsonl"), "--output", str(report), "--artifact", str(artifact)]
    assert experiment.main(args) == 0
    assert json.loads(report.read_text())["artifact"] == json.loads(artifact.read_text())
    before = report.read_bytes(), artifact.read_bytes()
    with pytest.raises(SystemExit): experiment.main(args)
    assert (report.read_bytes(), artifact.read_bytes()) == before
    assert "promotion_eligible: false" in capsys.readouterr().out
    args[-1] = str(tmp_path / "private/new.json")
    args[-3] = str(tmp_path / "second.json")
    with pytest.raises(SystemExit): experiment.main(args)
    assert not (tmp_path / "private/new.json").exists()
    assert not (tmp_path / "second.json").exists()


def test_direct_script_help_version_and_strict_flags():
    for flag in ("--help", "--version"):
        result = subprocess.run([sys.executable, str(SCRIPT), flag], capture_output=True, text=True)
        assert result.returncode == 0
        assert "Traceback" not in result.stderr
    with pytest.raises(SystemExit) as exc:
        experiment.main(["--unknown"])
    assert exc.value.code == 2
