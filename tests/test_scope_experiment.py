"""Experiment integrity: disjoint calibration, immutable read-only inputs."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from dailydigest.rank.scope import anchor_records, fingerprint


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/experiment_scope.py"
spec = importlib.util.spec_from_file_location("experiment_scope", SCRIPT)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
FIXTURE = Path(__file__).parent / "fixtures/research_scope_benchmark.json"


def fake_encoder(benchmark):
    dimensions = list(benchmark["definitions"]["core"]) + ["context", "off_interest", "uncertain"]
    text_vectors = {}
    for record in anchor_records(benchmark["definitions"]):
        key = record["facet"] if record["label"] == "core" else record["label"]
        text_vectors[record["text"]] = np.eye(len(dimensions))[dimensions.index(key)]
    for row in benchmark["examples"]:
        text = experiment._candidate_text({"metadata": row})
        key = row["facet"] if row["label"] == "core" else row["label"]
        text_vectors[text] = np.eye(len(dimensions))[dimensions.index(key)]
    return lambda texts, is_query=False: np.array([text_vectors[text] for text in texts], dtype=np.float32)


def test_evaluation_labels_cannot_choose_thresholds():
    benchmark = json.loads(FIXTURE.read_text())
    encoder = fake_encoder(benchmark)
    first = experiment.run_experiment(benchmark, embed=encoder, document_signature="doc", query_signature="query")
    changed = deepcopy(benchmark)
    for example in changed["examples"]:
        if example["split"] == "evaluation":
            example.update(label="off_interest", facet="")
    second = experiment.run_experiment(changed, embed=encoder, document_signature="doc", query_signature="query")
    assert first["policy"]["calibration"] == second["policy"]["calibration"]
    assert first["policy"]["thresholds"] == second["policy"]["thresholds"]
    assert first["benchmark_metrics"]["evaluation"] != second["benchmark_metrics"]["evaluation"]
    assert not first["policy"]["promotion_eligible"]


@pytest.mark.parametrize("bad", ["document", "query", "profile", "text", "facets"])
def test_snapshot_fingerprint_mismatch_fails_before_embedding(bad):
    benchmark = json.loads(FIXTURE.read_text())
    profile = {"canonical_facets": benchmark["definitions"]["core"]}
    row = {"title": "Real article", "abstract": "Description"}
    snapshot = {"embedding_signature": "doc", "query_embedding_signature": "query",
                "profile": profile, "profile_signature": fingerprint(profile),
                "candidates": [{"metadata": row, "text_signature": fingerprint(list(row.values()))}]}
    if bad == "document":
        snapshot["embedding_signature"] = "changed"
    elif bad == "query":
        snapshot["query_embedding_signature"] = "changed"
    elif bad == "profile":
        snapshot["profile"]["name"] = "edited"
    elif bad == "text":
        row["abstract"] = "edited"
    else:
        snapshot["profile"]["canonical_facets"] = {}
    def no_embedding(*args, **kwargs):
        pytest.fail("invalid inputs must be rejected before embedding")
    with pytest.raises(ValueError):
        experiment.run_experiment(benchmark, snapshot=snapshot, embed=no_embedding,
                                  document_signature="doc", query_signature="query")


def test_cache_reads_are_hash_checked_and_never_written(tmp_path):
    path = tmp_path / "copy.db"
    rows = [{"item_id": i, "metadata": {"title": f"Paper {i}", "abstract": "Detailed abstract"}}
            for i in range(1, 5)]
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE item_embeddings (item_id INTEGER, model TEXT, text_hash TEXT, dim INTEGER, vector BLOB)")
        for row in rows:
            iid = row["item_id"]
            text_hash = hashlib.sha256(experiment._candidate_text(row).encode()).hexdigest()
            db.execute("INSERT INTO item_embeddings VALUES (?, ?, ?, ?, ?)",
                       (iid, "doc" if iid != 3 else "wrong-model", text_hash if iid != 2 else "stale",
                        2, np.array([1, 0] if iid != 4 else [np.nan, 0], dtype=np.float32).tobytes()))
    before = path.read_bytes()
    calls = []
    def embed(texts):
        calls.extend(texts)
        return np.tile([0., 1.], (len(texts), 1))
    vectors, counts = experiment.snapshot_vectors(rows, "doc", path, embed)
    assert counts == {"cached": 1, "embedded": 3}
    assert len(calls) == 3
    assert vectors[0].tolist() == [1, 0]
    assert vectors[1:].tolist() == [[0, 1]] * 3
    assert path.read_bytes() == before


def test_missing_cache_is_not_created(tmp_path):
    path = tmp_path / "missing.db"
    with pytest.raises(sqlite3.OperationalError):
        experiment.snapshot_vectors([], "doc", path, lambda texts: [])
    assert not path.exists()


@pytest.mark.parametrize("symlink", [False, True])
def test_cli_protects_alternate_production_directory(tmp_path, monkeypatch, capsys, symlink):
    from dailydigest.config import get_settings
    data = tmp_path / "private-data"
    data.mkdir()
    monkeypatch.setenv("DB_PATH", str(data / "live.db"))
    get_settings.cache_clear()
    target = data
    if symlink:
        target = tmp_path / "alias"
        target.symlink_to(data, target_is_directory=True)
    with pytest.raises(SystemExit) as exc:
        experiment.main(["--output", str(target / "report.json")])
    assert exc.value.code == 2
    assert "configured production" in capsys.readouterr().out
    assert not (data / "report.json").exists()


def test_cli_unknown_flags_fail_without_loading_encoder(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), "--outpu", str(tmp_path / "report.json")],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "error:" in result.stdout
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("flag", ["-v", "-V", "--version"])
def test_version_is_a_dependency_free_fast_path(flag):
    result = subprocess.run([sys.executable, "-S", str(SCRIPT), flag], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip() == experiment.VERSION
