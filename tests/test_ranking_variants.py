"""Offline variants use fixed contrasts and never learn from held-out judgments."""
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "experiment_ranking_variants.py"
    spec = importlib.util.spec_from_file_location("ranking_variants", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def data():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [SimpleNamespace(id=i, title=f"paper {i}", source="Nature", section="research", url="") for i in range(1, 41)]
    labels = [1 if i % 2 else -1 for i in range(40)]
    stamps = [base + timedelta(days=i // 2) for i in range(40)]
    grades = [100 if value > 0 else 10 for value in labels]
    matrices = {"legacy": np.array([[1., 0., 0.]], dtype=np.float32),
                "canonical": np.array([[0., 1., 0.]], dtype=np.float32),
                "additive": np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)}
    return rows, labels, stamps, grades, matrices


def install_embeddings(monkeypatch):
    from dailydigest.rank import embedding_cache
    def embed(rows):
        return np.asarray([[1., 0.9 if row.id % 2 == 0 else 0., 0.9 if row.id % 2 else 0.] for row in rows], dtype=np.float32)
    monkeypatch.setattr(embedding_cache, "embed_item_rows", embed)


def test_variants_are_predeclared_small_and_repeated_folds_do_not_overlap(monkeypatch):
    mod = module()
    install_embeddings(monkeypatch)
    args = data()
    first = mod.run_experiments(*args)
    second = mod.run_experiments(*args)
    assert first == second
    assert len(mod.VARIANTS) <= 9
    assert {v["representation"] for v in mod.VARIANTS} == {"legacy", "canonical", "additive"}
    assert {v["learning"] for v in mod.VARIANTS} == {"disabled", "appended", "bounded"}
    assert {v["preference_weight"] for v in mod.VARIANTS} == {0, 0.5, 0.75, 1}
    assert set(first["folds"][0]["test_ids"]).isdisjoint(first["folds"][1]["test_ids"])
    day_by_id = {row.id: stamp.date() for row, stamp in zip(args[0], args[2], strict=True)}
    assert {day_by_id[i] for i in first["folds"][0]["test_ids"]}.isdisjoint(
        {day_by_id[i] for i in first["folds"][1]["test_ids"]})
    for fold in first["folds"]:
        assert set(fold["train_ids"]).isdisjoint(fold["test_ids"])
        assert set(fold["variants"][0]["head"]) == {"5", "11", "15"}
    assert "component" in first["scope"]


def test_heldout_label_changes_do_not_change_train_learned_rankings(monkeypatch):
    mod = module()
    install_embeddings(monkeypatch)
    rows, labels, stamps, grades, matrices = data()
    first = mod.run_experiments(rows, labels, stamps, grades, matrices)
    changed = grades.copy()
    for index in range(32, 40):
        changed[index] = 70 if labels[index] > 0 else 40
    second = mod.run_experiments(rows, labels, stamps, changed, matrices)
    for before, after in zip(first["folds"], second["folds"], strict=True):
        for arm_a, arm_b in zip(before["variants"], after["variants"], strict=True):
            assert arm_a["ranking_ids"] == arm_b["ranking_ids"]
            assert arm_a["score_signature"] == arm_b["score_signature"]


def test_component_counts_within_test_versions_without_claiming_family_recall(monkeypatch):
    mod = module()
    install_embeddings(monkeypatch)
    rows, labels, stamps, grades, matrices = data()
    rows[32].metadata_json = '{"doi": "10.1234/shared-test-family"}'
    rows[34].metadata_json = rows[32].metadata_json
    result = mod.run_experiments(rows, labels, stamps, grades, matrices)
    final_fold = result["folds"][1]
    assert final_fold["test_repeated_family_count"] == 1
    assert final_fold["test_extra_family_versions"] == 1
    assert final_fold["test_family_count"] == len(final_fold["test_ids"]) - 1
    assert any("observed TEST items" in message for message in result["limitations"])


def test_tied_ndcg_uses_declared_item_id_order_under_input_permutation(monkeypatch):
    import math
    from dailydigest.rank import embedding_cache
    mod = module()
    rows, labels, stamps, grades, matrices = data()
    rows[32].id, rows[33].id = rows[33].id, rows[32].id
    monkeypatch.setattr(embedding_cache, "embed_item_rows", lambda items: np.tile(
        np.array([[1., 0., 0.]], dtype=np.float32), (len(items), 1)))
    original = mod.run_experiments(rows, labels, stamps, grades, matrices)
    permuted = mod.run_experiments(rows[::-1], labels[::-1], stamps[::-1], grades[::-1], matrices)
    label_by_id = {row.id: label for row, label in zip(rows, labels, strict=True)}
    for first, second in zip(original["folds"], permuted["folds"], strict=True):
        for arm, changed in zip(first["variants"], second["variants"], strict=True):
            assert arm["ranking_ids"] == changed["ranking_ids"]
            gains = [int(label_by_id[iid] > 0) for iid in arm["ranking_ids"]]
            actual = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains[:10]))
            ideal = sum(gain / math.log2(i + 2) for i, gain in enumerate(sorted(gains, reverse=True)[:10]))
            assert arm["observed_ndcg_at_10"] == actual / ideal
            assert arm["observed_ndcg_at_10"] == changed["observed_ndcg_at_10"]
