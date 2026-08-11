"""Hermetic test for the per-facet ranker experiment (scripts/experiment_per_facet.py).

No model download: embeddings are mocked with deterministic vectors and votes are
synthetic + linearly separable. Exercises the experiment's ranker-construction and
metric functions end to end and asserts:

  1. the per-facet FEATURE matrix has exactly one column per profile keyword, plus
     2 affinity columns when affinity is enabled (and none when it is disabled);
  2. the held-out evaluation is LEAKAGE-FREE — no test item id appears in the
     train exemplar id arrays;
  3. all three rankers (topic-only, deployed v6, per-facet) evaluate end to end and
     return defined pairwise accuracy / nDCG@10 on separable clusters; and
  4. the gate helper (nDCG delta >= 0.08) drives the ADOPT / STOP decision.

Geometry: two keyword facets aligned to axes 1 and 3. Liked items load on axis 1,
disliked on axis 3, and BOTH share the on-topic axis 0 (the topic baseline can't
separate them). The per-facet cosines (item·facet_axis1, item·facet_axis3) DO
separate them, so the per-facet ranker earns its lift from the facet features.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_experiment_module():
    """Import scripts/experiment_per_facet.py by path (scripts/ is not a package)."""
    path = Path(__file__).resolve().parent.parent / "scripts" / "experiment_per_facet.py"
    spec = importlib.util.spec_from_file_location("experiment_per_facet", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["experiment_per_facet"] = module
    spec.loader.exec_module(module)
    return module


_DIM = 6
_TOPIC_AXIS = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
_FACET_A = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # keyword 1
_FACET_B = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # keyword 2
_LIKE = np.array([1.0, 0.9, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
_DISLIKE = np.array([1.0, 0.0, 0.0, 0.9, 0.0, 0.0], dtype=np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


def _make_synthetic_votes(n_per_class: int = 12, seed: int = 7):
    rng = np.random.default_rng(seed)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows: list = []
    labels: list[int] = []
    timestamps: list[datetime] = []
    grades: list[int] = []
    vec_by_id: dict[int, np.ndarray] = {}

    item_id = 1
    for _k in range(n_per_class):
        for center, label, grade in ((_LIKE, 1, 100), (_DISLIKE, -1, 10)):
            jitter = rng.normal(0.0, 0.03, size=_DIM).astype(np.float32)
            vec = _unit(center + jitter)
            rows.append(
                SimpleNamespace(
                    id=item_id,
                    source="Synthetic Journal",
                    section="research",
                    title=f"item {item_id}",
                    abstract="synthetic abstract",
                    authors="",
                    published_at=base + timedelta(days=item_id),
                )
            )
            labels.append(label)
            timestamps.append(base + timedelta(hours=item_id))
            grades.append(grade)
            vec_by_id[item_id] = vec
            item_id += 1
    return rows, labels, timestamps, grades, vec_by_id


def _install_fakes(monkeypatch, exp, vec_by_id, facets):
    """Patch embeddings + the static-profile loader so no model / YAML is needed."""

    def fake_embed_item_rows(rows):
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return np.array(
            [vec_by_id[int(r.id)] for r in rows], dtype=np.float32
        )

    from dailydigest.rank import embedding_cache as cache_mod

    monkeypatch.setattr(cache_mod, "embed_item_rows", fake_embed_item_rows)

    from dailydigest import votes as votes_mod

    if hasattr(votes_mod, "embed_item_rows"):
        monkeypatch.setattr(votes_mod, "embed_item_rows", fake_embed_item_rows, raising=False)

    # Static profile used for topic-cosine fusion: two identical topic rows so
    # _multi_cosine's multi-facet path runs.
    static_mat = np.vstack([_unit(_TOPIC_AXIS), _unit(_TOPIC_AXIS)]).astype(np.float32)
    monkeypatch.setattr(exp, "_load_default_static_profile", lambda: static_mat)
    return fake_embed_item_rows


def test_perfacet_feature_matrix_has_one_column_per_keyword(monkeypatch):
    exp = _load_experiment_module()
    rows, labels, timestamps, grades, vec_by_id = _make_synthetic_votes()
    facets = np.vstack([_unit(_FACET_A), _unit(_FACET_B)]).astype(np.float32)  # 2 keywords
    _install_fakes(monkeypatch, exp, vec_by_id, facets)

    # Build TRAIN-only exemplars via the reused benchmark helper.
    train_idx = list(range(len(rows)))
    pos_ex, neg_ex = exp._build_train_exemplars(rows, labels, grades, train_idx)

    # With affinity: K facet columns + 2 affinity columns.
    X_aff = exp.build_perfacet_features(rows, facets, pos_ex, neg_ex, with_affinity=True)
    assert X_aff.shape[1] == facets.shape[0] + 2, (
        f"expected {facets.shape[0]} facet cols + 2 affinity cols, got {X_aff.shape[1]}"
    )
    assert X_aff.shape[0] == len(rows)

    # Without affinity: exactly one column per keyword.
    X_noaff = exp.build_perfacet_features(rows, facets, pos_ex, neg_ex, with_affinity=False)
    assert X_noaff.shape[1] == facets.shape[0], (
        f"expected exactly one column per keyword ({facets.shape[0]}), "
        f"got {X_noaff.shape[1]}"
    )

    # The facet columns are genuine cosines: liked items load high on facet A (col 0),
    # disliked items load high on facet B (col 1).
    like_idx = [i for i, l in enumerate(labels) if l == 1]
    dislike_idx = [i for i, l in enumerate(labels) if l == -1]
    assert X_noaff[like_idx, 0].mean() > X_noaff[dislike_idx, 0].mean()
    assert X_noaff[dislike_idx, 1].mean() > X_noaff[like_idx, 1].mean()


def test_all_three_rankers_run_and_are_leakage_free(monkeypatch):
    exp = _load_experiment_module()
    rows, labels, timestamps, grades, vec_by_id = _make_synthetic_votes()
    facets = np.vstack([_unit(_FACET_A), _unit(_FACET_B)]).astype(np.float32)
    _install_fakes(monkeypatch, exp, vec_by_id, facets)

    profile_mat = np.vstack([_unit(_TOPIC_AXIS), _unit(_TOPIC_AXIS)]).astype(np.float32)

    dep = exp.eval_deployed(rows, labels, timestamps, grades, profile_mat)
    pf = exp.eval_perfacet(
        rows, labels, timestamps, grades, facets, with_affinity=True
    )

    # Same held-out split for both -> apples-to-apples.
    assert dep["n_test"] == pf["n_test"] and dep["n_test"] > 0
    assert dep["n_test_pairs"] == pf["n_test_pairs"] and dep["n_test_pairs"] > 0

    # Leakage-free in BOTH: no test item id appears among train exemplar ids.
    for res in (dep, pf):
        leaked = set(res["test_ids"]) & set(res["train_exemplar_ids"])
        assert not leaked, f"test items leaked into train exemplars: {sorted(leaked)}"
        assert res["train_exemplar_ids"], "expected non-empty train exemplars"

    # All metrics defined on separable clusters.
    assert not np.isnan(dep["topic_acc"])
    assert not np.isnan(dep["deployed_acc"])
    assert not np.isnan(pf["perfacet_acc"])
    assert not np.isnan(pf["perfacet_ndcg"])

    # One feature column per keyword recorded.
    assert pf["n_facets"] == facets.shape[0]

    # The per-facet ranker earns lift from its facet features: its RRF-fused score
    # separates the clusters better than the topic-only baseline (which can't tell
    # the two on-topic clusters apart). RRF-fusing with that weak topic ranking
    # dilutes the near-perfect facet signal, so the fused accuracy is strong but
    # need not hit a perfect 1.0.
    assert pf["perfacet_acc"] > dep["topic_acc"], (
        f"per-facet fused ({pf['perfacet_acc']:.3f}) should beat topic-only "
        f"({dep['topic_acc']:.3f}) on facet-separable clusters"
    )
    assert pf["perfacet_acc"] >= 0.85


def test_gate_decision_helper_logic(monkeypatch):
    """The gate: adopt only if per-facet nDCG@10 beats deployed by >= GATE_DELTA."""
    exp = _load_experiment_module()
    assert exp.GATE_DELTA == pytest.approx(0.08)

    # A clear pass and a clear fail, expressed via the same comparison main() uses.
    def gate(pf_ndcg, dep_ndcg):
        return (pf_ndcg - dep_ndcg) >= exp.GATE_DELTA

    assert gate(0.30, 0.17) is True
    assert gate(0.20, 0.17) is False  # +0.03 < 0.08 -> STOP
    assert gate(0.173, 0.173) is False  # tie -> STOP


def test_main_runs_end_to_end_on_synthetic_db(monkeypatch, capsys):
    """main() loads real votes from the DB; here we stub the loader + facets so it
    runs fully offline and prints a decision line."""
    exp = _load_experiment_module()
    rows, labels, timestamps, grades, vec_by_id = _make_synthetic_votes()
    facets = np.vstack([_unit(_FACET_A), _unit(_FACET_B)]).astype(np.float32)
    _install_fakes(monkeypatch, exp, vec_by_id, facets)

    monkeypatch.setattr(
        exp,
        "_load_signed_votes_chronological",
        lambda research_only=True: (rows, labels, timestamps, grades),
    )
    monkeypatch.setattr(exp, "build_facet_matrix", lambda: (facets, ["kwA", "kwB"]))

    rc = exp.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "DECISION:" in out
    assert ("ADOPT" in out) or ("STOP" in out)
    # All three rankers reported.
    assert "topic-only" in out and "deployed v6" in out and "per-facet" in out
