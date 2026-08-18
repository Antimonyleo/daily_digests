"""Reproducible-evidence artifact for the preference ranker.

Runs the committed leakage-free chronological benchmark
(`scripts/benchmark_ranker.py`) on a SMALL SYNTHETIC vote set with deterministic
embeddings (no model download), and asserts:

  1. the preference-feature probe's HELD-OUT pairwise accuracy beats the
     topic-cosine-only baseline on clearly-separable liked/disliked clusters;
  2. the PRODUCTION-FAITHFUL mode (graded kNN preference + RRF fuse) also beats topic-only
     on the same clusters and reports a defined nDCG@10;
  3. exemplar construction is leakage-free in BOTH modes — no test-set item id
     appears in the train exemplar id arrays; and
  4. the benchmark's DEFAULT profile path is STATIC — it calls
     `rank.profile.build_profile_matrix`, NOT
     `build_profile_matrix_with_rocchio` (which loads a vote-derived
     `learned_profile.npz` rebuilt from ALL votes and would leak the held-out
     test votes into the profile).

The real-DB numbers are reported by `scripts/benchmark_ranker.py` and depend on
live votes.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_benchmark_module():
    """Import scripts/benchmark_ranker.py by path (scripts/ is not a package)."""
    path = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_ranker.py"
    spec = importlib.util.spec_from_file_location("benchmark_ranker", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_ranker"] = module
    spec.loader.exec_module(module)
    return module


# Geometry chosen so TOPIC-COSINE is a WEAK discriminator but the full model
# (via neg_affinity) is a STRONG one — the exact case the memory model exists for:
# an on-profile-but-unwanted subtopic. Both clusters sit high on axis 0 (the
# profile direction, "on-topic"), so profile cosine cannot tell them apart. They
# differ only on their signature axes (1 = liked, 3 = disliked), which the pos/neg
# exemplar affinity keys off. A ranker with only topic-cosine hovers near chance;
# the full model separates the clusters cleanly.
_DIM = 6
_PROFILE_AXIS = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
_LIKE = np.array([1.0, 0.9, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
_DISLIKE = np.array([1.0, 0.0, 0.0, 0.9, 0.0, 0.0], dtype=np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def _make_synthetic_votes(n_per_class: int = 12, seed: int = 7):
    """Build (rows, labels, timestamps, grades, vec_by_id) for n items per class.

    Each item gets a stable integer id and a deterministic embedding = cluster
    center + small jitter, so runs are reproducible and clusters stay separable.
    """
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
            # Interleave chronologically so the newer-25% test split has both signs.
            timestamps.append(base + timedelta(hours=item_id))
            grades.append(grade)
            vec_by_id[item_id] = vec
            item_id += 1
    return rows, labels, timestamps, grades, vec_by_id


def _install_fake_embeddings(monkeypatch, vec_by_id: dict[int, np.ndarray]):
    """Patch embed_item_rows everywhere it is imported to serve our fixed vectors."""

    def fake_embed_item_rows(rows):
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return np.array(
            [vec_by_id[int(r.id)] for r in rows], dtype=np.float32
        )

    # The benchmark imports embed_item_rows locally from embedding_cache in
    # several helpers, so patch the source module.
    from dailydigest.rank import embedding_cache as cache_mod

    monkeypatch.setattr(cache_mod, "embed_item_rows", fake_embed_item_rows)
    # votes._affinity/_pack use their own local import too — patch that name if
    # already bound on the module.
    from dailydigest import votes as votes_mod

    if hasattr(votes_mod, "embed_item_rows"):
        monkeypatch.setattr(votes_mod, "embed_item_rows", fake_embed_item_rows, raising=False)
    return fake_embed_item_rows


def test_full_model_beats_topic_baseline_and_is_leakage_free(monkeypatch):
    bench = _load_benchmark_module()
    rows, labels, timestamps, grades, vec_by_id = _make_synthetic_votes()
    _install_fake_embeddings(monkeypatch, vec_by_id)

    # Profile points at the shared on-topic axis (axis 0), NOT the liked/disliked
    # signature axes — so topic-cosine cannot separate the clusters and the full
    # model must earn its lift from neg_affinity. Two identical rows so
    # _multi_cosine's multi-facet path runs. Passed explicitly to avoid a profile
    # file / model download.
    profile_mat = np.vstack([_unit(_PROFILE_AXIS), _unit(_PROFILE_AXIS)]).astype(np.float32)

    result = bench.run_benchmark(
        rows,
        labels,
        timestamps,
        grades=grades,
        profile_mat=profile_mat,
        train_frac=0.75,
        random_state=0,
    )

    # Held-out set is non-empty and has pairs to score.
    assert result["n_test"] > 0
    assert result["n_test_pairs"] > 0

    # (2) Leakage-free: no test item id appears among the train exemplar ids.
    leaked = set(result["test_ids"]) & set(result["train_exemplar_ids"])
    assert not leaked, f"test items leaked into train exemplars: {sorted(leaked)}"
    assert result["train_exemplar_ids"], "expected non-empty train exemplars"

    # (1) Probe's held-out pairwise accuracy > topic-only baseline.
    assert not np.isnan(result["topic_acc"])
    assert not np.isnan(result["full_acc"])
    assert result["full_acc"] > result["topic_acc"], (
        f"probe ({result['full_acc']:.3f}) should beat topic-only "
        f"({result['topic_acc']:.3f}) on separable synthetic clusters"
    )
    # On cleanly-separated clusters the probe should be near-perfect.
    assert result["full_acc"] >= 0.9


def test_production_benchmark_beats_topic_and_is_leakage_free(monkeypatch):
    """Mode B: graded kNN + RRF fuse beats topic-only, leakage-free, with nDCG@10."""
    bench = _load_benchmark_module()
    rows, labels, timestamps, grades, vec_by_id = _make_synthetic_votes()
    _install_fake_embeddings(monkeypatch, vec_by_id)

    profile_mat = np.vstack([_unit(_PROFILE_AXIS), _unit(_PROFILE_AXIS)]).astype(np.float32)

    result = bench.run_production_benchmark(
        rows,
        labels,
        timestamps,
        grades=grades,
        profile_mat=profile_mat,
        train_frac=0.75,
    )

    assert result["n_test"] > 0
    assert result["n_test_pairs"] > 0

    # Leakage-free: no test item id in the train exemplar ids.
    leaked = set(result["test_ids"]) & set(result["train_exemplar_ids"])
    assert not leaked, f"test items leaked into train exemplars: {sorted(leaked)}"

    # The deployed-style (pairwise + RRF-fused) ranker beats topic-only, and both
    # metrics are defined on the separable clusters.
    assert not np.isnan(result["topic_acc"])
    assert not np.isnan(result["fused_acc"])
    assert result["fused_acc"] > result["topic_acc"], (
        f"fused ({result['fused_acc']:.3f}) should beat topic-only "
        f"({result['topic_acc']:.3f}) on separable synthetic clusters"
    )
    assert not np.isnan(result["fused_ndcg"])
    assert result["fused_ndcg"] >= result["topic_ndcg"] or np.isnan(result["topic_ndcg"])
    # Cleanly separated clusters -> a good top-10.
    assert result["fused_ndcg"] >= 0.9


def test_default_profile_path_is_static_not_rocchio(monkeypatch):
    """The default (profile_mat=None) path must use the STATIC profile matrix.

    Guards the third-audit fix: the benchmark must call
    `rank.profile.build_profile_matrix` and must NOT call
    `build_profile_matrix_with_rocchio` (which loads the vote-derived
    learned_profile.npz rebuilt from ALL votes, leaking the held-out test set).
    """
    bench = _load_benchmark_module()
    rows, labels, timestamps, grades, vec_by_id = _make_synthetic_votes(n_per_class=8)
    _install_fake_embeddings(monkeypatch, vec_by_id)

    import dailydigest.rank.profile as profile_mod

    calls = {"static": 0, "rocchio": 0}
    static_mat = np.vstack(
        [_unit(_PROFILE_AXIS), _unit(_PROFILE_AXIS)]
    ).astype(np.float32)

    def spy_static(profile):
        calls["static"] += 1
        return static_mat

    def spy_rocchio(profile, vote_count=0):
        calls["rocchio"] += 1  # must never happen in the default path
        return static_mat

    monkeypatch.setattr(profile_mod, "build_profile_matrix", spy_static)
    monkeypatch.setattr(
        profile_mod, "build_profile_matrix_with_rocchio", spy_rocchio, raising=False
    )
    # Avoid loading a real profile YAML / settings: stub the loaders the default
    # path calls before build_profile_matrix.
    import dailydigest.config as config_mod
    from dailydigest.models import Profile

    monkeypatch.setattr(
        config_mod,
        "load_settings",
        lambda: SimpleNamespace(profile_path="unused.yaml"),
        raising=False,
    )
    monkeypatch.setattr(
        Path, "read_text", lambda self, *a, **k: "bio: x\nkeywords: []\n", raising=False
    )
    monkeypatch.setattr(
        "yaml.safe_load", lambda text: {"bio": "x", "keywords": []}, raising=False
    )
    monkeypatch.setattr(Profile, "__init__", lambda self, **kw: None, raising=False)

    # profile_mat=None -> exercise the default profile-construction path.
    result = bench.run_benchmark(
        rows, labels, timestamps, grades=grades, profile_mat=None, random_state=0
    )
    assert result["n_test"] > 0

    assert calls["static"] >= 1, "default path must build the STATIC profile matrix"
    assert calls["rocchio"] == 0, (
        "default path must NOT use build_profile_matrix_with_rocchio "
        "(it leaks held-out test votes via learned_profile.npz)"
    )


def test_benchmark_does_not_import_rocchio_builder():
    """The script must not reference the Rocchio profile builder at all."""
    path = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_ranker.py"
    source = path.read_text(encoding="utf-8")
    assert "build_profile_matrix_with_rocchio" not in source, (
        "benchmark must not use the vote-derived Rocchio profile (leakage)"
    )
    assert "build_profile_matrix" in source, "benchmark must use the static profile matrix"


def test_pairwise_accuracy_helper_orders_and_handles_ties():
    bench = _load_benchmark_module()
    # liked (label +1) at indices 0,1; disliked (-1) at 2,3.
    labels = np.array([1, 1, -1, -1], dtype=np.int32)
    # Perfect ordering: liked scored above disliked.
    acc, n = bench._pairwise_accuracy(np.array([0.9, 0.8, 0.2, 0.1]), labels)
    assert n == 4
    assert acc == pytest.approx(1.0)
    # Fully inverted ordering.
    acc, _ = bench._pairwise_accuracy(np.array([0.1, 0.2, 0.8, 0.9]), labels)
    assert acc == pytest.approx(0.0)
    # All ties -> 0.5.
    acc, _ = bench._pairwise_accuracy(np.array([0.5, 0.5, 0.5, 0.5]), labels)
    assert acc == pytest.approx(0.5)
    # No valid pair (single class) -> NaN.
    acc, n = bench._pairwise_accuracy(np.array([0.5, 0.6]), np.array([1, 1]))
    assert n == 0 and np.isnan(acc)


def test_ndcg_at_k_helper():
    bench = _load_benchmark_module()
    labels = np.array([1, 1, -1, -1], dtype=np.int32)
    # Perfect ranking: both positives on top -> nDCG@10 == 1.0.
    assert bench._ndcg_at_k(np.array([0.9, 0.8, 0.2, 0.1]), labels, k=10) == pytest.approx(1.0)
    # Inverted: positives at bottom -> < 1.0.
    assert bench._ndcg_at_k(np.array([0.1, 0.2, 0.8, 0.9]), labels, k=10) < 1.0
    # No positive label -> undefined (NaN).
    assert np.isnan(bench._ndcg_at_k(np.array([0.5, 0.6]), np.array([-1, -1])))
