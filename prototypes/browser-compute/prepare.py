"""Prepare public fictional fixtures and a portable Python ranking-code bundle.

Reference inference happens once during preparation. The served site is static;
the browser subsequently performs its own inference and ranking independently.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "assets"
RANK_FUNCTIONS = {
    "_multi_cosine", "_cosine_sim", "_scored_sort_key", "_item_text", "_reason_penalty_for",
    "_row_feature_key", "_feature_payload", "_freshness_penalty", "_apply_quality_adjustments_with_features",
    "_pick_news_balanced", "pick_top_per_section", "_pick_research_balanced",
    "_apply_final_score_cutoff", "_available",
}
SETTINGS = (
    "semantic_matching", "research_quality_weight", "venue_relevance_credit", "max_preprint_research_frac",
    "max_low_impact_research_frac", "low_impact_relevance_floor", "research_final_score_floor_frac",
    "research_final_score_min_keep", "adaptive_section_sizes", "min_topic_relevance",
    "min_news_quality", "min_opportunity_relevance", "adaptive_relevance_floor",
)


def _source(path):
    return (ROOT / path).read_text()


def _extract(path, names):
    source = _source(path)
    tree = ast.parse(source)
    selected = {node.name: ast.get_source_segment(source, node) for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names}
    if selected.keys() != names:
        raise ValueError(f"Missing portable functions: {names - selected.keys()}")
    return "\n\n".join(selected.values()) + "\n"


def build_fixture():
    corpus = json.loads((ROOT / "tests/fixtures/research_scope_benchmark.json").read_text())
    examples = corpus["examples"]
    venues = ["Nature", "Nature Methods", "Science", "ACS Nano", "bioRxiv", "Unlisted Journal"]
    papers = [{"id": index + 1, "title": example["title"], "abstract": example["abstract"],
               "authors": "Alex Example; Sam Fixture", "source": venues[index % len(venues)],
               "external_id": f"fixture-{index + 1}", "url": f"https://example.invalid/fixture/{index + 1}",
               "metadata": {}, "synthetic_scope": example["label"]}
              for index, example in enumerate(examples)]
    # Identical text isolates venue/DOI effects from embedding changes.
    extra = len(papers) + 1
    papers.extend([
        {**papers[0], "id": extra, "source": "Unlisted Journal", "external_id": "10.9999/prototype.low",
         "url": "https://example.invalid/low", "metadata": {"doi": "10.9999/prototype.low"}},
        {**papers[0], "id": extra + 1, "source": "Nature", "title": "Editorial: " + papers[0]["title"],
         "external_id": "fixture-editorial", "url": "https://example.invalid/editorial",
         "metadata": {"publication_types": ["Editorial"]}},
        {**papers[0], "id": extra + 2, "source": "arXiv q-bio", "external_id": "2610.12345",
         "url": "https://arxiv.org/abs/2610.12345", "metadata": {"published_doi": "10.9999/prototype.journal"}},
        {**papers[0], "id": extra + 3, "source": "Nature Methods", "title": "Transferable architecture for programmable molecular hinges",
         "external_id": "10.9999/prototype.journal", "url": "https://example.invalid/journal",
         "metadata": {"doi": "10.9999/prototype.journal"}},
    ])
    core = list(corpus["definitions"]["core"])
    return {"schema_version": 1, "provenance": "Fictional development papers and example interests; no user data or accuracy ground truth.",
            "queries": core + ["Natural protein structures and cellular signaling", "Cellular condensates and chromatin regulation"],
            "query_weights": [1.0] * len(core) + [.45, .45], "cap": 6, "papers": papers,
            "query_prefix": "Represent this sentence for searching relevant passages: ", "document_prefix": ""}


def build_bundle():
    from dailydigest.config import Settings, load_sources
    quality_path = "src/dailydigest/rank/source_quality.py"
    dedupe_path = "src/dailydigest/dedupe.py"
    rank_path = "src/dailydigest/rank/ranker.py"
    pipeline_path = "src/dailydigest/pipeline.py"
    quality = _source(quality_path)
    dedupe = _source(dedupe_path)
    # Preserve the actual algorithms; replace only unavailable ORM/model types.
    rank_header = '''from __future__ import annotations
import math
from collections.abc import Mapping
from datetime import UTC
from typing import Any
from types import SimpleNamespace as ItemRow
import numpy as np
from .source_quality import (RANKER_VERSION, quality_adjusted_score, should_skip_item, source_bucket,
    publication_date_is_known, source_diversity_key, is_arxiv_cs_source, is_high_quality_journal_source,
    is_low_impact_research, is_preprint_source, is_published_journal_source)
DOWNWEIGHT_PENALTY = 0.20
ScoreFeatureMap = dict
def item_text(row):
    return row.title.strip() + (". " + row.abstract.strip() if row.abstract.strip() else "")
'''
    pipe_tree = ast.parse(_source(pipeline_path))
    blocklist = next(ast.get_source_segment(_source(pipeline_path), node) for node in pipe_tree.body
                     if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_TITLE_BLOCKLIST" for target in node.targets))
    gate_header = '''from __future__ import annotations
import logging
import re
from typing import Any
from types import SimpleNamespace as ItemRow
from .config import get_settings
from .rank.ranker import _row_feature_key
from .rank.source_quality import should_skip_item, source_bucket, is_high_quality_journal_source, venue_relevance_credit
ScoreFeatureMap = dict
logger = logging.getLogger(__name__)
'''
    names = {"Nature", "Nature Methods", "Science", "ACS Nano", "bioRxiv", "Unlisted Journal", "arXiv q-bio"}
    sources = [{name: getattr(spec, name) for name in ("name", "kind", "category", "quality_tier", "prestige_score", "impact_floor", "promo_risk", "paywalled")}
               for spec in load_sources() if spec.name in names]
    settings = {name: Settings.model_fields[name].default for name in SETTINGS}
    settings["adaptive_relevance_floor"] = False  # Learned calibration is outside this prototype.
    return {"schema_version": 1, "settings": settings, "sources": sources,
            "quality_source": quality, "dedupe_source": dedupe,
            "ranker_source": rank_header + _extract(rank_path, RANK_FUNCTIONS),
            "gates_source": gate_header + blocklist + "\n" + _extract(pipeline_path, {"_audit_item", "_quality_gate", "_research_relevant", "_filter_off_topic"}),
            "production_source_sha256": {path: hashlib.sha256(_source(path).encode()).hexdigest()
                                         for path in (quality_path, dedupe_path, rank_path, pipeline_path)}}


def build_policy_cases():
    """Controlled vectors separate policy portability from semantic inference."""
    import math
    base = {"title": "Programmable DNA origami hinges establish transferable molecular design rules",
            "abstract": "We introduce a transferable engineering method and measure designed molecular assembly geometry.",
            "authors": "Alex Example; Sam Fixture", "metadata": {}, "synthetic_scope": "core"}
    def row(i, source, **updates):
        return {**base, "id": i, "source": source, "external_id": f"case-{i}",
                "url": f"https://example.invalid/case/{i}", **updates}
    def case(name, papers, scores, cap=4):
        return {"name": name, "fixture": {"papers": papers, "queries": ["DNA nanotechnology"],
                                           "query_weights": [1.], "cap": cap},
                "vectors": {"queries": [[1., 0.]], "documents": [[score, math.sqrt(1-score**2)] for score in scores]}}
    quality = case("Venue limits, explicit editorial and off-topic prestige", [
        row(1, "Nature"), row(2, "Nature Methods"), row(3, "Science"), row(4, "ACS Nano"),
        row(5, "Unlisted Journal", metadata={"doi": "10.9999/prototype.low"}),
        row(6, "Nature", metadata={"publication_types": ["Editorial"]}), row(7, "Nature")],
        [.86, .86, .86, .86, .86, .86, .2])
    sparse = case("Sparse pool retains quality gates", quality["fixture"]["papers"][4:], [.86, .86, .2])
    versions = case("Renamed preprint and journal share one publication identity", [
        row(1, "arXiv q-bio", external_id="2610.12345", url="https://arxiv.org/abs/2610.12345", metadata={"published_doi": "10.9999/prototype.published"}),
        row(2, "Nature Methods", title="Engineering molecular hinges with reusable architectural rules", metadata={"doi": "10.9999/prototype.published"}),
        row(3, "Science", title="A different programmable RNA assembly architecture")], [.88, .88, .88], 3)
    unicode = case("Alpha and beta targets remain distinct papers", [
        row(1, "arXiv q-bio", title="Programmable molecular assemblies selectively target receptor α complexes", external_id="2610.11111", url="https://arxiv.org/abs/2610.11111"),
        row(2, "Nature", title="Programmable molecular assemblies selectively target receptor β complexes")], [.88, .88], 2)
    context = case("Context-only similarity cannot outrank core engineering", [row(1, "Nature"), row(2, "Science")], [.9, .9], 2)
    context["fixture"].update(queries=["DNA engineering", "Native protein signaling"], query_weights=[1., .45])
    context["vectors"] = {"queries": [[1., 0.], [0., 1.]], "documents": [[1., 0.], [0., 1.]]}
    return [quality, sparse, versions, unicode, context]


def _kernel():
    spec = importlib.util.spec_from_file_location("prototype_kernel", HERE / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_rank(fixture, vectors, bundle):
    """Independent imports of real modules are the policy-parity reference."""
    from dailydigest import dedupe, pipeline
    from dailydigest.rank import ranker, source_quality
    settings = SimpleNamespace(**bundle["settings"])
    sources = {row["name"].lower(): SimpleNamespace(**row) for row in bundle["sources"]}
    with patch("dailydigest.config.get_settings", return_value=settings), patch.object(pipeline, "get_settings", return_value=settings), patch.object(source_quality, "_configured_sources_cached", return_value=sources):
        return _kernel().rank_fixture(fixture, vectors, (source_quality, dedupe, ranker, pipeline))


def main():
    if len(sys.argv) > 1:
        if sys.argv[1:] == ["--help"]:
            print("description: Prepare fictional fixtures, pinned model assets and native parity references.\nhelp: Run python prepare.py; no production data or live feeds are used.")
            return 0
        print('error: "Unknown arguments"\nhelp: "Run python prepare.py or python prepare.py --help"')
        return 2
    from fastembed import TextEmbedding
    import numpy as np
    fixture, bundle = build_fixture(), build_bundle()
    OUTPUT.mkdir(exist_ok=True)
    # Dedicated development cache; no production embedding cache is read/written.
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=OUTPUT / "native-model-cache", threads=2)
    documents = [row["title"].strip() + (". " + row["abstract"].strip() if row["abstract"].strip() else "") for row in fixture["papers"]]
    queries = [fixture["query_prefix"] + text for text in fixture["queries"]]
    def encode(texts):
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        encoded = np.asarray(list(model.embed([texts[index] for index in order], batch_size=4)), dtype=np.float32)
        result = np.empty_like(encoded)
        result[order] = encoded
        return result / np.linalg.norm(result, axis=1, keepdims=True)
    vectors = {"documents": encode(documents).tolist(), "queries": encode(queries).tolist()}
    # Reuse the very same graph and tokenizer in the browser, not a different export.
    cache = Path(model.model._model_dir)
    target = OUTPUT / "models/bge-small"
    (target / "onnx").mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copyfile(cache / name, target / name)
    graph = cache / model.model.model_description.model_file
    shutil.copyfile(graph, target / "onnx/model.onnx")
    manifest = {"model": "BAAI/bge-small-en-v1.5", "backend": "fastembed-native-vs-onnx-wasm",
                "graph_sha256": hashlib.sha256(graph.read_bytes()).hexdigest(), "graph_bytes": graph.stat().st_size,
                "pooling": "cls", "normalize": True, "max_length": 512, "dimension": 384,
                "query_prefix": fixture["query_prefix"], "document_prefix": fixture["document_prefix"],
                "transformers_js": "3.8.1", "pyodide": "0.28.3", "production_source_sha256": bundle["production_source_sha256"]}
    result = native_rank(fixture, vectors, bundle)
    policy_cases = build_policy_cases()
    for case in policy_cases:
        case["reference"] = native_rank(case["fixture"], case["vectors"], bundle)
    for name, content in (("fixture.json", fixture), ("bundle.json", bundle),
                          ("reference.json", {"vectors": vectors, "ranking": result, "manifest": manifest}),
                          ("policy-cases.json", policy_cases)):
        (OUTPUT / name).write_text(json.dumps(content, allow_nan=False) + "\n")
    (OUTPUT / "kernel.py").write_text((HERE / "kernel.py").read_text())
    # Pyodide's npm package includes the runtime but not its NumPy wheels. Store
    # verified public dependencies locally so the browser needs only static GETs.
    runtime_dir = HERE / "node_modules/pyodide"
    lock = json.loads((runtime_dir / "pyodide-lock.json").read_text())
    pending, seen = ["numpy"], set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        record = lock["packages"][name]
        target = runtime_dir / record["file_name"]
        if not target.exists():
            urllib.request.urlretrieve("https://cdn.jsdelivr.net/pyodide/v0.28.3/full/" + record["file_name"], target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Runtime dependency checksum mismatch: " + name)
        pending.extend(record.get("depends", []))
    print(f'status: "fixture_prepared"\npapers: {len(documents)}\nqueries: {len(queries)}\nmodel_bytes: {graph.stat().st_size}\nselected: {len(result["selected_ids"])}\nprivate_user_data_used: false')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
