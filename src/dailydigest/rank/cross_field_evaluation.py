"""Frozen, provisional scientific task-alignment evaluation; never a serving gate.

Run offline against the real scoring/selection path in an isolated child process::

    uv run python -m dailydigest.rank.cross_field_evaluation \
        --fixture tests/fixtures/cross_field_alignment.json \
        --model-cache prototypes/browser-compute/assets/native-model-cache \
        --state /path/to/new/evaluation-state --output /path/to/report.json

``--reader-detail`` measures explicitly supplied intent descriptions, not a model
change. Labels are agent metadata judgments; neither split is independent gold.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

from .cold_start_evaluation import (
    INPUT_MODES, evaluate_first_slate, prepare_profile, review_packet,
)


def load_fixture(path: str | Path) -> dict:
    corpus = json.loads(Path(path).read_text())
    if corpus.get("schema_version") != 1:
        raise ValueError("Unsupported alignment fixture")
    profiles = {p["id"]: p for p in corpus["profiles"]}
    if len(profiles) != len(corpus["profiles"]):
        raise ValueError("Duplicate profile")
    ids, paper_splits = set(), {}
    for case in corpus["cases"]:
        if case["id"] in ids:
            raise ValueError("Duplicate case")
        ids.add(case["id"])
        if case["label"] not in {"aligned", "off_task", "uncertain"}:
            raise ValueError("Unknown alignment label")
        if case["split"] not in {"dev", "heldout"}:
            raise ValueError("Unknown split")
        profile = profiles[case["profile_id"]]
        if case["url"] not in {item["url"] for item in profile["pool"]}:
            raise ValueError("Judgment outside captured pool")
        if not case.get("reviewer") or not case.get("review_status") or not case.get("rationale"):
            raise ValueError("Judgment requires review provenance and rationale")
        identity = str(case.get("doi") or case["url"]).casefold()
        if identity in paper_splits and paper_splits[identity] != case["split"]:
            raise ValueError("Same paper leaks across splits")
        paper_splits[identity] = case["split"]
    return corpus


def evaluate_alignment(corpus: dict, decisions: list[dict]) -> dict:
    """Count retrieved, eligible and selected separately; never infer recall upstream.

    Decisions must account for every judged pair. Unknown annotations are reported
    separately, rather than silently counted as negatives or positives.
    """
    by_pair = {}
    for row in decisions:
        key = row["profile_id"], row["url"]
        if key in by_pair:
            raise ValueError("Duplicate decision")
        if row["selected"] and not row["eligible"]:
            raise ValueError("Selected item must be eligible")
        by_pair[key] = row
    report = {"label_policy": corpus["label_policy"], "split_policy": corpus["split_policy"],
              "coverage_limit": corpus["coverage_limit"], "splits": {}}
    for split in ("dev", "heldout"):
        cases = [case for case in corpus["cases"] if case["split"] == split]
        groups = {"all": cases}
        groups.update({p["id"]: [c for c in cases if c["profile_id"] == p["id"]]
                       for p in corpus["profiles"]})
        results = {}
        for name, group in groups.items():
            counts = Counter(case["label"] for case in group)
            metrics = {"judged": len(group), "labels": dict(counts), "stages": {}}
            for stage in ("retrieved", "eligible", "selected"):
                passing = []
                for case in group:
                    key = case["profile_id"], case["url"]
                    if key not in by_pair:
                        raise ValueError(f"Missing decision: {key}")
                    if stage == "retrieved" or by_pair[key][stage]:
                        passing.append(case)
                kept = Counter(case["label"] for case in passing)
                known = kept["aligned"] + kept["off_task"]
                metrics["stages"][stage] = {
                    "aligned": kept["aligned"], "off_task": kept["off_task"],
                    "uncertain": kept["uncertain"],
                    "precision_on_decisive_labels": kept["aligned"] / known if known else None,
                    "retained_aligned_pool_fraction": (
                        kept["aligned"] / counts["aligned"] if counts["aligned"] else None),
                }
            results[name] = metrics
        report["splits"][split] = results
    return report


def _source_configuration() -> Path:
    # The replay changes cwd; source priors must still use this checkout's config.
    path = Path(__file__).resolve().parents[3] / "config/sources.yaml"
    if not path.is_file():
        raise ValueError("Alignment replay requires the repository source configuration")
    return path


def _embedding_provenance() -> dict:
    from . import embed

    inspect_provenance = getattr(embed, "active_embedding_provenance", None)
    if inspect_provenance is not None:
        return inspect_provenance()
    from ..config import get_settings

    return {"model": embed.active_model_name(),
            "requested_backend": get_settings().embed_backend,
            "actual_backend": None, "status": "requested_not_independently_verified",
            "document_signature": embed.active_embedding_signature(),
            "query_signature": embed.active_embedding_signature(is_query=True)}


@contextmanager
def _isolated_profile_store(state: Path, index: int, settings):
    """Give each scenario fresh metadata, feedback and item embedding state."""
    from .. import store

    profile_state = state / f'profile-{index:03d}'
    profile_state.mkdir(parents=True, exist_ok=False)
    previous_path = settings.db_path
    if store._ENGINE is not None:
        store._ENGINE.dispose()
    store._reset_init_for_testing()
    settings.db_path = str(profile_state / 'digest.db')
    try:
        store.init_db()
        yield
    finally:
        if store._ENGINE is not None:
            store._ENGINE.dispose()
        store._reset_init_for_testing()
        settings.db_path = previous_path


def _score_captured_research(rows, profile, *, profile_matrix, paper_text):
    """Ablate ranking text only; serving dedupe retains title and full abstract."""
    from unittest.mock import patch
    from .. import pipeline
    from . import embedding_cache
    from .contribution_claims import contribution_text
    from .profile import build_attribution_context, build_profile_matrix

    original_item_text = embedding_cache.item_text

    def evaluation_item_text(row):
        return (contribution_text(row.title or '', row.abstract or '')
                if paper_text == 'claims' else original_item_text(row))

    with patch.object(embedding_cache, 'item_text', evaluation_item_text):
        return pipeline._score_items_for_pipeline(
            rows, build_profile_matrix(profile, mode=profile_matrix), profile.downweight,
            build_attribution_context(profile))


def _select_captured_research(scored, features, *, settings, digest_id):
    """Use serving deduplication, budgets, selection and reading-mode helpers."""
    import inspect

    from .. import pipeline
    from ..dedupe import cap_near_duplicates
    from .embedding_cache import embed_item_rows
    from .ranker import apply_exploration
    from .source_quality import is_high_quality_journal_source

    eligible = list(scored)
    dropped = []
    if settings.within_day_dedupe and len(eligible) > 1:
        rows = [row for row, _ in eligible]
        keep = set(cap_near_duplicates(
            rows, embed_item_rows(rows), settings.within_day_dedupe_threshold))
        dropped = [int(row.id) for index, (row, _) in enumerate(eligible) if index not in keep]
        eligible = [pair for index, pair in enumerate(eligible) if index in keep]
    ordered = pipeline._apply_topic_selection_preferences(
        eligible, features, digest_id=digest_id)
    caps = pipeline._dynamic_section_caps(
        ordered, features, research_ceiling=settings.top_research, settings=settings)
    selection = {"score_features": features, "settings": settings}
    if hasattr(pipeline, "capture_selection_facts"):
        selection["selection_facts"] = pipeline.capture_selection_facts(
            [row for row, _ in ordered])
    picked = pipeline.pick_top_per_section(ordered, caps, **selection)
    slots = int(getattr(settings, "exploration_slots", 0))
    uncertainty = {key: 1.0 - 2.0 * abs(float(feat.get("learned_score", 0.0)) - .5)
                   for key, feat in features.items()
                   if feat.get("scoring_mode") in {"hybrid_lr", "hybrid_knn"}}
    if slots > 0 and uncertainty:
        exploration = {"slots": slots, "eligible": is_high_quality_journal_source}
        # Older native branches expose the same default no-exploration path,
        # with a smaller optional exploration interface.
        if "caps" in inspect.signature(apply_exploration).parameters:
            exploration.update(caps=caps, **selection)
        picked = apply_exploration(picked, ordered, uncertainty, **exploration)
    picked = pipeline._restore_research_order(picked, ordered)
    picked = pipeline.apply_reading_mode(picked, "full", preserve_research_order=True)
    effective_settings = {name: getattr(settings, name) for name in (
        "scoring_mode", "semantic_matching", "adaptive_section_sizes", "top_research",
        "min_research", "min_topic_relevance", "within_day_dedupe",
        "within_day_dedupe_threshold", "max_preprint_research_frac",
        "max_low_impact_research_frac", "low_impact_relevance_floor",
        "research_final_score_floor_frac", "research_final_score_min_keep",
        "research_quality_weight", "venue_relevance_credit", "topic_priority_bonus_scale",
        "topic_coverage_bonus_scale", "citation_enrichment", "exploration_slots",
    )}
    return ordered, picked, {"section_budgets": caps, "within_day_dropped_ids": dropped,
                             "reading_mode": "full", "exploration_slots": slots,
                             "effective_settings": effective_settings}


def replay_offline(corpus: dict, *, state: Path, model_cache: Path,
                   reader_detail: bool = False, profile_input: str = 'provided',
                   profile_matrix: str = 'legacy', paper_text: str = 'full') -> list[dict]:
    """CLI-only replay: creates a new state directory and forbids socket access.

    Run in a dedicated process: settings, working directory and embedding lifecycle
    are process globals in the production path. Never point this at user state.
    """
    import datetime as datetime_module
    import os
    import socket
    from unittest.mock import patch

    if reader_detail:
        if profile_input != 'provided':
            raise ValueError('Use reader_detail or profile_input, not both')
        profile_input = 'intent'
    if profile_matrix not in {'legacy', 'detailed', 'canonical', 'additive'}:
        raise ValueError('Unknown profile matrix mode')
    if paper_text not in {'full', 'claims'}:
        raise ValueError('Unknown paper text mode')
    # Validate every scenario before creating state or loading a model.
    prepared_profiles = {entry['id']: prepare_profile(entry, profile_input)
                         for entry in corpus['profiles']}
    if any(entry["profile"].get(key) for entry in corpus["profiles"]
           for key in ("negative_interests", "authors_of_interest")):
        raise ValueError("Captured-pool replay requires profiles without negative interests or watched authors")
    source_configuration = _source_configuration()
    state = state.resolve()
    model_cache = model_cache.resolve()
    if not model_cache.is_dir():
        raise ValueError("A populated local model cache is required")
    state.mkdir(parents=True, exist_ok=False)
    os.chdir(state)
    os.environ.update(FASTEMBED_CACHE_PATH=str(model_cache), HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1")
    from .. import config

    # Explicit settings prevent ambient .env/provider and private profile reuse.
    for name in config.Settings.model_fields:
        os.environ.pop(name.upper(), None)
    settings = config.Settings(_env_file=None, db_path=str(state / "data/digest.db"),
                               sources_path=str(source_configuration),
                               embed_backend="fastembed", embed_threads=2, embed_batch_size=4,
                               llm_backend="extractive", citation_enrichment=False,
                               llm_api_key="", resend_api_key="", contribution_reranker_path="",
                               scoring_mode="cosine", top_research=12, semantic_matching=False,
                               exploration_slots=0)
    config.SETTINGS = settings
    config.get_settings = config.load_settings = lambda: settings
    from .. import pipeline, store
    from ..dedupe import dedupe_ranking_candidates
    from ..models import Item, Profile

    original_datetime = datetime_module.datetime
    decision_at = original_datetime.fromisoformat(corpus["decision_at"])

    class DateMeta(type):
        def __instancecheck__(cls, value):
            return isinstance(value, original_datetime)

    class FrozenDateTime(original_datetime, metaclass=DateMeta):
        @classmethod
        def now(cls, tz=None):
            return decision_at.astimezone(tz) if tz else decision_at.replace(tzinfo=None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Alignment replay must remain offline")

    decisions = []
    with (patch.object(socket.socket, "connect", forbidden),
          patch.object(socket, "create_connection", forbidden),
          patch.object(datetime_module, "datetime", FrozenDateTime),
          patch.object(pipeline, "datetime", FrozenDateTime),
          patch.object(store, "datetime", FrozenDateTime)):
        for index, entry in enumerate(corpus["profiles"]):
            with _isolated_profile_store(state, index, settings):
                started = time.perf_counter()
                profile_data = prepared_profiles[entry['id']]
                profile = Profile.model_validate(profile_data)
                config.load_profile = lambda *args, profile=profile: profile
                items = [Item.model_validate(item) for item in entry["pool"]]
                store.upsert_items(items)
                urls = {item.url for item in items}
                rows = dedupe_ranking_candidates(
                    [row for row in store.recent_items(days=36500) if row.url in urls])
                quality, _ = pipeline._quality_gate(rows)
                scored, features = _score_captured_research(
                    quality, profile, profile_matrix=profile_matrix, paper_text=paper_text)
                relevant = pipeline._filter_off_topic(scored, features)
                selection_pool, picked, selection_trace = _select_captured_research(
                    relevant, features, settings=settings, digest_id=decision_at.date().isoformat())
                quality_urls = {row.url for row in quality}
                topic_urls = {row.url for row, _ in relevant}
                eligible = {row.url for row, _ in selection_pool}
                selected = {row.url for row, _ in picked}
                positions = {row.url: index for index, (row, _) in enumerate(picked, 1)}
                selection_trace.update(profile_input=profile_input, profile_matrix=profile_matrix,
                                       paper_text=paper_text, elapsed_seconds=time.perf_counter() - started,
                                       supplied_profile=profile_data,
                                       candidate_isolation='fresh_database_per_profile',
                                       dedupe_representation='full_title_abstract')
                rows_by_url = {row.url: row for row in rows}
                for item in items:
                    row = rows_by_url.get(item.url)
                    decisions.append({"profile_id": entry["id"], "url": item.url,
                                      "eligible": item.url in eligible,
                                      "selected": item.url in selected,
                                      "selected_position": positions.get(item.url),
                                      "stages": {"after_quality_gate": item.url in quality_urls,
                                                 "after_topic_gate": item.url in topic_urls,
                                                 "after_within_day_dedupe": item.url in eligible},
                                      "selection_trace": selection_trace,
                                      "features": features.get(row.id, {}) if row else {},
                                      "embedding_provenance": _embedding_provenance()})
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reader-detail", action="store_true")
    parser.add_argument("--profile-input", choices=INPUT_MODES, default='provided')
    parser.add_argument("--profile-matrix", choices=('legacy', 'detailed', 'canonical', 'additive'), default='legacy')
    parser.add_argument("--paper-text", choices=('full', 'claims'), default='full')
    parser.add_argument("--review-packet", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    packet_path = args.review_packet.resolve() if args.review_packet else None
    fixture = args.fixture.resolve()
    corpus = load_fixture(fixture)
    source_root = Path(__file__).resolve().parent
    source_paths = [source_root / name for name in (
        "cross_field_evaluation.py", "ranker.py", "source_quality.py", "profile.py",
        "embed.py", "embedding_cache.py", "cold_start_evaluation.py", "contribution_claims.py")]
    source_paths.extend(source_root.parent / name for name in (
        "pipeline.py", "dedupe.py", "config.py", "models.py"))
    source_configuration = _source_configuration()
    source_hashes = {str(path.relative_to(source_root.parent)): hashlib.sha256(
        path.read_bytes()).hexdigest() for path in source_paths}
    decisions = replay_offline(corpus, state=args.state, model_cache=args.model_cache,
                               reader_detail=args.reader_detail, profile_input=args.profile_input,
                               profile_matrix=args.profile_matrix, paper_text=args.paper_text)
    report = evaluate_alignment(corpus, decisions)
    report.update(fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                  effective_settings=(decisions[0]["selection_trace"]["effective_settings"]
                                      if decisions else {}),
                  evaluation_scope={"population": "captured pools, not live retrieval",
                                    "profile_scope": "fixture scenarios; no user profile, votes or history",
                                    "scoring_mode": "cosine", "semantic_matching": False,
                                    "negative_interests_and_watched_authors": "unsupported; rejected if supplied",
                                    "selection": "serving within-day dedupe, adaptive budgets, source caps, full reading mode",
                                    "comparison_limit": "Earlier fixed-cap/no-within-day-dedupe metrics are obsolete"},
                  reader_detail=args.reader_detail, decision_at=corpus["decision_at"],
                  profile_input='intent' if args.reader_detail else args.profile_input,
                  profile_matrix=args.profile_matrix, paper_text=args.paper_text,
                  first_session=evaluate_first_slate(corpus, decisions),
                  scoring_source_sha256=source_hashes,
                  source_configuration_sha256=hashlib.sha256(
                      source_configuration.read_bytes()).hexdigest(),
                  decisions=decisions)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if packet_path:
        packet_path.write_text(json.dumps(review_packet(
            corpus, profile_input='intent' if args.reader_detail else args.profile_input),
            ensure_ascii=False, indent=2) + '\n')


if __name__ == "__main__":
    main()
