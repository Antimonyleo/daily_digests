"""Serving-parity offline ranking evaluation on immutable ranking snapshots (R1.1).

Two modes are reported separately and never substitute for each other:

* ``selector_replay`` reruns serving *selection* on recorded scores
  (``replay.frozen_selection``). It proves nothing about rescoring.
* ``frozen_rescore`` reruns the serving scorer
  (``ranker.score_items_with_features``, the function ``_score_items_for_pipeline``
  calls) on each snapshot's scored pool, with every input frozen at that
  decision: the snapshot's own votes, profile matrix, settings, enrichment
  evidence, reason penalties and clock; cached vectors with matching text hashes
  for the recorded embedding signature. Network, current-state database access,
  the LR artifact and current profile/settings are blocked inside the boundary.
  Inputs that cannot be frozen are named per run as ``retrospective_inputs``.

Labels are the current latest signed votes, used only after scoring. Items the
reader had rated before a decision are excluded from that run's evaluation.
Unrated items are unknown, never negative. All retained history has already
been inspected, so every result is ``development`` evidence, not a holdout.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import socket
import sqlite3
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .replay import IncompleteReplay, _time, frozen_selection, recorded_slate

VERSION = "offline-serving-eval-v1"
STAGES = ("veteran", "first_shot_proxy")
# A first-shot run is new-reader evidence only when its learned row was removed.
VERIFIED_STATIC_BASES = frozenset({"static_rows_recorded", "static_rows_verified_by_profile_weights"})
FIRST_SHOT_UNVERIFIED = "static_profile_unverified"
DECISIONS = ("better", "non_inferior", "worse", "insufficient_evidence")
UTILITY_GAIN = {"very_relevant": 3.0, "relevant": 1.0, "legacy_positive": 1.0}
MAX_GAIN = 3.0
POSITIVE = {"very_relevant", "relevant", "legacy_positive"}
NEGATIVE = {"strict_negative", "weak_negative", "legacy_negative"}
CONTRASTS = {
    "relevant_or_better_vs_strict_negative": ("strict_negative",),
    "relevant_or_better_vs_weak_negative": ("weak_negative",),
    "relevant_or_better_vs_any_negative": tuple(sorted(NEGATIVE)),
}
PRIMARY_CONTRAST = "relevant_or_better_vs_strict_negative"
_ATTRIBUTION_KEYS = ("primary_facet", "secondary_facets", "primary_facet_score",
                     "topic_priority", "topic_priority_bonus", "topic_alignment")
_UNCAPTURED_SETTINGS = ("adaptive_section_sizes", "min_news_quality", "min_opportunity_relevance")
_PARITY_TOLERANCE = 1e-3


class OfflineEvalError(ValueError):
    """Evidence cannot support the requested offline evaluation."""


# --------------------------------------------------------------------------- #
# Read-only evidence access
# --------------------------------------------------------------------------- #


def _connect(database: str | Path) -> sqlite3.Connection:
    path = Path(database).resolve()
    if not path.exists():
        raise OfflineEvalError("database does not exist")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _snapshot_index(connection) -> list[tuple[str, str]]:
    rows = connection.execute(
        "SELECT run_id, created_at FROM ranking_snapshots WHERE schema_version = 1 ORDER BY created_at, run_id"
    ).fetchall()
    return [(str(run), str(created)) for run, created in rows]


def _load_snapshot(connection, run_id: str) -> dict:
    (payload,) = connection.execute(
        "SELECT payload_zlib FROM ranking_snapshots WHERE run_id = ?", (run_id,)
    ).fetchone()
    snapshot = json.loads(zlib.decompress(payload))
    if snapshot.get("schema_version") != 1 or snapshot.get("run_id") != run_id:
        raise OfflineEvalError("snapshot identity or schema differs from its row")
    return snapshot


def read_labels(connection) -> dict[int, dict]:
    """Latest current vote per item (latest created_at, then row id wins)."""
    labels: dict[int, dict] = {}
    for item_id, value, grade, created_at, _row in connection.execute(
        "SELECT item_id, value, grade, created_at, id FROM votes ORDER BY item_id, created_at, id"
    ):
        labels[int(item_id)] = {"value": int(value), "grade": grade,
                                "created_at": _time(created_at) if created_at else None}
    return labels


def label_class(label: dict | None) -> str:
    """Map a vote onto explicit strata; unrated and cleared stay unknown."""
    if not label or label.get("value") not in (-1, 1):
        return "unknown"
    grade = label.get("grade")
    if label["value"] == 1:
        if grade is None:
            return "legacy_positive"
        return "very_relevant" if grade >= 100 else "relevant"
    if grade is None:
        return "legacy_negative"
    return "strict_negative" if grade <= 10 else "weak_negative"


def _item_text(title, abstract) -> str:
    title, abstract = (title or "").strip(), (abstract or "").strip()
    return f"{title}. {abstract}" if abstract else title


class FrozenEmbeddings:
    """Read-only vector lookup keyed by item, recorded signature and text hash.

    A missing or text-changed vector is never silently replaced. With
    ``allow_local_encoder`` the local model re-encodes it offline (sockets stay
    blocked). If the recorded signature predates the active one, re-encoding
    is allowed only after cached vectors of that signature reproduce at cosine
    >= ``EQUIVALENCE_MIN_COSINE``; such runs are labeled retrospective.
    """

    EQUIVALENCE_MIN_COSINE = 0.999
    EQUIVALENCE_PROBE = 24

    def __init__(self, connection, *, allow_local_encoder: bool = False):
        self._connection = connection
        self.allow_local_encoder = allow_local_encoder
        self.signature: str | None = None
        self._memo: dict[tuple[str, int, str], np.ndarray] = {}
        self.stats: Counter = Counter()
        self.reencoded: Counter = Counter()
        self.equivalence: dict[str, dict] = {}

    def _equivalent(self, active: str) -> bool:
        """Probe whether the active encoder reproduces the recorded vector family."""
        if self.signature in self.equivalence:
            return self.equivalence[self.signature]["equivalent"]
        from .embed import embed_texts

        rows = self._connection.execute(
            "SELECT e.vector, e.dim, e.text_hash, i.title, i.abstract FROM item_embeddings e "
            "JOIN items i ON i.id = e.item_id WHERE e.model = ? ORDER BY e.item_id LIMIT 400",
            (self.signature,)).fetchall()
        probe = []
        for raw, dim, text_hash, title, abstract in rows:
            text = _item_text(title, abstract)
            if len(raw) == int(dim) * 4 and hashlib.sha256(text.encode()).hexdigest() == text_hash:
                probe.append((text, np.frombuffer(raw, dtype=np.float32)))
            if len(probe) >= self.EQUIVALENCE_PROBE:
                break
        cosine = None
        if probe:
            fresh = embed_texts([text for text, _vector in probe])
            old = np.stack([vector for _text, vector in probe])
            cosine = float(np.min(np.sum(fresh * old, axis=1) / (
                np.linalg.norm(fresh, axis=1) * np.linalg.norm(old, axis=1) + 1e-12)))
        equivalent = len(probe) >= self.EQUIVALENCE_PROBE and cosine is not None \
            and cosine >= self.EQUIVALENCE_MIN_COSINE
        self.equivalence[self.signature] = {"active_signature": active, "probe_vectors": len(probe),
                                            "min_cosine": cosine, "equivalent": equivalent}
        return equivalent

    def bind(self, signature: str) -> None:
        self.signature = signature

    def _fetch(self, ids: list[int]) -> dict[int, tuple[str, np.ndarray]]:
        found = {}
        for start in range(0, len(ids), 900):
            chunk = ids[start:start + 900]
            query = ("SELECT item_id, text_hash, dim, vector FROM item_embeddings WHERE model = ? "
                     f"AND item_id IN ({','.join('?' * len(chunk))})")
            for iid, text_hash, dim, raw in self._connection.execute(query, [self.signature, *chunk]):
                if 0 < int(dim) <= 8192 and len(raw) == int(dim) * 4:
                    found[int(iid)] = (text_hash, np.frombuffer(raw, dtype=np.float32).copy())
        return found

    def vectors(self, rows) -> np.ndarray:
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        if self.signature is None:
            raise OfflineEvalError("embedding store is not bound to a recorded signature")
        keys = []
        for row in rows:
            if not isinstance(getattr(row, "id", None), int):
                raise IncompleteReplay("frozen rescoring requires persistent item IDs")
            text = _item_text(row.title, row.abstract)
            keys.append((self.signature, int(row.id), hashlib.sha256(text.encode()).hexdigest()))
        wanted = sorted({key[1] for key in keys if key not in self._memo})
        if wanted:
            cached = self._fetch(wanted)
            for key in keys:
                if key in self._memo or key[1] not in cached:
                    continue
                text_hash, vector = cached[key[1]]
                if text_hash == key[2]:
                    self._memo[key] = vector
        missing = [index for index, key in enumerate(keys) if key not in self._memo]
        if missing:
            if not self.allow_local_encoder:
                self.stats["missing"] += len(missing)
                raise IncompleteReplay(f"{len(missing)} scored vectors are absent from the frozen cache")
            from .embed import active_embedding_signature, embed_texts

            active = active_embedding_signature()
            if active != self.signature and not self._equivalent(active):
                raise IncompleteReplay("local encoder does not reproduce the recorded vector family")
            encoded = embed_texts([_item_text(rows[i].title, rows[i].abstract) for i in missing])
            if active_embedding_signature() != active or encoded.shape[0] != len(missing):
                raise IncompleteReplay("local encoder resolved a different embedding signature")
            for index, vector in zip(missing, encoded, strict=True):
                self._memo[keys[index]] = np.asarray(vector, dtype=np.float32)
            self.stats["reencoded"] += len(missing)
            self.reencoded[self.signature] += len(missing)
        self.stats["cache_or_memo"] += len(keys) - len(missing)
        return np.stack([self._memo[key] for key in keys]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Configs and frozen state
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvalConfig:
    name: str
    preference_weight: float = 0.5
    settings: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, value: dict) -> "EvalConfig":
        from ..config import Settings

        if not isinstance(value, dict) or not isinstance(value.get("name"), str) or not value["name"]:
            raise OfflineEvalError("each config needs a nonempty name")
        unknown = set(value) - {"name", "preference_weight", "settings"}
        if unknown:
            raise OfflineEvalError(f"unknown config keys: {sorted(unknown)}")
        weight = float(value.get("preference_weight", 0.5))
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise OfflineEvalError("preference_weight must be within [0, 1]")
        settings = dict(value.get("settings") or {})
        missing = sorted(set(settings) - set(Settings.model_fields))
        if missing:
            raise OfflineEvalError(f"no existing settings knob for: {missing}")
        return cls(value["name"], weight, settings)


def _frozen_settings(snapshot: dict, config: EvalConfig, scoring_mode: str):
    """Recorded ranking settings plus explicit overrides; never .env or os.environ."""
    from ..config import Settings

    fields = Settings.model_fields
    base = {key: value for key, value in snapshot.get("ranking_settings", {}).items() if key in fields}
    payload = {**base, "scoring_mode": scoring_mode, "user_tz": "UTC", **config.settings}
    return Settings.model_validate(payload)


def _row(candidate: dict):
    from ..store import ItemRow

    metadata = candidate.get("metadata") or {}
    values = {}
    for name in ("source", "external_id", "url", "title", "abstract", "authors", "metadata_json"):
        values[name] = metadata.get(name)
    for name in ("published_at", "fetched_at"):
        raw = metadata.get(name)
        values[name] = datetime.fromisoformat(raw) if isinstance(raw, str) and raw else None
    values["metadata_json"] = values["metadata_json"] or "{}"
    values["abstract"] = values["abstract"] or ""
    values["authors"] = values["authors"] or ""
    return ItemRow(id=int(candidate["item_id"]), section=candidate.get("section") or "", **values)


def _static_rows(snapshot: dict) -> tuple[np.ndarray, str]:
    """Drop the appended learned row: by the recorded static row count, else by
    the frozen profile's own row weights under the current profile rules.

    An unverified basis returns the whole matrix; callers must not treat that
    as a first-use profile (see ``FIRST_SHOT_UNVERIFIED``).
    """
    from ..models import Profile
    from .profile import _profile_parts_with_weights

    matrix = np.asarray(snapshot["profile_matrix"], dtype=np.float32)
    recorded = snapshot.get("profile_static_rows")
    if isinstance(recorded, int) and not isinstance(recorded, bool) and 0 < recorded <= matrix.shape[0]:
        return matrix[:recorded].copy(), "static_rows_recorded"
    try:
        _parts, weights = _profile_parts_with_weights(Profile(**snapshot["profile"]))
    except Exception:  # noqa: BLE001 - malformed historical profile is reported, not guessed
        return matrix, "static_rows_unverified"
    n = len(weights)
    norms = np.linalg.norm(matrix, axis=1)
    if matrix.shape[0] in (n, n + 1) and np.allclose(norms[:n], weights, atol=1e-3):
        return matrix[:n].copy(), "static_rows_verified_by_profile_weights"
    return matrix, "static_rows_unverified"


def _frozen_votes(snapshot: dict, as_of: datetime) -> tuple[list[dict], int]:
    """Decision-time vote state, additionally clipped strictly before the decision."""
    state = snapshot.get("feedback_at_start")
    if not isinstance(state, dict) or not isinstance(state.get("votes"), list):
        raise IncompleteReplay("snapshot lacks frozen decision-time votes")
    kept, dropped = {}, 0
    for vote in state["votes"]:
        created = vote.get("created_at")
        if created is None or _time(created) >= as_of:
            dropped += 1
            continue
        kept[int(vote["item_id"])] = dict(vote)
    return [kept[iid] for iid in sorted(kept)], dropped


@dataclass
class FrozenRun:
    snapshot_meta: dict
    as_of: datetime
    stage: str
    rows: list
    recorded: dict[int, dict]
    actionable: set[int]
    votes: list[dict]
    reason_penalties: dict[int, float]
    profile_matrix: np.ndarray
    downweight: list[str]
    viewed_history: dict[str, datetime | None]
    scoring_mode: str
    retrospective_inputs: list[str]
    freeze_notes: dict
    recorded_facts: dict = field(default_factory=dict)
    exemplar_cache: tuple | None = None


def freeze_run(snapshot: dict, *, stage: str) -> FrozenRun:
    """Extract one decision's frozen inputs; the first-shot proxy removes feedback."""
    if stage not in STAGES:
        raise OfflineEvalError("unknown reader stage")
    if not snapshot.get("complete_candidate_pool"):
        raise IncompleteReplay("snapshot pool is incomplete")
    as_of = _time(snapshot["decision_at"])
    stages = snapshot.get("stage_results") or {}
    if "after_quality_gate" not in stages or "after_actionable_gate" not in stages:
        raise IncompleteReplay("snapshot lacks the scored-pool stage results")
    scored_ids = set(stages["after_quality_gate"])
    candidates = [c for c in snapshot["candidates"] if c["item_id"] in scored_ids]
    recorded = {int(c["item_id"]): json.loads(json.dumps(c.get("features") or {})) for c in candidates}
    modes = Counter(f.get("scoring_mode") for f in recorded.values() if f.get("scoring_mode"))
    if modes.get("hybrid_lr"):
        raise IncompleteReplay("hybrid_lr runs need a frozen LR artifact, which snapshots do not retain")
    retrospective = ["vote_reasons_current_state", "source_tiers_current_config",
                     "uncaptured_settings_current_defaults"]
    notes: dict[str, Any] = {"uncaptured_settings": list(_UNCAPTURED_SETTINGS),
                             "scored_pool": len(candidates)}
    votes, dropped = _frozen_votes(snapshot, as_of)
    notes["votes_dropped_at_or_after_decision"] = dropped
    matrix = np.asarray(snapshot["profile_matrix"], dtype=np.float32)
    penalties = {iid: float(f.get("reason_penalty", 0.0) or 0.0) for iid, f in recorded.items()}
    history = _viewed_history(snapshot, recorded, as_of)
    if stage == "first_shot_proxy":
        matrix, basis = _static_rows(snapshot)
        notes["static_profile"] = basis
        votes, penalties, history = [], {}, {}
        retrospective = [*retrospective, "pool_filtered_by_veteran_history",
                         "negative_interest_penalty_from_veteran_topic"]
        if basis not in VERIFIED_STATIC_BASES:
            retrospective.append(FIRST_SHOT_UNVERIFIED)
    return FrozenRun(
        snapshot_meta={key: snapshot.get(key) for key in (
            "run_id", "digest_id", "decision_at", "decision_local_date", "reading_mode", "catch_up",
            "exploration_slots", "section_budgets", "selection_settings", "ranking_settings",
            "reading_order_policy", "embedding_signature", "scorer_version")},
        as_of=as_of, stage=stage, rows=[_row(c) for c in candidates], recorded=recorded,
        actionable=set(stages["after_actionable_gate"]), votes=votes,
        reason_penalties=penalties, profile_matrix=matrix,
        downweight=list((snapshot.get("profile") or {}).get("downweight") or []),
        viewed_history=history, scoring_mode="hybrid_knn",
        retrospective_inputs=retrospective, freeze_notes=notes,
        recorded_facts={int(c["item_id"]): c.get("selection_facts") or {} for c in snapshot["candidates"]},
    )


def _viewed_history(snapshot: dict, recorded: dict[int, dict], as_of: datetime) -> dict:
    """Recover decision-time facet absence from recorded coverage bonuses.

    A recorded bonus b = scale * priority * absence; a qualified facet without a
    bonus had zero absence. The result reproduces the same bonuses when the
    strongest candidate per facet changes under another config.
    """
    settings = snapshot.get("ranking_settings") or {}
    scale = float(settings.get("topic_coverage_bonus_scale", 0.03) or 0.0)
    floor = float(settings.get("min_topic_relevance", 0.65))
    history: dict[str, datetime | None] = {}
    for feature in recorded.values():
        facet = str(feature.get("primary_facet") or "").strip()
        if not facet or float(feature.get("primary_facet_score", 0.0) or 0.0) < floor:
            continue
        key = facet.casefold()
        bonus = float(feature.get("topic_coverage_bonus", 0.0) or 0.0)
        priority = float(feature.get("topic_priority", 0.0) or 0.0)
        if bonus > 0 and scale > 0 and priority > 0:
            absence = min(1.0, bonus / (scale * priority))
            history[key] = None if absence >= 0.9999 else as_of - timedelta(days=7 * absence)
        else:
            history.setdefault(key, as_of)
    return history


# --------------------------------------------------------------------------- #
# Isolated serving boundary
# --------------------------------------------------------------------------- #


def _forbidden(name):
    def blocked(*_args, **_kwargs):
        raise IncompleteReplay(f"{name} is not available inside the frozen serving boundary")
    return blocked


def _frozen_datetime(clock: datetime):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock if tz is None else clock.astimezone(tz)
    return FrozenDatetime


def _pack_exemplars(run: FrozenRun, reasons: dict, embeddings: FrozenEmbeddings, item_rows) -> tuple:
    """Frozen equivalent of ``votes._load_vote_exemplars`` over decision-time votes."""
    from .. import votes as votes_mod

    pos, neg = [], []
    for vote in run.votes:
        iid, value = int(vote["item_id"]), int(vote["value"])
        if value not in (-1, 1) or iid not in item_rows:
            continue
        if not votes_mod.is_semantic_vote(value, reasons.get(str(iid)) or []):
            continue
        grade = vote.get("grade")
        grade = int(grade) if grade is not None else votes_mod.value_to_grade(value)
        weight = max(0.05, abs(grade - votes_mod.VOTE_GRADE_NEUTRAL) / 50.0)
        (pos if value > 0 else neg).append((iid, item_rows[iid], weight))

    def pack(items):
        if not items:
            return (np.zeros(0, dtype=np.int64), np.zeros((0, 1), np.float32), np.zeros(0, np.float32))
        vectors = embeddings.vectors([row for _iid, row, _w in items]).astype(np.float32)
        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
        return (np.array([iid for iid, _r, _w in items], dtype=np.int64), vectors,
                np.array([w for _i, _r, w in items], dtype=np.float32))

    return pack(pos), pack(neg)


@contextlib.contextmanager
def serving_boundary(*, settings, clock: datetime, embeddings: FrozenEmbeddings,
                     exemplars, signed_votes: int, viewed_history: dict):
    """Patch every current-state dependency of the serving scorer, then restore.

    The serving functions themselves run unchanged. Current settings, LR
    artifact, vote/feedback tables, embedding cache writes, wall clock and
    sockets are replaced or blocked for the duration.
    """
    from .. import config, pipeline, store
    from .. import votes as votes_mod
    from . import embedding_cache, ranker

    original_freshness = ranker._freshness_penalty

    def frozen_freshness(row):
        published = getattr(row, "published_at", None)
        if not isinstance(published, datetime):
            return original_freshness(row)
        reference = published if published.tzinfo else published.replace(tzinfo=timezone.utc)
        # Shift publication by (wall clock - decision) so the unchanged serving
        # function measures age at the decision. Sub-millisecond drift only.
        row.published_at = reference + (datetime.now(timezone.utc) - clock)
        try:
            return original_freshness(row)
        finally:
            row.published_at = published

    def no_network(*_args, **_kwargs):
        raise IncompleteReplay("network access is blocked inside the frozen serving boundary")

    def exemplar_loader():
        if exemplars is None:
            raise IncompleteReplay("vote exemplars were not frozen for this run")
        return exemplars()

    patches = [
        (config, "_cached_settings", lambda: settings),
        (config, "_recover_settings_files", lambda: False),
        (ranker, "embed_item_rows", embeddings.vectors),
        (votes_mod, "embed_item_rows", embeddings.vectors),
        (votes_mod, "_load_vote_exemplars", exemplar_loader),
        (ranker, "_vote_count", lambda: signed_votes),
        (ranker, "get_lr_ranker", lambda: None),
        (ranker, "_freshness_penalty", frozen_freshness),
        (pipeline, "recent_viewed_facet_dates", lambda **_kw: dict(viewed_history)),
        (pipeline, "datetime", _frozen_datetime(clock)),
        (socket.socket, "connect", no_network),
        (socket, "create_connection", no_network),
    ]
    for module in (store, votes_mod, pipeline, embedding_cache):
        for name in ("session_scope", "init_db"):
            if hasattr(module, name):
                patches.append((module, name, _forbidden(f"{module.__name__}.{name}")))
    saved = [(target, name, getattr(target, name)) for target, name, _value in patches]
    try:
        for target, name, value in patches:
            setattr(target, name, value)
        yield
    finally:
        for target, name, value in reversed(saved):
            setattr(target, name, value)


# --------------------------------------------------------------------------- #
# Frozen rescoring through the serving scorer
# --------------------------------------------------------------------------- #


def rescore_run(run: FrozenRun, config: EvalConfig, *, embeddings: FrozenEmbeddings,
                reasons: dict, item_rows: dict) -> dict:
    """Serving scorer and post-score path on frozen inputs; returns compact results."""
    from .. import pipeline
    from .enrich import _apply_venue_evidence, enrich_scored
    from .ranker import score_items_with_features
    from .source_quality import RANKER_VERSION

    meta = run.snapshot_meta
    settings = _frozen_settings(meta, config, run.scoring_mode)
    if getattr(settings, "semantic_matching", False) or getattr(settings, "within_day_dedupe", False) \
            or float(getattr(settings, "contribution_order_strength", 0.0)) \
            or getattr(settings, "contribution_reranker_path", ""):
        raise IncompleteReplay("semantic matching, within-day dedupe and contribution ordering are not frozen")
    embeddings.bind(meta["embedding_signature"])
    signed = sum(1 for vote in run.votes if int(vote["value"]) in (-1, 1))
    rows = [_fresh_row(row) for row in run.rows]
    recorded = run.recorded

    def offline_enrichment(items, _topic_scores):
        for row in items:
            entry = (recorded.get(int(row.id)) or {}).get("enrichment")
            if not isinstance(entry, dict) or not entry.get("quality_resolved_before_scoring"):
                continue
            evidence = {key: entry.get(key) for key in (
                "cited_by_count", "venue_impact", "venue", "status", "observed_at", "refresh_status")}
            evidence["status"] = entry.get("status") or "fresh"
            evidence["doi"] = entry.get("doi")
            evidence["quality_resolved_before_scoring"] = True
            row.enrichment_evidence = evidence
            _apply_venue_evidence(row, evidence)

    def exemplars():
        # Packed lazily inside the boundary; identical for every config of a run.
        if run.exemplar_cache is None:
            run.exemplar_cache = _pack_exemplars(run, reasons, embeddings, item_rows)
        return run.exemplar_cache

    from ..votes import MIN_VOTES_FOR_LR

    expected_mode = run.scoring_mode if signed >= MIN_VOTES_FOR_LR else "cosine"
    with serving_boundary(settings=settings, clock=run.as_of, embeddings=embeddings,
                          exemplars=exemplars if run.votes else None,
                          signed_votes=signed, viewed_history=run.viewed_history):
        scored, features = score_items_with_features(
            rows, run.profile_matrix, run.downweight, reason_penalty_map=run.reason_penalties,
            attribution=None, preference_weight=config.preference_weight,
            prepare_quality=offline_enrichment,
        )
        modes = {feature.get("scoring_mode") for feature in features.values()}
        if modes - {expected_mode}:
            # The serving scorer swallows preference failures and falls back to
            # cosine; a frozen evaluation must not report that as the config.
            raise IncompleteReplay(f"serving scorer fell back from {expected_mode} to {sorted(modes)}")
        for row, _score in scored:
            feature, old = features.setdefault(int(row.id), {}), recorded.get(int(row.id), {})
            for key in _ATTRIBUTION_KEYS:
                if key in old:
                    feature[key] = json.loads(json.dumps(old[key]))
        pre_gate = _post_score(scored, features, recorded)
        pickable = [(row, score) for row, score in pre_gate if int(row.id) in run.actionable]
        pickable = pipeline._filter_off_topic(pickable, features)
        gated = {int(row.id) for row, _score in pickable}
        pickable = enrich_scored(pickable, settings=settings, now=run.as_of, prepared_only=True)
        for row, score in pickable:
            features[int(row.id)]["final_score"] = round(float(score), 4)
        pickable = pipeline._apply_topic_selection_preferences(
            pickable, features, digest_id=meta["digest_id"])
        counterfactual = _counterfactual(meta, settings, pickable, features, RANKER_VERSION, run)
        slate = frozen_selection(counterfactual)
    order = {int(row.id): index for index, (row, _score) in enumerate(pickable)}
    score_after = {int(row.id): float(score) for row, score in pickable}
    pre = {int(row.id): float(score) for row, score in pre_gate}
    research = {}
    for row, _score in scored:
        if (row.section or "") != "research":
            continue
        iid, feature = int(row.id), features.get(int(row.id), {})
        research[iid] = {
            "topic_score": float(feature.get("topic_score", 0.0)),
            "preference_signal": float(feature.get("preference_signal", 0.0)),
            "confidence_score": float(feature.get("confidence_score", 0.0)),
            "ranking_score": score_after.get(iid, pre.get(iid, 0.0)),
            "eligible": iid in order, "topic_gate": iid in gated,
            "input_position": order.get(iid), "scoring_mode": feature.get("scoring_mode"),
            "primary_facet": str(feature.get("primary_facet") or "") or "unassigned",
            "primary_facet_score": float(feature.get("primary_facet_score", 0.0) or 0.0),
        }
    return {"config": config.name, "stage": run.stage, "slate": slate.get("research", []),
            "full_slate": slate, "research": research,
            "settings_signature": hashlib.sha256(
                json.dumps(settings.model_dump(mode="json"), sort_keys=True, default=str).encode()).hexdigest()}


def _fresh_row(row):
    from ..store import ItemRow

    copy = ItemRow(**{column: getattr(row, column) for column in (
        "id", "source", "section", "external_id", "url", "title", "abstract", "authors",
        "published_at", "fetched_at", "metadata_json")})
    return copy


def _post_score(scored, features, recorded):
    """Pipeline order after scoring, reusing recorded config-independent nudges."""
    adjusted = []
    for row, score in scored:
        old = recorded.get(int(row.id), {})
        if (row.section or "") in {"opportunities", "events"} and "opportunity_profile_score" in old:
            similarity = float(old["opportunity_profile_score"])
            features[int(row.id)]["opportunity_profile_score"] = similarity
            score = min(1.0, float(score) + 0.10 * similarity)
        adjusted.append((row, float(score)))
    if any("negative_interest_penalty" in recorded.get(int(row.id), {}) for row, _ in adjusted):
        penalized = []
        for row, score in adjusted:
            penalty = float(recorded.get(int(row.id), {}).get("negative_interest_penalty", 0.0) or 0.0)
            score = max(0.0, score - penalty)
            feature = features[int(row.id)]
            feature["negative_interest_penalty"] = round(penalty, 4)
            feature["confidence_score"] = round(max(0.0, feature["confidence_score"] - penalty), 4)
            feature["final_score"] = round(score, 4)
            penalized.append((row, score))
        adjusted = sorted(penalized, key=lambda pair: pair[1], reverse=True)
    if any(recorded.get(int(row.id), {}).get("author_boost") for row, _ in adjusted):
        boosted = []
        for row, score in adjusted:
            boost = float(recorded.get(int(row.id), {}).get("author_boost", 0.0) or 0.0)
            if boost > 0:
                score = min(1.0, score + boost)
                features[int(row.id)].update(author_boost=boost, final_score=round(score, 4))
            boosted.append((row, score))
        adjusted = sorted(boosted, key=lambda pair: pair[1], reverse=True)
    return adjusted


def _counterfactual(meta, settings, pickable, features, version, run) -> dict:
    """Minimal snapshot that lets ``frozen_selection`` select from new scores."""
    from .snapshot import SELECTION_SETTINGS

    by_id = {int(row.id): row for row in run.rows}
    candidates = []
    for position, (row, score) in enumerate(pickable):
        iid = int(row.id)
        candidates.append({
            "item_id": iid, "section": row.section or "", "eligible": True,
            "input_position": position, "ranking_score": float(score),
            "features": features.get(iid, {}), "family_id": f"item:{iid}",
            "metadata": {"source": by_id[iid].source, "title": by_id[iid].title,
                         "abstract": by_id[iid].abstract, "url": by_id[iid].url,
                         "external_id": by_id[iid].external_id,
                         "metadata_json": by_id[iid].metadata_json},
            "selection_facts": run.recorded_facts[iid],
        })
    selection = {name: getattr(settings, name) for name in SELECTION_SETTINGS}
    return {"schema_version": 1, "run_id": meta["run_id"], "digest_id": meta["digest_id"],
            "reading_mode": meta["reading_mode"], "catch_up": bool(meta["catch_up"]),
            "exploration_slots": int(meta.get("exploration_slots") or 0),
            "decision_local_date": meta["decision_local_date"],
            "reading_order_policy": meta.get("reading_order_policy") or "selection_priority",
            "complete_candidate_pool": True, "section_budgets": meta["section_budgets"],
            "selection_settings": selection, "scorer_version": version, "candidates": candidates}


# --------------------------------------------------------------------------- #
# Parity, funnel and slate metrics
# --------------------------------------------------------------------------- #


def rescore_parity(snapshot_research: dict[int, dict], recorded_slate_ids: list[int], result: dict) -> dict:
    """Compare a serving-default rescoring with what was actually recorded."""
    diffs = defaultdict(list)
    eligible_mismatch = 0
    for iid, new in result["research"].items():
        old = snapshot_research.get(iid)
        if old is None:
            continue
        for key in ("topic_score", "preference_signal", "confidence_score"):
            if key in old["features"]:
                diffs[key].append(abs(new[key] - float(old["features"][key])))
        if old["eligible"] and new["eligible"] and old["ranking_score"] is not None:
            diffs["ranking_score"].append(abs(new["ranking_score"] - float(old["ranking_score"])))
        eligible_mismatch += bool(old["eligible"]) != new["eligible"]
    summary = {key: {"n": len(values), "max_abs": max(values), "p99_abs": float(np.quantile(values, .99))}
               for key, values in diffs.items() if values}
    components_ok = all(summary.get(key, {"max_abs": 0.0})["max_abs"] <= _PARITY_TOLERANCE
                        for key in ("topic_score", "preference_signal"))
    slate = result["slate"]
    return {"component_parity": "ok" if components_ok else "failed",
            "tolerance": _PARITY_TOLERANCE, "abs_differences": summary,
            "eligibility_mismatches": eligible_mismatch,
            "research_slate_matches_recorded": slate == recorded_slate_ids,
            "research_slate_overlap": len(set(slate) & set(recorded_slate_ids)),
            "basis": ("Topic and kNN preference must reproduce (reconstruction check). Final score "
                      "and slate differences also include scorer code changed since the decision.")}


def _labels_for_run(labels: dict[int, dict], prior: set[int], as_of: datetime) -> dict[int, str]:
    out = {}
    for iid, label in labels.items():
        created = label.get("created_at")
        if iid in prior or (created is not None and created < as_of):
            out[iid] = "pre_decision"
        else:
            out[iid] = label_class(label)
    return out


def _slate_counts(ids, classes) -> dict:
    counts = Counter(classes.get(iid, "unknown") for iid in ids)
    counts.pop("pre_decision", None)
    n = len(ids)
    positive = sum(counts[c] for c in POSITIVE)
    unknown = n - sum(counts[c] for c in POSITIVE | NEGATIVE)
    gain = sum(UTILITY_GAIN.get(classes.get(iid, ""), 0.0) for iid in ids)
    return {"served": n, "rated": n - unknown, "unknown": unknown, "positive": positive,
            **{c: counts[c] for c in sorted(POSITIVE | NEGATIVE)},
            "precision_bounds": [positive / n, (positive + unknown) / n] if n else [None, None],
            "utility_per_slot_bounds": [gain / n, (gain + MAX_GAIN * unknown) / n] if n else [None, None]}


def _facet_errors(slates_by_run, research_by_run, classes_by_run) -> dict:
    table: dict[str, dict] = defaultdict(lambda: {"top3": Counter(), "top5": Counter(),
                                                  "runs_with_qualified_supply": 0, "runs_covered": 0})
    for run, slate in slates_by_run.items():
        research, classes = research_by_run[run], classes_by_run[run]
        for depth in (3, 5):
            for iid in slate[:depth]:
                bucket = classes.get(iid, "unknown")
                bucket = "unknown" if bucket == "pre_decision" else bucket
                table[research[iid]["primary_facet"]][f"top{depth}"][bucket] += 1
        qualified = {v["primary_facet"] for v in research.values() if v["eligible"]
                     and v["primary_facet"] != "unassigned"}
        covered = {research[iid]["primary_facet"] for iid in slate}
        for facet in qualified:
            table[facet]["runs_with_qualified_supply"] += 1
            table[facet]["runs_covered"] += facet in covered
    return {facet: {"top3": dict(row["top3"]), "top5": dict(row["top5"]),
                    "top3_errors": sum(v for k, v in row["top3"].items() if k in NEGATIVE),
                    "top5_errors": sum(v for k, v in row["top5"].items() if k in NEGATIVE),
                    "runs_with_qualified_supply": row["runs_with_qualified_supply"],
                    "runs_covered": row["runs_covered"]}
            for facet, row in sorted(table.items())}


# --------------------------------------------------------------------------- #
# Clustered (pigeonhole) bootstrap
# --------------------------------------------------------------------------- #


@dataclass
class _Design:
    runs: list[str]
    families: dict[tuple[str, int], int]
    n_families: int


def _weights(design: _Design, n_boot: int, seed: int):
    """Independent Poisson(1) weights for date clusters and work families.

    Owen's pigeonhole bootstrap for crossed effects: one day's items share a
    date weight and repeated exposures of a work share a family weight, so
    neither 11 items per day nor repeated works count as independent samples.
    Row 0 is the unit-weight point estimate.
    """
    rng = np.random.default_rng(seed)
    runs = rng.poisson(1.0, size=(n_boot + 1, len(design.runs))).astype(np.float64)
    fams = rng.poisson(1.0, size=(n_boot + 1, max(design.n_families, 1))).astype(np.float64)
    runs[0], fams[0] = 1.0, 1.0
    return runs, fams


def _summarize(values: np.ndarray, *, reference: float, reference_meaning: str) -> dict:
    point, draws = values[0], values[1:]
    draws = draws[np.isfinite(draws)]
    if not np.isfinite(point) or draws.size < 10:
        return {"estimate": None, "ci95": None, "reference": reference,
                "reference_meaning": reference_meaning}
    return {"estimate": round(float(point), 4),
            "ci95": [round(float(np.quantile(draws, .025)), 4), round(float(np.quantile(draws, .975)), 4)],
            "reference": reference, "reference_meaning": reference_meaning}


def _slate_mean(design, run_w, fam_w, slates, classes_by_run, value_of) -> np.ndarray:
    total = np.zeros(run_w.shape[0])
    weight = np.zeros(run_w.shape[0])
    for r, run in enumerate(design.runs):
        ids = slates.get(run) or []
        if not ids:
            continue
        idx = [design.families[(run, iid)] for iid in ids]
        wf = fam_w[:, idx]
        x = np.array([value_of(classes_by_run[run].get(iid, "unknown")) for iid in ids])
        den = wf.sum(axis=1)
        mean = np.divide((wf * x).sum(axis=1), den, out=np.full_like(den, np.nan), where=den > 0)
        valid = np.isfinite(mean)
        total[valid] += run_w[valid, r] * mean[valid]
        weight[valid] += run_w[valid, r]
    return np.divide(total, weight, out=np.full_like(total, np.nan), where=weight > 0)


def _delta_bounds(design, run_w, fam_w, base, cand, classes_by_run, gain_of, max_gain):
    """Partial-identification bounds on a paired slate difference.

    A shared unrated item has the same unknown label in both slates. Each
    unknown takes the label that minimizes (lower) or maximizes (upper) the
    difference, so no unrated paper is ever scored as negative or positive.
    """
    lo_total, hi_total, weight = (np.zeros(run_w.shape[0]) for _ in range(3))
    for r, run in enumerate(design.runs):
        b, c = base.get(run) or [], cand.get(run) or []
        if not b or not c:
            continue
        union = list(dict.fromkeys([*b, *c]))
        idx = [design.families[(run, iid)] for iid in union]
        wf = fam_w[:, idx]
        in_b = np.array([iid in b for iid in union], dtype=float)
        in_c = np.array([iid in c for iid in union], dtype=float)
        classes = [classes_by_run[run].get(iid, "unknown") for iid in union]
        known = np.array([cls in POSITIVE | NEGATIVE for cls in classes], dtype=float)
        x = np.array([gain_of(cls) for cls in classes])
        wb, wc = (wf * in_b).sum(axis=1), (wf * in_c).sum(axis=1)
        ok = (wb > 0) & (wc > 0)
        coef = np.zeros_like(wf)
        coef[ok] = wf[ok] * in_c / wc[ok, None] - wf[ok] * in_b / wb[ok, None]
        known_part = (coef * x * known).sum(axis=1)
        unknown = 1.0 - known
        lo = known_part + max_gain * (np.minimum(coef, 0.0) * unknown).sum(axis=1)
        hi = known_part + max_gain * (np.maximum(coef, 0.0) * unknown).sum(axis=1)
        lo_total[ok] += run_w[ok, r] * lo[ok]
        hi_total[ok] += run_w[ok, r] * hi[ok]
        weight[ok] += run_w[ok, r]
    safe = weight > 0
    lo = np.divide(lo_total, weight, out=np.full_like(weight, np.nan), where=safe)
    hi = np.divide(hi_total, weight, out=np.full_like(weight, np.nan), where=safe)
    return lo, hi


def _pairs(design, results_by_run, classes_by_run, negatives):
    """Within-run (relevant-or-better, negative) pairs on the labeled scored pool."""
    pairs = []
    for r, run in enumerate(design.runs):
        research, classes = results_by_run[run], classes_by_run[run]
        pos = [iid for iid in research if classes.get(iid) in POSITIVE]
        neg = [iid for iid in research if classes.get(iid) in negatives]
        pairs.extend((r, run, i, j) for i in pos for j in neg)
    return pairs


def _pair_correct(research, i, j) -> float:
    """Eligible-aware serving order: a gated-out item can never outrank an eligible one."""
    def key(iid):
        row = research[iid]
        return (1 if row["eligible"] else 0,
                -row["input_position"] if row["input_position"] is not None else row["ranking_score"])
    a, b = key(i), key(j)
    return 1.0 if a > b else 0.5 if a == b else 0.0


def _auc(design, run_w, fam_w, pairs, correct) -> np.ndarray:
    if not pairs:
        return np.full(run_w.shape[0], np.nan)
    r = np.array([p[0] for p in pairs])
    fi = np.array([design.families[(p[1], p[2])] for p in pairs])
    fj = np.array([design.families[(p[1], p[3])] for p in pairs])
    w = run_w[:, r] * fam_w[:, fi] * fam_w[:, fj]
    den = w.sum(axis=1)
    return np.divide((w * correct).sum(axis=1), den, out=np.full_like(den, np.nan), where=den > 0)


# --------------------------------------------------------------------------- #
# Ship rule
# --------------------------------------------------------------------------- #


def decide(comparison: dict, *, margins: dict, min_runs: int, min_strict_pairs: int) -> dict:
    """Decide better / non_inferior / worse / insufficient_evidence.

    ``margins`` must name ``precision`` and ``auc`` explicitly; zero tolerance
    is margin 0. Precision uses the conservative interval [lower CI of the
    worst-case difference, upper CI of the best-case difference]. The strict
    contrast (relevant-or-better vs Not relevant) uses its paired difference CI,
    whose null reference is 0, not the 0.5 chance level of a single AUC.
    """
    if not isinstance(margins, dict) or set(margins) != {"precision", "auc"}:
        raise OfflineEvalError("margins must explicitly define precision and auc")
    m_prec, m_auc = (float(margins[key]) for key in ("precision", "auc"))
    if not all(math.isfinite(v) and v >= 0 for v in (m_prec, m_auc)):
        raise OfflineEvalError("margins must be finite and nonnegative")
    reasons = []
    if comparison.get("n_runs", 0) < min_runs:
        reasons.append(f"fewer than {min_runs} comparable runs")
    if comparison.get("parity_failures"):
        reasons.append("reconstruction parity failed on some runs")
    prec = comparison.get("precision_delta_interval")
    strict = comparison.get("contrasts", {}).get(PRIMARY_CONTRAST, {})
    delta = strict.get("delta", {})
    if prec is None or prec[0] is None or prec[1] is None:
        reasons.append("precision difference interval undefined")
    if strict.get("pairs", 0) < min_strict_pairs or delta.get("ci95") is None:
        reasons.append(f"fewer than {min_strict_pairs} strict-negative pairs")
    rule = {"margins": {"precision": m_prec, "auc": m_auc}, "min_runs": min_runs,
            "min_strict_pairs": min_strict_pairs}
    if reasons:
        return {"decision": "insufficient_evidence", "reasons": reasons, "rule": rule}
    lo, hi = prec
    a, b = delta["ci95"]
    if hi < -m_prec or b < -m_auc:
        return {"decision": "worse", "reasons": ["CI lies entirely beyond the margin on the losing side"],
                "rule": rule}
    noninferior = lo >= -m_prec and a >= -m_auc
    if noninferior and (lo > 0 or a > 0):
        return {"decision": "better", "reasons": ["non-inferior on both and CI excludes zero on one"],
                "rule": rule}
    if noninferior:
        return {"decision": "non_inferior", "reasons": ["CIs stay within the explicit margins"],
                "rule": rule}
    return {"decision": "insufficient_evidence",
            "reasons": ["CIs cross the non-inferiority margin; unrated new picks stay unknown"], "rule": rule}


def combine_decisions(by_stage: dict[str, dict]) -> str:
    values = [row["decision"] for row in by_stage.values()]
    if "worse" in values:
        return "worse"
    if "insufficient_evidence" in values or not values:
        return "insufficient_evidence"
    return "better" if "better" in values else "non_inferior"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _compare(design, run_w, fam_w, base, cand, classes_by_run) -> dict:
    lo, hi = _delta_bounds(design, run_w, fam_w, base["slates"], cand["slates"], classes_by_run,
                           lambda c: 1.0 if c in POSITIVE else 0.0, 1.0)
    ulo, uhi = _delta_bounds(design, run_w, fam_w, base["slates"], cand["slates"], classes_by_run,
                             lambda c: UTILITY_GAIN.get(c, 0.0), MAX_GAIN)
    lo_s, hi_s = (_summarize(v, reference=0.0, reference_meaning="no difference") for v in (lo, hi))
    ulo_s, uhi_s = (_summarize(v, reference=0.0, reference_meaning="no difference") for v in (ulo, uhi))
    contrasts = {}
    for name, negatives in CONTRASTS.items():
        pairs = _pairs(design, base["research"], classes_by_run, set(negatives))
        cb = np.array([_pair_correct(base["research"][p[1]], p[2], p[3]) for p in pairs])
        cc = np.array([_pair_correct(cand["research"][p[1]], p[2], p[3]) for p in pairs])
        auc_b, auc_c = _auc(design, run_w, fam_w, pairs, cb), _auc(design, run_w, fam_w, pairs, cc)
        contrasts[name] = {"pairs": len(pairs), "runs_with_pairs": len({p[0] for p in pairs}),
                           "delta": _summarize(auc_c - auc_b, reference=0.0,
                                               reference_meaning="no difference between configs")}
    kept, lost, gained = Counter(), Counter(), Counter()
    changed = unrated_new = 0
    for run in design.runs:
        b, c, classes = set(base["slates"][run]), set(cand["slates"][run]), classes_by_run[run]
        changed += len(c - b)
        for iid in b | c:
            cls = classes.get(iid, "unknown")
            if iid in c - b and cls not in POSITIVE | NEGATIVE:
                unrated_new += 1
            if cls not in POSITIVE:
                continue
            (kept if iid in b & c else lost if iid in b else gained)[cls] += 1
    return {"n_runs": int(sum(bool(base["slates"][r]) and bool(cand["slates"][r]) for r in design.runs)),
            "parity_failures": base.get("parity_failures", 0) + cand.get("parity_failures", 0),
            "precision_delta_bounds": {"worst_case": lo_s, "best_case": hi_s},
            "precision_delta_interval": [lo_s["ci95"][0] if lo_s["ci95"] else None,
                                         hi_s["ci95"][1] if hi_s["ci95"] else None],
            "utility_delta_bounds": {"worst_case": ulo_s, "best_case": uhi_s},
            "contrasts": contrasts,
            "slate_changes": {"items_replaced": changed, "new_items_unrated": unrated_new},
            "known_positives": {"kept": dict(kept), "lost": dict(lost), "gained": dict(gained)}}


def _config_summary(design, run_w, fam_w, data, classes_by_run, funnel) -> dict:
    slates = data["slates"]
    def stat(fn):
        return _slate_mean(design, run_w, fam_w, slates, classes_by_run, fn)
    rated = stat(lambda c: 1.0 if c in POSITIVE | NEGATIVE else 0.0)
    p_lo, p_hi = stat(lambda c: 1.0 if c in POSITIVE else 0.0), stat(lambda c: 0.0 if c in NEGATIVE else 1.0)
    u_lo = stat(lambda c: UTILITY_GAIN.get(c, 0.0))
    u_hi = stat(lambda c: UTILITY_GAIN.get(c, MAX_GAIN if c not in NEGATIVE else 0.0))
    totals = Counter()
    for run in design.runs:
        totals.update({k: v for k, v in _slate_counts(slates[run], classes_by_run[run]).items()
                       if isinstance(v, int)})
    contrasts = {}
    for name, negatives in CONTRASTS.items():
        pairs = _pairs(design, data["research"], classes_by_run, set(negatives))
        correct = np.array([_pair_correct(data["research"][p[1]], p[2], p[3]) for p in pairs])
        contrasts[name] = {"pairs": len(pairs), "pairwise_accuracy": _summarize(
            _auc(design, run_w, fam_w, pairs, correct), reference=0.5,
            reference_meaning="chance ordering of a single config")}
    return {"top_k_totals": dict(totals),
            "rated_fraction": _summarize(rated, reference=1.0, reference_meaning="fully rated"),
            "precision_lower": _summarize(p_lo, reference=0.0, reference_meaning="unrated counted as not useful"),
            "precision_upper": _summarize(p_hi, reference=1.0, reference_meaning="unrated counted as useful"),
            "utility_per_slot_lower": _summarize(u_lo, reference=0.0, reference_meaning="unrated gain 0"),
            "utility_per_slot_upper": _summarize(u_hi, reference=MAX_GAIN, reference_meaning="unrated gain 3"),
            "contrasts": contrasts,
            "per_facet": _facet_errors(slates, data["research"], classes_by_run),
            "funnel": funnel}


def _funnel(snapshot: dict, result: dict, classes: dict[int, str], families: dict) -> dict:
    """Research candidate supply and loss by stage, with unique works and known positives."""
    research = {c["item_id"] for c in snapshot["candidates"] if c["section"] == "research"}
    stages = [("captured_pool", research)]
    for name in ("after_reviewed", "after_shown_and_known", "after_identity_dedupe",
                 "after_cross_day_dedupe", "after_quality_gate", "after_actionable_gate"):
        stages.append((name, research & set(snapshot["stage_results"].get(name, []))))
    rescored = result["research"]
    stages.append(("scored_by_serving_scorer", set(rescored)))
    stages.append(("after_topic_gate", {i for i, v in rescored.items() if v["topic_gate"]}))
    stages.append(("eligible_for_selection", {i for i, v in rescored.items() if v["eligible"]}))
    stages.append(("selected", set(result["slate"])))
    out, previous = [], None
    for name, ids in stages:
        fams = {families.get(i, f"item:{i}") for i in ids}
        positives = sum(classes.get(i) in POSITIVE for i in ids)
        out.append({"stage": name, "candidates": len(ids), "unique_works": len(fams),
                    "lost_from_previous": None if previous is None else len(previous - ids),
                    "known_positives": positives})
        previous = ids
    return out


def evaluate(database: str | Path, configs: list[dict], *, baseline: str, margins: dict,
             stages=STAGES, n_boot: int = 2000, seed: int = 0, min_runs: int = 3,
             min_strict_pairs: int = 20, split_at: str | None = None, run_ids=None,
             reasons_path: str | Path | None = None, allow_local_encoder: bool = False) -> dict:
    """Evaluate configs on every retained snapshot; nothing here changes defaults."""
    parsed = [EvalConfig.parse(config) for config in configs]
    names = [config.name for config in parsed]
    if len(set(names)) != len(names) or baseline not in names:
        raise OfflineEvalError("config names must be unique and include the baseline")
    reasons = json.loads(Path(reasons_path).read_text()) if reasons_path else {}
    connection = _connect(database)
    try:
        labels = read_labels(connection)
        embeddings = FrozenEmbeddings(connection, allow_local_encoder=allow_local_encoder)
        index = [entry for entry in _snapshot_index(connection) if run_ids is None or entry[0] in run_ids]
        report_runs, family_input = [], {}
        for run_id, _created in index:
            snapshot = _load_snapshot(connection, run_id)
            info = _evaluate_snapshot(snapshot, parsed, stages, labels, reasons, embeddings, connection)
            info["segment"] = ("late" if split_at and _time(snapshot["decision_at"]) >= _time(split_at)
                               else "early" if split_at else "all")
            family_input[run_id] = {"candidates": [
                {"item_id": c["item_id"], "family_id": c.get("family_id"),
                 "identity_keys": c.get("identity_keys", [])}
                for c in snapshot["candidates"] if c["section"] == "research"]}
            # Per-run unique works use the snapshot's own family identities.
            own = {c["item_id"]: c.get("family_id") or f"item:{c['item_id']}"
                   for c in snapshot["candidates"]}
            info["funnel_by_stage"] = {
                stage: {name: _funnel(snapshot, result, info["classes"][stage], own)
                        for name, result in by_config.items() if "research" in result}
                for stage, by_config in info["results"].items()}
            report_runs.append(info)
            del snapshot
    finally:
        connection.close()
    from .digest_evaluation import _families

    joined = _families(family_input)
    segments = ["all"] + (["early", "late"] if split_at else [])
    analyses = {}
    for segment in segments:
        selected = [info for info in report_runs if segment == "all" or info["segment"] == segment]
        analyses[segment] = _analyse(selected, joined, parsed, baseline, stages, margins,
                                     n_boot=n_boot, seed=seed, min_runs=min_runs,
                                     min_strict_pairs=min_strict_pairs)
    return {
        "schema_version": 1, "version": VERSION, "evidence_class": "development",
        "evidence_basis": "All retained snapshots and votes were previously inspected; not a holdout.",
        "baseline": baseline, "configs": [vars(config) for config in parsed],
        "margins": margins, "bootstrap": {"replicates": n_boot, "seed": seed,
                                          "method": "pigeonhole Poisson weights over date clusters and work families, per reader stage"},
        "split_at": split_at, "label_basis": ("current latest votes; items rated before a decision are "
                                              "excluded from that run; unrated stays unknown"),
        "embedding_stand_in": dict(embeddings.stats) | {"local_encoder_allowed": allow_local_encoder,
                                                        "equivalence_probes": embeddings.equivalence},
        "runs": [_run_report(info) for info in report_runs],
        "analyses": analyses,
        "defaults_changed": False,
        "limitations": [
            "Labels exist only on items the reader was shown; the comparison is exposure-biased.",
            "Unrated new picks keep the precision interval wide; no unrated paper is scored either way.",
            "Rescoring uses current scorer code on frozen inputs; final scores can differ from history.",
            "Vote reasons, source tiers and a few uncaptured settings use current state (retrospective).",
            "The first-shot proxy is the same reader with feedback removed, filtered by veteran history.",
            "First-shot runs whose learned profile row cannot be removed are diagnostics only and never "
            "enter a decision; snapshots before profile_static_rows was recorded are usually such runs.",
            "One reader and a handful of days cannot establish general accuracy improvements.",
        ],
    }


def _evaluate_snapshot(snapshot, parsed, stages, labels, reasons, embeddings, connection) -> dict:
    run_id = snapshot["run_id"]
    as_of = _time(snapshot["decision_at"])
    research_recorded = {
        c["item_id"]: {"features": c.get("features") or {}, "eligible": bool(c["eligible"]),
                       "ranking_score": c.get("ranking_score")}
        for c in snapshot["candidates"] if c["section"] == "research"}
    recorded_research = recorded_slate(snapshot).get("research", [])
    info = {"run_id": run_id, "digest_id": snapshot["digest_id"], "decision_at": snapshot["decision_at"],
            "scorer_version": snapshot.get("scorer_version"), "recorded_research_slate": recorded_research,
            "results": {}, "classes": {}, "freeze": {}, "parity": {}, "errors": {}}
    item_rows = _exemplar_rows(snapshot, connection)
    signature = snapshot.get("embedding_signature")
    reencoded_before = embeddings.reencoded.get(signature, 0)  # memo is shared across runs
    for stage in stages:
        try:
            frozen = freeze_run(snapshot, stage=stage)
        except IncompleteReplay as exc:
            info["errors"][stage] = str(exc)
            continue
        # Items the reader had rated before the decision are training state,
        # never outcomes, for every stage (the first-shot pool is the same).
        prior = {int(v["item_id"]) for v in _frozen_votes(snapshot, as_of)[0] if int(v["value"]) in (-1, 1)}
        info["classes"][stage] = _labels_for_run(labels, prior, as_of)
        info["freeze"][stage] = {"retrospective_inputs": frozen.retrospective_inputs, **frozen.freeze_notes,
                                 "frozen_votes": len(frozen.votes)}
        info["results"][stage] = {}
        for config in parsed:
            try:
                result = rescore_run(frozen, config, embeddings=embeddings, reasons=reasons,
                                     item_rows=item_rows)
            except IncompleteReplay as exc:
                info["results"][stage][config.name] = {"error": str(exc)}
                continue
            info["results"][stage][config.name] = result
            if stage == "veteran" and config.preference_weight == 0.5 and not config.settings:
                info["parity"][config.name] = rescore_parity(research_recorded, recorded_research, result)
    reencoded = embeddings.reencoded.get(signature, 0) - reencoded_before
    if reencoded:
        probe = embeddings.equivalence.get(signature)
        extra = ["vectors_reencoded_locally"] + (["embedding_signature_superseded"] if probe else [])
        for notes in info["freeze"].values():
            notes["retrospective_inputs"] = [*notes["retrospective_inputs"], *extra]
            notes["reencoded_vectors"] = reencoded
            notes["equivalence_probe"] = probe
    info["selector_replay"] = _selector_replay(snapshot)
    return info



def _exemplar_rows(snapshot: dict, connection) -> dict:
    """Voted items' text for exemplar vector lookup (read-only current text)."""
    from ..store import ItemRow

    ids = sorted({int(v["item_id"]) for v in (snapshot.get("feedback_at_start") or {}).get("votes", [])})
    rows = {}
    for start in range(0, len(ids), 900):
        chunk = ids[start:start + 900]
        for iid, title, abstract in connection.execute(
                f"SELECT id, title, abstract FROM items WHERE id IN ({','.join('?' * len(chunk))})", chunk):
            rows[int(iid)] = ItemRow(id=int(iid), title=title or "", abstract=abstract or "")
    return rows


def _selector_replay(snapshot: dict) -> dict:
    """Mode A: real selection on recorded scores; says nothing about rescoring."""
    from .source_quality import RANKER_VERSION

    recorded = recorded_slate(snapshot)
    try:
        status = "matched" if frozen_selection(snapshot) == recorded else "mismatch"
        reason = None
    except IncompleteReplay as exc:
        status, reason = "incomplete", str(exc)
    try:
        current = frozen_selection({**snapshot, "scorer_version": RANKER_VERSION})
        diagnostic = "matched" if current == recorded else "mismatch"
    except IncompleteReplay as exc:
        diagnostic = f"incomplete: {exc}"
    return {"status": status, "reason": reason,
            "current_policy_on_recorded_scores": diagnostic,
            "scope": "selection from recorded scores and eligibility; not rescoring"}


def _run_report(info: dict) -> dict:
    out = {key: info[key] for key in ("run_id", "digest_id", "decision_at", "scorer_version", "segment",
                                      "freeze", "parity", "errors", "selector_replay")}
    out["evidence_class"] = "development"
    out["retrospective"] = any(row.get("retrospective_inputs") for row in info["freeze"].values())
    out["slates"] = {stage: {name: result.get("slate", result.get("error"))
                             for name, result in by_config.items()}
                     for stage, by_config in info["results"].items()}
    out["funnel"] = info.get("funnel_by_stage", {})
    return out


def _analyse(infos, joined, parsed, baseline, stages, margins, *, n_boot, seed, min_runs,
             min_strict_pairs) -> dict:
    result = {}
    for stage in stages:
        rescored = [info for info in infos if all(
            "research" in info["results"].get(stage, {}).get(config.name, {}) for config in parsed)]
        # A first-shot run that may still carry the learned row is a diagnostic
        # (its slates stay in the run report), never first-use evidence.
        unverified = [info["run_id"] for info in rescored if FIRST_SHOT_UNVERIFIED in
                      info["freeze"].get(stage, {}).get("retrospective_inputs", [])]
        usable = [info for info in rescored if info["run_id"] not in unverified]
        if not usable:
            status, reason = (("static_profile_unverified",
                               "learned row could not be removed from any run's profile")
                              if unverified else ("no_rescorable_runs", "no rescorable runs"))
            result[stage] = {"status": status, "unverified_runs": unverified,
                             "decisions": {c.name: {"decision": "insufficient_evidence",
                                                    "reasons": [reason]}
                                           for c in parsed if c.name != baseline}}
            continue
        runs = [info["run_id"] for info in usable]
        families: dict[tuple[str, int], int] = {}
        codes: dict[str, int] = {}
        for info in usable:
            for config in parsed:
                for iid in info["results"][stage][config.name]["research"]:
                    key = (info["run_id"], iid)
                    fam = joined.get(key, f"{info['run_id']}:{iid}")
                    families[key] = codes.setdefault(fam, len(codes))
        design = _Design(runs, families, len(codes))
        run_w, fam_w = _weights(design, n_boot, seed)
        classes = {info["run_id"]: info["classes"][stage] for info in usable}
        data = {}
        for config in parsed:
            parity_failures = sum(info["parity"].get(config.name, {}).get("component_parity") == "failed"
                                  for info in usable) if stage == "veteran" else 0
            data[config.name] = {
                "slates": {info["run_id"]: info["results"][stage][config.name]["slate"] for info in usable},
                "research": {info["run_id"]: info["results"][stage][config.name]["research"] for info in usable},
                "parity_failures": parity_failures,
            }
        summaries = {}
        for config in parsed:
            funnel = _sum_funnels([info["funnel_by_stage"][stage][config.name] for info in usable])
            summaries[config.name] = _config_summary(design, run_w, fam_w, data[config.name], classes, funnel)
        comparisons, decisions = {}, {}
        for config in parsed:
            if config.name == baseline:
                continue
            comparison = _compare(design, run_w, fam_w, data[baseline], data[config.name], classes)
            comparisons[config.name] = comparison
            decisions[config.name] = decide(comparison, margins=margins, min_runs=min_runs,
                                            min_strict_pairs=min_strict_pairs)
        result[stage] = {"runs": runs, "excluded_runs": [i["run_id"] for i in infos if i not in usable],
                         "unverified_runs": unverified,
                         "families": design.n_families, "configs": summaries,
                         "comparisons_vs_baseline": comparisons, "decisions": decisions}
    overall = {}
    for config in parsed:
        if config.name != baseline:
            by_stage = {stage: result[stage]["decisions"][config.name] for stage in stages}
            overall[config.name] = combine_decisions(by_stage)
    result["overall_decisions"] = overall
    return result


def _sum_funnels(funnels: list[list[dict]]) -> list[dict]:
    totals: dict[str, Counter] = {}
    order = []
    for funnel in funnels:
        for row in funnel:
            if row["stage"] not in totals:
                totals[row["stage"]] = Counter()
                order.append(row["stage"])
            totals[row["stage"]].update({k: v or 0 for k, v in row.items() if k != "stage"})
    return [{"stage": stage, **dict(totals[stage]),
             "basis": "sum over runs; unique works counted per run"} for stage in order]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _protected_output(path: Path, database: Path | None = None) -> bool:
    from ..config import get_settings

    try:
        data_dir = Path(get_settings().db_path).resolve().parent
    except Exception:  # noqa: BLE001
        data_dir = None
    repo_data = Path(__file__).resolve().parents[3] / "data"
    resolved = path.resolve()
    source_dir = Path(database).resolve().parent if database is not None else None
    return any(directory is not None and resolved.is_relative_to(directory)
               for directory in (data_dir, repo_data, source_dir))


def run_cli(*, database: Path, configs_path: Path, baseline: str, margin_precision: float,
            margin_auc: float, output: Path, **kwargs) -> dict:
    if _protected_output(output, database):
        raise OfflineEvalError("refusing to write evaluation output inside a data directory")
    configs = json.loads(Path(configs_path).read_text())
    report = evaluate(database, configs, baseline=baseline,
                      margins={"precision": margin_precision, "auc": margin_auc}, **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False, default=str) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True,
                        help="JSON list of {name, preference_weight, settings}")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--margin-precision", type=float, required=True,
                        help="Explicit non-inferiority margin; 0 means zero tolerance")
    parser.add_argument("--margin-auc", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; never overwritten")
    parser.add_argument("--reasons", type=Path)
    parser.add_argument("--split-at")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-runs", type=int, default=3)
    parser.add_argument("--min-strict-pairs", type=int, default=20)
    parser.add_argument("--allow-local-encoder", action="store_true")
    args = parser.parse_args(argv)
    report = run_cli(database=args.database, configs_path=args.configs, baseline=args.baseline,
                     margin_precision=args.margin_precision, margin_auc=args.margin_auc,
                     output=args.output, reasons_path=args.reasons, split_at=args.split_at,
                     n_boot=args.bootstrap, seed=args.seed, min_runs=args.min_runs,
                     min_strict_pairs=args.min_strict_pairs,
                     allow_local_encoder=args.allow_local_encoder)
    print(json.dumps(report["analyses"]["all"]["overall_decisions"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
