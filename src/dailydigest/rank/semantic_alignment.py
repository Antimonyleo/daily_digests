"""Bounded local topic verification with explicit text-coverage evidence.

Raw logits are not calibrated ranking scores. Existing quality/preference
ordering survives; the serving gate consumes only conservative match evidence.
"""
from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import re

from .semantic_assets import SEMANTIC_FILES, SEMANTIC_REPOSITORY, SEMANTIC_REVISION
from .semantic_windows import WINDOW_PROTOCOL

_BROWSER_MODEL_DIR = Path(__file__).resolve().parents[3] / "browser/.build-cache/semantic-model"
_BIBLIOGRAPHIC_STUB = re.compile(
    r'^[^.!?]+,\s*(?:Volume\s+\d+,\s*Issue\s+[^.!?]+|Ahead\s+of\s+Print)[.!?]?$', re.I)
_PAIR_CACHE = OrderedDict()


def _usable_abstract(row):
    text = " ".join((getattr(row, "abstract", "") or "").split())
    return len(text) >= 80 and not _BIBLIOGRAPHIC_STUB.fullmatch(text)


def _validate_summary(summary):
    try:
        logit = float(summary["logit"])
        windows, complete = summary.get("windows"), summary.get("complete")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Invalid semantic encoder result") from exc
    if (not math.isfinite(logit) or isinstance(windows, bool)
            or not isinstance(windows, int) or not 1 <= windows <= 3
            or not isinstance(complete, bool)):
        raise ValueError("Semantic encoder must return finite logits and valid window coverage")
    result = {"logit": logit, "windows": windows, "complete": complete}
    if "query_truncated" in summary:
        if not isinstance(summary["query_truncated"], bool) or summary["query_truncated"] and complete:
            raise ValueError("Truncated queries must report incomplete coverage")
        result["query_truncated"] = summary["query_truncated"]
    return result


def _score_records(backend, records, check_cancel):
    # Model object identity namespaces entries; changing verified model assets
    # creates a different backend. Full supplied text and query identify a pair.
    keys = [(backend, record.get("topic", ""), record["query"], record["title"], record["abstract"])
            for record in records]
    missing = [index for index, key in enumerate(keys) if key not in _PAIR_CACHE]
    fresh = backend.score_records([records[index] for index in missing],
                                  check_cancel=check_cancel) if missing else []
    if len(fresh) != len(missing):
        raise ValueError("Semantic encoder must return one result per article")
    validated = [_validate_summary(summary) for summary in fresh]
    if check_cancel is not None:
        check_cancel()
    for index, summary in zip(missing, validated, strict=True):
        _PAIR_CACHE[keys[index]] = summary
    result = []
    for key in keys:
        result.append(_PAIR_CACHE[key])
        _PAIR_CACHE.move_to_end(key)
    while len(_PAIR_CACHE) > 512:
        _PAIR_CACHE.popitem(last=False)
    return result


def _verify_files(root):
    for name, expected in SEMANTIC_FILES.items():
        path = root / name
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Semantic model checksum mismatch: {name}")


@lru_cache(maxsize=2)
def _build_backend(model_dir=None):
    """Resolve the pinned public model; failed downloads/verification abort work."""
    from .cross_encoder import LocalCrossEncoder

    if model_dir is not None:
        root = Path(model_dir).expanduser().resolve(strict=True)
    elif _BROWSER_MODEL_DIR.is_dir():
        root = _BROWSER_MODEL_DIR
    else:
        from huggingface_hub import hf_hub_download

        paths = {name: Path(hf_hub_download(
            repo_id=SEMANTIC_REPOSITORY, filename=name, revision=SEMANTIC_REVISION,
        )) for name in SEMANTIC_FILES}
        root = paths["config.json"].parent
        if any(path != root / name for name, path in paths.items()):
            raise ValueError("Pinned semantic model files must share a snapshot directory")
    _verify_files(root)
    return LocalCrossEncoder(root)


def _queries(profile):
    canonical = getattr(profile, "canonical_facets", {}) or {}
    topics = canonical or {name: None for name in getattr(profile, "keywords", [])}
    queries = {}
    for name, spec in topics.items():
        topic = str(name).strip()
        if not topic:
            continue
        parts = [getattr(spec, "description", ""), topic,
                 *(getattr(spec, "aliases", []) or []),
                 *(getattr(spec, "anchors", []) or [])]
        parts = list(dict.fromkeys(str(part).strip() for part in parts if str(part).strip()))
        queries[topic.casefold()] = (topic, ". ".join(parts))
    return queries


def apply_topic_alignment(scored, features, profile, *, model_dir=None,
                          check_cancel=None, limit=48):
    """Check at most one existing primary topic per shortlisted article.

    One best candidate per interest enters the shortlist, followed by up to 24
    leaders by current score, then raw topic score. Unknown primary assignments in
    multi-interest profiles keep their existing scores rather than invent intent.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 48:
        raise ValueError("Topic alignment limit must be an integer from 0 to 48")
    scored = list(scored)
    if any(not math.isfinite(float(score)) for _, score in scored):
        raise ValueError("Original topic alignment scores must be finite")
    queries = _queries(profile)
    candidates = []
    statuses = {}
    for index, (row, score) in enumerate(scored):
        if row.section != "research":
            continue
        if not _usable_abstract(row):
            statuses[index] = "insufficient_text"
            continue
        statuses[index] = "unassigned_interest"
        feature = features.get(row.id, {})
        primary = str(feature.get("primary_facet", "") or "").strip().casefold()
        assignment = queries.get(primary)
        if assignment is None and len(queries) == 1:
            assignment = next(iter(queries.values()))
        if assignment is None:
            continue
        statuses[index] = "outside_shortlist"
        raw = float(feature.get("topic_score", score))
        if not math.isfinite(raw):
            raise ValueError("Raw topic scores must be finite")
        candidates.append((index, assignment, raw, float(score)))
    candidates.sort(key=lambda candidate: (candidate[2], candidate[3]), reverse=True)
    selected, topics = [], set()
    for candidate in candidates:
        if len(selected) >= limit:
            break
        topic = candidate[1][0].casefold()
        if topic not in topics:
            selected.append(candidate)
            topics.add(topic)
    selected_indices = {candidate[0] for candidate in selected}
    for candidate in sorted(candidates, key=lambda candidate: candidate[3], reverse=True)[:24]:
        if len(selected) >= limit:
            break
        if candidate[0] not in selected_indices:
            selected.append(candidate)
            selected_indices.add(candidate[0])
    for candidate in candidates:
        if len(selected) >= limit:
            break
        if candidate[0] not in selected_indices:
            selected.append(candidate)
    if not selected:
        return scored
    if check_cancel is not None:
        check_cancel()
    records = [{"query": assignment[1], "topic": assignment[0],
                "title": str(scored[index][0].title or ""),
                "abstract": str(scored[index][0].abstract or "")}
               for index, assignment, _, _ in selected]
    summaries = _score_records(_build_backend(model_dir), records, check_cancel)
    updates = []
    for (index, assignment, _, old_score), summary in zip(selected, summaries, strict=True):
        logit = summary["logit"]
        windows, complete = summary["windows"], summary["complete"]
        score = old_score
        alignment = {
            "topic": assignment[0], "query": assignment[1], "logit": logit,
            "windows": windows, "complete": complete, "score": score,
            "previous_score": old_score, "kind": "local_joint_match_v1",
            "mode": "verification_only",
            "model_repository": SEMANTIC_REPOSITORY, "model_revision": SEMANTIC_REVISION,
            "window_protocol": WINDOW_PROTOCOL,
            "uncertain": logit < -9.5 and complete,
        }
        if "query_truncated" in summary:
            alignment["query_truncated"] = summary["query_truncated"]
        updates.append((index, score, alignment))
    if check_cancel is not None:
        check_cancel()
    for index, score, alignment in updates:
        row = scored[index][0]
        features.setdefault(row.id, {})["topic_alignment"] = alignment
        statuses[index] = "matched"
        scored[index] = (row, score)
    for index, status in statuses.items():
        features.setdefault(scored[index][0].id, {})["topic_alignment_status"] = status
    return scored
