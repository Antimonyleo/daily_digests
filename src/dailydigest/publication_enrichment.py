"""Bounded, opt-in enrichment of publication identity with per-item cache state."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
import json

from .brew import BrewCancelled
from .publications import _get, _metadata, normalize_doi

_VERSION = 1
_LIMIT = 12
_TTLS = {'fresh': timedelta(days=7), 'missing': timedelta(days=1),
         'failed': timedelta(minutes=15)}
_EVIDENCE_KEYS = ('publication_types', 'publication_type_evidence',
                  'companion_of_dois', 'companion_relation_evidence')


def merge_publication_metadata(existing: dict, incoming: dict) -> dict:
    """Add trusted identity evidence without overwriting unrelated or prior facts."""
    merged = dict(existing)
    for key in _EVIDENCE_KEYS:
        additions = incoming.get(key)
        if not isinstance(additions, list):
            continue
        previous = merged.get(key, [])
        values = list(previous) if isinstance(previous, list) else [previous]
        for value in additions:
            if value not in values:
                values.append(value)
        if values:
            merged[key] = values
    if incoming.get('is_retracted') is True:
        merged['is_retracted'] = True
    return merged


def _timestamp(value):
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    except (TypeError, ValueError):
        return None


def _cached(metadata, now):
    cache = metadata.get('publication_enrichment')
    if not isinstance(cache, dict) or cache.get('version') != _VERSION:
        return False
    status = cache.get('status')
    timestamp = _timestamp(cache.get('last_attempt_at' if status == 'failed' else 'checked_at'))
    ttl = _TTLS.get(status)
    if status == 'fresh' and cache.get('incomplete') is True:
        ttl = _TTLS['failed']
    return bool(timestamp and ttl and timedelta(0) <= now - timestamp < ttl)


def _write(row, metadata):
    if _metadata(row) == metadata:
        return
    encoded = json.dumps(metadata, ensure_ascii=False)
    if isinstance(row, dict):
        row['metadata_json'] = encoded
        if isinstance(row.get('metadata'), dict):
            row['metadata'] = metadata
    else:
        row.metadata_json = encoded


def enrich_publications(rows, *, settings, fetcher=None, now=None, check_cancel=None):
    """Mutate detached ItemRow objects or dictionaries; caller persists them.

    Input order defines relevance priority. A single batch requests at most twelve
    unique DOIs; profiles and article text are never sent. Failed attempts have a
    short backoff and preserve both prior facts and the last successful check.
    Partial successful lookups retain their facts but retry after fifteen minutes.
    """
    diagnostics = {'enabled': bool(getattr(settings, 'publication_enrichment', False)),
                   'requested': 0, 'cached': 0, 'fresh': 0, 'missing': 0,
                   'failed': 0, 'without_doi': 0, 'deferred': 0}
    if not diagnostics['enabled']:
        return diagnostics
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    groups = {}
    for row in rows:
        metadata = _metadata(row)
        doi = next((normalized for value in (metadata.get('doi'), _get(row, 'external_id'),
                                            _get(row, 'url'))
                    if (normalized := normalize_doi(value))), None)
        if not doi:
            diagnostics['without_doi'] += 1
            continue
        groups.setdefault(doi, []).append(row)
    pending = {}
    for doi, duplicates in groups.items():
        # A cached same-DOI record also supplies evidence to a newly fetched copy.
        cached = next((_metadata(row) for row in duplicates if _cached(_metadata(row), now)), None)
        if cached is not None:
            diagnostics['cached'] += 1
            for row in duplicates:
                metadata = merge_publication_metadata(_metadata(row), cached)
                metadata['publication_enrichment'] = dict(cached['publication_enrichment'])
                _write(row, metadata)
        elif len(pending) < _LIMIT:
            pending[doi] = duplicates
        else:
            diagnostics['deferred'] += 1
    if not pending:
        return diagnostics
    if check_cancel:
        check_cancel()
    if fetcher is None:
        from .publication_fetch import fetch_publication_metadata
        fetcher = fetch_publication_metadata
    diagnostics['requested'] = len(pending)
    try:
        results = fetcher(list(pending), deadline_seconds=8.0)
    except BrewCancelled:
        raise
    except Exception:
        # A provider failure must not become a verified negative cache entry.
        results = {}
    if check_cancel:
        check_cancel()
    results = results if isinstance(results, Mapping) else {}
    for doi, duplicates in pending.items():
        result = results.get(doi, {})
        result = result if isinstance(result, Mapping) else {}
        status = result.get('status')
        evidence = result.get('metadata')
        if status not in _TTLS or not isinstance(evidence, dict):
            status, evidence = 'failed', {}
        diagnostics[status] += 1
        combined = {}
        for row in duplicates:
            combined = merge_publication_metadata(combined, _metadata(row))
        if status == 'fresh':
            combined = merge_publication_metadata(combined, evidence)
        for row in duplicates:
            metadata = merge_publication_metadata(_metadata(row), combined)
            old = metadata.get('publication_enrichment', {})
            cache = {'version': _VERSION, 'status': status}
            if status == 'fresh' and result.get('incomplete') is True:
                cache['incomplete'] = True
            if status == 'failed':
                if isinstance(old, dict) and old.get('checked_at'):
                    cache['checked_at'] = old['checked_at']
                cache['last_attempt_at'] = now.isoformat()
            else:
                cache['checked_at'] = now.isoformat()
            metadata['publication_enrichment'] = cache
            _write(row, metadata)
    return diagnostics
