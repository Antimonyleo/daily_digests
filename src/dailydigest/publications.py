"""Publication identity and explicit commentary relationships, without similarity guesses."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from urllib.parse import unquote, urlsplit


_LABELS = {
    'news and views': 'News & Views', 'news & views': 'News & Views',
    'research highlight': 'Research Highlight', 'research highlights': 'Research Highlight',
    'perspective': 'Perspective', 'review': 'Review', 'systematic review': 'Review',
    'meta-analysis': 'Review', 'meta analysis': 'Review',
    'commentary': 'Commentary', 'comment': 'Commentary', 'editorial': 'Editorial',
    'news': 'News', 'research article': 'Research article', 'preprint': 'Preprint',
    'protocol': 'Protocol', 'primer': 'Primer', 'letter': 'Letter',
}
_COMMENTARY_LABELS = {'News & Views', 'Research Highlight', 'Perspective',
                      'Commentary', 'Editorial', 'News', 'Review'}
RELATION_NAMES = frozenset({'is-commentary-on', 'is-review-of'})
_DOI = re.compile(r'10\.\d{4,9}/[^\s<>"?#]+', re.IGNORECASE)


def _get(row, key, default=None):
    return row.get(key, default) if isinstance(row, Mapping) else getattr(row, key, default)


def _metadata(row) -> dict:
    value = _get(row, 'metadata')
    if isinstance(value, dict):
        return value
    try:
        value = json.loads(_get(row, 'metadata_json', '') or '{}')
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def normalize_doi(value) -> str | None:
    """Normalize an identifier, not a DOI-like substring inside arbitrary text."""
    if not isinstance(value, str):
        return None
    value = unquote(value.strip())
    if value.lower().startswith(('http://', 'https://')):
        try:
            parsed = urlsplit(value)
            host = parsed.hostname
        except ValueError:
            return None
        if host not in {'doi.org', 'dx.doi.org'}:
            return None
        value = parsed.path.lstrip('/')
    value = re.sub(r'^doi:\s*', '', value, flags=re.IGNORECASE)
    value = value.rstrip('.,;')
    # Preserve balanced parentheses that are legitimate DOI suffixes.
    while value.endswith(')') and value.count(')') > value.count('('):
        value = value[:-1]
    return value.lower() if _DOI.fullmatch(value) else None


def retraction_label(row) -> str | None:
    """Identify explicit withdrawal/retraction evidence, without title guesses."""
    metadata = _metadata(row)
    values = metadata.get('publication_types', [])
    values = values if isinstance(values, (list, tuple)) else [values]
    kinds = {' '.join(value.casefold().split())
             for value in [*values, metadata.get('work_type', ''), metadata.get('preprint_type', '')]
             if isinstance(value, str)}
    if 'withdrawn' in kinds:
        return 'Withdrawn'
    if metadata.get('is_retracted') is True or 'retracted publication' in kinds:
        return 'Retracted'
    if kinds & {'retraction', 'retraction notice', 'retraction of publication'}:
        return 'Retraction notice'
    return None


def publication_label(row) -> str:
    """Prefer specific, declared genres; generic article records remain unknown."""
    if status := retraction_label(row):
        return status
    metadata = _metadata(row)
    values = metadata.get('publication_types', [])
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        values = []
    labels = {_LABELS.get(' '.join(v.casefold().split()))
              for v in [*values, metadata.get('work_type', '')] if isinstance(v, str)}
    for label in ('News & Views', 'Research Highlight', 'Perspective', 'Review',
                  'Commentary', 'Editorial', 'News', 'Protocol', 'Primer', 'Letter',
                  'Research article', 'Preprint'):
        if label in labels:
            return label
    # Dedicated preprint-server URLs establish status without title inference.
    try:
        host = urlsplit(_get(row, 'url', '') or '').hostname
    except (ValueError, TypeError):
        host = None
    if host in {'arxiv.org', 'www.arxiv.org', 'biorxiv.org', 'www.biorxiv.org',
                'medrxiv.org', 'www.medrxiv.org'}:
        return 'Preprint'
    return 'Publication'


def companion_target_dois(row) -> set[str]:
    """Read only directed, explicit about-work evidence, never bibliography links."""
    metadata = _metadata(row)
    values = [metadata.get('companion_of_doi')]
    multiple = metadata.get('companion_of_dois', [])
    if isinstance(multiple, (list, tuple)):
        values.extend(multiple)
    relation = metadata.get('relation', {})
    if isinstance(relation, dict):
        for name in RELATION_NAMES:
            entries = relation.get(name, [])
            if isinstance(entries, list):
                values.extend(e.get('id') for e in entries if isinstance(e, dict)
                              and e.get('id-type', 'doi') == 'doi')
    return {doi for value in values if (doi := normalize_doi(value))}


def _explicit_relation_targets(metadata, names=RELATION_NAMES) -> set[str]:
    values = []
    relation = metadata.get('relation', {})
    if isinstance(relation, dict):
        for name in names:
            entries = relation.get(name, [])
            if isinstance(entries, list):
                values.extend(e.get('id') for e in entries if isinstance(e, dict)
                              and e.get('id-type', 'doi') == 'doi')
    evidence = metadata.get('companion_relation_evidence', [])
    if isinstance(evidence, list):
        values.extend(e.get('doi') for e in evidence if isinstance(e, dict)
                      and e.get('rel') in names)
    return {doi for value in values if (doi := normalize_doi(value))}


def is_companion(row) -> bool:
    """A linked commentary, never an ordinary review's bibliography or self-link."""
    label = publication_label(row)
    targets = companion_target_dois(row) - _identity_dois(row)
    if label not in _COMMENTARY_LABELS or not targets:
        return False
    return label != 'Review' or bool(targets & _explicit_relation_targets(_metadata(row)))


def companion_metadata(row) -> dict:
    """Canonical directed evidence safe to copy only between identical works."""
    targets = companion_target_dois(row) - _identity_dois(row)
    if not targets:
        return {}
    if publication_label(row) == 'Review':
        targets &= _explicit_relation_targets(_metadata(row))
        if not targets:
            return {}
    review_targets = _explicit_relation_targets(_metadata(row), {'is-review-of'})
    return {'companion_of_dois': sorted(targets), 'companion_relation_evidence': [
        {'rel': 'is-review-of' if doi in review_targets else 'is-commentary-on', 'doi': doi}
        for doi in sorted(targets)
    ]}


def _identity_dois(row) -> set[str]:
    values = [_metadata(row).get('doi'), _get(row, 'external_id'), _get(row, 'url')]
    return {doi for value in values if (doi := normalize_doi(value))}


def _retracted(row) -> bool:
    return retraction_label(row) is not None


def companion_entries(parents, candidates) -> dict[int, list[dict]]:
    """Attach declared commentary to exact parent DOI identities without merging."""
    owners: dict[str, list[int]] = {}
    for parent in parents:
        parent_id = _get(parent, 'id', _get(parent, 'item_id'))
        if not isinstance(parent_id, int) or publication_label(parent) in _COMMENTARY_LABELS:
            continue
        parent_dois = _identity_dois(parent)
        published_doi = normalize_doi(_metadata(parent).get('published_doi'))
        if published_doi:
            parent_dois.add(published_doi)
        for doi in parent_dois:
            owners.setdefault(doi, []).append(parent_id)
    result: dict[int, list[dict]] = {}
    seen: set[tuple[int, str]] = set()
    for candidate in candidates:
        label = publication_label(candidate)
        if not is_companion(candidate) or retraction_label(candidate):
            continue
        title, url = _get(candidate, 'title', ''), _get(candidate, 'url', '')
        if not isinstance(title, str) or not title.strip() or not isinstance(url, str):
            continue
        try:
            parsed = urlsplit(url)
            safe = parsed.scheme in {'https', 'http'} and bool(parsed.hostname) and not parsed.username
        except ValueError:
            safe = False
        if not safe:
            continue
        own_dois = _identity_dois(candidate)
        identity = min(own_dois) if own_dois else url
        targets = companion_target_dois(candidate) - own_dois
        if label == 'Review':
            targets &= _explicit_relation_targets(_metadata(candidate))
        for doi in sorted(targets):
            for parent_id in owners.get(doi, []):
                if _get(candidate, 'id') == parent_id or (parent_id, identity) in seen:
                    continue
                result.setdefault(parent_id, []).append({
                    'title': title.strip(), 'url': url, 'publication_label': label,
                    'item_id': _get(candidate, 'id'),
                })
                seen.add((parent_id, identity))
    return result
