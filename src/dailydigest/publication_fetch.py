"""Bounded public DOI metadata lookup; never infer relationships from citations."""
from __future__ import annotations

import json
import math
import time
from urllib.parse import quote

import httpx

from .brew import BrewCancelled
from .publications import normalize_doi, publication_label

_MAX_DOIS = 12
_MAX_REQUESTS = 4
_MAX_BYTES = 2 * 1024 * 1024
_EPMC = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
_CROSSREF = 'https://api.crossref.org/works/'


def _types(metadata, values, source, url):
    if not isinstance(values, list):
        return
    for value in values:
        if not isinstance(value, str) or not value.strip() or len(value) > 200:
            continue
        raw_value = value
        value = value.strip()
        if value.casefold() == 'research-article':
            value = 'Research Article'
        if value not in metadata.setdefault('publication_types', []):
            metadata['publication_types'].append(value)
            metadata.setdefault('publication_type_evidence', []).append(
                {'source': source, 'field': 'europepmc_pub_type' if source == 'Europe PMC' else 'crossref_type',
                 'value': raw_value, 'url': url})
        if value.casefold() == 'retracted publication':
            metadata['is_retracted'] = True


def _relation(metadata, own, target, relation, source, url):
    target = normalize_doi(target)
    if not target or target == own:
        return
    targets = metadata.setdefault('companion_of_dois', [])
    if target not in targets:
        targets.append(target)
    evidence = {'rel': relation, 'doi': target, 'source': source, 'url': url}
    if evidence not in metadata.setdefault('companion_relation_evidence', []):
        metadata['companion_relation_evidence'].append(evidence)


def fetch_publication_metadata(dois, *, deadline_seconds=8.0) -> dict[str, dict]:
    """Return fresh/missing/failed results for at most twelve normalized DOIs.

    One Europe PMC exact-DOI batch, optionally one explicit PMID-link batch,
    then Crossref lookups within a shared four-request budget. A fresh result
    means metadata was checked, not that its genre or parent is known. No
    redirects, retries, publisher crawling, reference-list inference or secrets.
    The eight-second budget is checked before requests and between body chunks;
    network phase/inactivity timeouts are at most two seconds. Native DNS/header
    handling can exceed this soft wall-clock budget. Fresh results with an
    interrupted or deferred relationship lookup carry incomplete=True so the
    caller can retry sooner without losing successfully retrieved type facts.
    """
    selected = []
    for raw in dois:
        doi = normalize_doi(raw)
        if doi and len(doi) <= 512 and '\\' not in doi and doi not in selected:
            selected.append(doi)
        if len(selected) >= _MAX_DOIS:
            break
    results = {doi: {'status': 'failed', 'metadata': {}} for doi in selected}
    if not selected:
        return results
    if not isinstance(deadline_seconds, (int, float)) or not math.isfinite(deadline_seconds):
        return results
    deadline = time.monotonic() + min(8.0, max(0.0, deadline_seconds))
    requests = 0

    def get_json(client, url, params=None):
        nonlocal requests
        remaining = deadline - time.monotonic()
        if remaining <= 0 or requests >= _MAX_REQUESTS:
            raise TimeoutError('Publication metadata budget exhausted')
        requests += 1
        with client.stream('GET', url, params=params,
                           timeout=httpx.Timeout(min(remaining, 2.0))) as response:
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_bytes():
                if time.monotonic() >= deadline:
                    raise TimeoutError('Publication metadata deadline exceeded')
                body.extend(chunk)
                if len(body) > _MAX_BYTES:
                    raise ValueError('Publication metadata response too large')
            value = json.loads(body)
            if not isinstance(value, dict):
                raise ValueError('Publication metadata response is not an object')
            return value

    def search(client, query):
        payload = get_json(client, _EPMC, {'query': query, 'format': 'json',
                                          'resultType': 'core', 'pageSize': 100})
        container = payload.get('resultList')
        if not isinstance(container, dict) or not isinstance(container.get('result'), list):
            raise ValueError('Missing Europe PMC results')
        records = container['result']
        # Truncated or malformed responses cannot establish absence.
        count = payload.get('hitCount')
        if (not isinstance(count, int) or count < 0 or count > len(records)
                or any(not isinstance(record, dict) for record in records)):
            raise ValueError('Incomplete Europe PMC results')
        return records

    failures = (httpx.HTTPError, ValueError, TypeError, TimeoutError, RuntimeError)
    with httpx.Client(follow_redirects=False, trust_env=False,
                      headers={'User-Agent': 'DailyDigest/0.1', 'Accept': 'application/json'}) as client:
        records = []
        try:
            records = search(client, ' OR '.join(f'DOI:"{doi}"' for doi in selected))
            for result in results.values():
                result['status'] = 'missing'
        except BrewCancelled:
            raise
        except failures:
            pass
        initial_records = []
        for record in records:
            doi = normalize_doi(record.get('doi'))
            if doi not in results:
                continue
            initial_records.append(record)
            result = results[doi]
            result['status'] = 'fresh'
            types = record.get('pubTypeList', {})
            if isinstance(types, dict):
                _types(result['metadata'], types.get('pubType', []), 'Europe PMC',
                       f'https://europepmc.org/article/{record.get("source", "MED")}/{record.get("id", "")}')

        # Only explicit MED CommentOn/CommentIn relationships qualify. Version,
        # citation, erratum and computational title-match links do not.
        links = []
        for record in initial_records:
            container = record.get('commentCorrectionList', {})
            entries = container.get('commentCorrection', []) if isinstance(container, dict) else []
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict) or entry.get('source') != 'MED':
                    continue
                kind = str(entry.get('type', '')).replace(' ', '').casefold()
                pmid = entry.get('id')
                if kind in {'commenton', 'commentin'} and isinstance(pmid, str) and pmid.isdigit():
                    links.append((normalize_doi(record.get('doi')), kind, pmid))
        pmids = list(dict.fromkeys(link[2] for link in links))[:_MAX_DOIS]
        link_lookup_failed = False
        if pmids:
            try:
                records += search(client, 'SRC:MED AND (' + ' OR '.join(f'EXT_ID:{p}' for p in pmids) + ')')
            except BrewCancelled:
                raise
            except failures:
                link_lookup_failed = True
        identities = {}
        for record in records:
            doi = normalize_doi(record.get('doi'))
            pmid = record.get('id')
            if doi and record.get('source') == 'MED' and isinstance(pmid, str):
                identities.setdefault(pmid, set()).add(doi)
        for own, kind, pmid in links:
            candidates = identities.get(pmid, set())
            if len(candidates) != 1:
                if link_lookup_failed or pmid not in pmids:
                    results[own]['incomplete'] = True
                continue
            other = next(iter(candidates))
            child, parent = (own, other) if kind == 'commenton' else (other, own)
            if child in results:
                _relation(results[child]['metadata'], child, parent, 'is-commentary-on',
                          'Europe PMC', f'https://europepmc.org/article/MED/{pmid}')

        # Commentary candidates first; arbitrary bibliography links never qualify.
        priority = [doi for doi in selected if publication_label(
            {'metadata': results[doi]['metadata']}) in {
                'Review', 'Commentary', 'Editorial', 'News', 'News & Views',
                'Perspective', 'Research Highlight',
            }]
        for doi in priority:
            if results[doi]['metadata'].get('companion_of_dois'):
                continue
            if requests >= _MAX_REQUESTS or time.monotonic() >= deadline:
                results[doi]['incomplete'] = True
                continue
            try:
                payload = get_json(client, _CROSSREF + quote(doi, safe=''))
                record = payload.get('message')
                if not isinstance(record, dict) or normalize_doi(record.get('DOI')) != doi:
                    raise ValueError('Crossref DOI identity mismatch')
                results[doi]['status'] = 'fresh'
                metadata = results[doi]['metadata']
                # Crossref's journal-article is not evidence of original research.
                if record.get('type') == 'peer-review':
                    _types(metadata, ['Review'], 'Crossref', _CROSSREF + quote(doi, safe=''))
                relations = record.get('relation', {})
                if isinstance(relations, dict):
                    for rel in ('is-commentary-on', 'is-review-of'):
                        entries = relations.get(rel, [])
                        if isinstance(entries, list):
                            for entry in entries:
                                if isinstance(entry, dict) and entry.get('id-type') == 'doi':
                                    _relation(metadata, doi, entry.get('id'), rel, 'Crossref',
                                              _CROSSREF + quote(doi, safe=''))
            except BrewCancelled:
                raise
            except failures:
                results[doi]['incomplete'] = True
    return results
