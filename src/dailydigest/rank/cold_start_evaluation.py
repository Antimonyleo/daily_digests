"""First-session evaluation on captured pools, never a production relevance gate.

Labels in the bundled corpus are provisional. Unknowns remain unknown; smaller
slates cannot manufacture a successful top-five result. No live retrieval,
provider calls, user state, or fitting belongs in this module.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import re
from urllib.parse import unquote


INPUT_MODES = ('provided', 'context', 'intent', 'anchors', 'examples')


def _check_example_overlap(entry: dict, examples: list[str]) -> None:
    from ..dedupe import _extract_doi, canonicalize_url

    def normalized(text):
        return ' '.join(str(text).casefold().split())

    for example in examples:
        title_text = normalized(example)
        doi = _extract_doi(example)
        urls = {canonicalize_url(unquote(url).rstrip('.,;)'))
                for url in re.findall(r'https?://[^\s<>]+', example)}
        for item in entry['pool']:
            title = normalized(item.get('title', ''))
            item_doi = _extract_doi(str(item.get('metadata', {}).get('doi', '')),
                                    item.get('url', ''), item.get('external_id', ''))
            if ((title and (title == title_text or f'{title}.' == title_text
                            or title_text.startswith(title + '. ')))
                    or (doi and doi == item_doi)
                    or canonicalize_url(item.get('url', '')) in urls):
                raise ValueError('Representative examples overlap the evaluated candidate pool')


def prepare_profile(entry: dict, mode: str) -> dict:
    """Construct an explicit reader-input ablation, without inspecting labels.

    Examples must be separately supplied, never mined from judged positives.
    A single goal cannot be copied onto unrelated topics in a multi-topic reader.
    """
    if mode not in INPUT_MODES:
        raise ValueError('Unknown cold-start input mode')
    profile = deepcopy(entry['profile'])
    if mode == 'provided':
        _check_example_overlap(entry, profile.get('seed_works', []))
        return profile
    if mode == 'context':
        if not str(entry.get('intent', '')).strip():
            raise ValueError('An explicit research goal is required')
        profile['bio'] = entry['intent']
        _check_example_overlap(entry, profile.get('seed_works', []))
        return profile
    topics = list(profile.get('canonical_facets') or profile.get('keywords', []))
    intents = entry.get('topic_intents', {})
    if not intents and len(topics) == 1:
        intents = {topics[0]: entry.get('intent', '')}
    if not topics or any(not str(intents.get(topic, '')).strip() for topic in topics):
        raise ValueError('Explicit per-topic intent is required for every research topic')
    facets = profile.setdefault('canonical_facets', {})
    for topic in topics:
        facet = facets.setdefault(topic, {})
        if mode == 'anchors':
            facet['anchors'] = [intents[topic]]
        else:
            facet['description'] = intents[topic]
    if mode == 'examples':
        examples = entry.get('example_works')
        if not isinstance(examples, list) or not examples or any(
                not isinstance(text, str) or not text.strip() for text in examples):
            raise ValueError('Separately supplied representative example_works are required')
        profile['seed_works'] = list(examples)
    # Exact normalized identity only; paraphrases and alternate versions still
    # require independent review. This is not proof of semantic independence.
    _check_example_overlap(entry, profile.get('seed_works', []))
    return profile


def evaluate_first_slate(corpus: dict, decisions: list[dict]) -> dict:
    """Measure the first five in actual selection order, with full denominators."""
    expected = {(p['id'], item['url']) for p in corpus['profiles'] for item in p['pool']}
    observed = {(row['profile_id'], row['url']) for row in decisions}
    if observed != expected or len(observed) != len(decisions):
        raise ValueError('Decision population must match the complete captured pool')
    labels = {}
    for case in corpus['cases']:
        key = case['profile_id'], case['url']
        if key in labels or case['label'] not in {'aligned', 'off_task', 'uncertain'}:
            raise ValueError('Ambiguous or invalid pair labels')
        if key not in expected:
            raise ValueError('Judgment outside captured population')
        labels[key] = case['label']
    result = {
        'scope': 'Captured candidate pools; upstream retrieval recall is unmeasured',
        'label_status': 'Provisional fixture judgments, not independent reader validation',
        'target': 'Five decisively judged picks, at least four aligned, first three none off-task',
        'release_approved': False,
        'profiles': {},
    }
    for profile in corpus['profiles']:
        pid = profile['id']
        rows = [row for row in decisions if row['profile_id'] == pid]
        chosen = [row for row in rows if row['selected']]
        positions = [row.get('selected_position') for row in chosen]
        if (any(type(pos) is not int for pos in positions)
                or sorted(positions) != list(range(1, len(chosen) + 1))
                or any(row.get('selected_position') is not None for row in rows if not row['selected'])):
            raise ValueError('Selected positions must be complete, unique, and one-based')
        if any(row['selected'] and not row['eligible'] for row in rows):
            raise ValueError('Selected papers must be eligible')
        first = sorted(chosen, key=lambda row: row['selected_position'])[:5]
        slate = [{'url': row['url'], 'position': row['selected_position'],
                  'label': labels.get((pid, row['url']), 'unjudged')} for row in first]
        counts = Counter(row['label'] for row in slate)
        counts = {label: counts[label] for label in ('aligned', 'off_task', 'uncertain', 'unjudged')}
        unknown = counts['uncertain'] + counts['unjudged']
        first_three_unknown = sum(row['label'] in {'uncertain', 'unjudged'} for row in slate[:3])
        first_three_off = sum(row['label'] == 'off_task' for row in slate[:3])
        positives = [row for row in rows if labels.get((pid, row['url'])) == 'aligned']
        result['profiles'][pid] = {
            'selected_count': len(chosen), 'first_slate': slate, 'counts': counts,
            'useful_fraction_bounds': ([counts['aligned'] / len(first),
                                        (counts['aligned'] + unknown) / len(first)]
                                       if first else [None, None]),
            'first_three_off_task': first_three_off, 'first_three_unknown': first_three_unknown,
            'aligned_pool_retained': {'total': len(positives),
                                      'eligible': sum(row['eligible'] for row in positives),
                                      'selected': sum(row['selected'] for row in positives)},
            'screen_target': ('met_on_provisional_labels' if len(first) == 5
                              and counts['aligned'] >= 4 and not unknown and not first_three_off
                              and not first_three_unknown else 'not_demonstrated'),
        }
    return result


def compare_slates(corpus: dict, baseline: list[dict], candidate: list[dict]) -> dict:
    before = evaluate_first_slate(corpus, baseline)
    after = evaluate_first_slate(corpus, candidate)
    differences = {}
    for pid in before['profiles']:
        selected_before = {r['url'] for r in baseline if r['profile_id'] == pid and r['selected']}
        selected_after = {r['url'] for r in candidate if r['profile_id'] == pid and r['selected']}
        aligned = {c['url'] for c in corpus['cases'] if c['profile_id'] == pid and c['label'] == 'aligned'}
        eligible_before = {r['url'] for r in baseline if r['profile_id'] == pid and r['eligible']}
        eligible_after = {r['url'] for r in candidate if r['profile_id'] == pid and r['eligible']}
        first_before = {r['url'] for r in before['profiles'][pid]['first_slate']}
        first_after = {r['url'] for r in after['profiles'][pid]['first_slate']}
        differences[pid] = {'before': before['profiles'][pid], 'after': after['profiles'][pid],
                            'lost_aligned': sorted((selected_before - selected_after) & aligned),
                            'lost_eligible_aligned': sorted((eligible_before - eligible_after) & aligned),
                            'lost_first_five_aligned': sorted((first_before - first_after) & aligned),
                            'added_aligned': sorted((selected_after - selected_before) & aligned)}
    return {'profiles': differences, 'release_approved': False}


def review_packet(corpus: dict, *, profile_input: str = 'provided') -> dict:
    """Export all candidates in stable blinded order, without scores or labels.

    This prepares independent review; it does not pretend that review happened.
    Include unselected papers so judgments can expose missed positives.
    """
    cases = []
    for profile in corpus['profiles']:
        supplied = prepare_profile(profile, profile_input)
        for article in profile['pool']:
            identity = profile['id'] + '\0' + article['url']
            cases.append({'id': hashlib.sha256(identity.encode()).hexdigest()[:20],
                          'profile_id': profile['id'], 'intent': profile['intent'],
                          'topic_intents': deepcopy(profile.get('topic_intents', {})),
                          'supplied_profile': supplied, 'profile_input': profile_input,
                          'article': {key: article.get(key, '') for key in ('url', 'title', 'abstract')},
                          'label': None, 'rationale': '', 'reviewer': None})
    return {'status': 'awaiting_independent_review',
            'labels': ['aligned', 'off_task', 'uncertain'],
            'cases': sorted(cases, key=lambda case: case['id'])}


def reorder_selected_by_intent(corpus: dict, baseline: list[dict], alternative: list[dict]) -> list[dict]:
    """Offline ordering-only ablation; retain every baseline eligibility/pick.

    Alternative goal scores must exist for all selected items. This is explicitly
    semantic ordering, not a validated utility model or scientific quality score.
    No alternate threshold, source budget, or captured candidate may leak in.
    """
    import math

    evaluate_first_slate(corpus, baseline)
    evaluate_first_slate(corpus, alternative)
    by_pair = {(row['profile_id'], row['url']): row for row in alternative}
    result = deepcopy(baseline)
    for row in result:
        other = by_pair[row['profile_id'], row['url']]
        quality = row.get('stages', {}).get('after_quality_gate')
        if type(quality) is not bool or other.get('stages', {}).get('after_quality_gate') != quality:
            raise ValueError('Ordering study requires the same recorded quality gate')
        if row['selected']:
            score = other.get('features', {}).get('topic_score')
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError('Every selected candidate requires a finite intent score')
            row['ordering_experiment'] = {'intent_score': score,
                                         'baseline_position': row['selected_position']}
    for profile in corpus['profiles']:
        chosen = [row for row in result if row['profile_id'] == profile['id'] and row['selected']]
        chosen.sort(key=lambda row: (-row['ordering_experiment']['intent_score'],
                                    row['ordering_experiment']['baseline_position']))
        for position, row in enumerate(chosen, 1):
            row['selected_position'] = position
    return result
