"""Extract current author claims with their task and object, without scope labels.

This small extractive parser recognizes explicit author language. It does not
establish novelty, entailment or scientific quality. Unrecognized prose retains
the full abstract for ranking, so extraction uncertainty never becomes a gate.
"""
from __future__ import annotations

import re


FEATURE_PROTOCOL = {'version': 'author-claims-v1', 'max_claims': 3,
                    'fallback': 'original_title_abstract', 'role_is_scope_gate': False}
_BREAK = re.compile(r'(?<=[.!?;])\s+')
_VERBS = {
    'introduced_method': r'introduce(?:d)?|develop(?:ed)?|propose(?:d)?|present(?:ed)?',
    'engineered_system': r'engineer(?:ed)?|design(?:ed)?|construct(?:ed)?|program(?:med)?|synthesize(?:d)?',
    'application': r'use(?:d)?|apply|applied|employ(?:ed)?|evaluate(?:d)?|benchmark(?:ed)?',
    'characterization': r'characterize(?:d)?|characterise(?:d)?|reveal(?:ed)?|investigate(?:d)?|determine(?:d)?|demonstrate(?:d)?|show(?:ed)?|measure(?:d)?',
    'review': r'review(?:ed)?|survey(?:ed)?|summarize(?:d)?|summarise(?:d)?',
}
_TASK = re.compile(
    r'\b(?P<owner>we|this (?:work|study|paper|review))\s+'
    r'(?:(?:here|also|now|systematically|experimentally|computationally|rationally)\s+)*'
    r'(?P<task>' + '|'.join(_VERBS.values()) + r')\b', re.I)
_PRIOR = re.compile(r'^\s*(?:previously|earlier|before this study|in (?:our )?(?:previous|prior) work|in \d{4}|(?:for|in) future work)\b', re.I)
_AMBIGUOUS_PREFIX = re.compile(r'\b(?:not true|false|incorrect|hypothesiz\w*|speculat\w*|previous authors|prior authors|claim that)\b|["“‘]', re.I)


def contribution_claims(abstract: str) -> dict:
    """Keep exact spans; role describes a stated action, not a relevance judgment."""
    if not isinstance(abstract, str):
        raise ValueError('Abstract must be a string')
    claims, start, ambiguous = [], 0, False
    for boundary in [match.start() for match in _BREAK.finditer(abstract)] + [len(abstract)]:
        raw = abstract[start:boundary]
        sentence = raw.strip()
        offset = start + len(raw) - len(raw.lstrip())
        start = boundary
        if not sentence:
            continue
        if _PRIOR.match(sentence):
            ambiguous = True
            continue
        for match in _TASK.finditer(sentence):
            if _AMBIGUOUS_PREFIX.search(sentence[:match.start()]):
                ambiguous = True
                continue
            task = match.group('task')
            role = next(role for role, pattern in _VERBS.items()
                        if re.fullmatch(pattern, task, re.I))
            tail = sentence[match.end():]
            obj = tail.strip()
            if not obj or not re.search(r'\w', obj):
                continue
            # A characterization may concern an engineered object or a native
            # mechanism; the role alone does not decide that distinction.
            claims.append({'start': offset, 'end': offset + len(sentence),
                           'quote': sentence, 'role': role,
                           'task': task, 'object': obj,
                           'ownership': 'explicit_author_statement',
                           'meaning': 'Stated task/object; originality and correctness unverified'})
    return {'version': FEATURE_PROTOCOL['version'], 'status': 'stated_claims' if claims else 'uncertain',
            'claims': claims, 'scope_inferred': False, 'ambiguous_ownership': ambiguous}


def contribution_text(title: str, abstract: str) -> str:
    """Focus features on bounded explicit claims; retain full text when uncertain."""
    if not isinstance(title, str):
        raise ValueError('Title must be a string')
    extracted = contribution_claims(abstract)
    claims = extracted['claims']
    # Multiple actions within one sentence may overlap; keep the earliest span.
    quotes, last_end = [], -1
    for claim in claims:
        if claim['start'] < last_end:
            continue
        quotes.append(claim['quote'])
        last_end = claim['end']
    # Dense author-claim abstracts retain the full input rather than silently
    # deleting a later task/object because it follows routine methods.
    focused = (' '.join(quotes) if not extracted['ambiguous_ownership']
               and 0 < len(quotes) <= FEATURE_PROTOCOL['max_claims'] else abstract)
    return title + '. ' + focused
