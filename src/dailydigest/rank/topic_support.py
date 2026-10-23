"""Conservative textual support for a semantic topic assignment.

This is display evidence, never an eligibility gate or a scientific-quality
assessment. Missing terminology does not disprove relevance. Only the reader's
topics and explicit aliases are used, never invented synonyms.
"""
from __future__ import annotations

import re
from functools import lru_cache
from threading import local

from snowballstemmer.english_stemmer import EnglishStemmer

_LOCAL = local()
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_STOP = frozenset({'a', 'an', 'the', 'of', 'in', 'and', 'for', 'to', 'on', 'with'})


@lru_cache(maxsize=16384)
def _stem(word: str) -> str:
    if not word.isascii():
        return word
    if not hasattr(_LOCAL, "stemmer"):
        _LOCAL.stemmer = EnglishStemmer()
    return _LOCAL.stemmer.stemWord(word)


def _tokens(text: str) -> list[str]:
    return [_stem(word) for word in _WORDS.findall(text.casefold()) if word not in _STOP]


def topic_support(row, topic: str, aliases=()) -> dict:
    """Return an exact source excerpt or explicitly abstain.

    Short phrases must retain word order: 'proximity-induced' is not evidence
    for the reader's 'induced proximity' topic. Longer topics allow intervening
    words within one sentence and its title. Stemming handles inflections, not synonyms.
    """
    if not topic or not str(topic).strip():
        return {'status': 'unavailable'}
    terms = [str(topic), *(term for term in list(aliases)[:3] if isinstance(term, str))]
    title = str(getattr(row, 'title', '') or '')
    abstract = str(getattr(row, 'abstract', '') or '')
    passages = [('title', title), *[('abstract', text) for text in re.split(r'(?<=[.!?])\s+', abstract)]]
    tokenized = [(origin, text, _tokens(text)) for origin, text in passages if text.strip()]
    title_tokens = set(_tokens(title))
    for term in terms:
        wanted = _tokens(term)
        if not wanted:
            continue
        for origin, text, tokens in tokenized:
            if re.search(r"[㐀-鿿぀-ヿ가-힯]", term):
                found = term.casefold() in text.casefold()
            elif len(wanted) <= 2:
                found = any(tokens[i:i + len(wanted)] == wanted for i in range(len(tokens) - len(wanted) + 1))
            else:
                found = set(wanted).issubset(set(tokens) | title_tokens)
            if found:
                return {'status': 'text_supported', 'topic': topic, 'term': term,
                        'quote': text, 'origin': origin, 'title_only': not abstract.strip()}
    return {'status': 'semantic_only', 'topic': topic, 'title_only': not abstract.strip()}


def annotate_topic_support(scored, features, profile) -> None:
    from .ranker import _row_feature_key

    facets = getattr(profile, 'canonical_facets', {})
    for row, _score in scored:
        if row.section != 'research':
            continue
        feature = features.get(_row_feature_key(row))
        if feature is None:
            continue
        topic = feature.get('primary_facet') or ''
        facet = facets.get(topic)
        support = topic_support(row, topic, getattr(facet, 'aliases', ()))
        if support['status'] == 'semantic_only':
            # An embedding's strongest facet can differ from the interest
            # actually named in the text. Report that evidence without changing
            # attribution scores, weights, or candidate eligibility.
            for other in (facets or getattr(profile, 'keywords', ())):
                if other == topic:
                    continue
                alternative = topic_support(row, other, getattr(facets.get(other), 'aliases', ()))
                if alternative['status'] == 'text_supported':
                    support = alternative
                    break
        feature['topic_support'] = support
