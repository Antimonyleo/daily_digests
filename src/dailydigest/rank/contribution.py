"""Small local contribution evidence signal for ordering qualified research.

This is an explicit text heuristic, not a trained semantic classifier. It quotes
abstract passages, allows multiple topics and leaves missing evidence neutral.
It never changes source quality, relevance scores or candidate eligibility.
"""
from __future__ import annotations

import math
import re
from copy import deepcopy
from types import SimpleNamespace


CONTRIBUTION_VERSION = 'contribution-evidence-v2'
_SENTENCES = re.compile(r'(?<=[.!?;])\s+')
_BACKGROUND = re.compile(r'\b(?:previous(?:ly)?|earlier work|prior work|recent studies|future work|could|might|may enable|motivates|commonly used)\b', re.I)
_NEGATION = re.compile(r'\b(?:do(?:es)? not|did not|neither|no new|not designed|not engineered|without (?:design|engineering|developing))\b', re.I)
_OWN = re.compile(r'\b(?:we|here|this (?:work|study|paper))\b', re.I)
_ENGINEERING = re.compile(r'\b(?:design(?:ed|ing)?|engineer(?:ed|ing)?|program(?:med|ming)?|construct(?:ed)?)\b', re.I)
_METHOD = re.compile(r'\b(?:develop(?:ed)?|introduc(?:e|ed)|propos(?:e|ed)|present(?:ed)?)\b', re.I)
_METHOD_OBJECT = re.compile(r'\b(?:method|model|platform|framework|algorithm|predictor|tool|approach|pipeline|library|toolkit)\b', re.I)
_NATURAL = re.compile(r'\b(?:native|natural(?:ly)?|cellular|signaling|signalling|mechanism|endogenous|physiological)\b', re.I)
_OBSERVATION = re.compile(r'\b(?:reveal|determine|characteri[sz]e|investigate|elucidate|discover|study|describe)(?:s|d)?\b', re.I)
_USE = re.compile(r'\b(?:use|using|apply|applied|employ)(?:s|d)?\b', re.I)
_REVIEW = re.compile(r'\b(?:this review|we review|in this perspective|we summarize|we survey|review article)\b', re.I)
_ACTOR = r'\b(?:we|this (?:work|study|paper))\s+(?:(?:also|now|rationally|computationally|experimentally|successfully|systematically)\s+)*'
_ENGINEERING_CLAIM = re.compile(_ACTOR + _ENGINEERING.pattern.removeprefix(r'\b'), re.I)
_METHOD_CLAIM = re.compile(_ACTOR + _METHOD.pattern.removeprefix(r'\b'), re.I)
_EXISTING = re.compile(r'\b(?:existing|established|off.the.shelf|previously published)\b', re.I)
_ENGINEERED_OBJECT = re.compile(r'^\s*(?:(?:a|an|the|new|novel|programmable|synthetic|de novo|self.assembling|modular|orthogonal|functional|fluorescent|patchy|active|colloidal|protein|rna|dna|delivery|gene.therapy)\s+)*(?:proteins?|enzymes?|peptides?|binders?|scaffolds?|origami|tiles?|nanostructures?|wireframes?|colloids?|particles?|nanoparticles?|crystals?|materials?|vectors?|assemblies|nanodevices?)\b', re.I)
_DESIGN_USE = re.compile(r'\bwe\s+used?\s+(?:structure.guided\s+)?(?:rational\s+)?(?:design|directed evolution)\b.*\bto\s+(?:produce|create|generate|engineer)\b', re.I)
_TO_ENGINEER = re.compile(r'\bwe\s+used?\b[^.;!?]*?\bto\s+(?:engineer|design|construct|program)\b', re.I)
_NATURAL_MODEL = re.compile(r'\b(?:chromatin|nucleosome|native|natural(?:ly)?|cellular|signaling|signalling)\b', re.I)
_CLINICAL_METHOD = re.compile(r'\b(?:diagnostic|prognostic|mortality|hospital|biomarkers?|disease risk)\b', re.I)
_METHOD_TARGETS = {
    'Protein engineering and design': r'\b(?:protein engineering|protein design|enzyme design|(?:protein|enzyme|peptide) optimization|(?:protein|enzyme|peptide|binder) variants?|(?:protein|enzyme|peptide|binder)s? librar\w*|library of .*?(?:proteins?|enzymes?|peptides?|binders?)|(?:fluorescent )?protein binders?)\b',
    'Self-assembled functional materials': r'\b(?:self.assembl\w* materials?|superlattices?|photonic crystals?|framework materials|programmable|engineered)\b',
    'Gene and RNA therapeutic engineering': r'\b(?:delivery|gene therapy|gene.editing|therapeutic engineering)\b',
    'Colloidal self-assembly': r'\b(?:self.assembl\w*|crystalli[sz]\w*|patchy|programmable|active colloids?|dna.mediated assembly)\b',
}

# Contribution-specific vocabularies supplement exact reader aliases. Other
# profiles use their own aliases directly, without assuming these topic names.
_DOMAINS = {
    'Protein engineering and design': r'\b(?:proteins?|enzymes?|peptides?|binders?)\b',
    'Protein structure and function prediction': r'\b(?:alphafold|protein (?:structure|function) prediction|predict(?:ing|ion of) protein)\b',
    'DNA nanotechnology': r'\b(?:dna (?:origami|tiles?|nanostructures?|nanotechnology|wireframes?)|dna-mediated assembly)\b',
    'RNA nanotechnology': r'\b(?:rna (?:origami|tectonics|nanostructures?|nanotechnology)|kissing-loop assembl\w*)\b',
    'Colloidal self-assembly': r'\b(?:colloids?|colloidal|patchy particles|dna-mediated colloidal)\b',
    'Self-assembled functional materials': r'\b(?:self-assembl\w*|superlattices?|photonic crystals?|framework materials|condensates?)\b',
    'Gene and RNA therapeutic engineering': r'\b(?:mrna|rna delivery|nucleic.acid delivery|gene therapy|gene.editing|delivery vectors?)\b',
    'Machine learning for biomolecular design': r'\b(?:biomolecular design|molecular design|generative protein|protein language model)\b',
}


def _patterns(facets: dict) -> dict[str, re.Pattern]:
    result = {}
    for name, definition in facets.items():
        aliases = definition.get('aliases', []) if isinstance(definition, dict) else getattr(definition, 'aliases', [])
        phrases = [name, *aliases]
        terms = [r'(?<!\w)' + re.escape(term) + r'(?!\w)' for term in phrases if isinstance(term, str) and term.strip()]
        if name in _DOMAINS:
            terms.append(_DOMAINS[name])
        result[name] = re.compile('|'.join(terms), re.I)
    return result


def analyze_contribution(title: str, abstract: str, facets: dict) -> dict:
    """Find a contribution clause with topic evidence in that same clause.

    Background, future and negated claims cannot provide a positive signal.
    Titles can help identify topics, but only abstract passages justify priority.
    Evidence means an author's stated contribution, not independent validation.
    """
    patterns = _patterns(facets)
    abstract = abstract or ''
    matches = lambda text: [name for name, pattern in patterns.items() if pattern.search(text)]
    result = {'version': CONTRIBUTION_VERSION, 'kind': 'unknown', 'priority': 0,
              'facets': matches((title or '') + ' ' + abstract), 'evidence': [],
              'evidence_meaning': 'Explicit abstract text heuristic; author claim, not quality or probability'}
    if len(abstract.strip()) < 80:
        result['reason'] = 'insufficient_abstract'
        return result
    sentences = [s.strip() for s in _SENTENCES.split(abstract) if s.strip()]
    if re.search(r'\b(?:this review|review article|in this perspective)\b', abstract, re.I):
        result.update(kind='review', priority=-1 if result['facets'] else 0,
                      evidence=[next(s for s in sentences if _REVIEW.search(s))], reason='review_contribution')
        return result
    context = None
    for sentence in sentences:
        if _BACKGROUND.search(sentence) or _NEGATION.search(sentence):
            continue
        topics = matches(sentence)
        if not topics or not _OWN.search(sentence):
            continue
        engineering, method, use = _ENGINEERING_CLAIM.search(sentence), _METHOD_CLAIM.search(sentence), _USE.search(sentence)
        design_use = _DESIGN_USE.search(sentence)
        tool_engineering = _TO_ENGINEER.search(sentence)
        tool_output = sentence[tool_engineering.end():] if tool_engineering else ''
        if design_use or (tool_engineering and _ENGINEERED_OBJECT.search(tool_output) and matches(tool_output)) or (engineering and _ENGINEERED_OBJECT.search(sentence[engineering.end():]) and matches(sentence[engineering.end():]) and (not use or engineering.start() < use.start())):
            result.update(kind='engineering', priority=1, facets=topics,
                          evidence=[sentence], reason='stated_engineering_contribution')
            return result
        target_text = sentence[method.end():] if method else ''
        method_topics = [name for name in matches(target_text) if name not in _METHOD_TARGETS or re.search(_METHOD_TARGETS[name], target_text, re.I)]
        method_object = re.split(r'\busing\b', target_text, maxsplit=1, flags=re.I)[0]
        if method and method_topics and _METHOD_OBJECT.search(target_text) and not _EXISTING.search(method_object) and not _CLINICAL_METHOD.search(method_object) and (not use or method.start() < use.start()):
            result.update(kind='method', priority=1, facets=method_topics,
                          evidence=[sentence], reason='stated_transferable_method')
            return result
        observation = _OBSERVATION.search(sentence)
        if (_NATURAL.search(sentence) and observation and (not use or observation.start() < use.start())) or (method and _NATURAL_MODEL.search(target_text)):
            context = context or ('natural_mechanism', sentence, topics)
        elif use:
            context = context or ('application', sentence, topics)
    if context:
        kind, sentence, topics = context
        result.update(kind=kind, priority=-1, facets=topics, evidence=[sentence], reason='contextual_contribution')
    elif any(_REVIEW.search(s) for s in sentences):
        result.update(kind='review', priority=-1 if result['facets'] else 0,
                      evidence=[next(s for s in sentences if _REVIEW.search(s))], reason='review_contribution')
    else:
        result['reason'] = 'no_explicit_contribution_evidence'
    return result


def rerank_contributions(scored, features: dict, facets: dict, *, strength: float):
    """Reorder with a bounded evidence signal, retaining every original score.

    Existing topic/coverage ordering bonuses are retained. Call after upstream
    eligibility gates, so text evidence cannot rescue an ineligible candidate.
    """
    if not math.isfinite(strength) or not 0 <= strength <= .1:
        raise ValueError('Contribution ordering strength must be between 0 and .1')
    ordering = []
    for row, score in scored:
        feature = features.setdefault(int(row.id), {})
        adjustment = 0.
        if row.section == 'research':
            evidence = analyze_contribution(row.title, row.abstract, facets)
            adjustment = strength * evidence['priority']
            feature['contribution'] = evidence
            feature['contribution_order_adjustment'] = adjustment
        ordering.append((row, score, float(feature.get('selection_order_bonus', 0.) or 0.) + adjustment))
    if strength == 0:
        return list(scored)
    return [(row, score) for row, score, _bonus in sorted(ordering, key=lambda pair: float(pair[1]) + pair[2], reverse=True)]


def rerank_snapshot_contributions(snapshot: dict, *, strength: float) -> dict:
    """Copy and reorder captured eligible inputs for actual selector replay."""
    result = deepcopy(snapshot)
    eligible = sorted((c for c in result['candidates'] if c['eligible']), key=lambda c: c['input_position'])
    scored = [(SimpleNamespace(**{**c['metadata'], 'id': c['item_id'], 'section': c['section']}), c['ranking_score']) for c in eligible]
    features = {c['item_id']: c['features'] for c in eligible}
    ordering = rerank_contributions(scored, features, result['profile'].get('canonical_facets', {}), strength=strength)
    positions = {row.id: i for i, (row, _score) in enumerate(ordering)}
    for candidate in eligible:
        candidate['input_position'] = positions[candidate['item_id']]
    result['contribution_policy'] = {'version': CONTRIBUTION_VERSION, 'strength': strength,
                                     'intervention': 'eligible input ordering; scores and eligibility retained'}
    return result
