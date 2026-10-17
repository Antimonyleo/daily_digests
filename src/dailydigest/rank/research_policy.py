"""Reader-derived input for offline semantic ranking experiments."""
from __future__ import annotations


RESEARCH_INSTRUCTION = (
    "Judge the present paper's central contribution against the reader's research interests. "
    "Use the actual scientific task and object, rather than overlapping vocabulary. "
    "Prioritize relevant engineering, design, prediction, programmable organization, and transferable methods for these interests. "
    "Design assessment, functional characterization, or method evaluation can contribute useful transferable evidence for the reader's targets. "
    "Using an existing tool can support a genuinely new engineered system. "
    "The listed interests are authoritative; do not impose an additional fabrication route or require a new model architecture. "
    "Descriptive native structures, signaling, cellular condensates, clinical outcomes, and broad reviews are lower-priority context when they supply no relevant engineering or transferable method. "
    "Literature assistants and generic research-workflow tools are context unless their central task designs or predicts relevant molecular properties. "
    "Diagnostic associations, medical-image classification, and electronic hardware do not satisfy the interests through biological words alone. "
    "Distinguish completed current work from background, cited prior work, negation, and future plans. "
    "Assess interest alignment independently of venue prestige and scientific quality, using only supplied evidence."
)


def reader_research_query(profile: dict) -> str:
    """Include every canonical interest's supplied aliases and anchors exactly.

    This builds text, not relevance labels or eligibility rules. Experimental
    definitions must not silently substitute narrower targets for the profile.
    """
    rows = []
    for facet, evidence in (profile.get('canonical_facets') or {}).items():
        terms = list(dict.fromkeys([facet, *evidence.get('aliases', []), *evidence.get('anchors', [])]))
        rows.append('- ' + '; '.join(terms))
    if not rows:
        rows = ['- ' + term for term in profile.get('keywords', []) if term.strip()]
    if not rows:
        raise ValueError('Semantic ranking requires explicit reader interests')
    return 'Reader research interests:\n' + '\n'.join(rows)
