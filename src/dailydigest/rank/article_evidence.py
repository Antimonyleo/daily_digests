"""Offline article-level evidence claims, separate from relevance and prestige.

This small text heuristic reports exact author assertions. It does not verify
experiments or links, judge scientific quality, or penalize missing evidence.
"""
from __future__ import annotations

import re


ARTICLE_EVIDENCE_VERSION = "article-evidence-v1"
_BREAK = re.compile(r"(?<=[.!?;])\s+")
_VALIDATION = re.compile(r"\b(?:validat(?:e|ed|ions?|ing)|evaluat(?:e|ed|ions?|ing)|test(?:ed|ing)?|benchmark(?:ed|ing)?)\b", re.I)
_PERFORMED = re.compile(r"\b(?:validated|evaluated|tested|benchmarked|we\s+(?:validate|evaluate|test|benchmark)|(?:validation|evaluation|testing)\s+(?:results|accuracy|performance|confirmed|confirms)|(?:validation|evaluation|testing)\s+was\s+(?:performed|conducted))\b", re.I)
_REPORTED_VALIDATION = re.compile(r"\bwe\s+report(?:ed)?\s+(?:(?:held[ -]out|external|independent|experimental|benchmark)\s+)+(?:evaluations?|validations?|testing)\b", re.I)
_INDEPENDENT = re.compile(r"\b(?:independent|external|held[ -]out|hold[ -]out|out.of.sample)\b", re.I)
_EXPERIMENT = re.compile(r"\b(?:experiment(?:al|ally|s)?|measurements?|assays?|in vitro|in vivo)\b", re.I)
_BENCHMARK = re.compile(r"\bbenchmarks?\b", re.I)
_FUTURE = re.compile(r"\b(?:future|will|would|could|might|may|planned|intend|proposed validation)\b", re.I)
_NEGATED = re.compile(r"\b(?:did not|do not|does not|not (?:validated|evaluated|tested|available|performed)|(?:no|without)\s+(?:\w+\s+){0,3}validation|unavailable)\b", re.I)
_BACKGROUND = re.compile(r"\b(?:previous studies|prior work|earlier work|other studies|related work)\b", re.I)
_RESOURCE = r"(?:source\s+code|code|data(?:sets?)?|software|scripts?)"
_AVAILABLE = re.compile(r"\b" + _RESOURCE + r"\b[^.;]{0,65}?\b(?:is|are|was|were|has been|have been)\s+(?:(?:publicly|freely|openly)\s+)?(?:available|deposited|released|accessible|provided|shared)\b", re.I)
_SHARED = re.compile(r"\bwe\s+(?:make|made|provide|release|share|deposit)(?:d)?\b[^.;]{0,60}\b" + _RESOURCE + r"\b", re.I)
_LINKED = re.compile(r"\b" + _RESOURCE + r"\s*:\s*https?://", re.I)
_USED_EXTERNAL = re.compile(r"\b(?:we (?:used?|employed?|applied)|using|downloaded)\b", re.I)
_LIMITATION = re.compile(r"\b(?:limitations?|limited to|small sample|may not generalize|lack of|not assessed|not evaluated|does not account|did not assess|did not evaluate)\b", re.I)
_URL = re.compile(r"https?://[^\s<>]+", re.I)
_BIBLIOGRAPHIC = re.compile(r"^\s*(?:doi\s*:|https?://|published online\b|publication date\s*:|copyright\b|©)", re.I)


def _spans(text):
    start = 0
    for end in [match.start() for match in _BREAK.finditer(text)] + [len(text)]:
        raw = text[start:end]
        quote = raw.strip()
        if quote:
            offset = start + len(raw) - len(raw.lstrip())
            yield {"quote": quote, "start": offset, "end": offset + len(quote)}
        start = end


def assess_article_evidence(title, abstract, *, full_text=None):
    """Report supplied-text evidence; quoted offsets refer to the named source.

    Nonempty supplied full text is assessed instead of the abstract. Missing,
    sparse or bibliographic-only text remains unknown. ``reported`` means an
    author assertion was found, never that its validity was independently checked.
    """
    if (not isinstance(title, str) or not isinstance(abstract, str)
            or (full_text is not None and not isinstance(full_text, str))):
        raise ValueError("Title, abstract and supplied full text must be strings")
    source = "full_text" if full_text and full_text.strip() else "abstract"
    text = full_text if source == "full_text" else abstract
    sufficient = len(text.strip()) >= 40 and not (source == "abstract" and _BIBLIOGRAPHIC.match(text))
    dimensions = {key: {"status": "unknown", "claims": []}
                  for key in ("validation", "data_code", "limitations")}
    result = {"version": ARTICLE_EVIDENCE_VERSION,
              "assessment_basis": source if sufficient else "insufficient",
              "dimensions": dimensions,
              "meaning": "Explicit author-reported claims, not verified scientific quality; unknown evidence carries no penalty."}
    if not sufficient:
        return result
    for span in _spans(text):
        quote = span["quote"]
        claim = {**span, "source": source, "verification": "author_reported_unverified"}
        background = bool(_BACKGROUND.search(quote))
        if not background and not _FUTURE.search(quote) and not _NEGATED.search(quote):
            kind = None
            if _VALIDATION.search(quote) and (_PERFORMED.search(quote) or _REPORTED_VALIDATION.search(quote)):
                if _INDEPENDENT.search(quote):
                    kind = "independent_validation"
                elif _EXPERIMENT.search(quote):
                    kind = "experimental_validation"
                elif _BENCHMARK.search(quote):
                    kind = "benchmark_validation"
            if kind:
                dimensions["validation"]["claims"].append({**claim, "kind": kind})
            if ((_AVAILABLE.search(quote) or _SHARED.search(quote) or _LINKED.search(quote))
                    and not _USED_EXTERNAL.search(quote)):
                dimensions["data_code"]["claims"].append({
                    **claim, "kind": "resource_availability",
                    "availability": "upon_request" if re.search(r"\b(?:upon|on)\s+(?:reasonable\s+)?request\b", quote, re.I) else "asserted_available",
                    "links": [match.group().rstrip(".,;)") for match in _URL.finditer(quote)],
                })
        if not background and _LIMITATION.search(quote):
            dimensions["limitations"]["claims"].append({**claim, "kind": "stated_limitation"})
    for dimension in dimensions.values():
        if dimension["claims"]:
            dimension["status"] = "reported"
    return result
