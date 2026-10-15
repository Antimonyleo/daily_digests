"""Optional, reader-requested spelling suggestions for short interest phrases."""
from __future__ import annotations

import re
from xml.etree.ElementTree import ParseError

import httpx
from defusedxml.common import DefusedXmlException
from defusedxml.ElementTree import fromstring

_TOKENS = re.compile(r"\w+(?:[-+/]\w+)*")


class SpellingUnavailable(RuntimeError):
    """The spelling service did not provide a usable response."""


def _identifiers(text: str) -> set[str]:
    tokens = _TOKENS.findall(text)
    return {token.casefold() for token in tokens
            if any(character.isdigit() for character in token)
            or (len(token) > 1 and token.isupper())}


def suggest_topic_spelling(topic: str) -> str | None:
    """Suggest wording; never save it or treat no suggestion as validation.

    Only the explicitly checked phrase goes to NCBI. Acronyms and numbered
    identifiers must survive a suggestion, even when nearby prose is corrected.
    A suggestion still requires the reader's acceptance before use.
    """
    topic = " ".join(topic.split())
    if not 1 <= len(topic) <= 160:
        raise ValueError("Enter an interest of 1–160 characters.")
    try:
        with httpx.Client(timeout=8.0, headers={"User-Agent": "dailydigest/0.1"}) as client:
            response = client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/espell.fcgi",
                params={"db": "pubmed", "term": topic},
            )
            response.raise_for_status()
        root = fromstring(response.text)
    except (httpx.HTTPError, ParseError, DefusedXmlException) as exc:
        raise SpellingUnavailable("Spelling service unavailable") from exc
    if (root.tag != "eSpellResult" or root.find("CorrectedQuery") is None
            or (root.findtext("ERROR") or "").strip()):
        raise SpellingUnavailable("Invalid spelling response")
    suggestion = " ".join((root.findtext("CorrectedQuery") or "").split())
    suggested_words = {token.casefold() for token in _TOKENS.findall(suggestion)}
    if (not suggestion or len(suggestion) > 160
            or suggestion.casefold() == topic.casefold()
            or not _identifiers(topic) <= suggested_words):
        return None
    return suggestion
