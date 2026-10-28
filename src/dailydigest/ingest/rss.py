from __future__ import annotations

import hashlib
import html
import logging
import re
from datetime import datetime, timezone
from urllib.parse import urldefrag, urlparse, urlunparse

import feedparser
import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ..models import Item, SourceSpec
from ..article_figure import feed_figure
from ._retry import transient_http_error

logger = logging.getLogger(__name__)
_MAX_FEED_BYTES = 16 * 1024 * 1024


# A browser-like UA + Accept headers. Several publishers (Wiley, some news
# sites) 403 a bare bot UA; this clears the naive filters. It does NOT defeat
# active bot protection (ACS/Cloudflare) — those journals are pulled via
# OpenAlex-by-venue instead — but it recovers feeds behind simple UA checks.
_RSS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
    ),
    # Keep Accept fully permissive: a specific RSS/XML Accept list makes strict
    # servers (e.g. eLife) return 406 Not Acceptable. The browser UA above is the
    # lever that clears naive bot filters; the Accept header must not narrow it.
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(transient_http_error),
    reraise=False,
)
def _http_get_bytes(url: str) -> bytes:
    """Fetch a feed body via httpx so we get retry control over feedparser."""
    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        with client.stream("GET", url, headers=_RSS_HEADERS) as resp:
            resp.raise_for_status()
            content_length = resp.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > _MAX_FEED_BYTES:
                        raise RuntimeError(
                            "RSS response exceeds the 16 MiB safety limit"
                        )
                except ValueError:
                    pass
            body = bytearray()
            for chunk in resp.iter_bytes():
                body.extend(chunk)
                if len(body) > _MAX_FEED_BYTES:
                    raise RuntimeError("RSS response exceeds the 16 MiB safety limit")
            return bytes(body)


_BLOCK_NOISE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_HTML_TAG = re.compile(r"<[^>]+>")
_TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content"}


def _strip_html(s: str) -> str:
    if not s:
        return ""
    s = _BLOCK_NOISE_RE.sub(" ", s)
    s = _HTML_TAG.sub(" ", s)
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


# Publisher feed headers that precede (or replace) the abstract. Each pattern is
# anchored at the start of the text and names explicit bibliographic fields, so
# body sentences that merely cite a DOI or contain numbers are never touched.
_DOI_TOKEN = (
    # Nature's compact DOIs are glued to the next sentence ("...-1A paper"),
    # so their fixed suffix shape bounds them; other DOIs end at whitespace.
    r"(?:10\.1038/[a-z]{1,2}\d{4,6}-\d{3}-\d{4,6}-[0-9a-z]"
    r"|10\.\d{4,9}/[^\s,;]+)"
)
_FEED_HEADERS = (
    # "Nature Nanotechnology, Published online: 13 October 2026; doi:10.1038/…"
    re.compile(r"^\s*(?:[^,;\n]{2,100},\s*)?Published online\s*:\s*[^;\n]{3,40};\s*doi\s*:\s*"
               + _DOI_TOKEN + r"\s*", re.I),
    # "Advanced Science, Volume 13, Issue 26, 8 May 2026." (Wiley, AAAS, PNAS, ACS)
    re.compile(r"^\s*[^,;\n]{2,100},\s*Volume\s+\d+\s*,\s*Issue\s+[\w-]+\s*"
               r"(?:,\s*Pages?\s+[\w-]+\s*)?(?:,\s*[A-Za-z0-9 ]{0,30}?\b\d{4})?\s*\.?\s*", re.I),
    # "New England Journal of Medicine, Ahead of Print."
    re.compile(r"^\s*[^,;\n]{2,100},\s*Ahead of Print\s*\.?\s*", re.I),
    # A bare leading identifier: "doi:10.…" or "https://doi.org/10.…".
    re.compile(r"^\s*(?:doi\s*:\s*|https?://(?:dx\.)?doi\.org/)" + _DOI_TOKEN + r"\s*", re.I),
)
# RSC: "Chem. Sci., 2026, Accepted Manuscript DOI: 10.1039/…, Edge Article Open
# Access This article is licensed under … Licence. A. Author, B. Author <body>"
_RSC_HEADER = re.compile(
    r"^\s*[^\n]{2,60}?,\s*\d{4}\s*,\s*(?:Advance Article|Accepted Manuscript|[\d ,–-]+?)\s*"
    r"DOI\s*:\s*" + _DOI_TOKEN + r"\s*,\s*(?P<kind>(?:[A-Z][a-z]+\s+){0,2}(?:Article|Review|Perspective|"
    r"Communication|Highlight|Feature|Tutorial|Paper|Comment|Editorial|Correction))?"
    r"(?:\s*Open Access)?\s*(?:This article is licensed under.{0,200}?Licen[cs]e\.)?\s*",
    re.I | re.S)
_RSC_TAIL = re.compile(r"\s*(?:To cite this article before page numbers are assigned[^.]*\.)?"
                       r"\s*The content of this RSS Feed \(c\)[^\n]*$", re.I)
_NAME_TOKEN = re.compile(r"^(?:[A-Z][\w'’.-]*|van|von|de|der|den|da|di|du|la|le)$")
# Hacker News style link-post fields carry no article text.
_LINK_POST_FIELDS = re.compile(
    r"(?:Article URL|Comments URL)\s*:\s*\S+|\bPoints\s*:\s*\d+|#\s*Comments\s*:\s*\d+", re.I)
# A leading abstract label, spaced or glued ("AbstractAlternative …").
_LEAD_LABEL = re.compile(r"^\s*(?:Abstract|ABSTRACT|Summary|SUMMARY|Significance|SIGNIFICANCE)"
                         r"\s*[:.]?\s*(?=[A-Z0-9])")
# A repeated abstract label between a publisher teaser and the abstract itself.
_INNER_LABEL = re.compile(r"\s+ABSTRACT\s+(?=[A-Z0-9])|(?<=[.!?])\s*(?:Abstract|Significance)(?=[A-Z][a-z])")


def _strip_author_list(text: str) -> str:
    """Drop a leading "A. Author, B. Author" list that RSC places before its teaser.

    Requires a comma-separated name group before the final name, which ends at a
    glued case change ("LouDeciphering") or before a capitalized word that is
    followed by lowercase prose. Anything less certain is left unchanged.
    """
    tokens = text.split(" ")
    index, groups = 0, 0
    while index < len(tokens):
        start = index
        while index < len(tokens) and _NAME_TOKEN.match(tokens[index].rstrip(",")) and not tokens[index].endswith(","):
            index += 1
            if index - start > 5:
                return text
        if index < len(tokens) and tokens[index].endswith(",") and _NAME_TOKEN.match(tokens[index][:-1]):
            index, groups = index + 1, groups + 1
            continue
        index = start
        break
    if groups < 1:
        return text
    for offset in range(index, min(index + 6, len(tokens))):
        token = tokens[offset]
        glued = re.search(r"(?<=[a-z])(?=[A-Z][a-z])", token)
        if glued:
            return " ".join([token[glued.start():], *tokens[offset + 1:]])
        following = tokens[offset + 1] if offset + 1 < len(tokens) else ""
        if offset > index and following[:1].islower():
            return " ".join(tokens[offset:])
        if not _NAME_TOKEN.match(token):
            return text
    return text


def strip_feed_metadata(text: str) -> str:
    """Remove recognizable feed header/label metadata from abstract text.

    Only anchored bibliographic headers, publisher boilerplate, link-post
    fields and abstract labels are removed. A repeated inner ``ABSTRACT`` label
    becomes a newline so a publisher teaser and the abstract are not fused into
    one sentence. Returns "" when nothing but metadata remains.
    """
    if not text:
        return ""
    cleaned = str(text)
    changed = True
    while changed:
        changed = False
        for pattern in _FEED_HEADERS:
            stripped = pattern.sub("", cleaned, count=1)
            if stripped != cleaned:
                cleaned, changed = stripped, True
    rsc = _RSC_HEADER.match(cleaned)
    if rsc:
        cleaned = _strip_author_list(cleaned[rsc.end():].strip())
        cleaned = _RSC_TAIL.sub("", cleaned)
    cleaned = _LINK_POST_FIELDS.sub(" ", cleaned)
    cleaned = _LEAD_LABEL.sub("", cleaned, count=1)
    cleaned = _INNER_LABEL.sub("\n", cleaned)
    lines = (re.sub(r"[ \t\xa0]+", " ", line).strip() for line in cleaned.split("\n"))
    return "\n".join(line for line in lines if line)


def feed_publication_kind(text: str) -> str:
    """Article kind declared inside an RSC feed header ("Review Article"), else ""."""
    match = _RSC_HEADER.match(text or "")
    return (match.group("kind") or "").strip() if match else ""


def canonicalize_url(url: str) -> str:
    if not url:
        return url
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    parsed = parsed._replace(scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower())
    # drop tracking query params
    if parsed.query:
        kept = [
            kv
            for kv in parsed.query.split("&")
            if kv and kv.split("=")[0].lower() not in _TRACKING_PARAMS
        ]
        parsed = parsed._replace(query="&".join(kept))
    return urlunparse(parsed).rstrip("/")


def _parsed_dt(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        val = entry.get(key)
        if val:
            try:
                return datetime(*val[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return None


def _external_id(entry, url: str) -> str:
    raw = entry.get("id") or entry.get("guid") or url
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _entry_metadata(entry) -> dict:
    """Preserve explicit work identifiers and distinguish updates from publication."""
    from ..rank.enrich import derive_doi
    from types import SimpleNamespace

    metadata = {}
    figure = feed_figure(entry)
    if figure:
        metadata['article_figure'] = figure
    # Only exact values in record-level genre fields carry publication identity.
    # Feed categories/subjects and relation links may instead describe a topic
    # or another work; they must never exclude this record.
    publication_types = {
        "editorial": "Editorial", "news": "News", "news and views": "News & Views",
        "news & views": "News & Views", "comment": "Comment", "commentary": "Commentary",
        "research highlight": "Research Highlight", "research highlights": "Research Highlights",
        "correction": "Correction", "erratum": "Erratum", "published erratum": "Published Erratum",
        "retraction": "Retraction", "retraction notice": "Retraction Notice",
        "retracted publication": "Retracted Publication", "expression of concern": "Expression of Concern",
        "review": "Review", "systematic review": "Systematic Review", "perspective": "Perspective",
        "primer": "Primer", "letter": "Letter", "article": "Journal Article",
        "journal article": "Journal Article", "research article": "Research Article", "protocol": "Protocol",
    }
    evidence, types = [], []
    for field in ("dc_type", "prism_genre", "prism_section"):
        raw = entry.get(field)
        values = raw if isinstance(raw, (list, tuple)) else [raw]
        for value in values:
            if not isinstance(value, str):
                continue
            normalized = publication_types.get(" ".join(value.casefold().split()))
            if normalized:
                evidence.append({"field": field, "value": value})
                if normalized not in types:
                    types.append(normalized)
    if types:
        metadata["publication_types"] = types
        metadata["publication_type_evidence"] = evidence
    values = [entry.get(key) for key in ("prism_doi", "dc_identifier", "doi", "id", "guid", "link")]
    values.extend(link.get("href") for link in entry.get("links", [])
                  if isinstance(link, dict) and link.get("rel", "alternate") in {"alternate", "self"})
    for value in values:
        if isinstance(value, str):
            doi = derive_doi(SimpleNamespace(url=value))
            if doi:
                metadata["doi"] = doi
                break
    # Directed publisher relation links identify the subject of commentary;
    # ordinary related/alternate/reference links are not sufficient evidence.
    from ..publications import RELATION_NAMES, normalize_doi

    targets, relation_evidence = set(), []
    for link in entry.get("links", []):
        if not isinstance(link, dict) or link.get("rel") not in RELATION_NAMES:
            continue
        doi = normalize_doi(link.get("href"))
        if doi:
            targets.add(doi)
            relation_evidence.append({"field": "links", "rel": link["rel"], "doi": doi})
    if targets:
        metadata["companion_of_dois"] = sorted(targets)
        metadata["companion_relation_evidence"] = relation_evidence
    for key, source in (("published_parsed", "published"), ("updated_parsed", "updated")):
        value = entry.get(key)
        if value:
            try:
                datetime(*value[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
            metadata["publication_date_source"] = source
            break
    return metadata


def _extract_abstract(entry) -> str:
    """Extract abstract text, preferring content:encoded (CDATA) over summary.

    Nature, Cell, ACS, and other publishers put full abstracts in
    ``content[0].value`` (the ``content:encoded`` element) while ``summary``
    contains only a short teaser or is empty.  Without this fallback, items
    from top journals embed only the title, which badly hurts ranking quality.
    """
    content = entry.get("content")
    if content and isinstance(content, list):
        value = content[0].get("value") if content[0] else None
        if value:
            return _strip_html(value)
    return _strip_html(entry.get("summary", entry.get("description", "")))


class RSSSource:
    def fetch(self, spec: SourceSpec, days: int = 2) -> list[Item]:
        # ``days`` is ignored: an RSS feed only exposes its current window of
        # entries. The ranking recency window trims older items by published date.
        if not spec.url:
            return []
        try:
            content = _http_get_bytes(spec.url)
        except Exception as e:
            name = getattr(spec, "name", "RSSSource")
            raise RuntimeError(
                f"{name} RSS fetch failed: {type(e).__name__}: {str(e)[:200]}"
            ) from e
        if not content:
            raise RuntimeError(f"{spec.name} RSS fetch failed: empty response")
        feed = feedparser.parse(content)
        if feed.bozo and not feed.entries:
            raise RuntimeError(
                f"{spec.name} RSS parse failed: "
                f"{getattr(feed, 'bozo_exception', 'invalid feed')}"
            )
        if feed.bozo:
            logger.info("%s: bozo flag set (%s); proceeding with %d entries",
                        getattr(spec, "name", "RSSSource"),
                        getattr(feed, "bozo_exception", "?"),
                        len(feed.entries))
        out: list[Item] = []
        for entry in feed.entries[:200]:
            url = canonicalize_url(entry.get("link", ""))
            if not url:
                continue
            title = _strip_html(entry.get("title", "")).strip()
            if not title:
                continue
            abstract = _extract_abstract(entry)
            authors = ""
            if entry.get("authors"):
                authors = ", ".join(a.get("name", "") for a in entry["authors"] if a.get("name"))
            elif entry.get("author"):
                authors = entry["author"]
            out.append(
                Item(
                    source=spec.name,
                    section=spec.section,
                    external_id=_external_id(entry, url),
                    url=url,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    published_at=_parsed_dt(entry),
                    metadata=_entry_metadata(entry),
                )
            )
        return out
