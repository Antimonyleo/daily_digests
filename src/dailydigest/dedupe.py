from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any
from urllib.parse import parse_qsl, quote, unquote, urldefrag, urlencode, urlparse, urlunparse

from .models import Item


def cap_near_duplicates(rows: list[Any], vecs: Any, threshold: float) -> list[int]:
    """Greedily suppress within-day topical near-duplicates by embedding cosine.

    ``rows`` must already be sorted by final score DESC and ``vecs`` (a 2D array
    of L2-normalizable embeddings) aligned by index. Walking in score order, an
    item is KEPT unless its maximum cosine similarity to any already-kept item is
    ``>= threshold`` — so the highest-scored representative of each near-duplicate
    cluster survives and lower-scored near-copies are dropped. Returns the list of
    indices to KEEP (a subset of ``range(len(rows))``, in ascending order). The
    top-scored item of any cluster is never dropped (nothing is kept before it).

    Complexity is O(n * k) over the kept set (a few hundred items is fine). Inputs
    are not mutated.
    """
    n = len(rows)
    if n == 0:
        return []
    try:
        import numpy as np
    except Exception:  # noqa: BLE001 - numpy always present in practice
        return list(range(n))

    arr = np.asarray(vecs, dtype=np.float32)
    # Defensive: if the embedding matrix is missing/misaligned, keep everything
    # so a degraded embedder never silently drops items.
    if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] == 0:
        return list(range(n))

    normed = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)

    kept_indices: list[int] = []
    kept_vecs: list[Any] = []
    for i in range(n):
        v = normed[i]
        is_dup = False
        for kv in kept_vecs:
            if float(np.dot(v, kv)) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept_indices.append(i)
            kept_vecs.append(v)
    return kept_indices


def _canonical_doi(raw: str) -> str:
    """Normalize a DOI to bare lowercase form."""
    if not raw:
        return ""
    value = raw.strip()
    value = re.sub(
        r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.IGNORECASE
    )
    value = re.sub(r"^doi:\s*", "", value, flags=re.IGNORECASE)
    return value.lower()


_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_ARXIV_RE = re.compile(
    r"(?:arxiv:)?((?:\d{4}\.\d{4,5})|(?:[a-z-]+(?:\.[A-Z]{2})?/\d{7}))(?:v\d+)?",
    re.IGNORECASE,
)
_PMID_URL_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", re.IGNORECASE)
_TRAILING_DOI_PUNCT = ".),;:"
_TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
_RSS_TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content"}
_JOURNAL_ISSUE_SUFFIX_RE = re.compile(
    r"\s*\((?:adv\.?\s*mater\.?|small|angew\.?\s*chem\.?|chem\.?\s*sci\.?|"
    r"jacs|acs\s*nano|nano\s*lett\.?|science|nature)[^)]*\b\d{1,3}/\d{4}\)$",
    re.IGNORECASE,
)


def canonicalize_url(url: str) -> str:
    """Canonicalize URLs without importing ingest.rss and creating an import cycle."""
    if not url:
        return url
    url, _fragment = urldefrag(url)
    parsed = urlparse(url)
    parsed = parsed._replace(scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower())
    if parsed.query:
        kept = [
            kv
            for kv in parsed.query.split("&")
            if kv and kv.split("=")[0].lower() not in _RSS_TRACKING_PARAMS
        ]
        parsed = parsed._replace(query="&".join(kept))
    return urlunparse(parsed).rstrip("/")


def _event_start_suffix(it: Any, metadata: Any) -> str:
    """Meetings without their own page share a listing or registration URL;
    their start date keeps different meetings and editions apart."""
    if (getattr(it, "section", "") in {"opportunities", "events"} and isinstance(metadata, dict)
            and metadata.get("record_type") == "event" and metadata.get("event_start")):
        return f"@{metadata['event_start']}"
    return ""


def dedupe_by_url(items: list[Item]) -> list[Item]:
    _preserve_preprint_publication_links(items)
    _preserve_research_exclusions(items)
    seen: set[str] = set()
    out: list[Item] = []
    for it in items:
        canon = canonicalize_url(it.url)
        if not canon:
            out.append(it)
            continue
        canon += _event_start_suffix(it, getattr(it, "metadata", None))
        if canon in seen:
            continue
        seen.add(canon)
        out.append(it)
    return out


def dedupe_ranking_candidates[T](items: list[T]) -> list[T]:
    """Collapse duplicate ranking candidates across stored sources.

    Stored rows are unique only within a source. Ranking works across all sources,
    so use durable cross-source keys first (DOI/PMID/arXiv/URL), then a cautious
    same-day title fallback for feeds that do not expose shared identifiers.
    When duplicates disagree, keep the richest/highest-quality representative
    rather than whichever source happened to be fetched first.
    """
    _preserve_preprint_publication_links(items)
    _preserve_research_exclusions(items)
    return _merge_publication_versions(_merge_record_copies(items))


def _merge_record_copies[T](items: list[T]) -> list[tuple[T, list[T]]]:
    """Collapse copies of one record; return (representative, members) slots.

    Publication links are not record identity: a preprint and its journal
    version are separate records, merged later with explicit version evidence.
    """
    # A key lists every slot it named: one same-day title can belong to
    # several distinct works, and a later copy must find its own.
    key_to_index: dict[str, list[int]] = {}
    # deleted[i] = True means out[i] was merged into another slot
    deleted: list[bool] = []
    out: list[T] = []
    members: list[list[T]] = []

    from .rank.source_quality import is_preprint_source

    for it in items:
        keys = candidate_identity_keys(it, include_publication_links=False)
        matched_keys: dict[int, list[str]] = {}
        for k in keys:
            for i in key_to_index.get(k, ()):
                if not deleted[i]:
                    matched_keys.setdefault(i, []).append(k)
        matched_indices = [i for i, shared in matched_keys.items()
                           if not all(k.startswith("title_day:") for k in shared)]
        if not matched_indices:
            # A same-day title alone cannot make a preprint and a journal record
            # copies; the version stage decides that pair with explicit evidence.
            # Nor does it join records whose DOIs or authors name other works,
            # or bridge two works it kept apart: it joins at most one slot.
            title_matches = [
                (_title_copy_evidence(it, keys, members[i]), -i) for i in matched_keys
                if is_preprint_source(it) == is_preprint_source(out[i])
            ]
            best = max((match for match in title_matches if match[0]), default=None)
            matched_indices = [-best[1]] if best else []

        if not matched_indices:
            idx = len(out)
            out.append(it)
            members.append([it])
            deleted.append(False)
            for key in keys:
                key_to_index.setdefault(key, []).append(idx)
            continue

        # Canonical slot is the first match; merge any additional matched slots
        idx = matched_indices[0]
        members[idx].append(it)
        for extra in matched_indices[1:]:
            # Absorb extra slot into idx: keep best representative
            if _candidate_representative_score(out[extra]) > _candidate_representative_score(out[idx]):
                out[idx] = out[extra]
            members[idx].extend(members[extra])
            deleted[extra] = True
            # Re-point all keys that referenced extra to idx
            for slots in key_to_index.values():
                if extra in slots:
                    slots[:] = list(dict.fromkeys(idx if v == extra else v for v in slots))

        current = out[idx]
        if _candidate_representative_score(it) > _candidate_representative_score(current):
            out[idx] = it
        for key in keys:
            slots = key_to_index.setdefault(key, [])
            if idx not in slots:
                slots.append(idx)

    return [(item, group) for item, group, d in zip(out, members, deleted, strict=True) if not d]


# Angewandte publishes one article under paired German/International DOIs
# (10.1002/ange.X and 10.1002/anie.X); the pair names the same work.
_ANGEWANDTE_TWIN_DOI_RE = re.compile(r"^10\.1002/an[gi]e\.")


def _own_doi(keys: list[str]) -> str:
    return next((key[4:] for key in keys if key.startswith("doi:")), "")


# Words in author strings that are not names ("W. Zhang et al.", "and colleagues").
_AUTHOR_FILLER = frozenset({"et", "al", "and", "the", "others", "colleagues", "coworkers",
                            "co", "workers", "group", "team", "for", "on", "behalf", "of"})

# One author: (name words, initials), casefolded; "Dao, T. M." -> ({"dao"}, {"t", "m"}).
type _Author = tuple[frozenset[str], frozenset[str]]


def _author_list(row: Any) -> list[_Author]:
    """Authors of a row in listed order, in any common name format.

    Lists separate authors by ";" or by "," ("Smith A, Lee B"); a comma part
    holding only initials belongs to the surname before it ("Dao, T. M.").
    """
    text = re.sub(r"\bet\s+al\b\.?", "", str(getattr(row, "authors", "") or ""), flags=re.I)
    parts = re.split(r";|\s+(?:and|&)\s+", text) if ";" in text else re.split(r",|\s+(?:and|&)\s+", text)
    authors: list[_Author] = []
    for part in parts:
        words, initials = set(), set()
        for word in re.findall(r"[^\W\d_]+", part):
            if len(word) == 1 or (word.isupper() and len(word) <= 3):
                initials.update(word.casefold())
            elif word.casefold() not in _AUTHOR_FILLER:
                words.add(word.casefold())
        if words:
            authors.append((frozenset(words), frozenset(initials)))
        elif initials and authors and ";" not in text:
            authors[-1] = (authors[-1][0], authors[-1][1] | initials)
    return authors


def _same_author(a: _Author, b: _Author) -> bool:
    """Whether two author entries can name one person.

    They must share a name word, and what one side adds beyond it must agree
    with the other: "W. Zhang" is "Wei Ming Zhang", "Alice Smith" is not
    "Alice Jones", and "Wei Zhang" is not "Min Zhang". A bare shared word
    ("Zhang et al.") is compatible.
    """
    shared = a[0] & b[0]
    if not shared:
        return False
    rest = [(a[0] - shared, a[1]), (b[0] - shared, b[1])]

    def covered(side: tuple[frozenset[str], frozenset[str]],
                other: tuple[frozenset[str], frozenset[str]]) -> bool:
        letters = {word[0] for word in other[0]} | other[1]
        return all(word[0] in other[1] for word in side[0]) and side[1] <= letters

    return covered(rest[0], rest[1]) or covered(rest[1], rest[0])


def _share_author(left: list[_Author], right: list[_Author]) -> bool:
    # Index by name word so consortium-length lists stay cheap to compare.
    by_word: dict[str, list[_Author]] = {}
    for b in right:
        for word in b[0]:
            by_word.setdefault(word, []).append(b)
    return any(_same_author(a, b) for a in left for word in a[0] for b in by_word.get(word, ()))


def _angewandte_twins(doi: str, other: str) -> bool:
    return bool(_ANGEWANDTE_TWIN_DOI_RE.match(doi) and _ANGEWANDTE_TWIN_DOI_RE.match(other)
                and _ANGEWANDTE_TWIN_DOI_RE.sub("", doi) == _ANGEWANDTE_TWIN_DOI_RE.sub("", other))


def _title_copy_evidence(it: Any, keys: list[str], group: list[Any]) -> int:
    """How well a same-day title match supports joining ``it`` to a copy group.

    0 means the records name different works: author lists with no author in
    common, or two DOIs (other than an Angewandte twin) whose first authors
    differ. 2 means a shared author supports the match; 1 means nothing
    contradicts it, as for feeds without identifiers or authors.
    """
    doi, authors = _own_doi(keys), _author_list(it)
    evidence = 1
    for other in group:
        other_doi = _own_doi(candidate_identity_keys(other, include_publication_links=False))
        other_authors = _author_list(other)
        if authors and other_authors:
            if not _share_author(authors, other_authors):
                return 0
            evidence = 2
        if doi and other_doi and doi != other_doi and not _angewandte_twins(doi, other_doi) and not (
            authors and other_authors and _same_author(authors[0], other_authors[0])
        ):
            return 0
    return evidence


# A published record whose abstract is below this length (empty, or a feed
# teaser) does not displace a version carrying at least twice as much text.
_THIN_ABSTRACT_CHARS = 200


def _abstract_chars(row: Any) -> int:
    return len(str(getattr(row, "abstract", "") or "").strip())


def _version_representative(current: Any, incoming: Any) -> Any:
    best = (incoming if _candidate_representative_score(incoming)
            > _candidate_representative_score(current) else current)
    other = incoming if best is current else current
    if _abstract_chars(best) < min(_THIN_ABSTRACT_CHARS, _abstract_chars(other) / 2):
        return other
    return best


def _merge_publication_versions[T](slots: list[tuple[T, list[T]]]) -> list[T]:
    """One slot per work: join record slots linked by explicit version evidence.

    Publication versions often have different DOIs and publication dates. Each
    record keeps its own genre, preprint and retraction status; the survivor
    gains only ``metadata.publication_versions`` links to the other versions.
    """
    reps: list[T] = []
    groups: list[list[T]] = []
    versions: list[list[tuple[T, dict[str, Any]]]] = []
    index = _VersionIndex()
    for rep, members in slots:
        match = None
        for slot in index.candidates(members, rep):
            evidence = next((found for left in members for right in groups[slot]
                             if (found := publication_version_relation(left, right))), None)
            if evidence is not None:
                match = slot, evidence
                break
        if match is None:
            index.add(len(reps), members, rep)
            reps.append(rep)
            groups.append(list(members))
            versions.append([(rep, {})])
            continue
        slot, evidence = match
        if not versions[slot][0][1]:
            versions[slot][0] = (versions[slot][0][0], evidence)
        versions[slot].append((rep, evidence))
        groups[slot].extend(members)
        reps[slot] = _version_representative(reps[slot], rep)
        index.add(slot, members, rep)
    for survivor, linked in zip(reps, versions, strict=True):
        links = [version_link(version, evidence) for version, evidence in linked if version is not survivor]
        if links:
            _attach_version_links(survivor, links)
    return reps


class _VersionIndex:
    """Lookup of slots that could hold another version of a record."""

    def __init__(self) -> None:
        self.keys: dict[str, list[int]] = {}

    @staticmethod
    def _keys(members: list[Any], rep: Any) -> set[str]:
        keys = {f"title:{title}"} if (title := publication_version_title_key(rep)) else set()
        for member in members:
            keys.update(_durable_keys(member))
            if published := _declared_published_doi(member):
                keys.add(f"doi:{published}")
        return keys

    def candidates(self, members: list[Any], rep: Any) -> list[int]:
        return sorted({slot for key in self._keys(members, rep) for slot in self.keys.get(key, [])})

    def add(self, slot: int, members: list[Any], rep: Any) -> None:
        for key in self._keys(members, rep):
            if slot not in (bucket := self.keys.setdefault(key, [])):
                bucket.append(slot)


def _durable_keys(row: Any) -> set[str]:
    """The record's own DOI/PMID/arXiv/URL identity (no publication links)."""
    try:
        keys = candidate_identity_keys(row, include_publication_links=False)
    except ValueError:  # malformed feed URL: no guessed identity
        return set()
    return {key for key in keys if key.startswith(("doi:", "pmid:", "arxiv:", "url:"))}


def _declared_published_doi(row: Any) -> str | None:
    from .rank.source_quality import _row_metadata

    value = _row_metadata(row).get("published_doi")
    return _extract_doi(value) if isinstance(value, str) else None


def _row_id(row: Any) -> int | None:
    value = getattr(row, "id", None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def publication_version_relation(left: Any, right: Any) -> dict[str, Any] | None:
    """Evidence that two distinct research records are versions of one work.

    Accepted evidence, in order: a source-declared ``published_doi`` naming the
    other record's DOI; two preprints declaring the same ``published_doi``; or
    the conservative identical-distinctive-title/author rule. A record that is
    itself ineligible (withdrawn, retracted, editorial...) is never joined, so
    its status stays visible on its own record instead of being hidden or copied.
    """
    from .rank.source_quality import should_skip_item

    if left is right or any(str(getattr(row, "section", "") or "").casefold() != "research"
                            for row in (left, right)):
        return None
    if _durable_keys(left) & _durable_keys(right):
        return None  # copies of one record, not versions
    pair = sorted(item_id for item_id in (_row_id(left), _row_id(right)) if item_id is not None)
    evidence: dict[str, Any] | None = None
    for declarer, target in ((left, right), (right, left)):
        published = _declared_published_doi(declarer)
        if published and f"doi:{published}" in _durable_keys(target):
            evidence = {"relation": "published_doi", "field": "metadata.published_doi",
                        "value": published, "declared_by_item_id": _row_id(declarer),
                        "declared_by_source": str(getattr(declarer, "source", "") or "")}
            break
    if evidence is None:
        published = _declared_published_doi(left)
        if published and published == _declared_published_doi(right):
            evidence = {"relation": "shared_published_doi", "field": "metadata.published_doi",
                        "value": published}
    if evidence is None and are_publication_versions(left, right):
        evidence = {"relation": "same_title_authors", "field": "title+authors",
                    "value": publication_version_title_key(left)}
    if evidence is None or should_skip_item(left) or should_skip_item(right):
        return None
    return {**evidence, "between_item_ids": pair}


def version_link(row: Any, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """JSON-safe description of one version, for an "Also published in" link."""
    from .publications import publication_label, retraction_label
    from .rank.source_quality import _row_metadata, display_source, is_preprint_source

    evidence = dict(evidence or {})
    doi = _extract_doi(str(_row_metadata(row).get("doi") or ""),
                       str(getattr(row, "external_id", "") or ""), str(getattr(row, "url", "") or ""))
    return {
        "item_id": _row_id(row),
        "label": str(getattr(row, "item_label", "") or ""),
        "source": str(getattr(row, "source", "") or ""),
        "venue": display_source(row),
        "doi": doi,
        "url": str(getattr(row, "url", "") or ""),
        "is_preprint": is_preprint_source(row),
        "publication_label": publication_label(row),
        "retraction": retraction_label(row),
        "relation": evidence.pop("relation", "same_identifier"),
        "evidence": evidence,
    }


_LINK_KEYS = ("item_id", "doi", "url")


def _attach_version_links(row: Any, links: list[dict[str, Any]]) -> None:
    """Record links in memory only; persist_publication_metadata ignores this key."""
    from .rank.source_quality import _row_metadata

    metadata = dict(_row_metadata(row))
    merged = [link for link in metadata.get("publication_versions", []) if isinstance(link, dict)] \
        if isinstance(metadata.get("publication_versions"), list) else []
    seen = {tuple(link.get(key) for key in _LINK_KEYS) for link in merged}
    for link in links:
        if (marker := tuple(link.get(key) for key in _LINK_KEYS)) not in seen:
            seen.add(marker)
            merged.append(link)
    metadata["publication_versions"] = merged
    if isinstance(getattr(row, "metadata", None), dict):
        row.metadata = metadata
    else:
        row.metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def publication_version_links(row: Any) -> list[dict[str, Any]]:
    """Version links recorded on a brew survivor (empty when none)."""
    from .rank.source_quality import _row_metadata

    links = _row_metadata(row).get("publication_versions")
    return [dict(link) for link in links if isinstance(link, dict)] if isinstance(links, list) else []


@dataclass(frozen=True)
class WorkGroup:
    """One unique work in an existing digest, for display only.

    ``primary`` is the first member in the caller's order (digest order), so
    the card keeps its historical position. ``versions`` are the other digest
    rows of the same work; every row, vote, bookmark and known flag stays on its
    own item ID. Ratings are never merged or averaged: the card shows
    ``primary_rating`` and exposes ``other_version_ratings`` for a "rated on the
    other version" note. ``saved``/``known`` are true when ANY member is.
    """

    primary: Any
    versions: tuple[Any, ...] = ()
    links: tuple[dict[str, Any], ...] = ()
    item_ids: tuple[int, ...] = ()
    primary_rating: Any = None
    other_version_ratings: tuple[tuple[int, Any], ...] = ()
    saved_item_ids: tuple[int, ...] = ()
    known_item_ids: tuple[int, ...] = ()

    @property
    def saved(self) -> bool:
        return bool(self.saved_item_ids)

    @property
    def known(self) -> bool:
        return bool(self.known_item_ids)

    @property
    def rated_on_other_version(self) -> bool:
        return self.primary_rating is None and bool(self.other_version_ratings)

    @property
    def rating_conflict(self) -> bool:
        return self.primary_rating is not None and any(
            value != self.primary_rating for _item_id, value in self.other_version_ratings)


def group_publication_versions(
    items: list[Any],
    *,
    ratings: Mapping[int, Any] | None = None,
    saved: Iterable[int] = (),
    known: Iterable[int] = (),
    recorded_links: Mapping[int, list[dict[str, Any]]] | None = None,
) -> list[WorkGroup]:
    """Group an existing digest's rows into unique works; pure and read-only.

    Rows join only within one section, by a shared DOI/PMID/arXiv/URL identity
    or by ``publication_version_relation`` evidence. No row is mutated, merged,
    added or removed, so ``len(result)`` is the digest's unique-work count.
    ``recorded_links`` maps a row ID to brew-time ``publication_versions``
    links (from digest features); those attach to the card but are not items.
    """
    ratings = ratings or {}
    saved_ids, known_ids = set(saved), set(known)
    rows = list(items)
    parent = list(range(len(rows)))
    joined_by: dict[int, dict[str, Any]] = {}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    index_keys: dict[tuple[str, str], list[int]] = {}
    for position, row in enumerate(rows):
        section = str(getattr(row, "section", "") or "")
        own = _durable_keys(row)
        lookup = set(own)
        if published := _declared_published_doi(row):
            lookup.add(f"doi:{published}")
        if title := publication_version_title_key(row):
            lookup.add(f"title:{title}")
        earlier = sorted({other for key in lookup for other in index_keys.get((section, key), [])})
        for other in earlier:
            if find(other) == find(position):
                continue
            shared = sorted(own & _durable_keys(rows[other]))
            if shared:
                evidence = {"relation": "same_identifier", "field": "identifier", "value": shared[0],
                            "between_item_ids": sorted(i for i in (_row_id(row), _row_id(rows[other]))
                                                       if i is not None)}
            else:
                evidence = publication_version_relation(row, rows[other])
            if evidence is None:
                continue
            low, high = sorted((find(other), find(position)))
            parent[high] = low
            joined_by.setdefault(position, evidence)
        for key in lookup:
            index_keys.setdefault((section, key), []).append(position)

    members: dict[int, list[int]] = {}
    for position in range(len(rows)):
        members.setdefault(find(position), []).append(position)
    groups: list[WorkGroup] = []
    for root in sorted(members):
        positions = members[root]
        primary, versions = rows[positions[0]], tuple(rows[p] for p in positions[1:])
        item_ids = tuple(i for i in (_row_id(rows[p]) for p in positions) if i is not None)
        links = [{**version_link(rows[p], joined_by.get(p) or next(
            (joined_by[q] for q in positions if q in joined_by), None)), "in_digest": True}
            for p in positions[1:]]
        seen = {tuple(link.get(key) for key in _LINK_KEYS) for link in links}
        for item_id in item_ids:
            for link in (recorded_links or {}).get(item_id, []) or []:
                if not isinstance(link, dict) or link.get("item_id") in item_ids:
                    continue
                if (marker := tuple(link.get(key) for key in _LINK_KEYS)) in seen:
                    continue
                seen.add(marker)
                links.append({**link, "in_digest": False})
        primary_id = _row_id(primary)
        groups.append(WorkGroup(
            primary=primary, versions=versions, links=tuple(links), item_ids=item_ids,
            primary_rating=ratings.get(primary_id) if primary_id is not None else None,
            other_version_ratings=tuple((i, ratings[i]) for i in item_ids[1:] if i in ratings),
            saved_item_ids=tuple(i for i in item_ids if i in saved_ids),
            known_item_ids=tuple(i for i in item_ids if i in known_ids),
        ))
    return groups


def publication_version_title_key(row: Any) -> str:
    """Distinctive exact title for a conservative preprint/journal fallback.

    Preserve Unicode letters and numbers: alpha/beta targets or different study
    numbers are substantive differences, not punctuation to discard.
    """
    if str(getattr(row, "section", "") or "").casefold() != "research":
        return ""
    title = _normalized_title_key(str(getattr(row, "title", "") or ""))
    return title if len(title) >= 40 and len(set(title.split())) >= 6 else ""


def are_publication_versions(left: Any, right: Any) -> bool:
    """Match a preprint and journal version by a distinctive unchanged title.

    Explicit DOI relationships are handled by durable identity keys separately.
    A vague shared topic, short title or conflicting author list is insufficient.
    """
    from .rank.source_quality import is_preprint_source, should_skip_item

    title = publication_version_title_key(left)
    if not title or title != publication_version_title_key(right):
        return False
    if is_preprint_source(left) == is_preprint_source(right):
        return False
    if should_skip_item(left) or should_skip_item(right):
        return False
    # Initials and separators differ between feeds (Dao TM vs Dao, T. M.).
    # Keep the author sequence and letters; remove formatting, not names.
    authors = [re.sub(r"[\W_]+", "", str(getattr(row, "authors", "") or "").casefold())
               for row in (left, right)]
    if not all(authors) or authors[0] == authors[1]:
        return True
    signatures = [_author_signatures(str(getattr(row, "authors", "") or "")) for row in (left, right)]
    if not all(signatures) or len(signatures[0]) != len(signatures[1]):
        return False
    return all(a[:2] == b[:2] and not (a[2] and b[2] and a[2] != b[2])
               for a, b in zip(*signatures, strict=True))


def _author_signatures(raw: str) -> list[tuple[str, str, str]]:
    """Surname/initials plus known first name for the adapters' author lists.

    Handles surname-first initials and given-name-first full names. Ordered
    lists must agree; conflicting known first names are evidence against a match.
    This is only supporting evidence for an already identical distinctive title.
    """
    parts = raw.split(";") if ";" in raw else raw.split(",")
    if len(parts) == 2 and re.fullmatch(r"[A-Z.\s]+", parts[1].strip()):
        parts = [raw]  # Single surname-first author, e.g. Dao, T. M.
    result = []
    for part in parts:
        words = re.findall(r"[^\W\d_]+", part)
        if len(words) < 2:
            return []
        if "," in part:
            surname, given = part.split(",", 1)
            surname = re.sub(r"[\W_]+", "", surname.casefold())
            given_words = re.findall(r"[^\W\d_]+", given)
        elif words[-1].isupper() and len(words[-1]) <= 4:
            surname, given_words = "".join(words[:-1]).casefold(), [words[-1]]
        else:
            surname, given_words = words[-1].casefold(), words[:-1]
        if not surname or not given_words:
            return []
        initials = "".join(word.casefold() if word.isupper() and len(word) <= 4
                           else word[0].casefold() for word in given_words)
        first_name = given_words[0].casefold() if len(given_words[0]) > 1 and not given_words[0].isupper() else ""
        result.append((surname, initials, first_name))
    return result


def _preserve_preprint_publication_links(items: list[Any]) -> None:
    """Retain a unique explicit publication link across copies of one preprint.

    Group by the preprint's own identifier, never its published DOI or title.
    Conflicting links stay unresolved. This copies no venue/genre/quality claims.
    """
    from .rank.source_quality import _row_metadata, is_preprint_source

    groups: dict[str, list[tuple[Any, dict]]] = {}
    for item in items:
        if getattr(item, "section", "") != "research" or not is_preprint_source(item):
            continue
        metadata = _row_metadata(item)
        url, external = str(getattr(item, "url", "") or ""), str(getattr(item, "external_id", "") or "")
        arxiv = _extract_arxiv_id(url, external, str(getattr(item, "source", "")).lower())
        doi = _extract_doi(str(metadata.get("doi") or ""), external, url) if not arxiv else None
        key = f"arxiv:{arxiv}" if arxiv else f"doi:{doi}" if doi else "url:" + _normalized_url(url)
        if key != "url:":
            groups.setdefault(key, []).append((item, metadata))
    for group in groups.values():
        links = {_canonical_doi(metadata["published_doi"]) for _, metadata in group
                 if isinstance(metadata.get("published_doi"), str)}
        links = {link for link in links if re.fullmatch(r"10\.\d{4,9}/\S+", link)}
        if len(links) != 1:
            continue
        link = next(iter(links))
        for item, metadata in group:
            if metadata.get("published_doi"):
                continue
            metadata = {**metadata, "published_doi": link}
            if isinstance(getattr(item, "metadata", None), dict):
                item.metadata = metadata
            else:
                item.metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def _preserve_research_exclusions(items: list[Any]) -> None:
    """Attach explicit genres/exclusions to copies sharing a durable identity.

    A rich RSS row must not erase a database record's explicit retraction or
    publication genre when chosen as representative. Only DOI, PMID, arXiv and URL
    links connect evidence; title-based dedupe never transfers it. Existing
    metadata is retained, and this does not write to the database.
    """
    from .rank.source_quality import _row_metadata, research_publication_metadata

    from .publications import _retracted, companion_metadata, is_companion, publication_label

    if not any(research_publication_metadata(item) or companion_metadata(item) for item in items):
        return
    parent = list(range(len(items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    by_key: dict[str, int] = {}
    for index, item in enumerate(items):
        # A preprint's published_doi names a different record (its journal
        # version); it must not carry withdrawal or preprint genre across.
        for key in candidate_identity_keys(item, include_publication_links=False):
            if not key.startswith(("doi:", "pmid:", "arxiv:", "url:")):
                continue
            if key in by_key:
                parent[find(index)] = find(by_key[key])
            else:
                by_key[key] = index

    evidence_by_root: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(items):
        evidence = {**research_publication_metadata(item), **companion_metadata(item)}
        if companion_metadata(item) and _retracted(item):
            evidence["is_retracted"] = True
        if is_companion(item):
            evidence["publication_types"] = sorted(set(evidence.get("publication_types", []))
                                                    | {publication_label(item).casefold()})
        if not evidence:
            continue
        shared = evidence_by_root.setdefault(find(index), {})
        shared.setdefault("companion_of_dois", set()).update(evidence.get("companion_of_dois", []))
        relations = shared.setdefault("companion_relation_evidence", [])
        for record in evidence.get("companion_relation_evidence", []):
            if record not in relations:
                relations.append(record)
        if evidence.get("is_retracted") is True:
            shared["is_retracted"] = True
        shared.setdefault("publication_types", set()).update(evidence.get("publication_types", []))
        provenance = shared.setdefault("publication_type_evidence", [])
        for record in evidence.get("publication_type_evidence", []):
            if record not in provenance:
                provenance.append(record)

    for index, item in enumerate(items):
        evidence = evidence_by_root.get(find(index))
        if not evidence:
            continue
        metadata = dict(_row_metadata(item))
        if evidence.get("companion_of_dois"):
            metadata["companion_of_dois"] = sorted(evidence["companion_of_dois"])
            prior = metadata.get("companion_relation_evidence", [])
            prior = list(prior) if isinstance(prior, list) else []
            metadata["companion_relation_evidence"] = prior + [
                record for record in evidence["companion_relation_evidence"] if record not in prior
            ]
        if evidence.get("is_retracted") is True:
            metadata["is_retracted"] = True
        if evidence.get("publication_types"):
            current_types = metadata.get("publication_types", [])
            if not isinstance(current_types, list):
                current_types = []
            current_types = [value for value in current_types if isinstance(value, str)]
            known_types = {value.strip().casefold() for value in current_types}
            metadata["publication_types"] = current_types + sorted(
                evidence["publication_types"] - known_types
            )
        if evidence.get("publication_type_evidence"):
            provenance = metadata.get("publication_type_evidence", [])
            provenance = list(provenance) if isinstance(provenance, list) else []
            for record in evidence["publication_type_evidence"]:
                if record not in provenance:
                    provenance.append(record)
            metadata["publication_type_evidence"] = provenance
        if isinstance(getattr(item, "metadata", None), dict):
            item.metadata = metadata
        else:
            item.metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def _candidate_representative_score(it: Any) -> float:
    """Prefer direct, high-quality, information-rich source rows for duplicates."""
    try:
        from .rank.source_quality import infer_source_quality, source_bucket

        bucket = source_bucket(it)
        quality = infer_source_quality(
            str(getattr(it, "source", "") or ""),
            str(getattr(it, "section", "") or ""),
        )
        bucket_bonus = {
            "published_journal": 0.24,
            "published_database": 0.14,
            "other_research": 0.08,
            "bio_med_preprint": 0.02,
            "arxiv_other": 0.02,
            "preprint_other": 0.02,
            "arxiv_cs": 0.01,
            "aggregator": -0.12,
        }.get(bucket, 0.0)
        source_score = float(quality.prestige_score) + bucket_bonus
    except Exception:
        source_score = 0.0
    abstract_len = len(str(getattr(it, "abstract", "") or "").strip())
    title_len = len(str(getattr(it, "title", "") or "").strip())
    url = str(getattr(it, "url", "") or "").lower()
    direct_url_bonus = 0.04 if "doi.org" in url or "pubmed" in url else 0.0
    issue_suffix_penalty = (
        0.02 if _JOURNAL_ISSUE_SUFFIX_RE.search(str(getattr(it, "title", "") or "")) else 0.0
    )
    return (
        source_score
        + min(0.16, abstract_len / 2400)
        + min(0.04, title_len / 2500)
        + direct_url_bonus
        - issue_suffix_penalty
    )


def candidate_identity_keys(it: Any, *, include_publication_links: bool = True) -> list[str]:
    """Return durable cross-source identity keys for an item-like object."""
    url = str(getattr(it, "url", "") or "")
    external_id = str(getattr(it, "external_id", "") or "")
    source = str(getattr(it, "source", "") or "").lower()
    title = str(getattr(it, "title", "") or "")

    keys: list[str] = []
    metadata = getattr(it, "metadata", None)
    if not isinstance(metadata, dict):
        try:
            metadata = json.loads(str(getattr(it, "metadata_json", "") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
    metadata_doi = str(metadata.get("doi") or "") if isinstance(metadata, dict) else ""
    doi = _extract_doi(metadata_doi, external_id, url)
    if doi:
        keys.append(f"doi:{doi}")
    # A source-declared publication relationship identifies the same work even
    # when the journal version has a new DOI, title, or publication date.
    published_doi = metadata.get("published_doi") if isinstance(metadata, dict) else None
    if include_publication_links and isinstance(published_doi, str):
        published_doi = _extract_doi(published_doi)
        if published_doi and f"doi:{published_doi}" not in keys:
            keys.append(f"doi:{published_doi}")

    pmid = _extract_pmid(url, external_id, source)
    if pmid:
        keys.append(f"pmid:{pmid}")

    arxiv_id = _extract_arxiv_id(url, external_id, source)
    if arxiv_id:
        keys.append(f"arxiv:{arxiv_id}")

    normalized_url = _normalized_url(url)
    if normalized_url:
        keys.append(f"url:{normalized_url}{_event_start_suffix(it, metadata)}")

    if isinstance(metadata, dict) and getattr(it, "section", "") in {"opportunities", "events"}:
        # Deadline records carry no publication date. NIH Guide and Grants.gov
        # share the official notice number; EMBO and EMBL both list joint
        # symposia under the same title and start date.
        if metadata.get("record_type") == "event":
            if (start := str(metadata.get("event_start") or "")) and (name := _normalized_title_key(title)):
                keys.append(f"event:{name}:{start}")
        elif official := str(metadata.get("official_id") or "").strip().upper():
            keys.append(f"opportunity:{official}")

    title_key = _same_day_title_key(title, _effective_date(it))
    if title_key:
        keys.append(f"title_day:{title_key}")
        # Also emit the previous 12h bucket so items near a boundary still match
        parts = title_key.split(":", 2)
        if len(parts) == 3:
            try:
                keys.append(f"title_day:t12h:{int(parts[1]) - 1}:{parts[2]}")
            except ValueError:
                pass

    return keys


def _extract_doi(*values: str) -> str | None:
    for value in values:
        text = unquote(value or "").strip()
        text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^doi:\s*", "", text, flags=re.IGNORECASE)
        # Try to reconstruct DOI from publisher article URL patterns
        # nature.com/articles/s41586-2026-01234-5 -> 10.1038/s41586-2026-01234-5
        _nature_m = re.search(r"nature\.com/articles/([a-z0-9]+-[0-9]+-[0-9]+-[\da-z]+)", text, re.IGNORECASE)
        if _nature_m:
            doi = f"10.1038/{_nature_m.group(1)}".lower()
            if doi.startswith("10.") and "/" in doi:
                return doi
        # science.org DOI in URL
        _science_m = re.search(r"science\.org/doi/(10\.[0-9]{4,}/\S+)", text, re.IGNORECASE)
        if _science_m:
            doi = _science_m.group(1).strip().rstrip(_TRAILING_DOI_PUNCT).lower()
            if doi.startswith("10.") and "/" in doi:
                return doi
        match = _DOI_RE.search(text)
        doi = (match.group(0) if match else text if text.lower().startswith("10.") else "")
        doi = doi.strip().rstrip(_TRAILING_DOI_PUNCT).lower()
        if doi.startswith("10.") and "/" in doi:
            return doi
    return None


def _extract_pmid(url: str, external_id: str, source: str) -> str | None:
    match = _PMID_URL_RE.search(url)
    if match:
        return match.group(1)
    if "pubmed" in source and external_id.isdigit():
        return external_id
    return None


def _extract_arxiv_id(url: str, external_id: str, source: str) -> str | None:
    values: list[str] = []
    parsed = urlparse(unquote(url or ""))
    host = (parsed.hostname or "").lower()
    if host == "arxiv.org" or host.endswith(".arxiv.org"):
        # Preserve legacy identifiers (e.g. cond-mat/9901234), and only read
        # document paths. Digits in unrelated publisher URLs are not arXiv IDs.
        match = re.fullmatch(r"/(?:abs|pdf)/(.+?)(?:\.pdf)?/?", parsed.path)
        if match:
            values.append(match.group(1))
    if "arxiv" in source:
        values.append(external_id)
    for value in values:
        match = _ARXIV_RE.fullmatch(unquote(value or "").strip())
        if match:
            return match.group(1).lower()
    return None


def _normalized_url(url: str) -> str:
    canon = canonicalize_url(url)
    if not canon:
        return ""
    parsed = urlparse(canon)
    if not parsed.scheme or not parsed.netloc:
        return canon.rstrip("/")
    query = urlencode(
        sorted(
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS
        ),
        doseq=True,
    )
    path = quote(unquote(parsed.path), safe="/:@")
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower().removeprefix("www."),
            path.rstrip("/") or "/",
            "",
            query,
            "",
        )
    ).rstrip("/")


def _effective_date(it: Any) -> datetime | None:
    published_at = getattr(it, "published_at", None)
    if isinstance(published_at, datetime):
        return published_at
    fetched_at = getattr(it, "fetched_at", None)
    if isinstance(fetched_at, datetime):
        return fetched_at
    return None


def _same_day_title_key(title: str, dt: datetime | None) -> str:
    """Return a dedup key bucketed by 12-hour epoch windows.

    Using 12-hour buckets (vs calendar-day) means items published within
    ~12h of a midnight boundary (e.g. Nature RSS at 23:00 UTC and OpenAlex
    at 02:00 UTC next day) still land in adjacent buckets and can be matched
    via the two-key approach in _candidate_keys.
    """
    if dt is None:
        return ""
    normalized = _normalized_title_key(title)
    normalized = re.sub(r"\s+", " ", normalized)
    if len(normalized) < 12:
        return ""
    import calendar
    ts = int(dt.timestamp()) if dt.tzinfo is not None else calendar.timegm(dt.timetuple())
    bucket = ts // (12 * 3600)
    return f"t12h:{bucket}:{normalized}"


def _normalized_title_key(title: str) -> str:
    title = _JOURNAL_ISSUE_SUFFIX_RE.sub("", title.strip())
    return re.sub(r"[\W_]+", " ", title.casefold()).strip()


# Language filter rule. Precomputed decisions (CI public-source snapshots) carry
# language_rule_version() and are ignored entirely on any mismatch. Bump the
# revision whenever the sample cleaning, thresholds or keep/drop rule changes.
_LANGUAGE_RULE_REVISION = 1
_LANGUAGE_MIN_CHARS = 200
_LANGUAGE_MEMO_MAX_CHARS = 8192
_LANGUAGE_DROP_PROBABILITY = 0.95
_LANGUAGE_SEED = 0
_LANGUAGE_KEY_RE = re.compile(r"[0-9a-f]{64}")
# Decisions registered by the browser snapshot loader: (usable map, status).
_registered_language_decisions: tuple[dict[str, bool] | None, str] = (None, "absent")
# Counts from the most recent filter_english call, for diagnostics/profiling.
last_language_filter_stats: dict[str, Any] = {}


@lru_cache(maxsize=4096)
def _language_allowed(sample: str, detector) -> bool:
    """Reuse deterministic decisions for identical text within this process."""
    languages = detector(sample)
    return not languages or languages[0].lang == 'en' or languages[0].prob < _LANGUAGE_DROP_PROBABILITY


def _language_sample(it: Any) -> str:
    return (it.title + " " + it.abstract).strip()


def _detect_language_allowed(sample: str, detector) -> bool:
    # Bound retained text, but always classify the complete sample.
    # Exceptions are not cached, so temporary failures are retried.
    detect = _language_allowed if len(sample) <= _LANGUAGE_MEMO_MAX_CHARS else _language_allowed.__wrapped__
    return detect(sample, detector)


def _load_language_detector():
    from langdetect import DetectorFactory, detect_langs

    DetectorFactory.seed = _LANGUAGE_SEED
    return detect_langs


def language_sample_key(sample: str) -> str:
    """Key a decision by the exact detector input (UTF-8 sha256)."""
    import hashlib

    return hashlib.sha256(sample.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _langdetect_fingerprint() -> str:
    """Hash langdetect's code and language profiles (no dist-info in the browser bundle)."""
    import hashlib
    from pathlib import Path

    import langdetect

    root = Path(langdetect.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*")
                       if p.is_file() and "__pycache__" not in p.relative_to(root).parts):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()[:24]


def language_rule_version() -> str | None:
    """Identify the exact rule behind a decision, or None when undeterminable."""
    try:
        fingerprint = _langdetect_fingerprint()
    except Exception:
        return None
    return (f"english-filter-r{_LANGUAGE_RULE_REVISION};sample=strip(title+' '+abstract);"
            f"min_chars={_LANGUAGE_MIN_CHARS};memo_max_chars={_LANGUAGE_MEMO_MAX_CHARS};"
            f"drop=top!=en&prob>={_LANGUAGE_DROP_PROBABILITY};seed={_LANGUAGE_SEED};"
            f"langdetect-sha256={fingerprint}")


def language_decisions_status(payload: Any, *, max_entries: int | None = None) -> str:
    """Return absent, malformed, version-mismatch or current for a decision payload."""
    if payload is None:
        return "absent"
    allowed = payload.get("allowed") if isinstance(payload, dict) else None
    version = payload.get("rule_version") if isinstance(payload, dict) else None
    if (not isinstance(version, str) or not version or len(version) > 512
            or not isinstance(allowed, dict)
            or (max_entries is not None and len(allowed) > max_entries)
            or not all(isinstance(key, str) and _LANGUAGE_KEY_RE.fullmatch(key) and type(value) is bool
                       for key, value in allowed.items())):
        return "malformed"
    current = language_rule_version()
    return "current" if current is not None and version == current else "version-mismatch"


def _usable_language_decisions(payload: Any) -> tuple[dict[str, bool] | None, str]:
    status = language_decisions_status(payload)
    return (payload["allowed"] if status == "current" else None), status


def register_language_decisions(payload: Any) -> str:
    """Install precomputed decisions for later filter_english calls without a map."""
    global _registered_language_decisions
    try:
        _registered_language_decisions = _usable_language_decisions(payload)
    except Exception:
        _registered_language_decisions = (None, "malformed")
    return _registered_language_decisions[1]


def precompute_language_decisions(items: list[Item]) -> dict[str, Any] | None:
    """Record definite current-rule decisions for long samples, keyed by text hash.

    Short samples are decided by length alone. Detector failures are omitted,
    never stored as a result, so consumers retry them exactly as before.
    """
    try:
        detector = _load_language_detector()
    except Exception:
        return None
    version = language_rule_version()
    if version is None:
        return None
    allowed: dict[str, bool] = {}
    for it in items:
        sample = _language_sample(it)
        if len(sample) < _LANGUAGE_MIN_CHARS:
            continue
        try:
            key = language_sample_key(sample)
            allowed[key] = bool(_detect_language_allowed(sample, detector))
        except Exception:
            continue
    return {"rule_version": version, "allowed": dict(sorted(allowed.items()))}


def filter_english(items: list[Item], language_decisions: dict[str, Any] | None = None) -> list[Item]:
    """Drop only confident non-English long text; technical titles are ambiguous.

    ``language_decisions`` is an optional precomputed payload; without one, any
    registered snapshot payload is used. Only a current-rule payload is used, and
    then only as a lookup: a miss runs the detector exactly as without a map.
    """
    global last_language_filter_stats
    if language_decisions is None:
        allowed, status = _registered_language_decisions
    else:
        allowed, status = _usable_language_decisions(language_decisions)
    stats = {"items": len(items), "short": 0, "precomputed": 0, "detector": 0,
             "errors": 0, "decisions": status}
    last_language_filter_stats = stats
    try:
        detect_langs = _load_language_detector()
    except Exception:
        stats["decisions"] = "detector-unavailable"
        return items

    out: list[Item] = []
    for it in items:
        sample = _language_sample(it)
        if len(sample) < _LANGUAGE_MIN_CHARS:
            stats["short"] += 1
            out.append(it)
            continue
        if allowed:
            try:
                decision = allowed.get(language_sample_key(sample))
            except Exception:
                decision = None
            if decision is not None:
                stats["precomputed"] += 1
                if decision:
                    out.append(it)
                continue
        stats["detector"] += 1
        try:
            if _detect_language_allowed(sample, detect_langs):
                out.append(it)
        except Exception:
            stats["errors"] += 1
            out.append(it)
    return out
