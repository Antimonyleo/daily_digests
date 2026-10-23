from __future__ import annotations

import json
import re
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
    key_to_index: dict[str, int] = {}
    # deleted[i] = True means out[i] was merged into another slot
    deleted: list[bool] = []
    out: list[T] = []

    for it in items:
        keys = candidate_identity_keys(it)
        matched_indices: list[int] = list(dict.fromkeys(
            key_to_index[k] for k in keys if k in key_to_index
        ))
        # Strip slots that were already merged away
        matched_indices = [i for i in matched_indices if not deleted[i]]

        if not matched_indices:
            idx = len(out)
            out.append(it)
            deleted.append(False)
            for key in keys:
                key_to_index[key] = idx
            continue

        # Canonical slot is the first match; merge any additional matched slots
        idx = matched_indices[0]
        for extra in matched_indices[1:]:
            # Absorb extra slot into idx: keep best representative
            if _candidate_representative_score(out[extra]) > _candidate_representative_score(out[idx]):
                out[idx] = out[extra]
            deleted[extra] = True
            # Re-point all keys that referenced extra to idx
            for k, v in key_to_index.items():
                if v == extra:
                    key_to_index[k] = idx

        current = out[idx]
        if _candidate_representative_score(it) > _candidate_representative_score(current):
            out[idx] = it
        for key in keys:
            key_to_index[key] = idx

    distinct = [item for item, d in zip(out, deleted, strict=True) if not d]
    # Publication versions often have different DOIs and publication dates.
    # Keep this title fallback separate from record identity: it must not copy
    # a preprint's publication genre or retraction status onto a journal record.
    version_titles: dict[str, list[int]] = {}
    result: list[T] = []
    for item in distinct:
        title = publication_version_title_key(item)
        match = next((index for index in version_titles.get(title, [])
                      if are_publication_versions(item, result[index])), None)
        if match is None:
            if title:
                version_titles.setdefault(title, []).append(len(result))
            result.append(item)
        elif _candidate_representative_score(item) > _candidate_representative_score(result[match]):
            result[match] = item
    return result


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
        for key in candidate_identity_keys(item):
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
        keys.append(f"url:{normalized_url}")

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


@lru_cache(maxsize=4096)
def _language_allowed(sample: str, detector) -> bool:
    """Reuse deterministic decisions for identical text within this process."""
    languages = detector(sample)
    return not languages or languages[0].lang == 'en' or languages[0].prob < 0.95


def filter_english(items: list[Item]) -> list[Item]:
    """Drop only confident non-English long text; technical titles are ambiguous."""
    try:
        from langdetect import DetectorFactory, detect_langs

        DetectorFactory.seed = 0
    except Exception:
        return items

    out: list[Item] = []
    for it in items:
        sample = (it.title + " " + it.abstract).strip()
        if len(sample) < 200:
            out.append(it)
            continue
        try:
            # Bound retained text, but always classify the complete sample.
            # Exceptions are not cached, so temporary failures are retried.
            detect = _language_allowed if len(sample) <= 8192 else _language_allowed.__wrapped__
            if detect(sample, detect_langs):
                out.append(it)
        except Exception:
            out.append(it)
    return out
