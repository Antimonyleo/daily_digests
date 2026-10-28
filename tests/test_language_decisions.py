"""Precomputed language decisions must be exactly equivalent to running the detector."""

from __future__ import annotations

import hashlib
import inspect
import sys

import pytest

from dailydigest import dedupe
from dailydigest.dedupe import (
    filter_english,
    language_decisions_status,
    language_rule_version,
    language_sample_key,
    precompute_language_decisions,
    register_language_decisions,
)
from dailydigest.models import Item

ENGLISH = ("This study presents a programmable DNA origami scaffold that assembles protein cargo "
           "with nanometre precision. The results show improved yield and stability across buffers. ")
FRENCH = ("Cette étude présente une nouvelle méthode de recherche pour assembler des protéines. "
          "Les résultats montrent une amélioration importante de la précision et du rendement. ")
GERMAN = ("Diese Studie stellt eine neue Methode zur Herstellung von Proteinkomplexen vor. "
          "Die Ergebnisse zeigen eine deutliche Verbesserung der Genauigkeit und der Ausbeute. ")
SPANISH = ("Este estudio presenta un nuevo método de investigación para ensamblar proteínas. "
           "Los resultados muestran una mejora importante de la precisión y del rendimiento. ")


def _item(external_id: str, title: str, abstract: str = "") -> Item:
    return Item(source="fixture", section="research", external_id=external_id,
                url=f"https://example.org/{external_id}", title=title, abstract=abstract)


def _fixture() -> list[Item]:
    return [
        _item("en", "Programmable DNA origami", ENGLISH * 2),
        _item("fr", "Une nouvelle méthode", FRENCH * 2),
        _item("de", "Eine neue Methode", GERMAN * 2),
        _item("es", "Un nuevo método", SPANISH * 2),
        _item("short", "Perovskite solar cells"),
        _item("digits", "1234 5678", "0123456789 " * 30),  # detector has no features: raises
        _item("long", "A long English abstract", ENGLISH * 60),  # beyond the 8192 memo bound
        _item("dup", "Programmable DNA origami", ENGLISH * 2),  # same sample as "en"
    ]


def _ids(items: list[Item]) -> list[str]:
    return [it.external_id for it in items]


@pytest.fixture(autouse=True)
def _no_registered_decisions(monkeypatch):
    monkeypatch.setattr(dedupe, "_registered_language_decisions", (None, "absent"))


@pytest.fixture
def counted(monkeypatch):
    """Count real detector calls; a new function object never hits earlier LRU entries."""
    import langdetect

    real = langdetect.detect_langs
    calls: list[str] = []

    def detect(text):
        calls.append(text)
        return real(text)

    monkeypatch.setattr(langdetect, "detect_langs", detect)
    return calls


def test_precomputed_hit_miss_mismatch_and_no_map_retain_identical_ids(counted):
    items = _fixture()
    baseline = _ids(filter_english(items))
    assert baseline == ["en", "short", "digits", "long", "dup"]  # meaningful: three drops
    payload = precompute_language_decisions(items)
    # Short text is decided by length and the featureless sample raised: neither is stored.
    assert len(payload["allowed"]) == 5
    assert payload["rule_version"] == language_rule_version()

    counted.clear()
    assert _ids(filter_english(items, payload)) == baseline
    assert dedupe.last_language_filter_stats["precomputed"] == 6
    assert counted == [(items[5].title + " " + items[5].abstract).strip()]  # only the failure retries

    partial = {**payload, "allowed": dict(list(payload["allowed"].items())[:2])}
    counted.clear()
    assert _ids(filter_english(items, partial)) == baseline
    assert dedupe.last_language_filter_stats["decisions"] == "current"
    assert counted  # misses fall back to the detector

    flipped = {"rule_version": payload["rule_version"] + "-old",
               "allowed": {key: not value for key, value in payload["allowed"].items()}}
    counted.clear()
    assert _ids(filter_english(items, flipped)) == baseline
    assert dedupe.last_language_filter_stats["decisions"] == "version-mismatch"
    assert dedupe.last_language_filter_stats["precomputed"] == 0

    assert register_language_decisions(payload) == "current"
    assert _ids(filter_english(items)) == baseline
    assert dedupe.last_language_filter_stats["precomputed"] == 6
    assert register_language_decisions(flipped) == "version-mismatch"
    assert _ids(filter_english(items)) == baseline
    assert register_language_decisions(None) == "absent"
    assert _ids(filter_english(items)) == baseline


def test_detector_failures_are_never_precomputed_and_still_retain(monkeypatch):
    import langdetect

    real = langdetect.detect_langs
    failing = (FRENCH * 2).strip()

    def flaky(text):
        if FRENCH.strip() in text:
            raise RuntimeError("temporary detector failure")
        return real(text)

    monkeypatch.setattr(langdetect, "detect_langs", flaky)
    row = _item("fr", "Une nouvelle méthode", failing)
    payload = precompute_language_decisions([row])
    assert payload["allowed"] == {}
    assert filter_english([row], payload) == [row]
    monkeypatch.setattr(langdetect, "detect_langs", lambda text: real(text))
    # The retained failure was not a cached English verdict: the next check drops it.
    assert filter_english([row], payload) == []


@pytest.mark.parametrize("payload", [
    "not a map",
    {"rule_version": None, "allowed": {}},
    {"allowed": {}},
    {"rule_version": "x", "allowed": []},
])
def test_malformed_payload_shapes(payload):
    assert language_decisions_status(payload) == "malformed"


def test_malformed_entries_ignore_the_whole_map(counted):
    items = _fixture()
    baseline = _ids(filter_english(items))
    payload = precompute_language_decisions(items)
    for bad in ({"not-a-sha": True}, {"0" * 64: 1}, {"A" * 64: False}):
        poisoned = {**payload, "allowed": {**{k: not v for k, v in payload["allowed"].items()}, **bad}}
        assert language_decisions_status(poisoned) == "malformed"
        assert _ids(filter_english(items, poisoned)) == baseline
        assert dedupe.last_language_filter_stats["precomputed"] == 0
        assert register_language_decisions(poisoned) == "malformed"
        assert _ids(filter_english(items)) == baseline


def test_unavailable_detector_keeps_everything_even_with_a_current_map(monkeypatch):
    items = _fixture()
    payload = precompute_language_decisions(items)
    monkeypatch.setitem(sys.modules, "langdetect", None)
    assert filter_english(items, payload) is items
    assert precompute_language_decisions(items) is None


def test_decisions_key_the_exact_cleaned_sample():
    row = _item("fr", "  Une nouvelle méthode", FRENCH * 2 + "  ")
    payload = precompute_language_decisions([row])
    assert list(payload["allowed"]) == [language_sample_key((row.title + " " + row.abstract).strip())]
    assert language_sample_key("é") == hashlib.sha256("é".encode("utf-8")).hexdigest()


def test_rule_version_covers_thresholds_seed_and_detector_build():
    version = language_rule_version()
    for part in ("sample=strip(title+' '+abstract)", "min_chars=200", "memo_max_chars=8192",
                 "prob>=0.95", "seed=0", "langdetect-sha256="):
        assert part in version
    # Editing the decision rule must bump _LANGUAGE_RULE_REVISION and this pin.
    rule_source = "".join(inspect.getsource(fn) for fn in (
        dedupe._language_allowed.__wrapped__, dedupe._language_sample,
        dedupe._detect_language_allowed, dedupe._load_language_detector))
    assert (dedupe._LANGUAGE_RULE_REVISION, hashlib.sha256(rule_source.encode()).hexdigest()[:16]) == (
        1, "9c6593a21c8081aa")
