from __future__ import annotations

from types import SimpleNamespace

from dailydigest import summarize as sm
from dailydigest.store import ItemRow
from dailydigest.summarize import _build_prompt, _extractive, _filter_to_batch_ids


def _row(title: str, abstract: str, source: str = "Nature") -> ItemRow:
    return ItemRow(
        id=1,
        source=source,
        section="research",
        external_id="summary-test",
        url="https://example.com/summary-test",
        title=title,
        abstract=abstract,
    )


def test_extractive_summary_prefers_informative_sentences_over_title_paraphrase():
    item = _row(
        "RNA delivery platform improves tissue targeting",
        (
            "RNA delivery platform improves tissue targeting. "
            "The study reports a lipid nanoparticle screening method across 120 formulations. "
            "In mice, the lead formulation increased liver-sparing spleen delivery by 4-fold."
        ),
    )

    summary = _extractive(item)

    assert summary.startswith("Key finding:")
    assert "120 formulations" in summary
    assert "4-fold" in summary
    assert "Why read:" in summary
    assert "Caveat:" in summary
    assert "RNA delivery platform improves tissue targeting. The study" not in summary


def test_prompt_requests_substance_and_why_read_context():
    _sys, user = _build_prompt([_row("Title", "Abstract")])
    sys_prompt, _user_prompt = _build_prompt([_row("Title", "Abstract")])

    assert "do not paraphrase the title" in sys_prompt
    assert "Why read" in sys_prompt and "BRIDGE" in sys_prompt
    assert "Key finding" in sys_prompt
    assert "Caveat" in sys_prompt and "limitation" in sys_prompt
    assert '"title": "Title"' in user
    assert '"source": "Nature"' in user


def test_filter_to_batch_ids_drops_hallucinated_summary_ids():
    item = _row("Title", "Abstract")
    item.id = 7

    filtered = _filter_to_batch_ids({7: "real", 999: "wrong item"}, [item])

    assert filtered == {7: "real"}


def test_unknown_legacy_backend_uses_extractive_without_calling_api(monkeypatch):
    """Removed backend names in an old .env must not break a public install."""
    from dailydigest import config

    monkeypatch.setattr(
        config,
        "get_settings",
        lambda: SimpleNamespace(llm_backend="legacy_cli", llm_api_key="unused"),
    )
    monkeypatch.setattr(
        sm,
        "_summarize_via_api",
        lambda _items: (_ for _ in ()).throw(AssertionError("API was called")),
    )

    summary = sm.summarize_items([_row("Title", "A concrete finding.")])

    assert summary[1].startswith("Key finding:")
