from __future__ import annotations

import subprocess
import time

import pytest

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


def _item(i: int) -> ItemRow:
    return ItemRow(
        id=i, source="Nature", section="research", external_id=f"e{i}",
        url=f"https://example.com/{i}", title=f"Title {i}",
        abstract="A sentence describing the finding. Another sentence with detail.",
    )


def test_call_cli_timeout_kills_grandchildren_and_does_not_hang(monkeypatch):
    """The CLI's backgrounded child keeps the stdout pipe open; without a
    process-group kill, communicate() would block until that child exits (30s)
    despite the short timeout. _call_cli must SIGKILL the whole group and return
    within a bound close to the timeout."""
    monkeypatch.setattr(sm, "_CLI_TIMEOUT", 1)
    # sh backgrounds a 30s sleep that inherits stdout, then blocks another 30s.
    cmd = ["sh", "-c", "sleep 30 & sleep 30"]
    t = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        sm._call_cli([_item(1)], cmd)
    elapsed = time.monotonic() - t
    assert elapsed < 12, f"hung for {elapsed:.1f}s — process group not killed"


def test_summarize_via_cli_falls_back_to_extractive_on_timeout(monkeypatch):
    monkeypatch.setattr(sm, "_CLI_TIMEOUT", 1)
    items = [_item(1), _item(2)]
    out = sm._summarize_via_cli(items, ["sh", "-c", "sleep 30 & sleep 30"])
    # Every item still gets a (extractive) summary; the brew is never blocked.
    assert set(out) == {1, 2}
    assert all(out[i] for i in (1, 2))


def test_summarize_total_budget_switches_to_extractive(monkeypatch):
    monkeypatch.setattr(sm, "_CLI_TIMEOUT", 1)
    monkeypatch.setattr(sm, "_CLI_TOTAL_BUDGET", 0)  # exhausted immediately
    monkeypatch.setattr(sm, "_BATCH_SIZE", 1)
    calls = {"n": 0}
    orig = sm._call_cli
    def _spy(batch, cmd):
        calls["n"] += 1
        return orig(batch, cmd)
    monkeypatch.setattr(sm, "_call_cli", _spy)
    items = [_item(i) for i in range(1, 5)]
    out = sm._summarize_via_cli(items, ["sh", "-c", "sleep 30"])
    assert set(out) == {1, 2, 3, 4}          # all summarized (extractively)
    assert calls["n"] == 0                    # budget skipped the CLI entirely
