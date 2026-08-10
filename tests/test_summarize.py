from __future__ import annotations

from pathlib import Path
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


def test_anthropic_backend_uses_native_messages_api(monkeypatch):
    from dailydigest import config

    item = _row("RNA assembly", "A programmable RNA assembly was demonstrated.")
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"content": [{"type": "text", "text": '{"1":"Native Claude summary"}'}]}

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, url, *, headers, json):
            calls.append((url, headers, json))
            return FakeResponse()

    monkeypatch.setattr(
        config,
        "get_settings",
        lambda: SimpleNamespace(
            llm_backend="anthropic",
            llm_api_key="anthropic-key",
            llm_base_url="https://api.anthropic.com/v1",
            llm_model="claude-haiku-4-5-20251001",
        ),
    )
    monkeypatch.setattr(sm.httpx, "Client", FakeClient)

    summaries = sm.summarize_items([item])

    assert summaries == {1: "Native Claude summary"}
    url, headers, body = calls[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "anthropic-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "claude-haiku-4-5-20251001"
    assert body["messages"][0]["role"] == "user"


def test_claude_cli_backend_disables_tools_and_session_persistence(monkeypatch):
    from dailydigest import config

    item = _row("RNA assembly", "A programmable RNA assembly was demonstrated.")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"1":"Claude CLI summary"}', stderr="")

    monkeypatch.setattr(
        config,
        "get_settings",
        lambda: SimpleNamespace(
            llm_backend="claude_cli",
            llm_api_key="",
            llm_base_url="",
            llm_model="haiku",
        ),
    )
    monkeypatch.setattr(sm.shutil, "which", lambda name: f"/tools/{name}")
    monkeypatch.setattr(sm.subprocess, "run", fake_run)

    summaries = sm.summarize_items([item])

    assert summaries == {1: "Claude CLI summary"}
    command, kwargs = calls[0]
    assert command[0] == "/tools/claude"
    assert "--safe-mode" in command
    assert "--no-session-persistence" in command
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--model") + 1] == "haiku"
    assert kwargs["cwd"] != str(Path.cwd())
    assert kwargs["timeout"] > 0
    assert "Summarize each" in kwargs["input"]


def test_codex_cli_backend_is_ephemeral_read_only_and_disables_agent_tools(monkeypatch):
    from dailydigest import config

    item = _row("DNA origami", "A new DNA origami method was demonstrated.")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"1":"Codex CLI summary"}', stderr="")

    monkeypatch.setattr(
        config,
        "get_settings",
        lambda: SimpleNamespace(
            llm_backend="codex_cli",
            llm_api_key="",
            llm_base_url="",
            llm_model="",
        ),
    )
    monkeypatch.setattr(sm.shutil, "which", lambda name: f"/tools/{name}")
    monkeypatch.setattr(sm.subprocess, "run", fake_run)

    summaries = sm.summarize_items([item])

    assert summaries == {1: "Codex CLI summary"}
    command, kwargs = calls[0]
    assert command[:2] == ["/tools/codex", "exec"]
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in command
    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    disabled = {
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--disable"
    }
    assert {"shell_tool", "unified_exec", "apps", "browser_use", "plugins"} <= disabled
    assert "--dangerously-bypass-approvals-and-sandbox" not in command
    assert kwargs["cwd"] != str(Path.cwd())


def test_missing_signed_in_cli_falls_back_to_extractive(monkeypatch):
    from dailydigest import config

    monkeypatch.setattr(
        config,
        "get_settings",
        lambda: SimpleNamespace(
            llm_backend="claude_cli",
            llm_api_key="",
            llm_base_url="",
            llm_model="haiku",
        ),
    )
    monkeypatch.setattr(sm.shutil, "which", lambda _name: None)

    summaries = sm.summarize_items(
        [_row("RNA delivery", "The study reports a screen of 80 RNA formulations.")]
    )

    assert summaries[1].startswith("Key finding:")
