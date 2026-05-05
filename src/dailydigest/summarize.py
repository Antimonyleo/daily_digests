"""Per-item summarization.

Backends, selected by ``SETTINGS.llm_backend``:

* ``api`` (default): OpenAI-compatible HTTP API. Requires ``LLM_API_KEY``;
  silently falls through to ``extractive`` if no key is configured.
* ``claude_code``: shells out to the local ``claude`` CLI in non-interactive
  print mode. Uses your Anthropic subscription quota instead of API credits.
* ``codex``: shells out to the local ``codex`` CLI (``codex exec``). Uses your
  OpenAI subscription / login.
* ``extractive``: no LLM. Returns the first 1-2 sentences of each abstract.
  Also the per-batch fallback whenever any other backend fails.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess

import httpx

from .config import SETTINGS
from .store import ItemRow

logger = logging.getLogger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_BATCH_SIZE = 10
_TIMEOUT = 60.0
_CLI_TIMEOUT = 120  # seconds per batch for subprocess backends

# One-shot guard so a missing CLI does not spam ERROR logs once per batch.
_cli_missing_warned: set[str] = set()


def _extractive(row: ItemRow) -> str:
    abstract = (row.abstract or "").strip()
    if not abstract:
        return (row.title or "").strip()
    sents = _SENT_SPLIT.split(abstract)
    return " ".join(sents[:2]).strip()


def _build_prompt(batch: list[ItemRow]) -> tuple[str, str]:
    payload = [
        {
            "id": row.id,
            "title": (row.title or "").strip(),
            "abstract": (row.abstract or "").strip()[:1500],
        }
        for row in batch
    ]
    sys = (
        "You are a concise neutral summarizer for a daily research digest. "
        "Summarize each item in 1-2 sentences, factual and non-promotional. "
        "Return strict JSON: an object mapping the item's integer id (as a string) "
        "to its summary. No prose, no markdown."
    )
    user = (
        "Summarize each of the following items. Output JSON only.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    return sys, user


def _build_cli_prompt(batch: list[ItemRow]) -> str:
    """Single-string prompt for stdin-fed CLI backends.

    The system instructions are inlined since some CLIs do not expose a stable
    system-prompt flag for stdin mode. The JSON-only requirement is repeated
    twice to make refusals less likely.
    """
    sys, user = _build_prompt(batch)
    return (
        f"{sys}\n\n"
        f"{user}\n\n"
        "Respond ONLY with a JSON object mapping item ids (as string keys) "
        "to one-sentence summaries. No prose, no markdown fences, no explanation."
    )


# ANSI color/cursor escapes that CLIs sometimes emit even when piped.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# Best-effort balanced-brace JSON object extractor. Greedy on outermost { ... }.
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
# Markdown code fences such as ```json\n{...}\n```
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> str:
    """Pull the first JSON object out of free-form CLI output.

    Strips ANSI escapes, prefers fenced ```json``` blocks, otherwise greedily
    matches the outermost ``{...}``. Raises ``json.JSONDecodeError`` indirectly
    by returning a string the caller will pass to ``json.loads``.
    """
    cleaned = _ANSI_RE.sub("", text)
    fence = _FENCE_RE.search(cleaned)
    if fence:
        return fence.group(1)
    match = _JSON_OBJ_RE.search(cleaned)
    if match:
        return match.group(0)
    return cleaned.strip()


def _parse_id_summary_map(raw: str) -> dict[int, str]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("LLM output was not a JSON object")
    out: dict[int, str] = {}
    for k, v in parsed.items():
        try:
            val = str(v).strip()
            if val:
                out[int(k)] = val
        except (TypeError, ValueError):
            continue
    return out


def _call_llm(batch: list[ItemRow]) -> dict[int, str]:
    from .config import get_settings
    cfg = get_settings()
    sys_prompt, user_prompt = _build_prompt(batch)
    body = {
        "model": cfg.llm_model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {cfg.llm_api_key}",
        "Content-Type": "application/json",
    }
    url = cfg.llm_base_url.rstrip("/") + "/chat/completions"
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return _parse_id_summary_map(content)


def _call_cli(batch: list[ItemRow], cli_cmd: list[str]) -> dict[int, str]:
    """Run a CLI subprocess with the prompt fed via stdin, parse JSON stdout."""
    prompt = _build_cli_prompt(batch)
    completed = subprocess.run(  # noqa: S603 - cli_cmd is hard-coded, no shell
        cli_cmd,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=_CLI_TIMEOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            cli_cmd,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    raw = _extract_json_object(completed.stdout or "")
    return _parse_id_summary_map(raw)


def _summarize_via_cli(
    items: list[ItemRow], cli_cmd: list[str]
) -> dict[int, str]:
    """Generic subprocess summarizer.

    On per-batch failure (missing CLI, timeout, non-zero exit, malformed JSON)
    falls back to extractive for that batch only — never aborts the digest.
    """
    out: dict[int, str] = {}
    cli_name = cli_cmd[0]
    for i in range(0, len(items), _BATCH_SIZE):
        batch = items[i : i + _BATCH_SIZE]
        try:
            out.update(_call_cli(batch, cli_cmd))
        except FileNotFoundError:
            if cli_name not in _cli_missing_warned:
                logger.error(
                    "CLI %r not installed; falling back to extractive for "
                    "all batches in this run",
                    cli_name,
                )
                _cli_missing_warned.add(cli_name)
            for row in batch:
                out.setdefault(row.id, _extractive(row))
        except subprocess.TimeoutExpired:
            logger.warning(
                "%s timed out after %ds for batch %d; falling back",
                cli_name, _CLI_TIMEOUT, i,
            )
            for row in batch:
                out.setdefault(row.id, _extractive(row))
        except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "%s failed for batch %d: %s; falling back to extractive",
                cli_name, i, e,
            )
            for row in batch:
                out.setdefault(row.id, _extractive(row))

    for row in items:
        if not out.get(row.id):  # fill missing or empty-string summaries
            out[row.id] = _extractive(row)
    return out


def _summarize_via_api(items: list[ItemRow]) -> dict[int, str]:
    out: dict[int, str] = {}
    for i in range(0, len(items), _BATCH_SIZE):
        batch = items[i : i + _BATCH_SIZE]
        try:
            out.update(_call_llm(batch))
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM summarize failed for batch %d: %s; falling back", i, e)
            for row in batch:
                out.setdefault(row.id, _extractive(row))
    for row in items:
        if not out.get(row.id):
            out[row.id] = _extractive(row)
    return out


def _summarize_extractive(items: list[ItemRow]) -> dict[int, str]:
    return {row.id: _extractive(row) for row in items}


def summarize_items(items: list[ItemRow]) -> dict[int, str]:
    """Return ``{item.id: summary}``.

    Backend chosen by ``SETTINGS.llm_backend``. The ``api`` backend additionally
    requires ``LLM_API_KEY``; without it, falls through to extractive. Any
    backend failure is contained to the failing *batch*, not the whole run.
    """
    if not items:
        return {}

    from .config import get_settings
    s = get_settings()
    backend = (s.llm_backend or "api").lower()

    if backend == "extractive":
        return _summarize_extractive(items)
    if backend == "claude_code":
        cmd: list[str] = ["claude", "--print"]
        if s.llm_cli_model:
            cmd += ["--model", s.llm_cli_model]
        return _summarize_via_cli(items, cmd)
    if backend == "codex":
        # --color never: keep stdout free of ANSI escapes.
        # --skip-git-repo-check: work when called outside a git repo.
        # reasoning_effort=low: summarization is simple; high effort burns quota needlessly.
        cmd = [
            "codex", "exec",
            "--color", "never",
            "--skip-git-repo-check",
            "-c", "reasoning_effort=low",
        ]
        if s.llm_cli_model:
            cmd += ["--model", s.llm_cli_model]
        return _summarize_via_cli(items, cmd)

    # default: OpenAI-compatible HTTP API; no key -> extractive
    if not s.llm_api_key:
        return _summarize_extractive(items)
    return _summarize_via_api(items)
