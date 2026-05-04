# LLM backends

`dailydigest` writes ~20 one-sentence item summaries per digest. The summarizer
in `src/dailydigest/summarize.py` supports four interchangeable backends,
selected by the `LLM_BACKEND` env var. The same prompt and the same
`{item_id: sentence}` JSON contract are used everywhere; on any failure we
fall back to extractive for the failing *batch* only, never the whole run.

## Token budget (the same for every backend)

Roughly **~30k input tokens per day** total, regardless of which backend you
pick. Embedding, ranking, deduping and HTML rendering are all deterministic
Python — the LLM only writes the 20 summary sentences. There are 2 batches of
10 items per digest, so a single run is 2 short prompt round-trips.

> **Subscriptions are not API keys.** A Claude Pro/Max plan does *not* give
> you Anthropic API credit, and ChatGPT Plus does not give you OpenAI API
> credit. The CLI backends below piggyback on the *CLI's* OAuth login so the
> usage counts against your subscription's rate limits. Use the `api`
> backend only when you actually want to spend per-token credit.

## `api` (default)

OpenAI-compatible HTTP. Drives any endpoint that speaks
`/v1/chat/completions`: OpenAI proper, NanoGPT, OpenRouter, Ollama, vLLM,
LM Studio.

```env
LLM_BACKEND=api
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

- **Cost model:** per-token billing on whichever provider the base URL points
  at. With `gpt-4o-mini` the daily summarize step is well under one cent.
- **Latency:** ~300 ms per batch. Fastest of the four.
- **Setup:** drop a key in `.env`; nothing else.

## `claude_code` (Anthropic subscription via the `claude` CLI)

Shells out to `claude --print`, feeding the prompt on stdin and parsing JSON
from stdout. Uses your existing Claude Code session, so the request counts
against your Pro / Max plan rather than billed API tokens.

```env
LLM_BACKEND=claude_code
```

- **Install:** `npm install -g @anthropic-ai/claude-code`, then run `claude`
  once interactively to log in. Verify with `which claude` and
  `claude --print "ping" <<<""`.
- **Cost model:** counts against the *message limit* of your Claude
  subscription. Free if you have one; you'll hit the cap eventually if you
  also use Claude Code heavily for coding.
- **Latency:** ~5–15 s per batch (model thinking time + CLI startup).
- **Caveats:**
  - Requires an interactive login the first time; cannot run on a fresh
    GitHub Actions runner without provisioning credentials.
  - The CLI is a Node.js binary that boots a full agent harness on every
    call — startup overhead is real.
  - Output sometimes wraps the JSON in `\`\`\`json … \`\`\`` fences;
    `_extract_json_object` peels them off.

## `codex` (OpenAI / ChatGPT subscription via the `codex` CLI)

Shells out to `codex exec --color never --skip-git-repo-check`, feeds the
prompt on stdin, parses JSON from stdout. Uses your `codex login` session
(ChatGPT account) instead of an API key.

```env
LLM_BACKEND=codex
```

- **Install:** `npm install -g @openai/codex` (or follow the
  [codex CLI README](https://github.com/openai/codex)), then `codex login`.
- **Cost model:** counts against your ChatGPT plan limits. Same caveat as
  above re: subscriptions vs API credit.
- **Latency:** comparable to `claude_code` — single-digit-seconds per batch.
- **Caveats:**
  - The `--color never` flag is required to keep ANSI escapes out of stdout
    (the parser strips them anyway, but the flag is cheap insurance).
  - `--skip-git-repo-check` is set so the summarizer keeps working when run
    from a directory that isn't a git repo.
  - First run prompts for sandbox approval unless you've configured
    `codex` to auto-approve read-only or no-tool runs.

## `extractive` (no LLM)

```env
LLM_BACKEND=extractive
```

- Returns the first 1–2 sentences of each abstract via a regex sentence
  splitter. No network, no subprocess, no external dependency.
- **This is also the failure path** for every other backend on a per-batch
  basis, so you can think of it as the floor of summary quality rather than
  a separate product.

## Choosing

| Backend       | $ per digest | Latency | Needs login? | CI-friendly |
|---------------|--------------|---------|--------------|-------------|
| `api`         | <$0.001      | ~0.3 s  | API key      | Yes         |
| `claude_code` | sub usage    | ~5–15 s | OAuth        | Hard        |
| `codex`       | sub usage    | ~5–15 s | OAuth        | Hard        |
| `extractive`  | $0           | <1 ms   | No           | Yes         |

For automated GitHub Actions runs the only practical choices are `api` (cheap
key-based) or `extractive` (free). The CLI backends are best when you run
the digest manually from your laptop and want to avoid an LLM bill.

## Recommended models for each backend

| Backend | Cheap & fast (default rec) | Balanced | Premium |
|---|---|---|---|
| `api` (OpenAI-compat via NanoGPT, OpenAI, etc.) | gpt-4o-mini, claude-haiku-4-5 | gpt-4o, claude-sonnet-4-6 | gpt-5, claude-opus-4-7 |
| `claude_code` | claude-haiku-4-5-20251001 | claude-sonnet-4-6 | claude-opus-4-7 (default) |
| `codex` | gpt-5-mini | gpt-5-codex (default) | gpt-5 |

For a daily digest summarizing ~20 abstracts, **cheap & fast is plenty** — you're asking for one-sentence summaries of well-formed scientific abstracts. Premium models burn subscription quota for negligible gain.

Set via env: `LLM_CLI_MODEL=claude-haiku-4-5-20251001`. Leave empty to inherit your local CLI's default (which on Pro/Max plans is currently Opus 4.7 — overkill).
