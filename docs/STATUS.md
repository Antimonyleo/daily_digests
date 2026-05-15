# DailyDigest — Project Status

**Last updated:** 2026-05-12
**State:** v1.9-dev — graphical local digest UI, source/novelty ranking explanations, qualitative feedback reasons, and web-triggered ranking updates.
**End-to-end verified:** `NO_BROWSER=1 ./scripts/start.sh` syncs deps and boots FastAPI at `http://127.0.0.1:8765`. `GET /` returns 200 with must-read cards, source-mix bars, per-item ranking bars, and the updated personalization copy. **77 sources** (65 research incl. Nature/Science/Cell families, ACS/RSC/Wiley flagships, NAR). `133` pytest tests pass locally.

---

## TL;DR for picking this up

```bash
git clone <your-repo-url>
cd dailydigest
./scripts/start.sh
```

Open `http://127.0.0.1:8765`. First-time users are sent to `/setup`; the wizard writes the local ignored profile at `data/profile.yaml`, then brews with live progress. If you prefer no browser auto-open, use `NO_BROWSER=1 ./scripts/start.sh`.

---

## What works (verified live)

| Stage | Command | Status |
|---|---|---|
| Ingest | `uv run dd ingest` | ✅ 929 items from 26 active sources in ~37s |
| Rank | `uv run dd rank` | ✅ prints top-20 with freshness-filtered, deduped, topic + source-quality adjusted scores |
| Full pipeline | `uv run dd run-all --dry-run` | ✅ writes HTML to `data/digest-*.html` |
| Vote | `uv run dd vote "+R1 R2 -W1"` | ✅ resolves labels, writes to votes table |
| LR retrain | `uv run dd vote --train` | ✅ (skips if <30 votes) |
| Prune | `uv run dd prune` | ✅ deletes items >30d old |
| TZ gate | `uv run dd run-all --gate` | ✅ skips unless local hour == DIGEST_HOUR |
| Backfill | `uv run dd run-all --backfill 7` | ✅ widens recency window |
| Inbound replies | `uv run dd ingest-replies` | ✅ skip-path verified; IMAP path needs Gmail App Password to test |
| OPML export | `python -m dailydigest.freshrss_export` | ✅ writes `data/sources.opml` (20 RSS feeds) |
| GUI onboarding | `./scripts/start.sh` or `uv run dd start` | ✅ Boots loopback-only server + opens browser; routes users without `data/profile.yaml` to `/setup`, returning users to digest |
| Setup wizard | `GET /setup` | ✅ Bio + keywords + downweight + LLM backend + model picker (200 OK, 12 KB) |
| Brewing progress | `GET /run` | ✅ SSE stream from pipeline.run_all with stages: ingest → dedupe → rank → summarize → done |
| Done page | `GET /done` | ✅ "Your morning cup of tea is brewed" + "Open digest" → opens `/` |
| Local digest viewer | `GET /` | ✅ FastAPI app at 127.0.0.1:8765, graphical overview, must-read cards, source mix, ranking bars, Good / Neutral / Bad votes, reason chips; redirects to /setup if no profile |
| Claude Code backend | `LLM_BACKEND=claude_code uv run dd run-all` | ✅ `claude --print` (+ optional `--model <id>` via `LLM_CLI_MODEL`) |
| Codex backend | `LLM_BACKEND=codex uv run dd run-all` | ✅ `codex exec` (+ optional `--model`) |
| Tests | `.venv/bin/python -m pytest -q` | ✅ 133 passed |

Live verification artifacts:
- `data/digest-20260504-081032.html` — 28,224 bytes, all 4 sections (R×8, I×6, G×3, W×3) with emoji headers, inline CSS, vote-syntax footer.
- DB: `data/digest.db` ~929 items.

---

## Architecture (one-screen recap)

```
config/sources.yaml ─► ingest/* ─► dedupe ─► EN filter ─► upsert
                                                            │
                                            recent_items(2d)│
                                                            ▼
data/profile.yaml ─► embed (bge-small) ─► freshness + dedupe + cosine + source quality + (LR if trained)
                                                            │
                                                            ▼
                                            pick_top_per_section
                                            (R8 / I6 / G3 / W3)
                                                            │
                                                            ▼
                                            summarize (LLM or extractive)
                                                            │
                                                            ▼
                                            jinja2 → Resend email
```

- **Storage:** SQLite at `data/digest.db`. Tables: `items` `votes` `digests` `runs`. 30-day retention on items.
- **Ranker:** cosine vs profile vector by default; hybrid `0.5*cosine + 0.5*lr_prob` once ≥30 votes train the LR. Source quality is now a tie-breaker rather than the dominant force; novelty, access friction, promotional-risk, and low-information commentary filters are exposed in the web UI. Item embeddings are cached in SQLite and reused across brews until title/abstract text changes.
- **Summarizer:** OpenAI-compatible `/chat/completions`. Returns extractive (first 2 sentences) when no key is set. Never raises — failures fall back to extractive.
- **Email:** Resend. Sandbox `onboarding@resend.dev` works without domain. Dry-run writes to disk. Email and web templates force escaping for feed-controlled content.
- **Local UI safety:** write routes require a per-process CSRF token, reject foreign origins, and the CLI rejects non-loopback web binds.
- **Hosting:** GitHub Actions hourly cron, gates on user TZ; DB persisted across runs via workflow artifact.

---

## Phases — what landed

| Phase | Scope | Files |
|---|---|---|
| 0 | Foundation | `pyproject.toml`, `.env.example`, `.gitignore`, `config/sources.yaml`, `config/profile.example.yaml`, `src/dailydigest/{__init__,models,config,store}.py` |
| 1 | Ingest skeleton | `src/dailydigest/{dedupe.py, ingest/{__init__,base,rss,biorxiv}.py}` |
| 2 | Rank+summarize+email | `src/dailydigest/{rank/{__init__,embed,profile,ranker}.py, summarize.py, email_render.py, email_send.py, pipeline.py, cli.py}`, `templates/digest.html.j2` |
| 3 | Source coverage | `src/dailydigest/ingest/{arxiv,openalex,pubmed,fda,clinicaltrials}.py`, expanded `config/sources.yaml` |
| 4 | Voting + LR | `src/dailydigest/votes.py`, extended `rank/ranker.py` and `cli.py` |
| 5 | Automation | `.github/workflows/digest.yml`, `.github/workflows/prune.yml`, `src/dailydigest/prune.py` |
| 6 | Polish | tenacity retries on all adapters, `src/dailydigest/health.py`, backfill flag, health footer in template |
| 7a | Inbound replies | `src/dailydigest/inbound.py`, `dd ingest-replies` command, Reply-To header, `.github/workflows/ingest-replies.yml` |
| 7b | FreshRSS sidecar | `infra/{docker-compose.yml, Caddyfile, .env.example}`, `src/dailydigest/freshrss_export.py`, `docs/freshrss-setup.md` |
| 7c | Domain + tests | `docs/domain-setup.md`, `tests/` (68 tests, 5 files), `.github/workflows/test.yml`, pytest dev group |

---

## Configuration cheat sheet

### `data/profile.yaml` (per-user, ignored)
```yaml
name: ""
bio: |
  3–6 sentences describing research interests.
keywords: [list of 10–30 topics]
downweight: [list of penalty keywords]
```

### `.env` / GH secrets
```
PROFILE_PATH=data/profile.yaml
SOURCES_PATH=config/sources.yaml
LLM_BASE_URL=https://api.openai.com/v1   # or NanoGPT / Ollama / vLLM
LLM_API_KEY=                              # empty → extractive fallback
LLM_MODEL=gpt-4o-mini
RESEND_API_KEY=
DIGEST_FROM=onboarding@resend.dev
DIGEST_TO=you@example.com
USER_TZ=America/New_York
DIGEST_HOUR=8
DB_PATH=data/digest.db
TOP_RESEARCH=12
TOP_INDUSTRY=6
TOP_REGULATORY=3
TOP_WORLD=3
RETENTION_DAYS=30
```

### Sources (`config/sources.yaml`)
Currently 26 active feeds across 4 sections. Disabled in code (commented out): openFDA Drugs@FDA submissions (500s), EMA News RSS (404), Reuters World (returns 0 items). Replace EMA / openFDA URLs and uncomment to re-enable when working endpoints are found.

---

## Known issues / debt

| ID | Severity | File | Note |
|---|---|---|---|
| 1 | low | `config/sources.yaml` | EMA RSS still 404s on all known paths (commented out, TODO); openFDA reactivated with `submission_status:AP` (returns ~50 items live); Reuters World replaced with Al Jazeera. |
| 2 | low | `cli.py` | `--backfill 0` means "use default 2"; non-obvious. |
| 3 | low | `summarize.py` | LLM summary quality is untested with real keys (extractive fallback verified). |
| 4 | low | `ingest/pubmed.py` | `polite_email` is `noreply@example.com` — replace with real address for higher NCBI rate limits. |
| 5 | low | `email_send.py` | Resend sandbox sender often lands in spam; see `docs/domain-setup.md` for the fix. |
| 6 | none-blocking | DB persistence on GH Actions | Uses workflow artifacts; on first run download fails silently and starts fresh. Acceptable. |
| 7 | low | `votes.py` | Existing `data/digest.db` from before Wave 4 may have duplicate vote rows; new UNIQUE constraint won't be applied by `create_all`. Drop the votes table once if you need a clean slate. |

No blocker bugs.

---

## Quickstart for a fresh shell

```bash
git clone <your-repo-url>
cd dailydigest

# 1. Inspect what state already exists
git status
ls data/                  # ← DB and any prior dry-run HTMLs

# 2. Ensure deps are installed
uv sync

# 3. Configure
cp .env.example .env                                  # if not already
$EDITOR .env

# 4. Smoke test
uv run dd ingest          # should upsert hundreds of items
uv run dd rank            # should print top-20
uv run dd run-all --dry-run

# 5. Browse + vote (recommended)
./scripts/start.sh        # starts http://127.0.0.1:8765
# Open the URL in a browser; click Good / Neutral / Bad next to items.

# Or read as a static file:
xdg-open data/digest-*.html   # Linux
# or `open data/digest-*.html` on macOS
```

### LLM backends (pick one)

```bash
# Default: extractive fallback (no key, first 2 sentences)
LLM_BACKEND=extractive

# OpenAI-compatible API (NanoGPT, OpenAI, Anthropic-via-proxy):
LLM_BACKEND=api
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...

# Use your local Claude Code subscription (after `claude` login):
LLM_BACKEND=claude_code

# Use your local Codex subscription (after `codex login`):
LLM_BACKEND=codex
```

See `docs/llm-backends.md` for trade-offs.

To go live (send actual emails):
```bash
# fill RESEND_API_KEY and DIGEST_TO in .env
uv run dd run-all          # no --dry-run
```

To enable the GH Actions cron:
1. Push to a GitHub repo.
2. Settings → Secrets → add: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `RESEND_API_KEY`, `DIGEST_FROM`, `DIGEST_TO`, `USER_TZ`, `DIGEST_HOUR`, and optionally `PROFILE_PATH`.
3. The hourly workflow self-gates on `USER_TZ == DIGEST_HOUR` and only sends one email per day.

---

## Next-up ideas (not started)

Roughly in priority order if/when work resumes:

1. **Set up custom sender domain** — biggest deliverability win. Follow `docs/domain-setup.md`. ~30 min.
2. **Get a working LLM key** — NanoGPT or OpenAI. With key, `summarize.py` can produce richer two-sentence summaries; extractive mode now prefers informative abstract sentences and adds a short "why read" note. Test with `uv run dd run-all --dry-run` and inspect HTML.
3. **Test on real GH Actions runner** — 4 workflows written but never executed. Push to a private repo and trigger `workflow_dispatch` once for `digest`, `prune`, `ingest-replies`, `test`.
4. **Wire up Gmail App Password and test inbound replies** — set `IMAP_USER` / `IMAP_PASSWORD` / `REPLY_TO_EMAIL`, send yourself a digest, reply with `+R1 -I2`, run `uv run dd ingest-replies`.
5. **Replace dead feeds** — find current URLs for EMA News, openFDA Drugs@FDA, Reuters. Update `config/sources.yaml`.
6. **Train the LR ranker** — needs ~30 votes. Use the digest a few weeks first, then `uv run dd vote --train`.
7. **Stand up FreshRSS sidecar** (optional) — follow `docs/freshrss-setup.md`. Hetzner CX11 + a domain you own. ~$4/mo.
8. **Per-section length budgets in summary prompt** — currently each section caps item *count*; consider summary length too.

---

## Pointers for next session

- **Source of truth for design:** `CLAUDE.md` (architecture, decisions, repo layout).
- **Source of truth for status:** this file.
- **Build chronology:** `docs/CHANGELOG.md`.
- **Tests:** `uv run pytest -q`.

If you find yourself confused about why something is the way it is, the answer is almost always in `CLAUDE.md` § "Key decisions / conventions".
