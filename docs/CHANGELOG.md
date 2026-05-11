# DailyDigest — Build Log

Chronological log of how the MVP was built. Useful when you want to understand *why* something exists, not just *what* it is.

---

## 2026-05-03 — Scoping

- Surveyed the landscape (Feedly+Leo, Inoreader, Scholar Inbox, Particle, Refind, Readwise Reader, FreshRSS, OSS prior art including `giftedunicorn/ai-news-bot`, `AutoLLM/ArxivDigest`, "Paper Morning").
- Conclusion: no single tool unifies prestigious-journal TOCs + preprints + biotech press + FDA + general news under one personalization profile delivered by email. Hybrid build justified.
- User constraints: no VPS, no LLM key yet, no email domain, want any-user reusability, 8am local, English only, 30-day retention, ultra-simple reply syntax for feedback.

## 2026-05-04 — Foundation through MVP

### Phase 0 — Foundation
- `uv` project, Python 3.12+, deps: feedparser, httpx, sqlalchemy 2.x, pydantic 2.x, sentence-transformers, sklearn, jinja2, resend, typer, pyyaml, langdetect, tzlocal, tenacity.
- SQLite schema (`store.py`): `items`, `votes`, `digests`, `runs` with `(source, external_id)` unique constraint for idempotent ingest.
- `config/sources.yaml` (declarative source list), `config/profile.example.yaml` (bio + keywords + downweight terms).
- `.env.example` with all settings.

### Phase 1 — Ingest skeleton
- `ingest/rss.py` — feedparser-based RSS adapter with URL canonicalization (drop UTM, defrag, strip trailing slash) and HTML stripping.
- `ingest/biorxiv.py` — JSON API adapter for bioRxiv/medRxiv.
- `dedupe.py` — by canonical URL + langdetect English filter.

### Phase 2 — Ranking, summarization, email (parallel agent)
- `rank/embed.py` — singleton bge-small-en-v1.5; L2-normalized embeddings.
- `rank/profile.py` — profile vector = mean(embed(bio), embed(each keyword)).
- `rank/ranker.py` — cosine via dot-product; -0.05 penalty for any downweight term substring; per-section pick.
- `summarize.py` — OpenAI-compatible `/chat/completions` with batch-of-10, JSON response format. Extractive fallback (regex sentence split) on no key or any error. Never raises.
- `email_render.py` — Jinja `FileSystemLoader` rendering 4 sections with emoji headers (🧬💊📋🌍).
- `templates/digest.html.j2` — 640px max-width, inline-CSS, mobile-friendly.
- `email_send.py` — Resend client; dry-run writes `data/digest-<ts>.html`.
- `pipeline.py` — `ingest_all()` and `run_all()` orchestrators.
- `cli.py` — Typer app: `ingest`, `rank`, `send`, `run-all`, `vote`, `prune`. `should_run_now()` for TZ gating.

### Phase 3 — Source coverage (parallel agent)
- arXiv (Atom), OpenAlex (JSON, abstract reconstructed from inverted index), PubMed (E-utilities efetch XML), openFDA (drugsfda), ClinicalTrials.gov (v2 API).
- `dispatch_source` extended with lazy imports for all 7 kinds.
- `SourceSpec` extended with `category`, `query`, `condition`, `endpoint`, `polite_email`.

### Phase 4 — Voting + learning (parallel agent)
- `votes.py` — parses `"+R3 R7 -I5"` (unsigned defaults to +), resolves labels against latest digest, writes `VoteRow`. Returns `{up, down, unknown}` counts.
- `rank/ranker.py` extended with `LRRanker` (sklearn LR, persisted to `data/lr_ranker.npz`); hybrid `0.5*cosine + 0.5*lr_prob` once ≥30 votes; cosine fallback otherwise.
- `cli.py` extended with `dd vote "..."` and `dd vote --train`.

### Phase 5 — Automation (parallel agent)
- `.github/workflows/digest.yml` — hourly cron + workflow_dispatch; uv setup; runs `dd run-all --gate`; DB persisted via `actions/upload-artifact@v4` + `download-artifact@v4`; on failure creates a GH issue.
- `.github/workflows/prune.yml` — daily 03:15 UTC prune.
- `prune.py` — thin wrapper for the CLI.

### Phase 6 — Polish (parallel agent)
- All adapters wrapped with `tenacity` retry: `stop_after_attempt(3) + wait_exponential(2-10s) + retry_if HTTPError|Timeout`. Reraise=False; adapters return `[]` on final failure.
- RSS/arXiv use `httpx.get` first then hand bytes to `feedparser` (so retries actually work).
- `health.py` — per-source `IngestStats` collected per run, persisted to `data/health.json` with 7-day rolling history.
- Pipeline: backfill flag (`run_all(backfill_days=7)`); collects health stats; passes `health_summary` to template.
- Template: optional grey footer table with Items / Failures (red when nonzero); only shown when any source has ≥1 failure in last 7d.

### Verification fixes (parallel agent)
- All `DateTime` columns → `DateTime(timezone=True)` for proper UTC roundtrip.
- `_engine()` cached as module-level `_ENGINE` singleton.
- `config.py` added `get_settings()` with `lru_cache`; legacy `SETTINGS = load_settings()` retained for backwards compat.

### Final end-to-end verification
- `uv sync` succeeds (78 packages incl. torch 2.11).
- `dd ingest` → 903 items / 37s on first run; 929 on second.
- `dd rank` → top-20 prints correctly.
- `dd run-all --dry-run` → 28 KB HTML with all 4 sections.
- `dd vote "+R1 R2 -W1"` → `up=2 down=1 unknown=0`.
- `dd prune` → `pruned 0 items`.

### Bugs found by verification and fixed inline

1. **`templates/digest.html.j2` `section.items` collision** — Jinja resolved `section.items` to dict's `.items()` builtin, not the list at key `'items'`. Fixed by renaming `items` → `entries` in `email_render.py:76` and `digest.html.j2:28`.

2. **Pre-section global truncation starvation** — `pipeline.py` was applying `scored = scored[:60]` before `pick_top_per_section`. With biotech-tilted profile, all 60 were research, so industry/regulatory/world were empty. Fixed by removing the global slice; `pick_top_per_section` already caps at 8+6+3+3 = 20 max so LLM cost is bounded.

3. **CLI help mismatch** — `--backfill` default 0 means "use default of 2" but help text said "default 2". Clarified the help text in `cli.py`.

4. **Dead feeds** — openFDA submissions endpoint 500s, EMA News RSS 404, Reuters World feed returns 0 items. Commented out openFDA + EMA in `sources.yaml`; replaced Reuters with Al Jazeera (works).

---

## Notable decisions

- **No LLM in ranking hot path.** Ranking stays embedding + (eventual) LR. LLM is summary-only.
- **Reply-syntax voting via CLI**, not inbound email parsing — kept Phase 7-out so MVP needs zero public infra.
- **Self-gated cron over precise scheduling.** GH Actions UTC cron runs hourly; the script self-checks user TZ. Simpler than computing UTC offset for DST.
- **Workflow artifacts for DB persistence** — only one user, one digest/day, no concurrency contention. Beats setting up an external DB.
- **Pluggable LLM via OpenAI-compatible env** — works with NanoGPT / Ollama / vLLM / OpenAI without code changes.
- **Extractive fallback always available** — pipeline never blocks on missing API key.

---

## Skipped on purpose (not bugs)

- No type checking in CI. ruff lint configured but mypy/pyright not.

---

## 2026-05-04 (cont.) — Phase 7

Three parallel non-conflicting work streams.

### Phase 7a — Inbound email replies

- `src/dailydigest/inbound.py` — IMAP4_SSL poller (stdlib `imaplib` + `email`, no new deps). Searches for `UNSEEN SINCE <2d> SUBJECT "DailyDigest"` in user inbox.
- Body extraction strips HTML, removes quoted lines (`>`), removes "On ... wrote:" markers, removes `-- ` signatures. Only the user's authored top portion is parsed.
- `extract_vote_line` regex-matches `[+-]?\b[RIGW]\d+\b`; concatenates all lines containing tokens.
- `process_replies()` returns `{messages, votes_up, votes_down, votes_unknown, errors}`. Per-message exceptions logged and continued. Marks each fetched message Seen for idempotency.
- `email_send.py` now sets `Reply-To: $REPLY_TO_EMAIL` on outbound mail.
- `cli.py` adds `dd ingest-replies` (prints summary or "imap not configured" skip).
- `.github/workflows/ingest-replies.yml` — hourly `15 * * * *`, persists DB via `digest-db` artifact.
- `.env.example` adds `REPLY_TO_EMAIL`, `IMAP_HOST`, `IMAP_PORT`, `IMAP_USER`, `IMAP_PASSWORD`, `IMAP_MAILBOX` with App-Password note.

Verified: `dd ingest-replies` prints "imap not configured" when env unset.

### Phase 7b — FreshRSS sidecar (optional, Docker)

- `infra/docker-compose.yml` — FreshRSS + RSSHub + Caddy with healthchecks, named volumes, `restart: unless-stopped`. Internal network between services; Caddy only exposes 80/443.
- `infra/Caddyfile` — auto-HTTPS via Let's Encrypt for `${FRESHRSS_DOMAIN}` and `${RSSHUB_DOMAIN}`. Basic-auth on RSSHub.
- `infra/.env.example` — `FRESHRSS_DOMAIN`, `RSSHUB_DOMAIN`, `ACME_EMAIL`, `TZ`, `CRON_MIN`, RSSHub basic-auth user+hash.
- `src/dailydigest/freshrss_export.py` — reads `config/sources.yaml`, emits OPML 2.0 grouped by section. Skips non-RSS kinds (arxiv, biorxiv, openalex, pubmed, fda_api, clinicaltrials) since FreshRSS can't ingest them. Runnable via `python -m dailydigest.freshrss_export`.
- `docs/freshrss-setup.md` — step-by-step user guide (env, Caddy hash, compose up, OPML import, troubleshooting).

Decision: OPML import beats trying to push items into FreshRSS via Greader/Fever API — those APIs only manage subscriptions, not insert items. FreshRSS pulls the same RSS feeds we do, so OPML round-trips the user's source list with zero API complexity. Verified: 20 RSS feeds emitted to `data/sources.opml`.

### Phase 7c — Domain setup docs + pytest suite

- `docs/domain-setup.md` — Resend custom domain DKIM/SPF/DMARC walkthrough.
- `tests/` — 5 files, **68 tests, all passing in 0.43s**:
  - `test_votes_parse.py` (15) — `parse_vote_line` edge cases. Locked in observed behavior: `+R3 +R3` does NOT dedupe (returns both).
  - `test_dedupe.py` (16) — `canonicalize_url` UTM stripping, fragment removal, trailing-slash; `dedupe_by_url` order preservation. Locked in observed behavior: scheme/host case is preserved (not lowercased).
  - `test_ranker.py` (17) — score_items downweight penalty, pick_top_per_section caps, with `embed_texts` monkeypatched to a hash-based fake (no model load).
  - `test_config.py` (11) — load_sources/load_profile via `tmp_path` fixtures.
  - `test_health.py` (9) — weekly_summary aggregation.
- `tests/conftest.py` — auto-monkeypatches `LLM_API_KEY=""` so summarizer always uses extractive fallback; redirects `DB_PATH` to tmp.
- `pyproject.toml` — added `[dependency-groups] dev = ["pytest>=8.0", "pytest-cov>=5.0"]`.
- `.github/workflows/test.yml` — runs `uv run pytest -q` on push/PR.

Behavioral divergences from the brief that locked in observed-actual reality:
1. `parse_vote_line` does not dedupe duplicate labels.
2. `canonicalize_url` does not lowercase scheme/host.
3. `pick_top_per_section(scored, caps)` not `(caps, scored)`.
4. `score_items(items, profile_vec, downweight_terms)` is positional.

Tests assert these and flag them with comments.

---

## 2026-05-04 (cont.) — Wave 4: honest-review fixes

A 5-agent review team (goal-fit / code-correctness / security / runtime-stress / architecture) audited the v1 build and surfaced 8 actionable issues. Five parallel surgical-fix agents addressed them.

### Critical bugs fixed

1. **IMAP search query was dead-on-arrival** (`inbound.py`) — search criteria embedded literal quotes inside the subject term (`'"DailyDigest"'`), which IMAP servers interpreted as a literal-string match including the quotes → zero matches forever. Fixed by passing `cfg.subject_filter` as a bare token; `imaplib` handles quoting itself.

2. **Double-send race** (`pipeline.py`) — `run_all` never checked `DigestRow.sent_at` before sending. Two cron firings in the gate hour, or any retry, would send twice. Fixed by adding an early-exit check: if `sent_at IS NOT NULL` and not dry-run, log and return immediately.

3. **`digest_id` used UTC date instead of user-local TZ** (`pipeline.py`) — for Tokyo at 8am local (UTC-1), `digest_id` would be the previous calendar day. Fixed by anchoring `_digest_id()` to `ZoneInfo(SETTINGS.user_tz)`.

4. **`write_digest` clobbered `sent_at` to NULL on re-run** (`pipeline.py`) — `session.merge(DigestRow(id=..., item_count=...))` overwrites unset attributes to NULL, including `sent_at`. Added `_should_write_digest()` helper that skips the merge when the row is already `sent_at != None`.

### Security fix

5. **Sender authentication missing** (`inbound.py`) — `process_replies` accepted votes from any sender. Anyone who emailed the IMAP inbox with the right subject could poison the LR training set. Fixed by parsing `From:` with `email.utils.parseaddr` and matching against `SETTINGS.reply_to_email` (case-insensitive). Mismatches are skipped + logged at WARNING; counted as `skipped_unauthorized` in the return summary. When `REPLY_TO_EMAIL` unset, a one-time startup warning is logged and processing proceeds (back-compat).

### Goal gap closed

6. **TZ auto-detect** (`config.py`) — User asked "detect local timezone"; v1 only read `USER_TZ` env (defaulting to UTC). Added `tzlocal.get_localzone_name()` fallback chain in `load_settings()`: explicit env → tzlocal → UTC. Verified: detected `America/Phoenix` on the dev machine without any env config.

### Operational improvements

7. **Embedder offline cache** (`rank/embed.py`) — `SentenceTransformer` was making ~25 HuggingFace HEAD requests per CLI invocation even with primed cache. Tries `local_files_only=True` first; falls back to network on cache miss. Confirmed zero network calls on cached runs.

8. **Vote idempotency** (`votes.py` + `store.py`) — Multiple votes on the same item used to stack as duplicate rows. Added `UniqueConstraint("item_id")` on `VoteRow`. `record_votes` now upserts (query → update existing or insert new). Logs INFO on sign flips (`+1 -> -1`). `vote_dataset()` naturally yields ≤1 row per item, no double-counting in LR training.

9. **defusedxml** (`pubmed.py` + `pyproject.toml`) — Replaced `xml.etree.ElementTree.fromstring` on remote NCBI XML with `defusedxml.ElementTree.fromstring` to harden against billion-laughs / quadratic-blowup entity bombs.

### Source coverage improvements

10. **PubMed query** — old query returned 0 items. Simplified to `clinical trial[pt] AND 2024:2026[dp]`; live fetch now returns ~23 items in a 2-day window.

11. **openFDA reactivated** — added `query: submissions.submission_status:AP` to the YAML entry; FDASource consumes `spec.query` as the openFDA `search` clause. Live fetch now returns ~50 items per run.

12. **EMA News** — all four candidate URLs returned 404 or 429 (anti-bot). Left commented with the most plausible candidate (`/en/news-events/news/rss`) and a TODO listing what was tested. Will need manual re-discovery later.

### Verification

`uv run pytest -q` → 68/68 passing. `dd run-all --dry-run` produces a 28 KB HTML with all 4 sections. Idempotent re-run path verified: when `sent_at` was manually set, the second `dd run-all` correctly logged `"digest 2026-05-04 already sent at ...; skipping resend"` and exited without sending.

### Known issues remaining (low-severity, deferred)

- Test coverage stuck at ~29% — pipeline / cli / email / inbound / summarize / ingest adapters / web are covered only by smoke tests, not unit tests.
- LR/cosine 0.5/0.5 blend is calibration-arbitrary; ranker review noted it should start at 0.85/0.15 and shift as votes accumulate. Will fix when there are real votes to evaluate against.
- Architecture overbuild (Source Protocol, prune.py ceremony, two settings accessors, FreshRSS sidecar in same repo) is acknowledged but not blocking.
- Empty sections silently omitted from email — won't notice ingest gaps unless health footer triggers.

---

## 2026-05-04 (cont.) — Wave 5: pluggable LLM transports + local web UI

User wanted to use their existing Claude Code / Codex subscription instead of paying per-token API. Also wanted to validate content quality interactively before investing in email polish.

### Subprocess LLM backends (`summarize.py`, `config.py`)

- New env var `LLM_BACKEND` selects transport: `api` (default, OpenAI-compatible HTTP), `claude_code` (subprocess to `claude --print`), `codex` (subprocess to `codex exec --color never --skip-git-repo-check`), or `extractive` (no LLM).
- `_summarize_via_cli(items, cli_cmd)` is a generic subprocess shim: same batched prompt as the API path, fed via stdin, output parsed as JSON. Fences and ANSI escape codes stripped defensively (`_extract_json_object`).
- Per-batch fallback: if a CLI invocation fails (timeout, missing binary, malformed JSON), only that batch falls back to extractive — the rest of the digest continues. Whole-digest failure mode preserved.
- Both CLIs require interactive OAuth login first (`claude` once, `codex login`). Won't work on a fresh GitHub Actions runner without provisioned creds — documented in `docs/llm-backends.md`.

### Local web UI (`web.py`, `templates/digest_web.html.j2`, `dd serve`)

- New FastAPI app at `127.0.0.1:8765` (loopback only, no auth).
- Routes: `GET /` (renders today's digest with vote buttons), `POST /vote/{item_id}/{value}` (records vote), `POST /refresh` (kicks off `run_all(dry_run=True)` in background thread), `GET /healthz`.
- Click-to-vote UX: 👍 / 😐 / 👎 in top-right of each item. Active vote pre-highlighted from existing `VoteRow`. Click POSTs to `/vote/{id}/{1|0|-1}`, updates highlighting client-side without page reload. Failure flashes red.
- Refresh button regenerates today's digest in background.
- Vanilla JS only, ~30 lines. No React/jQuery.
- New `votes.record_vote_by_id(item_id, value)` for direct (label-less) voting; `value=0` deletes the vote (user changed mind).
- `dd serve [--host 127.0.0.1] [--port 8765]` starts uvicorn programmatically.
- New deps: `fastapi>=0.110.0`, `uvicorn[standard]>=0.30.0`.

### Verification

- 68/68 pytest passing.
- `dd serve --port 8765` boots in <2s, `GET /healthz` → `200 {"status":"ok"}`, `GET /` → `200, 24,034 bytes` HTML rendered from existing digest.
- All four backend selections importable; `LLM_BACKEND=extractive` falls through correctly when no key set (preserving prior behavior).

---

## 2026-05-04 (cont.) — Wave 6: GUI onboarding + journal expansion

User wanted to minimize CLI for non-technical use and expand journal coverage to a comprehensive prestige set.

### Journal expansion (`config/sources.yaml`)

- **16 → 65 research feeds** (49 added, all URL-verified live with curl).
- Nature family (9): Nanotechnology, Materials, Chemistry, Physics, Photonics, Catalysis, Energy, Sustainability, Communications.
- Nature Reviews (6): Drug Discovery, Chemistry, Materials, Genetics, Cancer, Microbiology.
- Science family (5): Advances, Robotics, Translational Medicine, Immunology, Signaling.
- Cell family (13): Reports, Stem Cell, Metabolism, Host & Microbe, Cancer Cell, Immunity, Neuron, Chemical Biology, Systems, Molecular Cell, Genomics, Med, Chem.
- ACS flagships (7): JACS, ACS Nano, Nano Letters, ACS Catalysis, Chem. of Materials, ACS Central Sci, ACS Chem. Biology.
- RSC (3): Chemical Science, Chem. Soc. Rev., Energy & Env. Sci. — used `feeds.rsc.org/rss/<code>` (the `pubs.rsc.org/en/Content/journalRss` path 404s).
- Wiley (5): Angew. Chem. Int. Ed., Adv. Materials, Adv. Functional Materials, Adv. Science, Small — `/feed/<issn>/most-recent` works unauthenticated.
- Other (1): NAR via Oxford Academic (`site_5127/3091.xml`).
- Total sources: 77 (65 research, 4 industry, 4 regulatory, 4 world).

### GUI onboarding (`web.py`, new templates)

- New routes: `GET /setup`, `POST /setup`, `GET /run`, `POST /run/start`, `GET /run/stream` (SSE), `GET /done`. `GET /` now redirects to `/setup` if `config/profile.yaml` doesn't exist.
- New templates: `_base.html.j2` (shared header/styles), `setup.html.j2` (~210 lines: chip tag inputs, LLM backend radios, model dropdown, CLI-installed badges), `run.html.j2` (~165 lines: CSS-animated steaming cup + SSE client), `done.html.j2` (~30 lines).
- Setup form writes both `config/profile.yaml` (bio/keywords/downweight) and `.env` (LLM_BACKEND, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_CLI_MODEL); preserves any existing `.env` keys not in the form.
- CLI-installed detection: server-side `which claude` / `which codex` at /setup load time; shows "✓ installed" or "✗ not found" badge with install link.
- Existing `digest_web.html.j2` refactored to extend `_base.html.j2` for consistency.

### Pipeline progress hooks (`pipeline.py`)

- Added optional `progress_callback: Callable[[str, dict], None] | None = None` to `run_all` and `ingest_all`.
- Stages emitted in order: `ingest_start` → `ingest_done` → `dedupe_done` → `rank_done` → `summarize_start` → `summarize_done` → `render_done` → `done`. Each stage payload carries counts / sources / digest_id.
- Backwards-compat: default `None` is a no-op; existing callers unaffected.
- Web layer feeds events from this callback into a per-run `asyncio.Queue` consumed by the SSE endpoint.

### `dd start` command (`cli.py`)

- New entry point for non-technical users: `dd start [--host 127.0.0.1] [--port 8765] [--no-browser]`.
- Schedules `webbrowser.open()` 1.5s after uvicorn starts (timer thread).
- Prints "First run? The browser will open the setup wizard. Press Ctrl-C to stop."

### Model pinning (`summarize.py`, `config.py`, `.env.example`)

- New env: `LLM_CLI_MODEL`. When set + `LLM_BACKEND=claude_code` → `claude --print --model <id>`. When set + `LLM_BACKEND=codex` → `codex exec --model <id>`. Empty = inherit CLI default (currently Opus on Claude Pro/Max — overkill).
- Recommended for daily digest: `claude-haiku-4-5-20251001` (fast/cheap) or `gpt-5-mini`. Documented in `docs/llm-backends.md`.

### Verification

- 68/68 pytest passing.
- `dd start --port 8766 --no-browser`: server boots in ~2s.
- `GET /setup` → 200 (12,063 B).
- `GET /` (with profile.yaml) → 200; without profile.yaml → 302 redirect to `/setup`.
- All 14 routes registered (incl. SSE stream, healthz, FastAPI auto-docs).

---

## 2026-05-05 — Wave 7: public-release hardening

Pre-release review focused on shipping the repo publicly rather than keeping it as a local prototype.

### Security / privacy

- Email renderer now forces Jinja autoescape for `.html.j2` templates; web UI uses an explicit autoescaped Jinja environment too.
- Feed-controlled links are sanitized to `http`/`https`; unsafe schemes render as `#`.
- Local write routes (`/setup`, `/profile/name`, `/vote`, `/refresh`, `/run/start`) require a per-process CSRF token and reject foreign `Origin` headers.
- `dd start` / `dd serve` reject non-loopback host binds because the local UI has no login system.
- `.env` writer now rejects CR/LF injection and quotes values with spaces, `#`, or `=`.
- Personal profile and Claude session logs removed from tracked release state; local profiles live under ignored `data/profile.yaml`.

### UX / onboarding

- Added `scripts/start.sh` as the public quickstart entry point.
- Removed Compact mode from the digest UI.
- Vote explainer now uses clearer copy: Good/Bad affect ranking after retraining; Neutral only marks reviewed items.
- Empty brewed digests now say no matching items were found instead of implying nothing was brewed.
- Run progress exposes a real progressbar, live status text, and alert-style error reporting.
- Setup chips are keyboard-removable buttons instead of mouse-only spans.

### Correctness

- Dry-run reruns refresh the local preview even if that digest was previously sent, while preserving `sent_at`.
- `send_digest()` returns success/failure; the pipeline only marks a digest sent after a successful real email send.
- Web summaries persist to the DB after summarization.
- Digest reruns clear stale item assignments for the same digest id.
- LR ranker weight persistence fixed and cache rechecks when weights appear after server start.

### Verification

- `uv run pytest -q` → 92/92 passing.
- Startup script smoke-tested with fake `uv` and with local server boot.

---

## 2026-05-11 — Wave 9: quality-aware ranking

The digest now ranks for "worth reading" instead of only "textually similar to my profile."

### Ranking quality

- Added local source-quality metadata and heuristics for journal/source reputation.
- Research ranking now blends topic fit, source prestige, and novelty/urgency signals.
- Low-prestige research is pushed down unless it is both highly relevant and clearly novel.
- Top journals such as Nature, Science, Cell, NEJM, and The Lancet can surface with moderate topic match.
- Industry and world-news ranking now penalizes promotional wording such as sponsored content, webinars, commercial launches, and press-release language.
- `config/sources.yaml` supports `quality_tier`, `prestige_score`, `impact_floor`, and `promo_risk` ranking hints.

### Verification

- Added tests for top-journal preference, novel low-prestige exceptions, promotional penalties, and source metadata loading.
- `uv run pytest -q` -> 99/99 passing.

---

## 2026-05-06 — Wave 8: lighter repeated local brews

Repeated ranking was re-embedding every recent item, which is expensive on small laptops.

### Embedding cache

- Added `item_embeddings` SQLite table keyed by item id + embedding model.
- Cached vectors store float32 bytes plus a SHA-256 hash of title+abstract text.
- Ranking and LR vote training now call `embed_item_rows()`, which only embeds new or changed items.
- Existing `prune()` removes old cached embeddings through SQLite foreign-key cascade.
- Rows without DB ids still fall back to direct embedding, keeping tests and ad-hoc callers simple.

### Verification

- Added cache tests for reuse, text-change invalidation, and pruning.
- `uv run pytest -q` → 95/95 passing.
