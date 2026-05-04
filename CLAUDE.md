# DailyDigest

Personalized morning research/news digest. Ingests RSS feeds and APIs across academic journals, preprints, biotech industry, regulatory bodies, and general news; ranks items against a user's interest profile; emails a condensed digest each morning.

**Owner / first user:** leoliuh127@gmail.com (biotech/research domain).
**Designed to be reusable:** profile is config-driven; any user can fork + plug in their own bio/keywords.

---

## Goals

1. Replace the daily chore of browsing many sites/journals.
2. Email-first: condensed digest at 8am local time, with section grouping and links.
3. Personalize from a free-text bio + thumbs feedback.
4. Adaptable: swap profile, keywords, source list per user without code changes.
5. Keep monthly cost ≤ $5 and infra simple enough to maintain solo.

## Non-goals

- Multi-user SaaS. Single profile per deployment; no auth.
- Mobile app. Email is enough.
- Real-time alerts. Daily cadence only.
- Building a recommender from scratch when an embedding + logistic regression baseline works.
- Full-text fetching. Title + abstract only.

---

## Architecture (v1)

```
sources.yaml ──► ingest ──► dedupe ──► rank ──► summarize ──► email
                              │                                  │
                              ▼                                  ▼
                            SQLite                         Resend (sandbox)
                          (30d retention)
```

### Pipeline stages

1. **Ingest** — pull from RSS feeds and APIs into a normalized `Item` schema. Filter to English via `langdetect`.
2. **Dedupe** — by canonical URL and title cosine similarity ≥ 0.92 (papers often hit journal RSS + OpenAlex + bioRxiv).
3. **Rank** — two-stage:
   - Stage A: embed `title + abstract`, cosine vs profile vector → top ~60.
   - Stage B (Phase 4): logistic regression on thumbs history, replacing cosine after first ~30 votes.
4. **Summarize** — top ~20 items grouped by section (Research 8 / Industry 6 / Regulatory 3 / World 3). LLM via OpenAI-compatible API; extractive (first 2 sentences) fallback when no key configured.
5. **Deliver** — Resend HTML email at 08:00 local time. Each item has a stable 2-char ID (`R3`, `I7`).
6. **Feedback** — `dd vote "+R3 R7 -I5"` CLI command updates the profile vector and trains the LR.

### Source categories

Configured in `config/sources.yaml`:
- **Academic journals** (TOC RSS): Nature, Science, Cell, NEJM, Lancet, PNAS, JAMA, eLife, Nature Biotech, Nature Methods, Nature Medicine.
- **Preprints**: bioRxiv, medRxiv, arXiv (q-bio, cs.LG), ChemRxiv.
- **Aggregators**: OpenAlex, PubMed E-utilities.
- **Industry**: Endpoints News, FierceBiotech, STAT, BioPharma Dive.
- **Regulatory**: FDA Press Announcements, FDA Drug Approvals, EMA news, ClinicalTrials.gov.
- **General/tech news**: Reuters, AP, BBC World, MIT Tech Review, Hacker News.

### Data store

SQLite at `data/digest.db`. Tables: `items`, `votes`, `profile`, `digests`, `runs`. 30-day item retention; votes preserved indefinitely.

### Models

- **Embeddings**: `BAAI/bge-small-en-v1.5` via `sentence-transformers`, local CPU. ~130MB.
- **Summaries**: pluggable, OpenAI-compatible. Configured via `LLM_BASE_URL` + `LLM_API_KEY` + `LLM_MODEL` env vars (works with NanoGPT, Ollama, vLLM, OpenAI). Extractive fallback when unset.
- **Ranker**: `sklearn` `LogisticRegression` after ~30 votes; cosine baseline before that.

### Hosting

- **GitHub Actions** scheduled workflow, runs hourly. Self-gates on `now.tz(USER_TZ).hour == 8` to send only at user-local 8am.
- Secrets via GH Actions secrets.
- No VPS. No public endpoint. No FreshRSS sidecar in v1.

### Sender

Resend sandbox `onboarding@resend.dev` initially (deliverability is mediocre — expect spam folder). Custom domain is a one-env-var swap later.

---

## Repo layout

```
dailydigest/
├── CLAUDE.md
├── README.md
├── pyproject.toml          # uv-managed
├── .env.example
├── .gitignore
├── config/
│   ├── sources.yaml        # feed list, grouped by category
│   └── profile.example.yaml
├── src/dailydigest/
│   ├── __init__.py
│   ├── cli.py              # typer: ingest, rank, send, run-all, vote, prune
│   ├── config.py           # settings + YAML loaders
│   ├── models.py           # Pydantic Item / Profile / Digest
│   ├── store.py            # SQLAlchemy schema + repo
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── base.py         # Source protocol
│   │   ├── rss.py
│   │   └── biorxiv.py
│   ├── dedupe.py
│   ├── rank/
│   │   ├── __init__.py
│   │   ├── embed.py
│   │   ├── profile.py
│   │   └── ranker.py
│   ├── summarize.py        # OpenAI-compatible client + extractive fallback
│   ├── email_render.py     # Jinja
│   ├── email_send.py       # Resend
│   ├── pipeline.py         # run-all orchestration
│   └── prune.py            # 30-day retention
├── templates/
│   └── digest.html.j2
├── .github/workflows/
│   └── digest.yml          # hourly cron, self-gates on user TZ
├── tests/
└── data/                   # gitignored: digest.db
```

---

## Build phases

### Phase 0 — Foundation
- `uv init`, deps locked.
- `config/sources.yaml`, `config/profile.example.yaml`, `.env.example`.
- SQLite schema + Pydantic models.

### Phase 1 — Ingest
- RSS adapter; bioRxiv JSON adapter as second source-type proof.
- `dd ingest` CLI; URL-canonicalization dedupe; langdetect EN filter.

### Phase 2 — Rank + summarize + email
- bge-small-en-v1.5 embeddings; profile vector from bio + keywords.
- Cosine ranker, per-section caps.
- LLM summary (OpenAI-compatible); extractive fallback.
- Jinja HTML; stable per-digest item IDs.
- Resend send; `dd run-all` end-to-end.

### Phase 3 — Coverage expansion
- Adapters: arXiv, OpenAlex, PubMed, FDA, ClinicalTrials.gov, EMA.
- Title-similarity dedup across sources.

### Phase 4 — Voting + learning
- `dd vote "+R3 R7 -I5"` CLI resolves IDs against last digest, writes votes.
- Weekly LR retrain; replaces cosine after 30 votes.

### Phase 5 — Automation
- `.github/workflows/digest.yml` hourly cron, self-gates on user TZ.
- Nightly prune (30-day retention).
- Failure → repo issue notification.

### Phase 6 — Polish
- Per-source rate limits + `tenacity` retries.
- Backfill flag.
- Per-source health stats appended to email weekly.

### Phase 7 (later, optional)
- Inbound email reply parsing for votes (Resend Inbound or Gmail IMAP).
- FreshRSS sidecar.
- Custom sender domain.

---

## Key decisions / conventions

- **Python 3.12+**, `uv` for env/deps, `ruff` for lint+format.
- **Declarative YAML** for sources and profile — adding sources doesn't require code changes.
- **No LLM in the ranking hot path** — keep it deterministic and cheap. LLM is for summaries only.
- **Idempotent ingest**: `(source, external_id)` unique. Re-runs do not duplicate.
- **Fail loud, partial OK**: per-source failures log and continue; never block the digest on one broken feed.
- **Pluggable LLM** via OpenAI-compatible env config; default extractive when unset.
- **Self-gated cron**: GH Actions runs hourly UTC; the script checks `USER_TZ` and only proceeds at 8am local.

---

## Costs (target)

| Item | Monthly |
|---|---|
| GitHub Actions (well under free minutes) | $0 |
| LLM (small hosted model, ~20 summaries/day) | $0–3 |
| Resend (1/day, well under 3000 free tier) | $0 |
| Embeddings (local, CPU, in GH runner) | $0 |
| **Total** | **$0–3/mo** |

---

## References

- Prior art: `giftedunicorn/ai-news-bot`, `AutoLLM/ArxivDigest`, "Paper Morning" (dev.to).
- Recommender pattern: Scholar Inbox (ACL 2025, arXiv:2504.08385) — embedding + active-learning LR.
- RSSHub journal routes: https://docs.rsshub.app/routes/journal
- OpenAlex: https://docs.openalex.org/
- bioRxiv API: https://api.biorxiv.org/
- FDA RSS: https://www.fda.gov/about-fda/contact-fda/get-email-updates
