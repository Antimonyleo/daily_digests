# DailyDigest

DailyDigest is a personalized morning research and news digest. It pulls from journals, preprint servers, biotech industry feeds, regulatory sources, and world news; ranks everything against your profile; summarizes the best items; and gives you a calm local UI for reading and feedback.

It is built for researchers who want signal without opening 40 tabs before breakfast.

**Status:** public-release hardened local app. Verified with `177` passing tests plus a live local web smoke test. See [docs/STATUS.md](docs/STATUS.md).

## Features

- One-command local web app with `./scripts/start.sh`
- First-run setup wizard for name, research profile, keywords, downweights, digest size, and summarizer backend
- Broad source coverage: journals, bioRxiv/medRxiv/arXiv, PubMed/OpenAlex, FDA/ClinicalTrials.gov, biotech news, and world news
- Profile-aware ranking with local embeddings, source-quality boosts, novelty signals, freshness gates, cross-source dedupe, anti-promo penalties, and source-balance caps
- Interest facet weights in `data/profile.yaml` so you can strengthen or soften topics without editing code
- Rank-feature snapshots, top-journal audit, and brew diagnostics so you can see when high-quality candidates were considered, filtered, or deduped
- SQLite embedding cache so repeated brews only embed new or changed items
- Graphical digest overview with scanned/selected counts, confidence-aware must-read cards, source-mix bars, diagnostic drawers, structured summaries, and per-item ranking signals
- Reader filters for priority, unread, published journals, preprints, AI/CS, and digest sections
- `Relevant` / `Seen` / `Not for me` feedback saved instantly, with optional toggleable reason chips like `Low impact`, `Promo`, `Access`, and `Duplicate`
- Optional learned ranking after enough Relevant/Not-for-me votes from the UI or `uv run dd vote --train`
- Extractive summaries by default, with optional OpenAI-compatible API, Claude Code CLI, or Codex CLI backends
- Dry-run HTML previews and optional Resend email delivery
- Local-first safety: loopback-only web server, CSRF-protected write routes, escaped templates, ignored user profile/data

## Quickstart

```bash
git clone <your-repo-url>
cd dailydigest
./scripts/start.sh
```

Open `http://127.0.0.1:8765`. First-time users go to setup, which writes the local profile to `data/profile.yaml`, updates `.env`, and starts a dry-run brew with live progress.

Run without auto-opening a browser:

```bash
NO_BROWSER=1 ./scripts/start.sh
```

DailyDigest expects to be run from a repo checkout. `uv` is required; the startup script will tell you if it is missing.

## First-Run Tutorial

1. Start the app with `./scripts/start.sh`.
2. Fill out Settings with your bio and interest keywords.
3. Choose `Extractive` for the easiest first run. It needs no login or API key.
4. Click `Brew my morning cup of tea`.
5. Read the `Must read first` cards, then scan each section.
6. Mark entries Relevant, Seen, or Not for me. Add a reason chip after Seen or Not for me when something is off.
7. When enough Relevant/Not-for-me feedback is collected, click `Update my ranking`.

Relevant and Not-for-me responses teach future ranking updates. Seen only marks an item reviewed unless you add a reason chip.

## Ranking Quality

DailyDigest balances personal topic fit with editorial quality. Top journals still help as a tie-breaker, but a weakly matched Nature or Science item should not beat a substantially better match from a less prestigious venue. Ranking candidates are freshness-filtered, deduped across sources while keeping the best publisher/high-quality representative, and checked for promotional wording and low-information commentary so stale items, repeated papers, press-release-style posts, and editorials without new methods/results are pushed below independent, substantive coverage.

You can tune source hints in `config/sources.yaml` with `quality_tier`, `prestige_score`, `impact_floor`, and `promo_risk`. Exact impact factors are not fetched during brewing; the app uses stable local tiers to stay fast and reproducible.

You can tune personal facets in `data/profile.yaml`:

```yaml
interest_weights:
  RNA therapeutics: 1.25
  clinical translation: 1.15
  arXiv CS methods: 0.45
```

Values above `1.0` strengthen a topic; values below `1.0` soften it. This is useful when a topic is relevant but should not crowd out published journal papers.

## Summarizer Backends

| Backend | Setup | Best for |
|---|---|---|
| `extractive` | None | First run, offline use, zero cost |
| `api` | `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | OpenAI-compatible hosted models |
| `claude_code` | Local `claude` CLI installed and logged in | Using a Claude subscription locally |
| `codex` | Local `codex` CLI installed and logged in | Using an OpenAI/Codex login locally |

Setup detects whether `claude` or `codex` are on `PATH`. If a CLI is installed but not logged in, the app may only discover that during brew time; failed batches fall back to extractive summaries instead of aborting the digest.

## Common Workflows

Preview a static HTML digest:

```bash
uv sync
uv run dd run-all --dry-run
```

Send a real email digest:

```bash
# Fill RESEND_API_KEY and DIGEST_TO in .env first.
uv run dd run-all
```

Retrain ranking from feedback:

```bash
uv run dd vote --train
```

Run only at the configured local digest hour:

```bash
uv run dd run-all --gate
```

## CLI Reference

```bash
uv run dd ingest                  # Pull all sources into SQLite
uv run dd rank                    # Print top ranked recent items
uv run dd run-all                 # Full pipeline + email send
uv run dd run-all --dry-run       # Render HTML to data/, no email
uv run dd run-all --backfill 7    # Widen recency window
uv run dd vote "+R3 R7 -I5"       # Record label-based feedback
uv run dd vote --train            # Train learned ranker, needs >=30 Relevant/Not-for-me votes
uv run dd prune                   # Delete old items
```

## Project Map

```mermaid
flowchart LR
    accTitle: DailyDigest Pipeline
    accDescr: Sources are ingested into SQLite, ranked against the user profile, summarized, shown in the local UI or email preview, and improved through feedback.

    sources["Sources"] --> ingest["Ingest"]
    ingest --> dedupe["Dedupe + English filter"]
    dedupe --> store["SQLite"]
    profile["data/profile.yaml"] --> rank["Quality-aware ranking"]
    store --> rank
    rank --> summarize["Summarize"]
    summarize --> ui["Local web UI"]
    summarize --> email["Email / HTML preview"]
    ui --> votes["Votes"]
    votes --> ranker["Optional LR retrain"]
    ranker --> rank
```

## Documentation

- [CLAUDE.md](CLAUDE.md) - architecture, conventions, and design notes
- [docs/STATUS.md](docs/STATUS.md) - current state and verification
- [docs/CHANGELOG.md](docs/CHANGELOG.md) - build log and fixes
- [docs/llm-backends.md](docs/llm-backends.md) - backend tradeoffs
- [docs/domain-setup.md](docs/domain-setup.md) - Resend sender domain setup

## Branches

- `main` is the clean public release branch for users.
- `dev` is the active development branch for ongoing work before release.
- Local agent folders such as `.claude/` and `.codex/`, plus `.env`, `data/`, and personal profiles, stay ignored and should not be committed.

## Cost

Typical personal use can stay very cheap:

- Local embeddings and SQLite: free
- Cached item embeddings are small, roughly 1.5 KB per item with the default model, and are pruned with old items
- GitHub Actions cron: free tier
- Resend daily email: free tier
- LLM summaries: optional, often a few dollars/month or zero with extractive mode

## License

Personal project. No license attached.
