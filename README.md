# DailyDigest

Personalized morning research/news digest. Ingests RSS feeds and APIs across academic journals (Nature, Science, Cell, NEJM, Lancet, ...), preprints (bioRxiv, medRxiv, arXiv), biotech industry press (Endpoints, FierceBiotech, STAT), regulatory bodies (FDA, ClinicalTrials.gov), and general news (BBC, Al Jazeera, MIT Tech Review, Hacker News). Ranks items against your interest profile. Emails a condensed digest each morning.

**Status:** MVP shipped, end-to-end verified in dry-run. See [`docs/STATUS.md`](docs/STATUS.md).

## Quickstart

```bash
uv sync
cp config/profile.example.yaml config/profile.yaml   # then edit
cp .env.example .env                                  # then edit
uv run dd run-all --dry-run                           # writes data/digest-<ts>.html
```

Open the resulting HTML in a browser. To go live, fill `RESEND_API_KEY` and `DIGEST_TO` in `.env` and drop `--dry-run`.

## Commands

```bash
dd ingest                  # pull all sources into SQLite
dd rank                    # print top 20 ranked recent items
dd run-all                 # full pipeline + send email
dd run-all --dry-run       # render HTML to disk, no send
dd run-all --gate          # only run if local hour == DIGEST_HOUR
dd run-all --backfill 7    # widen recency window to 7 days
dd vote "+R3 R7 -I5"       # thumbs feedback on the most recent digest
dd vote --train            # retrain the LR ranker (needs ≥30 votes)
dd prune                   # delete items older than RETENTION_DAYS
```

## Documentation

- [`CLAUDE.md`](CLAUDE.md) — full architecture, design decisions, conventions.
- [`docs/STATUS.md`](docs/STATUS.md) — what's done, what's broken, how to resume work.
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — chronological build log, decisions, bug fixes.

## Cost

Target ≤ $5/month all-in:
- GitHub Actions cron: free
- LLM (small hosted): $0–3
- Resend (1 email/day): free tier
- Embeddings: local CPU, free

## License

Personal project — no license attached.
