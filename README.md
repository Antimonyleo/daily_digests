# DailyDigest

DailyDigest is a local-first research and news reader for people who want a
small, personalized daily brief rather than another noisy feed. It collects
research, preprints, industry, regulatory, and world-news sources; ranks them
against your interests; and lets your feedback improve later digests.

Your profile, reading history, votes, and embedding cache stay in `data/` on
your machine. The app has no account system and listens only on localhost.

## What it does

- Guides each new reader through choosing 1–10 research interests and relative
  weights. Weights are preferences, not quotas: a high-quality paper from a
  lower-weight topic can still appear.
- Ranks candidates by topic fit, source quality, freshness, novelty, and
  duplicate/promotion signals. Weak matches do not bypass the relevance gate.
- Shows a web digest with concise summaries and feedback controls (`Must read`,
  `Relevant`, `Hmmm`, and `Not for me`).
- Uses free extractive summaries by default. An OpenAI-compatible API is an
  optional upgrade for richer summaries.
- Keeps data local in SQLite, supports a dry-run HTML digest, and can send
  email through Resend when configured.

## Install and start

Choose one option. The first brew downloads the default embedding model and can
take a few minutes; later runs reuse the local cache.

### Docker Desktop (macOS, Windows, or Linux)

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/), then
from a clone of this repository run:

```bash
docker compose up --build
```

Open <http://127.0.0.1:8765>. The Compose configuration deliberately exposes
the service only to your computer.

### Linux, macOS, or Windows through WSL2

DailyDigest requires Python 3.12 or newer and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone <repository-url>
cd dailydigest
uv sync
./scripts/start.sh
```

If you prefer a one-command installer, run `bash scripts/install.sh`. It
installs `uv` when missing, then starts the application. Native Windows shells
are not supported; use Docker Desktop or WSL2.

To run without automatically opening a browser:

```bash
NO_BROWSER=1 ./scripts/start.sh
```

## First use

1. Open <http://127.0.0.1:8765>.
2. Enter up to ten specific research interests, one per line, with a relative
   weight:

   ```text
   colloidal self-assembly | 18
   DNA nanotechnology | 15
   gene and RNA therapeutics | 10
   ```

   Higher weights gently break ties among already-qualified papers. They do not
   mean that a topic receives that percentage of the digest.
3. Choose **Extractive** for a free, no-key first run, then brew the digest.
4. Vote as you read. Positive votes are especially useful: they give the local
   ranker evidence about what matters within your stated interests.

Use the in-app settings page to update interests or weights. For a completely
new reader, stop the app and move the whole `data/` directory—not just the
profile—so votes and reading history are not shared:

```bash
mv data data.previous
```

## Optional configuration

Copy `.env.example` to `.env` only when you need non-default settings. The web
setup page can also write the basic settings for a local installation. For
Docker, edit the host `.env` and restart the container after changing settings.

```bash
cp .env.example .env
```

- `LLM_BACKEND=extractive` is the default and needs no key.
- For an OpenAI-compatible endpoint, set `LLM_BACKEND=api`, `LLM_BASE_URL`,
  `LLM_API_KEY`, and `LLM_MODEL`.
- For email delivery, configure `RESEND_API_KEY`, `DIGEST_FROM`, and
  `DIGEST_TO`; see [domain setup](docs/domain-setup.md).

Useful commands:

```bash
uv run dd run-all --dry-run  # generate a local HTML preview, never send email
uv run dd run-all            # send when email settings are configured
uv run dd vote --train       # retrain after collecting feedback
uv run dd eval               # inspect ranking metrics from recorded votes
```

## Testing an installation

The automated test suite checks application behavior:

```bash
uv sync --group dev
uv run pytest
```

Before each release, also test a **fresh clone** with no `.env` or `data/`:

1. Run `uv sync --frozen` and `NO_BROWSER=1 ./scripts/start.sh`.
2. Confirm `/healthz` returns `{"status":"ok"}` and `/` redirects to `/setup`.
3. Complete setup with extractive summaries and brew a dry run.
4. Repeat with `docker compose up --build` and an empty `data/` directory.
5. Check both a normal local user and a user behind a restrictive network or
   proxy. Ingest and first-model download need network access; the local UI and
   already-cached model do not.

Likely environment-specific issues are a port-8765 conflict, insufficient disk
space for dependencies and the embedding model, Docker bind-mount permissions
on `data/`, and a firewall/proxy blocking source feeds or model downloads. The
app handles missing profile data, invalid topic weights, old unsupported
summarizer settings (falls back to extractive), empty candidate pools, and
loopback-only access; it cannot make unavailable external feeds or networks
work. See the workflow in `.github/workflows/test.yml` for the supported
automated baseline.

## License

DailyDigest is released under the [MIT License](LICENSE).
