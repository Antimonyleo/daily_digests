# DailyDigest

DailyDigest is a private, local research and news recommender. It turns recent
papers and news into a small personalized brief that you can read and rate in
your browser.

**No paid AI subscription or API key is required.** The default setup is free,
runs on your computer, and opens at <http://127.0.0.1:8765>.

## What it does, who it is for, and how it works

DailyDigest is useful for researchers, engineers, students, and other curious
readers who want relevant new work without checking many journals and feeds by
hand.

It works in four steps:

1. Collects recent papers and preprints plus any optional streams you enable:
   funding opportunities, scientific events, industry updates, AI tools,
   clinical/regulatory news, and world news.
2. Compares each item with 1–10 interests that you choose and weight.
3. Builds a short digest using topic fit, source quality, freshness, novelty,
   and duplicate filtering. Optional sections can be switched off, and each
   brew can be served as a full, usual, or minimal reading session.
4. Learns from your `Must read`, `Relevant`, `Hmmm`, and `Not for me` feedback.

Your profiles, votes, reading history, database, and embedding cache stay in the
local `data/` directory. This includes the private opportunity-matching profile
when that feature is enabled. DailyDigest has no user-account system and listens
only on your computer by default. Fetching new publications requires internet access.
The default Extractive summarizer does not send article text or your profile to
an AI provider. Online API and signed-in CLI summarizers send selected article
metadata plus a short reader profile to the provider you choose. A local Ollama
endpoint keeps that processing on your computer.

## Quick start: browser app

### 1. Get DailyDigest

Either download the repository as a ZIP from GitHub and extract it, or clone it:

```bash
git clone https://github.com/Antimonyleo/daily_digests.git
cd daily_digests
```

### 2. Start it

Choose the instructions for your computer. The first launch installs the free
`uv` Python toolchain, installs DailyDigest, and opens your browser.

**Windows 10/11**

Double-click `DailyDigest-Windows.bat`, or run this in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
```

**macOS**

Double-click `scripts/DailyDigest.command`. If macOS blocks it, right-click it,
choose **Open**, and confirm. You can also use Terminal:

```bash
bash scripts/install.sh
```

**Linux**

```bash
bash scripts/install.sh
```

The first start can take several minutes because dependencies and the default
embedding model are downloaded once. Later starts reuse them.

### 3. Complete setup in the browser

The app opens <http://127.0.0.1:8765>. On the setup page:

1. Add 1–10 specific interests, one per line, with relative weights:

   ```text
   colloidal self-assembly | 18
   DNA nanotechnology | 15
   gene and RNA therapeutics | 10
   ```

   Weights are priorities, not quotas. A weight of 10 does not mean exactly 10%
   of the digest.
2. Keep **Extractive** selected for the easiest free setup.
3. Choose whether to include Funding & Opportunities, Events & Calls, Industry,
   AI tools & methods, Clinical & Regulatory, and World news. Research is always
   included.
4. If Funding or Events is enabled, provide your career stage, institution type,
   country, and applicant role. DailyDigest reuses your weighted research
   interests for topical matching; optional preferences cover opportunity type,
   event format and region, and preparation time. Structured fields reject only
   clearly incompatible calls. Unknown eligibility is shown as “needs
   verification,” never as confirmed eligibility.

5. Select **Save settings and choose today’s digest** to save the setup.
6. Choose how much you feel like reading today: **Energetic** keeps every
   qualified pick up to your section limits, **Usual** keeps the best 15, and
   **Tired** keeps up to 5 research picks. Both **Usual** and **Tired** keep the
   best pick from each other enabled section, so no section you switched on
   disappears. This changes only the final serving size; every enabled source is
   still scanned and filtered.

Keep the launcher or terminal window open while using DailyDigest. Press
`Ctrl+C` in that window to stop the server.

Settings records the browser's local timezone so digest dates and schedules
follow the reader on native and Docker installations. The greeting itself is
also calculated in the browser and updates immediately for local time.

## Everyday use

| System | Start DailyDigest |
| --- | --- |
| Windows | Double-click `DailyDigest-Windows.bat`, or run `powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1` |
| macOS | Double-click `scripts/DailyDigest.command`, or run `bash scripts/start.sh` |
| Linux | Run `bash scripts/start.sh` |
| Docker | Run `docker compose up` |

Then open <http://127.0.0.1:8765>. Use **Settings** in the app to change your
interests, weights, digest sections, summarizer, or item counts. Vote on items
as you read; positive and negative feedback both improve later ranking.

The digest starts with an estimated reading time and its main topics. Each
section opens to a compact list of picks; expand an item’s **Details** only when
you want its recommendation explanation and caveat. Use **Save for later** on
any item, then open **Saved reading** to search your personal archive. Saved
items are kept when old unsaved feed items are cleaned up.

Use **Night theme** on any page to switch between light and dark colors. On the
digest page, **Compact view** reduces card spacing when you want to scan more at
once. Both choices are remembered by that browser. The controls appear in the
lower-right corner on larger screens and near the end of the page on smaller
screens. Browsers that support local progressive-web-app installation also show
an **Install app** button. Installation creates an app-like launcher; the local
DailyDigest server still needs to be running because private digest pages are
deliberately not stored in the browser's offline cache.

Funding and event cards show the official source, current status, deadline,
available amount when stated, location/date for events, and a conservative
eligibility assessment. Use **Add to calendar** to download a standard `.ics`
file for the deadline or event. DailyDigest currently verifies US federal calls through
Grants.gov and scientific courses/conferences through EMBL's official event
pages. These sections are discovery aids: always read the linked official call
before spending time on an application.

## Accounts, subscriptions, and optional services

The default browser experience needs no login after the software is downloaded.

| Feature | Account or subscription | Alternative |
| --- | --- | --- |
| Research/news collection and recommendation | None | Included |
| Default extractive summaries | None | Included and recommended for first use |
| Richer AI summaries | OpenAI-compatible API, Anthropic API, or a signed-in Claude Code/Codex CLI | Keep Extractive, or run a local [Ollama](https://docs.ollama.com/api/openai-compatibility) model |
| Email delivery | Optional [Resend](https://resend.com/docs/api-reference/api-keys/create-api-key) account and API key | Read the digest in the browser |
| Installation | Free `uv` toolchain; Git only if cloning | Download the repository ZIP, or use Docker Desktop |

DailyDigest does **not** require Claude, Codex, ChatGPT, or another AI service.
Extractive summaries remain the private, free default. If Claude Code or Codex
is already installed and signed in, Settings can use that local login for
summaries; DailyDigest disables agent tools, uses an isolated temporary working
directory, and falls back to Extractive if the command fails.

If you choose an online model, you need developer API access from that provider:

- OpenAI API billing is separate from ChatGPT subscriptions. Configure an API
  key, base URL, and model in Settings. See the official
  [OpenAI billing explanation](https://help.openai.com/en/articles/8156019-how-can-i-move-my-chatgpt-subscription-to-the-api).
- Anthropic's native API option uses a Console API key and separate API billing.
  Alternatively, the Claude Code option uses a locally installed, signed-in
  `claude` command; Claude Code supports eligible Claude plans as documented in
  its [setup guide](https://code.claude.com/docs/en/getting-started).
- The Codex option uses a locally installed, signed-in `codex` command. Codex
  supports ChatGPT subscription login or API-key login; see the official
  [authentication guide](https://developers.openai.com/codex/auth/).
- Ollama is the simplest no-subscription option for richer local summaries. Use
  base URL `http://localhost:11434/v1`, API key `ollama` (required by the client
  but ignored by Ollama), and the name of a model you already pulled.

Email is optional. To send digests through Resend, add `RESEND_API_KEY`,
`DIGEST_FROM`, and `DIGEST_TO` to `.env`; see
[email domain setup](docs/domain-setup.md). Without email configuration,
DailyDigest keeps the result locally.

## Docker alternative

Docker Desktop provides the same browser app on Windows, macOS, and Linux. From
the repository directory, create the local settings file and start the app.

macOS/Linux:

```bash
cp .env.example .env
docker compose up --build
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Open <http://127.0.0.1:8765>. The Compose configuration maps only localhost and
persists both `data/` and `.env` across container rebuilds.

## Command-line use

The browser is the recommended interface, but the same installation includes a
CLI. Run these commands from the repository directory:

```bash
uv run dd brew                 # brew safely to a local HTML preview
uv run dd brew --backfill 7    # include the last seven days
uv run dd brew --send          # send by email when Resend is configured
uv run dd start                # start the browser app
uv run dd ingest               # fetch sources only
uv run dd rank                 # inspect the current ranking
uv run dd --help               # list every command
```

Local preview files are written under `data/`. The CLI uses the same interests,
votes, database, and settings as the browser app.

## Troubleshooting

- **The site cannot be reached:** keep the launcher window open and confirm it
  says `Uvicorn running on http://127.0.0.1:8765`. Then reopen that address.
- **Port 8765 is already in use:** stop the older DailyDigest process, or run
  `./scripts/start.sh` with a different `PORT` on macOS/Linux. On Windows use
  `.\scripts\start.ps1 -Port 8766`.
- **The first brew is slow:** the embedding model and article data are being
  downloaded. Later runs use the cache.
- **Few or no items appear:** check your internet connection, broaden the date
  window, and review source health in the app. Some publishers occasionally
  block or delay their feeds.
- **No funding or events appear:** confirm the section is enabled and complete
  its profile in Settings. A zero-result section means no official record passed
  status, deadline, eligibility, and topic checks; DailyDigest does not pad it
  with unrelated or closed calls.
- **An API key does not work:** a chat subscription is not necessarily API
  access. Use Extractive mode while checking the provider's API account.

For a new reader on the same computer, stop DailyDigest and move or back up the
whole `data/` directory so profiles, votes, and history are not mixed.

## Development and license

Run the automated tests with:

```bash
uv sync --frozen --group dev
uv run pytest
```

DailyDigest is released under the [MIT License](LICENSE).
