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

1. Add 1–10 specific interests, for example:

   ```text
   colloidal self-assembly
   DNA nanotechnology
   gene and RNA therapeutics
   ```

   Each interest has its own topic and percentage field. Shares must total 100%
   before saving; use **Normalize to 100%** or **Distribute evenly** to adjust them.
   Older relative weights are converted to the displayed whole percentages when
   you save. The no-JavaScript fallback still accepts `DNA nanotechnology | 3`.
   Percentages guide priority rather than guarantee digest slots. **Personal details & topics to see less often**
   holds optional bio and downweight preferences.
2. Keep **Extractive** selected for the easiest free setup. Expand the summary
   settings only when you want to change how summaries are made.
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
   disappears. Funding and events are never trimmed by the reading mode — a call
   you skip today may have closed by the time you next pick **Energetic**. This
   changes only the final serving size; every enabled source is still scanned
   and filtered. When qualified picks are trimmed only by the serving size, the
   digest lists them under “more qualified picks didn’t fit today’s reading
   mode” — open the list to read them now, or press **Save all for tomorrow**
   to give them one more ranking pass against the next brew’s fresh papers.

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
items are kept when old unsaved feed items are cleaned up. Use **Time Machine**
at the top of the digest to revisit the three most recent saved brews, including
across days when no digest was made.

**Saved reading** has pages covering your whole archive and lets you rate or
add reasons later. **Clear rating** removes the current preference and its
reasons while keeping the item saved. Under **Unrated**, a newly rated card
stays visible until you press **Done**, so you can finish optional reasons;
unfinished feedback also survives a reload in the same browser tab. **Hmmm**
is a mild negative preference; **Not for me** is a stronger one.

Research versions share a fourteen-day cooldown from their last digest
appearance. An explicit publication link or a distinctive unchanged title
matches a preprint to its journal version; the version-title fallback checks
author agreement. One brew keeps one representative, and a same-day rebrew can upgrade
to the journal version. A journal publication months later remains eligible.
Changed titles without publication links can still escape this check.

If brew status disconnects, use **Reconnect** to observe the same run or
**Read latest saved digest** to keep reading. Starting a new brew is a separate
action. **Stop before publishing** requests cancellation at the next safe
stage boundary; a running fetch or encoder call finishes first. Stage timings
and summary-cache/provider counts are recorded locally. Optional HTTP summaries
have an aggregate scheduling budget and stop after repeated failures, falling
back to local summaries; HTTP inactivity timeouts are not an absolute wall-time
deadline. Settings changes wait until brewing or ranking work finishes.

For explicit local backups and a recovery rehearsal that preserves saved items,
ratings and retained history, see [Local backup and recovery](docs/local-recovery.md).

Funding calls and events stay open for weeks, so DailyDigest treats them as a
standing shortlist and may re-show an active, relevant item while its deadline
remains actionable. When you have seen enough of a call, press **Mark known**
and it will not appear in any future digest. Nothing sets that flag automatically
— press it again to undo.

After each successful brew, DailyDigest removes expired digest history, old
HTML previews, and unreferenced feed items according to `RETENTION_DAYS`
(30 days by default). Voted items, saved items, items marked known, and
anything still referenced by a retained digest are preserved. Embeddings stay on disk rather than in memory;
only the active model is loaded during a compute job and released afterward.

Open **Display** on any page to switch **Night theme** on or off. On the
digest page, **Compact view** condenses the overview, filters, section headers,
recommendation metadata, and feedback controls so substantially more entries fit
on screen while keeping titles readable. **Show Pip** lets you hide or restore
the companion. These choices are remembered by that browser. The Display menu
appears in the lower-right corner on larger screens and near the end of the page
on smaller screens. Press Escape to close it and return focus to its button.
Browsers that support local progressive-web-app installation also show
**Add to desktop** inside Display. This creates an
app-like shortcut and standalone window; it does not install or start the
DailyDigest server. The server still needs to be running because private digest
pages are deliberately not stored in the browser's offline cache.

Pip, the tea-cup companion, serves a rotating daily deck of science facts
and lab jokes. On wide desktops Pip sits in the lower-left margin; on smaller
screens the tea break follows the digest, keeping the recommendations close to
the top. Select the leaf floating above Pip
to drop it into the cup and hear another note, up to ten leaves per daily digest.
After the tenth leaf, Pip calls the tea break finished. On wide screens Pip can
be dragged, moved through preset positions, or adjusted
with the arrow keys; the browser remembers the position. On mouse-equipped
devices Pip also reacts subtly to the pointer. Pip blinks once on arrival and
gives a small hop when you successfully save or rate an item while it is visible.
Its brief animations respect the operating system's reduced-motion preference.
The current note and leaf count are restored together after a refresh.

The **Unrated** filter shows picks that have not received feedback; it does not
track whether you opened their links. Touch devices use larger rating and save
controls, including in Compact view. Keyboard readers can use **Skip to
recommendations** to move past the header.

A slim green-to-gold bar at the top of the window tracks reading position through
the digest itself. It reaches the end when the bottom of the recommendation card
enters the reading area, independent of the setup and diagnostic panels around it.
It updates when you expand sections, filter picks, or change the display density.

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
persists `data/` across container rebuilds. The host `.env` is mounted read-only
as bootstrap configuration. Saving Settings creates overrides in `data/.env`;
these take precedence on later starts without changing or copying the host file.
Keep `data/` when updating the container. To return a saved setting to its host
bootstrap value, remove that override from `data/.env` while the container is
stopped. Native and browser installations continue using their root `.env`.

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

### Checking ranking quality

Normal brewing requires research papers to clear the reader's topic floor
independently of journal prestige. Quality evidence still orders eligible work.
Longer abstracts are retained and profile-driven OpenAlex searches include
preprints. Add aliases and describe useful papers in each interest's search
details to clarify ambiguous terms; no model guarantees literature coverage or
verifies scientific validity.

An experimental local joint topic check is available with `SEMANTIC_MATCHING=true`
but disabled by default: it has not demonstrated an additional selection gain.
It checks up to 48 papers, records partial text coverage, preserves existing
quality/preference scores, and never vetoes an otherwise-related paper. First use
caches about 92 MB of public weights; reader text stays local. `SEMANTIC_MODEL_DIR`
can provision the exact pinned files offline. Normal brewing avoids this download
and computation. The improved relevance gate remains active in either mode.

Every new brew captures the complete local candidate pool, rejected stages,
scores, source classifications, profile, selection settings, source diagnostics,
and final display order in a compressed local run record. Rebrewing the same day
keeps both records. Feedback changes also append timestamped events; older votes
remain available but are not presented as reconstructed historical events.
Run records expire with their digest's retention policy. Profiles and article
metadata in these records are private data, like the rest of `data/`.

```bash
uv run dd eval-ranking --section research --k 11 --json
uv run dd ranking-runs --limit 10
uv run dd replay-run RUN_ID
uv run dd replay-run RUN_ID --outcome-as-of 2026-10-15T00:00:00Z
```

`eval-ranking` uses displayed order and current latest feedback. It reports
judged coverage and precision bounds; unrated papers remain unknown. It cannot
establish how an older algorithm would have performed. `replay-run` independently
runs the real source caps, relevance floor, final-score cutoff, exploration, and
reading-mode rules against frozen inputs. It must match the recorded slate.
This verifies selection; it does not re-fetch sources, recompute embeddings, or
reconstruct earlier eligibility decisions. Feedback metrics require an explicit
outcome cutoff and use only recorded feedback events.

The October 2026 changes correct tied preference fusion, preserve quality
confidence through citation enrichment, distribute OpenAlex's fixed retrieval
budget across core interests before context, and prevent profile-driven PubMed
queries from independently admitting every clinical trial. Diagnostics distinguish
valid empty results, failures, and capped retrieval. These are correctness fixes;
improved personal relevance still needs measurement.

The follow-up quality review also replaced substring journal matching with
explicit identities, preserved known publication-type/retraction exclusions
through ingestion, deduplication and reingestion, and kept quality gates active
on sparse days. Exploration swaps must pass the same selection constraints.
Negative-interest weights scale the excess negative match after comparing it
with topic relevance. Venue prestige remains a source-level heuristic: neither
a prestigious journal nor a passing replay proves that a paper fits your work.
Supplied challenger slates that cannot be reproduced from their declared inputs
are explicitly unverified in evaluation reports.

Source diversity counts exact journal aliases together, including abbreviated
and full International Edition names. New run captures freeze this identity;
older captures without it retain their original feed-name caps during replay.
Configured preprint adapters and explicit preprint metadata cannot receive
published-journal protection. Metadata-only Ahead-of-Print placeholders do not
receive contribution scores. Populated seed papers, interest weights, and facet
weights bind local contribution artifacts to the reader profile; older artifacts
for weighted or seeded profiles require explicit revalidation and re-export.

Citation enrichment preserves article citation boosts, but does not add a second
positive venue bonus to journals already covered by the curated quality prior.
It still supplies venue evidence for previously unknown sources and honors
contradictory low-impact evidence. A DOI enables metadata lookup; its presence
does not establish article quality or research relevance.

When optional enrichment is enabled, raw topic scores bound the lookup batch;
venue evidence is resolved before quality scoring and preference fusion, without
another embedding pass. The later citation adjustment reuses that evidence.
Low-impact evidence already included in quality scoring does not incur a second
negative venue adjustment. Novelty keywords retain a small bonus but cannot waive
the low-impact penalty. Citation velocity requires a known publication date;
missing, future, and RSS update-only dates receive no velocity boost. RSS ingest
preserves explicit DOI fields/links and date provenance. New local run captures
include the venue-impact weight, resolved enrichment inputs, and applied boosts;
older captures retain their original selection replay scope.

The experimental semantic-scope classifier compares specific core-interest
anchors against contextual and off-interest anchors. It can abstain when text
is thin, similarity is weak, or scope classes are too close. Multiple core
facets can match; journal prestige and user usefulness grades are not classifier
inputs. It is **offline only**, with no change to daily serving defaults:

```bash
uv run python scripts/experiment_scope.py --output /tmp/scope-benchmark.json
uv run python scripts/experiment_scope.py --benchmark /tmp/labeled-scope.json \
  --snapshot /tmp/ranking-snapshot.json --cache-db /tmp/digest-copy.db \
  --output /tmp/scope-replay.json
```

The bundled 72-example corpus is fictional, independently authored development
data. Thresholds use only its calibration split; the separate evaluation split
reports abstentions, false core assignments, core retention, and facet accuracy.
Custom corpora use the same schema in `tests/fixtures/research_scope_benchmark.json`.
Keep paper families in one split and reader usefulness separate from scope labels.
Known failure cases can use `evaluation_group: known_regression` so they remain
distinct from newly sampled evaluation examples. `coverage_viable` only tests
recall and coverage constraints; it does not certify precision or justify release.

Optional replay requires an exported complete ranking snapshot and matching
encoder signatures/canonical facet names. The cache is read-only; stale vectors
are re-embedded locally without cache writes. A cached model must be available:
network access is blocked. Reports refuse existing output files and configured
production data/profile directories. Replay filters the full originally eligible
research pool, then reruns the actual source caps, exploration and reading-mode
selector, including replacements and shortfalls. Frozen upstream eligibility and
rank scores are unchanged; this does not test a new retrieval or scoring pipeline.

Initial October 2026 experiments did **not** justify enabling this classifier:
synthetic thresholds rejected valid real papers, while provisional real-abstract
calibration still produced many false core labels. Reports always retain
`promotion_eligible: false`. Further work needs stronger article–interest
discrimination and independent reader-grounded evaluation.

The follow-up pairwise experiment uses an explicit local ONNX cross-encoder
(`dailydigest.rank.cross_encoder.LocalCrossEncoder`) to score interest–article
pairs. It never downloads models or calls a provider. Its raw logits are not
relevance probabilities. A second fixed classifier learns scope classes from
these query scores using standardized logistic regression; ambiguous labels are
excluded, and thresholds use family-disjoint out-of-fold predictions. Facet
labels still come from query matches and require separate evaluation.

```bash
uv run python scripts/experiment_pair_scope.py --benchmark /tmp/labeled-scope.json \
  --scores /tmp/pair-scores.jsonl --manifest /tmp/scoring-manifest.json \
  --snapshot /tmp/ranking-snapshot.json --output /tmp/pair-scope-report.json
```

The runner checks internal consistency of cached model, query, protocol and
article fingerprints before fitting either arm; it does not independently
authenticate model assets or protocol contents. Model assets must be provisioned explicitly for
scoring; the experiment runner consumes previously computed local scores.
Unknown annotations are reported separately from confirmed non-core errors.
Benchmark validation also rejects cross-split long-title aliases, even when
feed dates or abstract versions differ. Previously inspected examples belong in
a separate regression group, not the fresh evaluation headline. These stronger
local alternatives remain experimental: neither is enabled in daily ranking.

The contribution follow-up adds a small local text-evidence signal in
`rank/contribution.py`. It distinguishes stated engineering/method contributions
from natural mechanisms, applications and reviews, and records supporting
abstract passages. It is a heuristic, with neutral results when evidence is
missing. `CONTRIBUTION_ORDER_STRENGTH=0.04` optionally nudges the consideration
order of eligible research; the default is `0`, and relevance/quality scores
and eligibility remain intact. On the inspected October snapshot this heuristic
did not improve the selected slate, so it is not a recommended serving setting.
Local summaries use the same evidence to explain contributions and context.

The learned contribution reranker in `rank/contribution_ranker.py` trains fixed
pairwise comparisons of core, context and off-interest papers from calibration
families. Its score changes final research scores through a fixed 70% contribution
signal / 30% original-score mixture. It retains raw topic gates, source limits
and incomplete-abstract neutrality; the signal is not a probability or a measure
of scientific quality. Calibration folds separate paper families before fitting
scalers and pairs. Only explicitly designated calibration examples fit its
coefficients or score scale. Previously assessed papers may become calibration
data in a subsequent experiment with a separate, unseen evaluation set.

Evaluate existing local logits without a database, provider call or live brew:

```bash
uv run python scripts/experiment_contribution_ranker.py \
  --benchmark /tmp/labeled-scope.json --scores /tmp/raw-pairs.jsonl \
  --manifest /tmp/pair-scoring-manifest.json --snapshot /tmp/day.json \
  --fresh-labels /tmp/blind-labels.json --output /tmp/new-contribution-report.json \
  --artifact /tmp/new-contribution-artifact.json
```

Normal ranking stays unchanged until both `CONTRIBUTION_RERANKER_PATH` and
`CONTRIBUTION_ENCODER_DIR` explicitly identify an evaluated artifact and its local
ONNX assets. `CONTRIBUTION_RERANKER_WEIGHT` defaults to `0.7`; setting it to `0`
disables this path. Serving verifies the reader's interest profile and exact
encoder/runtime identity, scores before within-day duplicate selection, and
persists the model fingerprint and signal with selected-item explanations.
Configured-model failures abort before replacing the digest. No model downloads
or automatic activation occur. CPU inference scores 24 pairs per eligible
research article and can add substantial brew time; the offline runner uses
cached logits. Offline replay preserves captured duplicate eligibility and
topic/coverage bonuses, so it cannot prove full live-pipeline parity.

On the single October 2 captured pool, a separate 56-paper assessment improved
balanced tier-pair ordering accuracy from 59.1% to 74.2% on 52 definitive
families. A separate post-selection scope audit found 9 core, 1 context and
1 uncertain paper versus 5 core, 4 context and 2 off-interest papers at the same
11-paper budget. These are analyst scope assessments, not reader usefulness or
article-quality results; context/off-interest separation and unrepresented
facets remain limitations. The experiment does not enable itself.

A subsequent frozen comparison expanded calibration to 275 definitive families
and assessed 53 newly blinded papers (48 definitive). On the same new holdout,
balanced tier-pair ordering was 67.5% for original scores, 78.2% for the previous
learned artifact, 79.4% for expanded training with the full abstract, and 74.5%
for expanded training with extracted author claims. The expanded model's 1.18
point gain affected only context/off-interest ordering; an exploratory paired
family bootstrap interval was -3.61 to +5.84 points. This does not establish
superiority. Claim-focused input also restored a known off-interest paper to
the replayed slate and is not recommended for ranking. Only one independent
captured day and no completed reader usefulness judgments were available.

`rank/contribution_claims.py` retains exact author statements and their stated
task/object for inspection; introducing a method does not imply core relevance.
`rank/article_evidence.py` separately records quoted validation, data/code
availability, and limitations from supplied text. Selected research shows this
evidence in Details, and summaries can use the same bounded article text.
These are unverified author claims. Unknown evidence remains unknown and does
not change article scores. Full text can be supplied explicitly to the evidence
helper; the digest does not retrieve it automatically. Facet changes also
invalidate cached personalized summaries.

The shared digest evaluator accepts multiple captured days and separate reader
judgments. It measures useful papers at top five and full digest size, observed
important-paper misses, source/topic coverage and repeated families. Analyst
scope annotations remain separate from reader usefulness. It can also produce
a blinded judgment packet with a private mapping back to snapshots:

```bash
uv run python scripts/evaluate_digest_variants.py \
  --snapshot /tmp/day1.json --snapshot /tmp/day2.json \
  --judgments /tmp/judgments.jsonl --challengers /tmp/challengers.json \
  --outcome-as-of 2026-10-05T00:00:00Z --output /tmp/new-comparison.json \
  --packet /tmp/new-blind-packet.jsonl
```

Omit `--judgments` to prepare a packet before rating. Fill its explicit `rater`,
`useful`, `important`, `scope`, `facets`, `contribution`, `evidence` and optional
`rejection_reason` fields; leave unresolved judgments null. To evaluate completed
blinded rows, supply their `.manifest.json` using `--judgment-manifest`. Outputs
must be new files outside configured production data/profile directories.
For a daily audit of the actual retained runs, use the existing SQLite database
read-only. This picks the latest saved brew per observed local day before the
explicit cutoff and uses only timestamped feedback events, including clears:

```bash
uv run python scripts/evaluate_digest_variants.py \
  --database data/digest.db --days 7 \
  --outcome-as-of 2026-10-14T00:00:00Z \
  --output reviews/week1.json --packet reviews/week1-papers.jsonl
```

The command does not brew, migrate, call providers, or read today's mutable votes.
Use the digest ratings as usual: **Must read** supplies an explicit importance
grade, **Relevant** supplies positive usefulness, and **Hmmm** or **Not for me**
supplies a negative preference. Unrated/cleared items stay unknown. For the packet,
review both selected papers and sampled omissions; write completed rows to a
separate JSONL file and use `--judgments` plus `--judgment-manifest` to reevaluate
with the same database and a later outcome cutoff. Omit `--days` on this second
command: the manifest pins the exact original runs even after later rebrews.
Missing or changed snapshots fail the manifest check. Saved feedback and packet
labels are outcomes, not independent
holdout labels when already used to tune the ranker.

The report separates observed important omissions in the eligible pool from
important ineligible records and grade-100 item IDs outside the supplied pools.
Ineligible records show their captured rejection stage for review; an expected
cooldown or another selected version can explain them. Outside-pool IDs are review leads, not
verified retrieval misses: they may be older papers or another publication
version. Papers never collected need explicit reader examples (title/DOI and the
day they should have appeared) before upstream coverage can be assessed. Do not
infer article quality or relevance from venue reputation or unrated items.

arXiv's explicit publication DOI and journal reference are retained as metadata.
The DOI connects renamed preprint/journal versions for deduplication and the
14-day reading cooldown. A journal reference does not promote the preprint's
venue or refresh its original submission date; no DOI is inferred from citations.

Counterfactual declarations contain a slate and, when scores/order change, its
modified snapshot; article/source evidence and original candidate restrictions
must match. Reuse `rerank_snapshot_contributions()` to generate the contribution
ordering experiment. Pure `audit_coverage()` and `audit_selection_snapshot()` helpers
report metadata/stage losses and score distributions from the same snapshots.

Feedback reasons now distinguish contextual or low-quality content from practical
rejections. Already-known, duplicate, timing, access and article-quality reasons
alone do not teach semantic dislike; mixed off-topic/contextual reasons still do.
Duplicate and article-quality reasons affect the specific item rather than its
publisher. Existing incompatible learned profile/LR artifacts are preserved and
ignored until normal feedback/training builds compatible versions; no data
migration is required.

Experimental canonical/additive profile anchors, bounded learned-profile
blending, and alternative fusion weights are available through offline helpers.
They are not new serving defaults. The evaluation sequence is:

1. Verify snapshot selection parity and source coverage on retained runs.
2. Compare a small predeclared set of challengers on identical pools and reading
   budgets. Keep explicit dislikes, quality gates, and opportunity eligibility.
3. Blindly judge the union of baseline and challenger picks plus a sampled set
   of rejected candidates. Track useful papers at actual slate size and top five,
   must-read recovery, strong rejections, topic coverage, and unknown-label bounds.
4. Review paired results by day and reading mode at a predeclared endpoint. A
   provisional target is one additional useful research paper per ordinary digest
   with a positive paired interval and no must-read or strong-rejection regression.
   Twenty days or 150 judgments are collection milestones, not guarantees of power.
5. Promote one challenger only after that evidence, keep its configuration
   reversible, and collect a fresh holdout before tuning again.

Historical rated-item benchmarks are component diagnostics with exposure bias
and current metadata. They cannot substitute for those prospective judgments.
For reproducible component contrasts, use a disposable database copy and frozen
profile with cached models:

```bash
uv run python scripts/experiment_ranking_variants.py \
  --db /tmp/ranking-review/digest.db \
  --profile /tmp/ranking-review/profile.yaml \
  --output /tmp/ranking-review/variants.json
```

This runner blocks model downloads and reports nine fixed variants at diagnostic
depths 5, 11, and 15 on two disjoint test windows. It omits downstream penalties
and selection rules, reports that limitation, and never selects a serving winner.

## Troubleshooting

- **The site cannot be reached:** keep the launcher window open and confirm it
  says `Uvicorn running on http://127.0.0.1:8765`. Then reopen that address.
- **Port 8765 is already in use:** stop the older DailyDigest process, or run
  `./scripts/start.sh` with a different `PORT` on macOS/Linux. On Windows use
  `.\scripts\start.ps1 -Port 8766`.
- **The first brew is slow:** the embedding model and article data are being
  downloaded. Later runs use the SQLite embedding cache. Local embedding uses
  bounded batches and four CPU threads by default; advanced users can lower
  `EMBED_BATCH_SIZE` or `EMBED_THREADS` in `.env` on a small computer.
- **A brew reports degraded research coverage:** DailyDigest received too few
  distinct papers or only one research-provider family. It deliberately keeps
  the existing digest instead of publishing a misleading partial result. Check
  your connection and source health, then retry. Redundant OpenAlex, PubMed,
  preprint, and journal channels mean one failed feed normally does not stop a
  brew.
- **No funding or events appear:** confirm the section is enabled and complete
  its profile in Settings. A successful zero-result scan is reported separately
  from a provider failure: it means no official record passed status, deadline,
  eligibility, and topic checks. DailyDigest does not pad it with unrelated or
  closed calls. If every configured provider for an enabled funding or events
  section fails, the brew stops and keeps the existing digest.
- **An API key does not work:** a chat subscription is not necessarily API
  access. Use Extractive mode while checking the provider's API account.

For a new reader on the same computer, stop DailyDigest and move or back up the
whole `data/` directory so profiles, votes, and history are not mixed.

## Local topic verification

Normal brewing requires research papers to clear the reader's topic floor
independently of journal prestige. Quality evidence still orders eligible work.
Longer abstracts are retained and profile-driven OpenAlex searches include
preprints. Add aliases and describe useful papers in each interest's search
details to clarify ambiguous terms; no model guarantees literature coverage or
verifies scientific validity.

An experimental local joint topic check is available with `SEMANTIC_MATCHING=true`
but disabled by default: it has not demonstrated an additional selection gain.
It checks up to 48 papers, records partial text coverage, preserves existing
quality/preference scores, and never vetoes an otherwise-related paper. First use
caches about 92 MB of public weights; reader text stays local. `SEMANTIC_MODEL_DIR`
can provision the exact pinned files offline. Normal brewing avoids this download
and computation. The improved relevance gate remains active in either mode.

## Development and license

Run the automated tests with:

```bash
uv sync --frozen --group dev
uv run pytest
uv run ruff check src tests
```

DailyDigest is released under the [MIT License](LICENSE).


Settings also offer optional example profiles, per-interest search details and
spelling suggestions, followed authors, representative works and background
interests. Test API connection performs an explicit small request without saving
settings or sending research context. Browser users need the companion for local
CLI providers; native users run those installed commands locally.

Unknown venue evidence is distinguished from observed low impact. Both retain
bounded selection and strong raw-topic requirements; a positive allowance permits
one candidate in small digests while explicit zero allowances remain exclusions.
Exact journal aliases avoid accidental downgrades of known venues. CS preprints
use the same publication-status policy as other fields. Source freshness and
limited searches remain visible; funding/event coverage is regionally limited.
