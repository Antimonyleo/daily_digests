# DailyDigest browser launch

This application packages the current Python DailyDigest implementation for
execution in a browser worker. The existing FastAPI interface renders pages in
memory; no listening Python server or remote ranking request is involved.
HTML forms, feedback, history and live brew progress use that same interface.

Setup shows muted research examples, which visitors can explicitly load and edit
before saving. Startup and brewing use tea animations and messages tied to real
processing stages; reduced-motion settings keep animations still. The digest's
source card shows the latest attempted check, source response coverage and
available public-feed update times separately from paper publication dates.
Unavailable or limited feeds remain visible, including after a failed brew.

Each interest has an optional **Check wording** action. It sends only that
phrase to PubMed for spelling suggestions; it does not run while typing.
Accepting a suggestion updates the editor, and saving applies the corrected
name to retrieval and ranking while retaining its weights and research details.
Specialist vocabulary may have no suggestion, and unavailable checks leave the
interest unchanged.

The **Compute** menu defaults to one CPU core. GPU acceleration is an explicit,
per-device opt-in and falls back to CPU if initialization or inference fails.
Apply and reload saves the choice for the next brew. Finish or stop an active
brew before switching. Compute shows selected versus completed work and CPU
fallback; acceleration affects new text comparisons, not fetching or an existing
digest. Cached comparisons need no acceleration.
Device-loss signals are watched directly because a lost GPU can leave the
runtime's inference promise pending without rejecting it.
It uses a checksum-pinned, unfused float32 export of the same BGE model because
the native optimized graph has an attention mask unsupported by this WebGPU
runtime. Hardware availability is checked before choosing the session's export.
That export and cache signature stay fixed even if inference later moves to
CPU, including when cached feedback vectors have already been read. The
133 MB export is downloaded by the browser
only when GPU acceleration is used. Physical GPU
performance and browser/device support vary; this remains experimental.
Multicore CPU uses at most four cores and is offered only with cross-origin
isolation and SharedArrayBuffer support. The Compute menu can opt in to a project-scoped service worker that adds
isolation headers, then reloads once before work starts. If the browser cannot
enable isolation, computation falls back to one core. Detection uses browser
capabilities on macOS, Windows and Linux, with no OS allowlist. Actual hardware
performance still needs device testing.

Brewing notices explain processor, memory, heat and battery use. The main brew
panel displays completed embedding counts and elapsed time independently of
Python's stage stream; remaining time is a rough estimate from recent batches.
Completed stage durations are displayed separately from the active work.
Cancellation is checked between embedding batches, preserving the native
publication gates and the previously saved digest.

## Architecture and feature transfer

* Native Python: all enabled ranking stages, profile weights/negative interests,
  graded preference memory, calibration, quality gates, source classification,
  novelty/history, fourteen-day publication cooldown, reading modes, summaries,
  opportunity eligibility, calendar files and persistence rules.
* Browser adapters: ONNX embeddings, HTTP transport, thread-free job execution,
  ASGI requests and streaming, IndexedDB snapshots, tab ownership, backups and
  project-subpath navigation. SQLAlchemy and SQLite remain the native store.
* Existing interface: setup, digest, saved reading, ratings/reasons, known items,
  Time Machine, progress/reconnect/cancellation, display controls and Pip.
* Public feed collection: GitHub Actions gathers configured non-profile-driven
  RSS/arXiv metadata twice daily. Matching successful records remain usable for
  48 hours, with a freshness warning after 24 hours. Expired or failed records
  remain genuine source failures. Scheduled and manual refreshes require fresh
  coverage of every section and provider kind. A partially failed push can retain
  the whole previous validated snapshot within 48 hours, preserving original
  timestamps and showing an earlier-snapshot warning. All-error collections fail.
  Profile-driven sources still run on the visitor's device. No personal profile,
  votes, reading history or private database enters the published artifact.

The public snapshot addresses CORS restrictions without moving personalization
to GitHub. The source audit observed 68 of 84 configured sources blocked in direct
browser access. Re-run the audit when sources or browser policies change.

Personalized OpenAlex searches use titles, abstracts and indexed keywords. The
original interest keeps stemming; explicit alternative terms remain quoted.
This reduces incidental full-text matches, but papers using different terminology
may be missed. Topic fit and venue quality are still checked separately.

## Build and preview

```bash
uv sync --frozen --group dev
cd prototypes/browser-compute
npm ci --ignore-scripts
npm run prepare-fixture
cd ../..
uv run python browser/collect_sources.py --output browser/.build-cache/public-sources.json --days 21
uv run python browser/build.py
python -m http.server 8767 --bind 127.0.0.1 --directory browser/dist
```

For an offline verification build, reuse an existing validated public snapshot in
`browser/.build-cache/public-sources.json` instead of running the collector. Then:

```bash
uv run python browser/build.py --output .full-review/clean-release/site
uv run python browser/make_test_fixture.py --output .full-review/clean-release/fixtures
# Serve the site at /daily_digests/ before running the acceptance check.
DAILYDIGEST_BROWSER_URL=http://127.0.0.1:8767/daily_digests/ \
DAILYDIGEST_BROWSER_FIXTURE=.full-review/clean-release/fixtures/fixture-backup.json \
DAILYDIGEST_BROWSER_FEED=.full-review/clean-release/fixtures/research-feed.xml \
DAILYDIGEST_BROWSER_REPORT=.full-review/clean-release/acceptance \
node browser/verify-browser.mjs
```

The browser check intercepts external requests with fictional fixtures. It also
reads the built public metadata; it does not perform a live brew or call providers.

Build output must be new or empty; `--output PATH` selects another directory.
Do not publish the entire repository, development caches, `data/`, or `.env`.
The positive build allowlist is covered by privacy regression tests.

## Operating requirements and remaining constraints

Use a browser supporting WebAssembly JavaScript Promise Integration and Web
Locks; runtime startup checks both. This implementation has been tested in
Chrome. Safari/Firefox and real phone performance are not claimed verified.
The static package is approximately 368 MB including public metadata, both embedding exports
and the optional topic-check model. Embedding models download on the first applicable
job; the additional topic-check model downloads only when explicitly enabled. Normal browser HTTP caching
applies; a storage-eviction-proof offline installation is not promised.

Data belongs to the browser and site origin. Export backups before clearing
storage. Import requires confirmation and replaces that browser's private state.
Cross-device state transfer is optional through an authenticated companion;
browser backup export/import also works without one. Default extractive
summaries are local. Optional HTTP model services require explicit configuration
and usable CORS; provider secrets are never built into the public application.

API, Anthropic and local CLI summaries, email delivery, inbound feedback and private state transfer
are implemented through `browser/companion.py`, enabled only when explicitly
configured. The browser applies email reply votes to its own dated digest before
acknowledging the mailbox. Tests mock providers, email and inboxes. Bounded live
Companion checks verified Claude and Codex CLI summaries using a public abstract;
API, email and inbox behavior still needs live credential checks. Optional local Ollama uses its HTTP
provider configuration and needs a usable browser/companion network path.
An explicitly connected companion advertises its configured summary providers;
their credentials stay on that server. The browser retains the native summary
budgets and extractive fallback, and invalidates cached summaries when the
companion's provider identity changes.

```bash
dd companion --allowed-origin https://antimonyleo.github.io
```

The command prints a private token file path, never the token itself. Enter the
endpoint and token in the site's Companion menu. The private default token file
is reused across launches; an explicit `--token-file` remains available. The default bind is localhost;
a remote companion needs an explicitly configured HTTPS reverse proxy. Native
local defaults are unchanged, and the browser never forwards a normal brew.
Explicit email and state-transfer actions are separate buttons.

Closed-browser scheduling additionally requires `--state-dir PATH --schedule`
and an explicit upload of browser state. Scheduled native work uses an isolated
copy, and the resulting snapshot can be restored in the browser. The companion
process must stay running. Server credentials never enter the returned snapshot.
Static Pages alone cannot execute local commands, host an inbox, protect server
secrets or guarantee background private computation.

The model's 512-token limit remains the same as the native default. Runtime
parity is not evidence of improved scientific relevance or article quality.

## Implementation and verification checklist

1. Preserve the native implementation and build a private-data-free artifact.
2. Verify compiled dependencies, native routes and real ONNX inference in Chrome.
3. Verify setup, persistence, backups, feedback, saved reading, history, calendar,
   display, cancellation and degraded-source preservation through actual pages.
4. Verify ranking/quality/version policies with existing native checks and the
   browser runtime; leave experimental disabled models disabled.
5. Collect public source metadata with explicit failure/staleness evidence.
6. Publish through `.github/workflows/browser-pages.yml`; bootstrap the same
   workflow on the default branch so scheduled refreshes actually execute.
7. Verify the HTTPS project URL and freshly published source snapshot.
   A severely degraded collection fails before replacing the published snapshot.
   Valid public records remain usable for 48 hours and warn after 24 hours;
   carried records retain their original collection time and latest error.
8. Record unsupported integrations and browser/device limitations honestly.

Run the full native suite before release. Browser bridge tests run with
`node --test --test-isolation=none browser/*.test.mjs`. Real-browser
acceptance is `node browser/verify-browser.mjs`; it expects isolated fictional
backup and feed artifacts and blocks arbitrary external services. It is not a
live brew against the reader's database or configured providers.
It also reads the published public snapshot through the actual browser HTTPX
transport, checking that browser-decoded responses are not decompressed twice.
Mobile viewport checks establish layout only; production-scale latency and
actual Android/iOS or other browser engines still need separate verification.

Run the large-checkpoint startup and recovery regression against a fresh build:

```bash
DAILYDIGEST_STORAGE_SITE=.full-review/clean-release/site \
DAILYDIGEST_STORAGE_REPORT=.full-review/clean-release/storage-startup \
node browser/verify-storage-startup.mjs
```

This check starts its own loopback server and an isolated Chromium session. It
uses only synthetic SQLite, embedding, and profile files, blocks non-local
requests, and never calls providers or runs a brew. It verifies startup, refresh,
checkpoint restore, and downloaded backup bytes without resetting the saved state.
The report is `DAILYDIGEST_STORAGE_REPORT/report.json`; synthetic fixtures stay in
the same directory. Chromium uses
`CHROME_PATH` when set, otherwise `/usr/bin/google-chrome`. The Pages build runs
this check against `browser/dist` before uploading its publication artifact.

The earlier 400-paper frozen replay (328 public records and 72 controlled scope cases)
matched the native ten-paper slate and ordering; maximum score drift was below
0.00015. On the desktop test machine, native cold/warm brewing took 22/2.4 seconds
and browser cold/warm brewing took 99/5.1 seconds. UI timer delays remained low
during processing. These are single-run development measurements with an
isolated fictional profile, not phone benchmarks or scientific-quality labels.
The replay also selected an uncertain controlled paper in both implementations;
matching native behavior does not settle contribution classification.
The 100 recorded ranking candidates also matched membership, selection flags
and primary facets; their maximum recorded score drift was 0.0002.

The larger 2,340-record replay (2,268 public records plus 72 controlled cases)
stored 2,331 unique items and encoded 1,424 qualified article inputs. Single-core
browser brewing took 22m 21s cold, 28s warm and 28s after reopening the site.
All three runs matched the native ten-paper slate and order, and all 1,474
recorded candidates matched membership, selection flags and primary facets.
The compatible GPU export, replayed through native ONNX CPU over that same pool,
also preserved those selections and facets; maximum candidate score drift was
0.00125. A bounded software-WebGPU check validated numerical agreement, while a
64-input software-GPU run exceeded its ten-minute limit. Production rejects
software adapters and uses CPU instead. A 64-input isolated-host CPU comparison
took 53s on one core versus 19s on four cores with identical vectors. These
development measurements do not establish physical GPU speed on macOS, Windows
or Linux; GPU mode remains experimental and opt-in.

## Recovery and portable data

Corrupt startup state exposes raw download, validated restore and an explicitly
confirmed fresh start. Replacement retains the original state in IndexedDB;
Browser data can download that retained copy. Raw recovery copies may contain
credentials and should be kept private. External imports are limited to 10,000 files
and 128 Mi characters of JSON; larger existing checkpoints remain restorable and
exportable without truncation, with a storage warning. Portable backups and Companion state
transfers exclude credential values by default; including them is explicit.
Internal checkpoints remain complete, and a redacted import preserves existing
credentials on the destination device. Imports that would attach an omitted API key
or mailbox password to a different endpoint or account are rejected before replacing state.

Writes are serialized only after private state changes, including reading
impressions and votes during a brew. Quota failures show a persistent warning;
read-only pages remain usable. Protect browser storage requests persistence,
but browser permission and backup practices still matter. The one-time reminder
requests browser persistence only on an explicit click. The browser manages the
storage directory; portable backups can be saved in a folder the reader chooses.
Other project sites
on the same hostname share a security origin; a dedicated hostname provides
actual isolation.

## Release identity and research interests

Application assets carry one release identity, and Python bundles are checked
before restoring state. Transitive runtime dependencies use a lock-derived path;
public snapshots are revalidated separately from model downloads. All configured
interests are shown separately from selected-paper topics. Editable alternative
terms and descriptions feed search and relevance; percentages are relative
priorities rather than guaranteed slot allocations. Research and venue-quality
gates remain in force.

## Broader-reader verification

Settings now keep each interest compact, with spelling, alternative terms and
intent behind Search details. Optional biology, chemistry and computing examples
load only into an empty editor. Advanced interests expose followed authors,
representative works and background topics. Save remains visible while scrolling;
private profile details are preserved. Displayed percentage shares must total
100% before saving, with frontend and server validation. Adding or removing a row
normalizes the shares; legacy relative weights convert only when settings are saved.

The browser reports which CLI providers its connected companion advertises.
Direct API settings have an explicit Test API connection action: one small,
potentially billable request, no reader profile or paper metadata, no save.
Anthropic Messages calls acquire the official browser-access header only in the
browser transport for the authenticated official endpoint. Native clients are
unchanged. Mocked success, failure, bad URLs and stale replies are checked; live
provider credentials and organization CORS policy remain unverified.

Unknown publication venues are distinct from observed low-impact venues. Both
remain bounded and require strong raw-topic relevance; missing venue metadata
cannot bypass those limits. Exact journal aliases prevent known venues from
falling into the unknown bucket. CS preprints use the same publication-status
policy as other fields. Explicit zero allowances remain zero, including fallback
selection. Venue priors do not verify article validity or scientific alignment.

The cross-field evaluation fixture contains 96 provisional metadata judgments
across eight scenarios. Its retrospective split is not independent expert gold,
and retrieved pools cannot establish upstream recall. Use
`python -m dailydigest.rank.cross_field_evaluation --help` for offline replay.
A failed semantic experiment is retained as evidence rather than enabled.

An optional experimental local topic check processes at most 48 research
papers, balanced across interests and including current ranking leaders. It uses
explicit per-interest descriptions and aliases; it does not invent a reader's
intended meaning. A separate pinned model (~92 MB) loads on first use and runs on
the CPU, even when embedding acceleration is selected. Native and browser paths
share the same title-preserving first/middle/last token windows and coverage flags.
Long descriptions retain the named topic and report partial coverage when bounded.

This check modestly lowers selection priority on strong model disagreement or
recovers a strongly matched borderline paper. It preserves eligibility for
otherwise-related papers, original quality/preference scores and all source-quality
floors and quotas. Journal reputation no longer admits otherwise off-topic work.
Raw logits are not relevance probabilities, and the check does not reliably
distinguish background mentions from the paper's contribution. Missing/unchecked
evidence keeps the existing ranking; a failed model pass stops publication.
The stronger score-replacement experiment and a slower NLI alternative were not
promoted after adversarial evaluation. `SEMANTIC_MATCHING=false` is the default:
the additional model has not demonstrated a separate selection gain, so normal
brewing avoids downloading/loading it. The corrected venue-independent topic
gate remains active. No online model provider receives reader text for checks.

The browser toolbar follows the reading desk's theme. Browser data shows site
storage estimates and the last backup-export attempt on this device. Separate
people should use separate browser profiles. Import and companion restore replace
state; they do not merge independently accumulated feedback. An installed shortcut
does not promise offline startup or background computation.

Configured research sources span different fields unevenly. Grants.gov covers US
funding and the EMBL event stream is limited; these sections are not exhaustive.
The interface states these limits before opt-in and retains conservative eligibility.

For operations, the Pages workflow reports build and deploy failures. An optional
repository secret `DAILYDIGEST_ALERT_WEBHOOK` enables a JSON webhook accepting a
`text` field containing only the public workflow link and a failure notice.
Without it, Actions emits configuration guidance; enable repository Actions
notifications to receive alerts. No webhook was configured or contacted in tests.
Native-only main retains a bootstrap workflow checking out the browser prototype
until browser deployment is integrated into main; keep that branch available.

Local Chrome CPU inference over 16 and 32 captured abstracts (three warm rounds)
measured approximately 3.2x and 3.1x speedups with four threads, identical vectors
and measured parallel CPU use. These are component measurements, not full-brew
latency targets. Physical Intel WebGPU inference was verified separately on two
inputs. Phone performance, Firefox/WebKit application behavior and production
cold-brew performance remain unverified. English is the validated interface and
model baseline; multilingual relevance has not been established.
