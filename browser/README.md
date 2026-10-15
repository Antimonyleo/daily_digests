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

The **Compute** menu defaults to one CPU core. GPU acceleration is an explicit,
per-device opt-in and falls back to CPU if initialization or inference fails.
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
The static package is approximately 276 MB including public metadata and both exports; model files
are downloaded lazily for the first embedding job. Normal browser HTTP caching
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
acknowledging the mailbox. Tests mock providers, email and inboxes; live credential
and provider behavior remains unverified. Optional local Ollama uses its HTTP
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
credentials and should be kept private. Portable backups and Companion state
transfers exclude credential values by default; including them is explicit.
Internal checkpoints remain complete, and a redacted import preserves existing
credentials on the destination device.

Writes are serialized only after private state changes, including reading
impressions and votes during a brew. Quota failures show a persistent warning;
read-only pages remain usable. Protect browser storage requests persistence,
but browser permission and backup practices still matter. Other project sites
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
