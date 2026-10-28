# DailyDigest browser launch

This application packages the current Python DailyDigest implementation for
execution in a browser worker. The existing FastAPI interface renders pages in
memory; no listening Python server or remote ranking request is involved.
HTML forms, feedback, history and live brew progress use that same interface.

After opening a digest, returning to the home page can display a small saved
reading view without starting Python or restoring the embedding cache. Article
links and summaries are available immediately. **Open full reading desk** enables
ratings, saves and settings; **Brew a new digest** opens the brewing controls.
Both start the existing runtime on demand. Other deep links start it normally.
The reading view is explicitly dated, read-only, and stored only in this browser.
It contains structured display text and optional publisher-image links, never
scripts, API keys or a second private database. Canonical state changes invalidate it in the same storage transaction;
the displayed digest can regenerate it only after its state is durably saved.
Missing, invalid or oversized reading copies use the ordinary startup path.
Backups and recovery continue to use the authoritative private snapshot.

Saved reading views can be opened in multiple tabs. Full controls have one
editable owner: a second tab offers **Use this tab instead**, with switch/close
instructions and Retry available immediately. A cooperative handoff waits for
in-flight actions (including cancelled UI waits), checkpoints the current owner,
then terminates its worker to release the browser lock. The receiving tab opens
a fresh runtime from that final checkpoint. The previous tab shows **Reading
desk moved to another tab** and can request it back. Unsaved Setup drafts, an
active brew, or failed storage commits refuse a handoff; no lock is forcibly
stolen. Older or unresponsive tabs may need to be closed manually. Broadcast
messages contain coordination metadata only, never profiles, keys or databases.

First-visit Setup keeps editable examples above the blank interest rows, with
biomolecular, engineering, chemistry, ecology and astronomy starting points.
Examples are never applied or saved automatically. Search and wording advice
stays in an optional disclosure; separate topic/percentage inputs and the 100%
total requirement remain unchanged.

A first-time visitor can type topics in the startup screen while Python warms
up. Continue transfers the draft into empty Setup rows with equal percentages;
it does not save a profile or start a brew. Drafts stay in this tab's session
storage until transferred. Existing profiles are never overwritten by this path.
Opening ordinary pages does not import the brew pipeline or NumPy. SciPy and
scikit-learn packages load before the first brew/training request, with a
retryable failure before any job is queued. Stop requests made during that
preparation are retained until the brew start is confirmed. This moves some
preparation to first compute use; it does not remove all runtime startup work.

Startup modules are precompiled at build time using the pinned Pyodide
interpreter. Source files remain in the bundle; CPython checks source hashes
before accepting cached bytecode. The optional dependency bytecode download
runs alongside runtime preparation and is abandoned if it is not ready when
imports begin. Missing, slow or damaged caches use the original source. Feed
adapters initialize once before the first compute request, after settings have
been restored. Compiler and Python requirements changes invalidate release URLs.

Repeated brews reuse successful language decisions for identical title/abstract
text and detector identity. The process-local cache holds at most 4,096 entries;
samples over 8,192 characters are classified in full without being retained.
Changed text and failed detection are rechecked. Language thresholds, candidate
coverage, embedding models, token limits and ranking rules remain unchanged.

When an article feed supplies an image, the digest and saved-paper archive offer
an optional collapsed **Graphical abstract** or **Article image**. Only explicit
image labels establish a graphical abstract; generic thumbnails keep the preview
label. Opening the disclosure loads the publisher image without cross-origin
cookies or referrer information; the image links to an enlarged view and the
publisher article. No image is fetched for a collapsed preview. Embedding/CORS
restrictions may prevent display, including under multicore isolation; the
publisher link remains available. Saved read-only views retain image links and
make no image requests. Images never add ranking weight. Existing metadata must
be refreshed from a feed before newly supported images appear; no publisher
scraping, proxy or rehosting is added.

The Python runtime and libraries are prepared when the full desk is opened,
instead of on every reading visit. Snapshot restoration decodes private files
once while retaining SQLite integrity checks before any restore writes.

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

On eligible desktop browsers, the **Compute** menu defaults to up to four CPU
cores. A new device with at least four logical cores and at least 4 GB reported
memory (or no memory estimate) attempts project-scoped browser isolation before
starting Python, with at most one preparation reload and a one-second deadline.
The topic draft and Continue action survive that reload. Existing compute or
isolation choices are preserved; unsupported, constrained or storage-restricted
devices retain the single-core path. Python initializes only once. GPU acceleration is an explicit,
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
Adapter detection has a three-second deadline; a stalled check falls back to CPU
instead of holding startup or the Compute controls indefinitely.
That export and cache signature stay fixed even if inference later moves to
CPU, including when cached feedback vectors have already been read. The
133 MB export is downloaded by the browser
only when GPU acceleration is used. Physical GPU
performance and browser/device support vary; this remains experimental.
Multicore CPU uses at most four cores and is offered only with cross-origin
isolation and SharedArrayBuffer support. The Compute menu can opt in to a project-scoped service worker that adds
isolation headers, then reloads once before work starts. The same mechanism is
used for the new-device CPU default. If the browser cannot
enable isolation, computation falls back to one core. Detection uses browser
capabilities on macOS, Windows and Linux, with no OS allowlist. Actual hardware
performance still needs device testing.

Brewing notices explain processor, memory, heat and battery use. The main brew
panel displays completed embedding counts and elapsed time independently of
Python's stage stream; remaining time is a rough estimate from recent batches.
Completed stage durations are displayed separately from the active work.
Cancellation is checked between embedding batches, preserving the native
publication gates and the previously saved digest.

Identical text comparisons also reuse a bounded memo within the active worker.
It keeps at most 256 entries, skips text longer than 8,192 characters, and uses
exact text including query/document prefixes. Editing research context or a topic
computes its new text; unchanged inputs can be reused. Model exports and worker
instances never share this memo, failed calls do not populate it, and returned
vectors are copied so callers cannot alter it. It adds no persisted profile data;
existing article-vector storage remains authoritative across reloads. Cancellation
is checked even when every requested vector is already available.

`verify-brew.mjs` exercises Standard, Short and Full brews, repeat use, changed
research context, reload, source failures, cancellation and a larger synthetic
candidate pool against a built site. It uses an isolated fictional profile and
intercepted feeds, with external services blocked. Set `DAILYDIGEST_BREW_SITE`,
`DAILYDIGEST_BREW_REPORT` and `DAILYDIGEST_BREW_FIXTURES` (a directory containing
`fixture-backup.json` and `research-feed.xml`) to select its inputs and output.
The harness fixes page and worker clocks five days after the fixture feed date,
while keeping a different browser timezone to catch silent date changes. Expired
production snapshots must not be relabelled fresh to make a release pass.
Existing profiles retain their configured timezone when other settings are saved;
only first-time Setup adopts the browser timezone automatically.
Latest release audit, feature coverage and deployment evidence:
[release review](../docs/release-review-2026-10-23.md).
Set `DAILYDIGEST_BREW_COMPUTE=multicore` to repeat the matrix with four CPU cores
(the harness supplies isolation headers); the default fixture explicitly uses
one core. `verify-cpu-startup.mjs` separately tests the actual service-worker
preparation/reload, preserved topic draft and explicit single-core preference.
It accepts `DAILYDIGEST_CPU_SITE` and `DAILYDIGEST_CPU_REPORT`.

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
# Independently verify refresh without Python, followed by real vote/save restore.
DAILYDIGEST_READING_SITE=.full-review/clean-release/site \
DAILYDIGEST_READING_FIXTURE=.full-review/clean-release/fixtures/fixture-backup.json \
DAILYDIGEST_READING_REPORT=.full-review/clean-release/reading-view \
node browser/verify-reading-view.mjs
# Three fresh startups, first-topic draft, and click-to-load images in normal
# and cross-origin-isolated Chromium. All external sources/providers are blocked.
DAILYDIGEST_STARTUP_SITE=.full-review/clean-release/site \
DAILYDIGEST_STARTUP_REPORT=.full-review/clean-release/startup-figures \
node browser/verify-startup-figures.mjs
```

The browser check intercepts external requests with fictional fixtures. It also
reads the built public metadata; it does not perform a live brew or call providers.
For an explicitly offline historical fixture only,
`DAILYDIGEST_BROWSER_SNAPSHOT_AT_COLLECTION=1` evaluates that transport check at
the snapshot's recorded collection time. Leave this unset for release checks;
an expired fixture is not a publishable release.

`verify-ranking-parity.mjs` runs complete local extractive Brews for eight captured
research fields and two opposite topic-priority configurations. It records every
candidate's numeric features, eligibility stages, selection and order, along with
the actual backend. Publication dates are rebased to the preceding fixture day;
article text and research metadata are retained. The source adapter is replaced
with the captured pool, so this tests computation rather than live retrieval.
Set `DAILYDIGEST_PARITY_SITE`, `DAILYDIGEST_PARITY_REPORT`,
`DAILYDIGEST_PARITY_MODE=cpu|multicore|gpu` and `DAILYDIGEST_PARITY_ROUNDS=1..5`.
`DAILYDIGEST_PARITY_PROFILES` restricts the comma-separated profile IDs.
The optional `watchlist_boundaries` profile uses three explicitly fictional
bylines to verify the real pipeline's author-match features and boosts.
`DAILYDIGEST_PARITY_HEADED=1` uses a temporary visible Chrome window without
experimental GPU or JSPI flags. `DAILYDIGEST_PARITY_GPU_FLAGS=1` is a separate
Linux Vulkan hardware diagnostic; its success does not establish default browser
support. All of these checks block external requests and use fictional desks.

`node browser/verify-pip.mjs` renders the current production template into an
isolated fictional desk and checks drag/release, retained conversation and leaf,
position/count persistence, keyboard controls and narrow-screen reduced motion.
Set `DAILYDIGEST_PIP_REPORT` for its screenshots and JSON evidence. It does not
start a browser Python runtime, call live sources or read personal state.

For startup download sensitivity, set `DAILYDIGEST_STARTUP_DOWNLOAD_MBPS=10` on
`verify-startup-figures.mjs`. Its local server shares one transfer budget across
page and worker requests, with an 80 ms response delay. Reports distinguish
Python startup from navigation-to-ready time. This is controlled network
simulation, not a measurement of production hosting or live source latency.
Optional dependency bytecode is prefetched only when the observed application
transfer reaches roughly 4 MB/s. Slower connections retain source compilation
and avoid that extra download; missing or damaged optional caches also fall back.

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
Backup can download that retained copy. Raw recovery copies may contain
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

### Reading desk, transfers and the quick guide

The Guide button opens a short walkthrough of setup, brewing, feedback, storage,
recovery, compute and Companion. New readers see a small introduction with a
restore option. They can dismiss it and reopen the guide at any time. The footer
storage button opens Backup & transfer. Setup also links to the guide and backups.

Each browser profile has its own working reading desk. Signing into Chrome or
installing the browser shortcut does not connect it to another profile or device.
DailyDigest has no account sync. Download a backup and restore it elsewhere to
transfer a reading desk. Restoring replaces the destination; it does not merge
independent edits. Companion transfers are also manual replacements.

Browsers with `showSaveFilePicker` can link one backup file in a folder the reader
chooses. The picker starts in Documents where supported; the app cannot scan a
default folder. File access and automatic updates require explicit permission.
The handle is remembered in a separate, project-scoped IndexedDB record. Startup
checks permission and file metadata without reading the whole file, showing a
picker, or starting Python just for the backup. If permission expired, the reader
can reconnect with Allow file access. Another browser profile must choose the file
separately. Browsers without this API use download and restore instead.

Automatic copies are off by default. When enabled, they run about a minute after
changes to saved settings, feedback or a completed brew, while the tab is open and
no brew is running. They reuse one file and exclude credentials. Reading a page
does not trigger a file copy. A copy may still be pending when the tab closes;
Save copy now writes the latest state explicitly. Full backups include local
caches and can be large, so copying does not run on each click. Copies beyond the
128 MB portable restore limit do not replace the last usable backup.

An existing file must be restored before automatic writing is allowed. Every
write takes an exclusive file lock and checks the previous content's SHA-256.
Changes made in another browser stop copying instead of being overwritten. A
failed write aborts the temporary stream, keeping the last committed file.
Disconnect forgets the handle and leaves the file untouched. Automatic copies
are backups, not synchronization, and the browser database remains authoritative.
Keep separate downloaded copies if you need older versions.

Run the guide and backup integration check against a built site:

```bash
DAILYDIGEST_GUIDE_SITE=browser/dist node browser/verify-guide-backup.mjs
```

It uses the fictional backup from `browser/make_test_fixture.py`. File selection
is replaced with an isolated browser file handle; the test never opens a reader's
folders. It checks a real rating, a matching full export, reload persistence and
restore in a separate profile. Native OS picker dialogs and permission prompts
still need manual checks on each supported platform.

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

The browser toolbar follows the reading desk's theme. Backup shows site
storage estimates and the last backup-export attempt on this device. Separate
people should use separate browser profiles. Import and companion restore replace
state; they do not merge independently accumulated feedback. An installed shortcut
does not promise offline startup or background computation.

Configured research sources span different fields unevenly. Funding covers the US
(Grants.gov, NIH Guide) and Europe (EU Funding & Tenders); events come from EMBL
and EMBO. These sections are not exhaustive.
The interface states these limits before opt-in and retains conservative eligibility.

For scheduled/manual source refreshes, the Pages workflow maintains one recurring
GitHub issue: failure creates or reopens it, subsequent failures update it, and a
successful strict refresh closes it. Only the reporting job has `issues: write`;
issue bodies contain public workflow links, never reader data or logs. Issue
support/permissions and future notification delivery still need verification after
publishing. Tests use a fake GitHub API and send no live alerts.
The optional `DAILYDIGEST_ALERT_WEBHOOK` remains available for build/deploy failure
notices; configure it explicitly if desired. Snapshot warnings use the original
source fetch time (24-hour warning, 48-hour expiry), not application build time.
Native-only main retains a bootstrap workflow checking out the browser prototype
until browser deployment is integrated into main; keep that branch available.

Local Chrome CPU inference over 16 and 32 captured abstracts (three warm rounds)
measured approximately 3.2x and 3.1x speedups with four threads, identical vectors
and measured parallel CPU use. These are component measurements, not full-brew
latency targets. The October 22 audit exercised physical Intel WebGPU over ten
profile configurations with cold and repeated Brews. It requires process-local
Vulkan diagnostic flags on this machine; ordinary isolated Chrome returned no
adapter and used CPU. GPU and CPU exports are not output-equivalent at every
selection boundary: see the comprehensive review before choosing GPU for a
quality-sensitive comparison. Phone performance, Firefox/WebKit application behavior and production
cold-brew performance remain unverified. English is the validated interface and
model baseline; multilingual relevance has not been established.


## Public comparisons and first use

A validated public source snapshot can ship CPU article comparisons, computed
once during the build using the same browser BGE export, tokenizer and pooling.
A new visitor still computes their profile and uncovered papers locally. Reader
interests, votes and query text are never uploaded for this optimization.

The builder caps this cache at 10,000 documents and 16 MB of float32 vectors from
the existing 21-day public snapshot. It binds vectors to model files, preprocessing
and dependency identity, and matches exact encoded text. Missing, changed,
corrupt or incompatible assets fall back to ordinary local computation. The
unfused GPU export, including its CPU fallback, does not mix these CPU vectors.

Before building, CI restores the previous public cache with
`uv run python browser/fetch_public_embeddings.py`. Local builds reuse
`browser/.build-cache/public-embeddings/public-embeddings.json` and its referenced
content-addressed `.f32` file. The bounded precompute checkpoints every 128 texts;
a 20-minute limit retains valid partial progress and leaves missing work local.
Missing Chromium or an unavailable public cache does not block a code hotfix.
Inspect `assets/build-manifest.json` for actual computed/reused/missing coverage.

Startup records named phase timings and skips micropip when no wheels require
installation. Python/runtime/package initialization still has a cost on every
opening; cached downloads do not imply instantaneous startup. Reload during a
brew keeps committed state, explains interruption, and waits for an explicit new
brew. A release notice never reloads a running brew automatically.

## Reader experience and evaluation

The footer keeps one status line and **Data & compute**, which opens a single
keyboard-accessible view for acceleration, backups, storage protection and the
optional Companion. First-visit Setup defaults to Research only, gives topics
equal shares and keeps individual percentage editing optional. Existing readers
keep their enabled sections and priorities. Example profiles remain editable.

Cards show available authors, publication status and the recorded matching topic.
Literal phrase evidence is quoted only when present in supplied text; semantic
matches never fabricate a quote. The reading study reports current explicit
usefulness ratings per recent research slate, with unrated papers left unknown.
It is descriptive and does not claim that a ranking algorithm improved.

Route responses return after application commits, before a coalesced background
checkpoint. Unchanged file encodings are reused using audited write versions;
unknown writes and path aliases invalidate reuse conservatively. Portable and
recovery snapshot formats remain unchanged. Explicit reload, transfer, import
and export still flush; visibility/page-hide request a flush and closing during
unsaved edits prompts. Browser shutdown can still interrupt an asynchronous save.

A provisional list appears after all topic and quality selection gates, while
summaries are prepared. It is not published or used as a replacement for the
last good digest when work is cancelled or fails. The UI gives stage progress
and measured text-comparison estimates rather than an invented total ETA.
