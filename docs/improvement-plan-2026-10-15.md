# Browser app improvement plan (2026-10-15, verified revision)

Status: revised scope implemented and locally verified on
`codex/browser-compute-prototype`, based on `5c00a4b`. This revision replaces
outdated findings with evidence and bounds the remaining implementation.
Startup latency, the dependency upgrade and external-device/live validation remain
explicit limitations, rather than claims of completed improvements.
The implementation passed local verification. The owner subsequently authorized
commit, push, deployment and bounded live evaluation with “go ahead.” Publication
results are recorded separately under `.full-review/release-2026-10-15/`.
The original draft is retained in `.full-review/improvement-plan-2026-10-15/original-plan.md`.

## Boundaries and decisions

- Preserve local extractive mode, localhost defaults, saved/voted/known items and
  retained history. Never write real `data/` or `.env` during tests.
- No live brews, configured provider calls, emails, or live alert creation.
  Public source/model artifacts and fictional fixtures are sufficient here.
- Leave `docs/evaluation-2026-09-11.md` unchanged and uncommitted.
- Publication was initially deferred pending a subsequent owner instruction.
  That instruction has now been received. Publish the verified changes and check
  CI, the deployed build, source age and isolated reader-state recovery.
- User selected **one recurring GitHub issue for failed refreshes**. Test with a
  fake GitHub API; future scheduled/manual workflows create/reopen/update the
  same issue, and a successful strict refresh closes it. No public reader data.
- One writer for browser lifecycle/UI files; a separate writer for public-vector
  construction; a separate writer for retrieval explanations/common native ports.
  Root owns integration, workflows, this document and dependency assessment.
- Keep ranking policy, journal-quality gates and semantic-model defaults unchanged.
  Do not manufacture domain synonyms, papers, vectors or successful coverage.

## Verified issue register

Evidence describes this checkout and existing saved reports. It does not certify
an alpha tester's hardware or establish absolute recall.

| ID | Verified finding and treatment | Execution |
|---|---|---|
| U1 | Python/runtime imports remain a substantial startup cost. Earlier empty-profile loads were about 10–12 s; those measurements do not prove downloads dominate. Profile named stages, shorten startup copy, retain cached assets. Do not promise warm<cold from one noisy sample. | Phase 2 |
| U2 | An exposed WebGPU API does not prove a usable hardware adapter or model. Existing CPU fallback works; missing no-adapter/software/device-loss detail remains. Probe without loading a model. | Phase 2 |
| U3 | `ownStorage()` immediately rejected an unavailable lock. Reload during a departing worker's brew could fail startup. Actual app mid-brew reload/recovery must be exercised, not just a stand-in lock. | Phase 1 |
| U4/U5 | Backup/Companion copy is long despite useful controls. Show one benefit line and controls; put secondary details behind disclosures. Rename visible Browser data to Backup and retain optional Companion. | Phase 2 |
| D1 | Fresh browser still encodes all uncached public papers locally. Repeated private-cache runs are faster; they do not solve first use. Publish compatible public CPU vectors with exact-text/model identity and fallback. | Phase 3 |
| D3 | **Already fixed before this plan:** missing JSPI/Web Locks are named and unsupported Retry hidden. Preserve tests; do not rewrite as an open defect. | Verified baseline |
| D5 | An optional webhook/failure-report helper already exists; alert delivery was not configured or tested live. Add the selected recurring issue and visible 24 h/48 h snapshot-age warning. | Phase 4 |
| D6 | **Already implemented:** explicit API connection test and authenticated Anthropic browser header; mocks cover correct and lookalike hosts. Live API-key providers remain unverified. | Verified baseline |
| D7 | Build stamps, mixed-release recovery and the large-snapshot stack-overflow fix already exist. Add safe new-release discovery after startup; never auto reload an active brew. | Phase 1 |
| D8 | **Original finding obsolete:** dirty-state writes, committed checkpoints, recoverable quota warnings and persistence consent already exist. Extend regression tests; do not restore full writes after every GET. | Verified baseline |
| E1 | Alternatives are manual. Offer explicit local wording variants only when evidenced by supplied acronym/full-name or hyphenation; do not imply a general scientific synonym engine. Broader ontology-backed suggestions need separate evidence. | Phase 5 |
| E2 | **600-expansion claim corrected:** it describes historical PubMed. Current NLM guidance says wildcards search all variations. Structured provider-warning logging already exists; preserve and display relevant warnings. | Phase 5 |
| E3 | Per-interest queries/stages and reasons already exist, but limited zero-return searches hid provider notices. Show observed returned counts with status/warnings and each paper's recorded best-matching interest; never infer global supply. | Phase 5 |
| E4 | **Prior live evidence exists:** exact nine-topic capture returned 81 PubMed/82 OpenAlex records; saved replay selected an LC3 degrader covering degradation/proximity. This is bounded metadata evidence, not expert gold labels or guaranteed coverage. Reuse it; no live brew needed. | Verified prior evidence |
| E5 | **Already fixed:** footer wraps and mobile320/390px acceptance passed. Recheck the new controls at those sizes. | Phase 2 regression |
| E6 | **Original finding obsolete:** Windows/macOS CI executed native launcher/process-tree fixtures. Physical browsers, real configured providers and alpha-device GPU speed are separate unverified claims. | Phase 6 evidence |
| E7 | Current npm audit still reports 6 findings (2 high/4 moderate), inherited through native-only sharp/onnxruntime-node chains. These are installed Node dependencies, not accurately described as all devDependencies. Audit published artifacts separately; do not silently force a model-runtime major upgrade. | Phase 6 assessment |

## Phase 1 — lifecycle and persistence

1. Replace immediate lock refusal with an abortable bounded 60 s wait. Show waiting
   state, then an actionable close-other-tab/retry error; retain exclusive ownership.
2. Record interrupted brew metadata separately from committed private snapshots.
   Reload preserves the last committed digest/feedback/cache; explain interruption
   and require explicit brew action. Do not silently restart or publish partial work.
3. Real Chromium acceptance: reload while a mocked source fetch is pending; compare
   committed SQLite/cache bytes, retained URL, no implicit collection, then successful
   explicit next brew. Include release reload refusal while work is active.
4. Discover a different release and offer an idle, checkpoint-protected reload.
5. Retain existing read-only, quota, transaction, large-state and recovery regressions.

Done when: actual runtime reload test and storage regressions pass without clearing
reader data or weakening publication/ownership guards.

## Phase 2 — concise interface and measured startup

1. Replace startup paragraphs with a short purpose line and honest stage progress;
   retain download-size, local-compute cost and backup information in a disclosure.
2. Record ownership, assets, Python, packages, import, restore and application setup
   timings. Compare isolated cold/warm runs; optimize only demonstrated bottlenecks.
3. Backup/Companion menus show benefits first and secondary controls behind More.
4. Test GPU reports no API, null adapter, software adapter, available adapter and
   probe failure; it requests no device/model and sends no reader content. Propagate
   actual device-loss/model-failure fallback reasons without claiming compatibility.
5. Recheck missing-browser-feature messages, reduced motion, keyboard access,
   mobile footer wrapping and all percentage/alias controls.

Done when: focused adapter/UX tests and real desktop/mobile acceptance pass; timing
results are reported with sample limits, including any unmet warm-speed target.

## Phase 3 — bounded public comparison reuse

Design reviewed before implementation. Use the exact production browser CPU BGE
export/tokenizer/pooling through the WASM engine, not native FastEmbed or unfused
GPU vectors. No profile or feedback enters public artifacts.

- Select exact title+abstract strings from successful records in the validated
  public snapshot's existing 21-day window. Cap 10,000 documents and 16 MB float32 vectors.
- Bind signature to CPU graph/config/tokenizer, preprocessing code and locked
  dependency identity. Index by exact encoded-text SHA256; validate dimensions,
  finite normalized vectors, binary checksum and safe content-addressed filename.
- Reuse only compatible prior public vectors and compute missing texts. Keep
  incremental cache under `browser/.build-cache/public-embeddings/`; CI restores
  validated public assets from the last published site before building.
- Missing/invalid/incomplete assets and nondefault prefixes fall back to local
  encoding. GPU/unfused-backend sessions do not mix these CPU vectors.
- Precomputation has a finite time budget and cannot block a hotfix when optional
  acceleration is unavailable. Report actual completed versus missing coverage.
- This is an internal 16 MB vector budget, **not** a claim that Pages has a 100 MB
  individual-file limit. GitHub documents a 1 GB published-site limit.

Done when: real CPU-engine fixtures verify vector/ranking parity; a fresh profile
reuses most covered public texts; mismatched signatures/corruption use local
encoding; incremental refresh recomputes only missing documents; artifact audit
finds no private files. Measure both build cost and browser compute saved.

## Phase 4 — operational feedback

1. A scheduled/manual strict-refresh failure maintains one recurring GitHub issue;
   repeat failures update it, a later strict success closes it. Grant issues:write
   only to this reporting job, never build/deploy. Mock create/update/reopen/resolve
   and denied-permission cases; do not send a live alert for tests.
2. Keep optional existing webhook compatibility and explicit public workflow links.
3. Show age from original successful fetched_at timestamps, with 24 h warning and 48 h
   expiry consistent with source gating. An old source snapshot is not made fresh
   by code deployment or rebuilding vectors.
4. Preserve verified API test/header behavior and strict scheduled/manual versus
   bounded push carry-forward gates. Unit fixtures simulate missing/expired/all-error
   snapshots; an actual future schedule and notification delivery remain unverified.

## Phase 5 — honest retrieval explanations

1. Explicitly add reader-evidenced wording variants; keep original topic, percentage
   total, aliases and saved settings unchanged until the reader chooses Add/Save.
2. Keep PubMed exact-first/thin-expansion budgets and structured warnings. Display
   notices for limited results; remove the obsolete 600-cap claim from guidance.
3. Distinguish completed zero results from failed/skipped/limited searches and
   scored/eligible/selected attribution. Explain recorded best-matching topic and
   source without inventing a global paper count or scientific-quality guarantee.
4. Re-evaluate the saved exact-profile capture. State provisional-label and time-window
   limitations; do not present fixture successes as guaranteed live personalized recall.
5. Port common template/pipeline fixes to both native worktrees without overwriting
   existing unrelated dirty work.

## Phase 6 — housekeeping and verification

- Re-run npm audit and inspect its affected dependency paths. The suggested
  Transformers 4.3.1 upgrade is a model-runtime major change; any decision to retain
  current pins must be explicit, with published-asset exposure checked separately.
  Do not call known findings fixed because vulnerable packages are unshipped.
- Verify Windows process-tree evidence from the prior successful CI logs; this local
  Linux session cannot physically retest Windows/macOS/Safari/Firefox/GPU hardware.
- Review returned diffs independently, run focused modules and lint first, then full
  Python/Node suites, new-directory static build and real Chromium acceptance.
- Report exact commands, pass/skip counts, public asset sizes, timing samples,
  unverified external behavior and any deferred dependency/runtime work.

## Evidence and execution log

New evidence: `.full-review/improvement-plan-2026-10-15/`.
Prior release evidence: `.full-review/compute-preferences-2026-10-15/report.json`
and `.full-review/alignment-fixes-2026-10-15/reporter-profile/`.

### Authorized publication and bounded live evaluation

The owner authorized the next release and evaluation. Browser implementation
commits `9a91871`/`b690cac` and native shared-fix commit `5f8bfe8` were pushed;
Linux CI and native launcher checks on macOS/Windows passed. A subsequent narrow
correction recognizes Advanced Science's exact NLM titles/abbreviation (Catalog
101664569, ISSN2198-3844), preserving its existing canonical tier and explicit
impact evidence. Similar-name lookalikes stay unknown; ranking cache identity
advances to v16. It does not change topic embeddings, thresholds or source weights.

The release acceptance probe now selects actually covered public texts. Partial
or empty optional precomputes verify local fallback instead of requiring a fully
populated accelerator. The earlier deployment run was superseded before publishing
so the latest release includes both corrections. Results and remaining limits
are recorded under `.full-review/release-2026-10-15/`.

Live public metadata capture returned184 URL records across catalysis, quantum
computing and solar-energy profiles. Actual offline scoring and selection were
replayed in isolated state with socket access blocked. Metadata inspection found
a publisher-confirmed News & Views item supplied as an article by OpenAlex; its
official RSS also omits genre. Providing the publisher genre makes the existing
exclusion gate reject it, but no automatic publisher-genre lookup is implemented.
A generic-repository preprint lacks independent scientific-quality evidence;
some quantum/photodetector selections are related context rather than direct task
contributions. The exact venue correction also exposes a broad biomedical quantum
opinion through the existing unknown-source quota. These are remaining evaluation
findings, not evidence that universal semantic alignment is solved. No title/DOI
blacklist or speculative scoring adjustment was added to conceal them.

A real Linux Chromium benchmark on24 public texts measured one-core encoding at
18–22s versus9–10s with four threads (two runs each, median2.11x speedup, identical
float32 vectors). This is not a Mac/Windows/mobile or full-brew benchmark. No GPU
device is exposed in this session. A separate Transformers4.3.1 install reports
zero npm findings but fails fresh offline loading of the existing tokenizer;
production retains3.8.1 pending a verified asset-loading migration.
Implementation and review results:

- Lifecycle: bounded exclusive lock handoff, interrupted-brew notice, committed
  checkpoint before explicit release reload, and a mutation fence during reload.
  Independent review also found a failed isolation-registration path could leave
  that fence active; cancellation now restores controls and actual feedback writes.
- Interface: compact startup stages, measured stage timings, concise Backup and
  optional Companion menus, adapter-only GPU test and precise fallback messages.
  Skipping the empty wheel installer removes work measured at about 0.49 s in two
  isolated trials. Python libraries and application setup still dominate startup;
  this is not evidence that all warm loads are faster or first brews are cheap.
- Public comparisons: all 3,836 public article texts received validated 384D CPU
  vectors, 5,892,096 bytes. Cold four-thread inference took 805.596 s; compatible later
  builds reused all 3,836 with zero recomputation. Only public snapshot article
  text enters inference. Private profiles and feedback never enter these artifacts.
- Parity: 12 existing fixture papers across degradation, asymmetric catalysis and
  quantum error correction retained score order and selected slates through the
  real ranker. The one-core fixture measured 2,442.6 ms local encoding versus 9.69 ms
  cache lookup, with maximum vector difference 1.49e-8. This is a small-fixture
  measurement, not a full personalized-brew benchmark or expert alignment label.
- Failure behavior: whole cache download has a 45 s bound and Stop polling;
  checksum/signature/backend mismatches use local encoding. A real 4 s build timeout
  stopped its detached Chromium and retained 128 complete vector rows. New Windows
  cleanup logic has not been executed on Windows here.
- Operations: compatible prior public comparisons are restored before building.
  Only scheduled/manual results maintain the recurring GitHub issue; its creation,
  update, reopening, resolution, spoof rejection and permission failures were
  tested with a fake API. No issue or webhook was sent. Common bootstrap workflow
  changes were applied to the native worktrees while retaining their intentional
  browser-branch checkout; deploy the browser branch before a future main bootstrap
  update. No push or deployment occurred in this task.
- Retrieval explanations: retained provider warnings and observed search status,
  best-matching interest, and explicit local wording variants. Real UI acceptance
  caught a missing initial suggestion button; an executed DOM regression failed
  before the one-line fix and passed afterward. Common template/pipeline fixes
  were applied surgically to both native worktrees, preserving their existing work.
- Saved personalized evidence was rechecked: 81 PubMed plus 82 OpenAlex records,
  138 after quality gates, 116 after relevance and 12 selected. These are URL-record
  stage counts, not an estimate of unique-paper recall. One selected LC3 degrader
  supports both degradation and biological proximity; those counts overlap. All
  six physical-proximity contrast records were unselected. Two selected preprints,
  one journal report without an abstract and nonexpert metadata judgments limit
  quality/alignment claims. No current ranking-policy change was made. See
  `reporter-evidence-review.json`; this saved 21-day capture is not a new live check.
- Dependency assessment: npm still reports 6 findings, 2 high/4 moderate. The current
  shipped JS bundle contains none of the implicated native module names, and those
  modules are excluded by the build allowlist. This is an exposure assessment,
  not a vulnerability fix. Retain current model-runtime pins; assess the proposed
  Transformers 4.x migration separately with model compatibility tests.
- Privacy limitation: the pre-existing Python build import can initialize ambient
  application settings and read `.env`. Those settings do not become public article
  inputs or allowlisted output; no private file was written. Artifact scanning is
  additional evidence, not proof against every possible encoded secret.

Final integrated verification:

- Browser-branch Python: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLM_API_KEY='' .venv/bin/python -m pytest -o addopts='' -q --junitxml=.full-review/improvement-plan-2026-10-15/python-final-after-ui.xml`
  — 2,322 passed, 3 skipped. One skip requires optional Hatchling; two require Windows
  PowerShell. Existing Starlette deprecation warning remains.
- Native integration worktree, with explicit `PYTHONPATH=src` and the same offline
  flags: full pytest— 1,093 passed, 2 Windows skips. Both native worktrees also passed
  the 33-test setup/reporting subset after the final template fix.
- `node --test --test-isolation=none browser/*.test.mjs`— 96 passed;
  `node --test --test-isolation=none prototypes/browser-compute/browser.test.mjs`— 5 passed.
- `.venv/bin/ruff check --select F,B src tests browser prototypes/browser-compute/prepare.py`
  and `git diff --check`—passed.
- New-directory build: `.venv/bin/python browser/build.py --output .full-review/improvement-plan-2026-10-15/final-v2/public/daily_digests`.
  Build `82e4bdaec411ccb8`, 76 files, 373,954,649 bytes including its manifest. Full
  file/hash inventory is in `artifact-audit-final.json`; no private/credential flags.
  Only the expected 133,093,490-byte GPU model exceeds 100 MB.
- `DAILYDIGEST_STORAGE_SITE=.full-review/improvement-plan-2026-10-15/final-v2/public/daily_digests DAILYDIGEST_STORAGE_REPORT=.full-review/improvement-plan-2026-10-15/storage-final node browser/verify-storage-startup.mjs`
  —passed. Synthetic 16,799,526-character state retains exact SQLite, NumPy, profile
  and environment bytes through restore/export/refresh; zero external requests.
  Local startup 13.364 s and refresh 17.515 s do not demonstrate a warm-load speedup.
- Final Chromium acceptance:42/42passed, zero page errors,150 route checkpoints
  equal full exports (66 mutating). The mid-brew check proves 76 real SQLite embedding-cache rows
  retain vector bytes/model/hash/dimension, plus identical committed database,
  feedback/digest, route, interruption notice and explicit next brew. Real
  isolation opt-in, four-thread fixture computation and opt-out passed; this is
  not a physical-device scaling benchmark. The390px layout and keyboard recovery
  passed. Fresh Python document encoding reused12 public vectors with zero local
  encoding calls. The GPU probe reported no usable adapter without loading ONNX.
  See `final-v2/acceptance-verified/browser-acceptance.json` for all checks.
- Final local runtime timing: fresh10.667s, restored10.801s. No warm-speed gain
  established. Python packages (~3.69s) and application setup (~4.4s) dominate.
- Older intentionally dirty native worktree:37 focused setup/reporting/workflow
  checks passed after the final port. Its entire historical dirty state was not
  retested or committed; the native integration worktree received the full suite.

An earlier full suite exposed a build-test fixture missing the new helper module;
the fixture was corrected and both full suites rerun. Browser acceptance found
the actual missing wording button and two test-harness defects (forbidden export
during a brew; a selector made ambiguous by nested storage details). Earlier
failed reports are preserved and are not counted as passing acceptance.

Exact commands and machine-readable results are in `report.json` in the new
evidence directory. The local preview remains available at
`http://127.0.0.1:35523/daily_digests/`, build`82e4bdaec411ccb8`.

Still unverified here: production Pages behavior after a future push; actual
scheduled issue delivery and notification settings; cross-device GPU/multicore
performance; the new Windows precompute cleanup path; real Safari/Firefox and
alpha-user hardware; new live providers or personalized recall. No live calls
were made for those checks. The saved capture and small parity fixtures cannot
establish universal research alignment or article quality. Six dependency audit
findings remain pending a separately tested runtime upgrade. No commits, pushes,
deployments, real reader-data writes or live alerts occurred.

Official references:

- [NLM wildcard update](https://www.nlm.nih.gov/pubs/techbull/mj24/mj24_pubmed_wildcard.html)
- [GitHub Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)
- [Anthropic official SDK browser header](https://raw.githubusercontent.com/anthropics/anthropic-sdk-typescript/main/src/client.ts)
- [GitHub issue API](https://docs.github.com/en/rest/issues/issues)

## Reader-experience review — 2026-10-16

This addendum supersedes the review's outdated assumptions where noted. Work is
on `codex/browser-compute-prototype`; shared native fixes are ported separately.
New evidence is kept in `.full-review/reader-experience-2026-10-16/`.

| Finding | Verified treatment |
|---|---|
| Blank / slow start | Saved reading already bypasses Python. Add a visible static skeleton for a fresh desk. Do not claim Python startup itself is instant. Preserve on-demand editing and model loading. |
| Synchronous checkpoint | Confirmed. Return committed route responses before a short coalesced checkpoint; reuse unchanged encodings using audited file-write identities. Keep portable format unchanged, full-export parity, SQL transaction deferral and forced flushes for transfers/reloads. Requests and file aliases that cannot be localized invalidate the encoding cache conservatively. Warn on actual pending edits when closing. |
| Voting waits | Paint the intended rating immediately, serialize rapid changes per card, ignore stale completions and restore the last confirmed rating on failure. Keep practical-reason semantics. A recorded response and a durable browser copy are distinct states. |
| Overcrowded shell | Replace three footer popovers with one accessible Data & compute view; move consent there. Keep a dot/status line in the footer and put stale-backup reminders in the body. Preserve every Backup, Compute and Companion function. |
| Save clipped; dead Back; 99/1 | Confirmed. Unclip the setup card, test sticky Save in the viewport, hide Back only when no profile exists, start new topics equally and preserve explicitly edited priorities. Percentages remain available and must sum to 100%. |
| Accidental heavy job | Remove home autostart fields. Run page shows heat/battery/time notice before an explicit Start. Preserve explicit recovery routes and reconnect behavior. |
| Conflicting timing copy | Remove the one-minute promise; describe 20+ minute worst cases for large first browser brews, caching and device dependence. Use measured text-comparison estimates; no fabricated whole-brew ETA. |
| Early picks | Show a bounded provisional preview only after every topic/quality selection gate. Keep the last good digest on cancellation, coverage failure or publication failure. Do not stream unchecked candidates as recommendations. |
| Generic why / missing metadata | Show the recorded topic, an exact supplied-text phrase when available, authors and evidence-supported publication status. Semantic matches do not invent literal evidence, and venue is not article-quality proof. |
| Learning threshold | Review was incomplete: early ratings can already refine the topic direction and reason penalties; 30 controls the additional preference-ranking layer. Explain this visibly without changing scoring policy. Never claim “N picks moved” without a counterfactual measurement. |
| One evaluation instead of experiments | Add a read-only personal reading study over recent saved research slates: useful/rated/unrated counts, current explicit ratings, unrated unknown, repeated-paper and causal limits. No new ranker experiment. |
| Scope / first run | Research only for new profiles; preserve existing opt-ins. Topic rows begin with equal shares; individual priorities and search details are optional. Editable examples remain prominent. Existing example-paper anchors are explained; bare DOIs are not silently fetched or treated as meaningful topic descriptions. |
| Visual system | One token source for browser and native templates, stronger control borders, 12 px text floor, smaller greeting with secondary navigation, stable CSS section marks and print rules. Keep reduced motion and keyboard focus indicators. |
| Navigation / access | Route-specific outer titles, one main landmark per rendered page, skip links, distinct field names, larger targets, inert/busy old content while navigating and restored reading position when returning. Put tab takeover immediately below its heading. |

Deliberate limits: retain the isolated iframe renderer rather than undertake a
large lifecycle/security rewrite to fix title, focus and scroll symptoms. No
scientific synonym guessing, automatic DOI retrieval, threshold changes or false
claims that saved votes prove the new ranker improved. Binary-per-file storage
migration is unnecessary to obtain the measured response improvement here: the
compatible snapshot adapter now coalesces saves and reuses encodings; remaining
whole-envelope allocation/storage cost is measured, not declared eliminated.

Acceptance: focused tests first, all Python/Node regressions, a fresh stamped
build, populated fixture digests, 320/390 px layouts, actual delayed-checkpoint
response timing, full-export byte comparisons, failed-save/rapid-rating recovery,
and tab handoff. No live brew, configured provider, email or alert. Real screen
reader, physical phone, Safari/Firefox and physical GPU/multicore remain external
validation gaps. Record commands/results and publication status in the release
report; do not change `docs/evaluation-2026-09-11.md`.

The new rapid-voting review found an additional lifecycle race: a second rating
could still be queued in the rendered page when navigation or tab transfer paused
the worker. The parent now drains the page's entire rating queue before pausing
or cancelling streams. A red regression reproduced the lost latest rating; the
final real-browser checks preserve both the latest rating and bookmarks across
navigation, transfer, failed checkpoint refusal and retry. Native pages warn when
navigating away with ratings still pending.

Fresh desk fixtures must be generated for the current date. An obsolete reused
fixture correctly rendered an empty day and caused two selectors to time out;
those runs are retained as failed checks. The reading verifier now bounds RPC
waits and explicitly waits for committed feedback before testing a clean refresh.
Its final 40 MiB SQLite timing fixture uses random bytes to avoid a misleading
compression advantage from zero-filled storage.

Exact commands: `.full-review/reader-experience-2026-10-16/commands.json`.
Build file inventory and stamped-import checks:
`.full-review/reader-experience-2026-10-16/release-inventory.json`.
Final local build `a2ba9944b8061ddf` has 79 allowlisted files, no private data paths,
and one expected file above 100 MB: the optional 133,093,490-byte GPU model.

Local regression results: prototype 2,341 passed / 3 skipped; native 1,110 passed
/ 2 skipped; browser unit tests 121 passed; prototype compute tests 5 passed;
populated Chromium acceptance 45 passed; handoff/navigation 8 passed;
large startup/recovery 5 passed; local CPU semantic bridge 5 passed; lint and
diff checks passed. Full-suite JUnit and exact-command records are retained in
the evidence directory. A final first-run validation-error fix was reproduced
red and tested green in all related routes: 111 prototype tests and 106 native
tests. The earlier full suites precede that one-line context fix; CI reruns the
full final tree before publication.

The incompressible-data test found a verifier race: its old worker could replace
the deliberately corrupted disposable copy before reload. Fence the old worker
first, then write the corrupt copy. This preserves the intended recovery test
instead of interpreting a successfully regenerated saved-reading view as a boot
failure. Unexpected unload prompts are recorded and fail the clean-refresh check.

Final local follow-ups: the shell skip link now focuses the reading desk without
changing the hash-router URL. Its pure behavior and actual Chromium keyboard
interaction are checked. Fresh-build readiness smoke: 5 checks passed. Final
40 MiB random-data result: route 48 ms while persistence is held; SQL checkpoints
329–332 ms; three read-only refreshes 96–126 ms with no Python workers or saved-byte
changes. Full-export equality and restored votes/bookmarks passed. No unexpected
unload dialog was needed. Synthetic measurements are specific to this host, not
a guarantee for real reader profiles, storage pressure or physical devices.

Scope decisions still stand: literal-topic evidence and descriptive reading
counts are shipped; a reader-rated controlled alignment study has not been run.
Full Python startup remains about 9 seconds here. The provisional list begins
after selection, not during the initial embedding pass. A renderer rewrite,
automatic bare-DOI interpretation and binary storage migration were deliberately
not added. Print styles are supplied; physical print and assistive-technology
behavior remain unverified. All existing local/private defaults remain intact.

## Next reader-experience pass — approved direction

1. Make first-use interest entry available while the Python runtime starts; keep
   drafts intact and make runtime readiness clear before submitting.
2. Offer a quick first digest with **at most 15 research papers**. Fifteen is a
   ceiling, not a quota: fewer qualifying papers means a shorter digest, and
   relevance/source-quality gates remain unchanged. New profiles remain
   Research only. Keep a broader brew available afterward. The existing Usual
   reading mode already caps non-deadline reading picks at 15; it does not reduce
   collection or scoring work. Measure any proposed candidate/time reduction
   against broad-brew topic coverage before claiming a faster first brew.
3. Explain Relevant, Save for later and Already know this with short action
   confirmations that accurately describe their different future effects.
4. Check one-handed rating/saving, article return and reading-position recovery
   on physical phones. Viewport-only checks do not establish these behaviors.
5. Make paper-title examples easier to find in basic setup. DOI lookup remains
   a later explicit action if observed onboarding difficulties justify it.
6. Observe 3–5 new readers across different disciplines using setup, brewing,
   rating, refresh and next-day return. Record hesitation, time to first useful
   paper and reader-rated useful picks; do not infer usefulness from clicks.

User clarification: “smaller should be 15 papers or less.” This is the accepted
maximum for the proposed quick first digest, not a promise that a new compute
mode has already shipped. Reader recruitment and physical-device evaluation
require actual participants/devices; they cannot be replaced by synthetic tests.

## Independent-review reconciliation — 2026-10-16

Remote tips checked read-only: browser `5153c0b`, native/main `03cab4c`. The
reported 19 main-only and 45 browser-only commits are accurate topology counts;
they do not by themselves identify missing functionality, because several shared
fixes were ported separately. Preserve both existing uncommitted documents.

| Review claim | Evidence and current interpretation |
|---|---|
| Full runtime startup still takes about 10 seconds | Confirmed. The previous live load took 10.6 seconds. Cached read-only viewing is a separate faster path. |
| Mid-brew refresh only tested with a stand-in worker | Outdated. `browser/verify-browser.mjs` runs the actual app, SQLite, browser worker and embedding cache with a controlled slow feed. It verifies exact committed DB/vector restoration, 76 cached rows, no automatic restart, and a successful explicit later brew. |
| Optimistic voting still open | Implemented. Ordered rapid intents, latest-intent UI, rollback on failure, and queue draining before navigation/handoff are covered by unit and real-browser checks. |
| Titles and scroll position still open | Implemented and browser-tested; an actual Settings round trip restored scroll from 668 to 668 and checked route titles and one main landmark. |
| Shared visual system still open | Shared tokens, stronger control borders and 12 px text floor are implemented. This establishes the code change and viewport checks, not physical-phone or screen-reader validation. |
| Precomputation partly implemented | The tested public catalog contained 4,206 vectors; a 12-text actual-browser probe reused all 12 with no new encoding. This does not quantify whole-first-brew savings or coverage of each personalized query. |
| Settings/back speed on a populated large profile unmeasured | The prior 40 MiB result timed an individual route response, not end-to-end navigation. New click-to-render measurements below close the synthetic-history gap; real-reader-device performance remains unverified. |

The new navigation experiment uses the already-built `5153c0b` application,
fictional profiles and synthetic records. It times actual Settings and Back
links through the iframe bridge, application render, document replacement and
two animation frames. It compares small and large desks with nine topics and
a full current slate of 15 research papers. The large case adds 20,000 historical
articles, 4,000 explicit ratings and feedback events, 12,000 cached vectors,
1,000 bookmarks, 90 historical digests and 5,400 impressions. Five round trips
per case; timing is descriptive, not a hardware-independent threshold.

Evidence and reproducible runner:
`.full-review/navigation-reconciliation-2026-10-16/`. No real reader data, live
brew, provider call, merge, commit, push or deployment is involved. An initial
15-card fixture attempt had an SQL placeholder-count error; its failed logs
remain separate and do not count as passing verification.

### Integration before further product development

1. Create an isolated integration branch/worktree; leave this prototype checkout
   and its uncommitted plans/evaluation document intact. Audit the native-only
   changes and duplicate ports against the merge base before resolving anything.
2. Integrate browser and native code together. Review overlapping ingestion,
   ranking, persistence, companion, settings and template changes by behavior;
   do not resolve whole files by blindly choosing one branch.
3. Move workflow changes with the integrated app. Main currently pins the
   browser branch, includes both alert jobs, and lacks the standalone reading
   and tab-handoff gates. The browser workflow checks out the event commit.
   Copying just that checkout policy onto unintegrated main is not a safe
   substitute for integration.
4. Make integrated main the production publication source. Development branches
   should run checks/produce previews without replacing the production Pages
   site on each experiment. Preserve strict scheduled refresh gates, bounded
   carry-forward and alerts.
5. Verify both native and browser behavior, full Python/Node/lint checks,
   large-state navigation/persistence, actual-app mid-brew recovery, a clean
   allowlisted build and workflow event/ref policy before merging/publishing.
6. Then pursue immediate first-use input during runtime startup, the at-most-15
   quick first digest, and controlled measurement of public-cache brew savings.
   Preserve per-interest coverage and article-quality gates; do not infer speed
   improvements merely from a shorter displayed digest.

Integration is planned here, not executed or approved by a review quotation.

Final navigation measurements (same host, five actual round trips per case):

| Desk | SQLite | Settings median / slowest | Back median / slowest |
|---|---:|---:|---:|
| Small history, nine topics, 15 current papers | 0.26 MiB | 254 / 359 ms | 342 / 365 ms |
| 20,000 extra articles, 4,000 ratings, 12,000 vectors, nine topics, 15 current papers | 57.19 MiB | 265 / 479 ms | 319 / 380 ms |

Both cases preserved the expected digest across navigation and matched tracked
checkpoints to full exports byte-for-byte. These are click-to-render measurements
after the runtime was ready, including actual link handling, frame replacement
and painting; they do not include Python startup or establish speed on the
alpha reader's machine. The earlier one-paper-current-slate measurements are
retained as preliminary evidence; the full 15-paper cases above supersede them.

Reproduction commands: `.full-review/navigation-reconciliation-2026-10-16/commands.json`.
Cohort counts, timings and checks: `.full-review/navigation-reconciliation-2026-10-16/navigation-summary.json`.
Only the plan was changed in the tracked tree during this verification pass;
production code, branch refs and deployment were not changed.

## Startup reduction and TOC figures — additional requested direction

The roughly ten-second startup is unfinished performance work, not an inherent
minimum. One recorded 8.96-second run spent 1.54 seconds starting Python,
3.15 seconds loading libraries, 0.56 seconds unpacking/importing the application,
and 3.69 seconds connecting/importing the full application stack. Asset download
caching does not retain the old worker's running interpreter across a reload.

The next startup change should expose first-use interest entry before Python is
ready, preserve drafts, and separate ordinary reading/settings from compute-only
initialization. Trace and defer numerical/ranking imports and packages to the
operations that require them; do not merely remove SciPy/scikit-learn from the
loader while web/runtime imports still depend on them. Start independent asset
work concurrently where measurements justify it. Measure first interaction and
compute readiness separately; moving work to a first brew does not eliminate
that work. Keep existing snapshot validation, feedback and single-writer safety.
No new startup time promise is made until the revised path is measured.

TOC figures / graphical abstracts are a useful optional reading enhancement.
Current RSS handling strips image tags from abstract HTML and does not retain
image metadata; the browser renderer also allows only data/blob image URLs in
its CSP. This needs an ingestion-to-display change, not just an image tag.

- Preserve article-specific publisher image URL, explicit type/provenance, and
  supplied caption/alt text when available. Distinguish a confirmed graphical
  abstract from a generic publisher thumbnail; do not relabel journal covers,
  logos, or arbitrary first figures as TOC graphics.
- Begin with feed-provided images; later support explicitly identified graphical
  abstracts in accessible publisher/JATS metadata where useful. Availability is
  source-dependent. JATS explicitly supports graphical abstract types:
  https://jats.nlm.nih.gov/archiving/tag-library/1.3/element/abstract.html
- Provide an optional compact preview, deferred until text renders, with lazy
  loading, stable dimensions, tap-to-enlarge, publisher credit/link, and a text
  fallback when images are absent or fail. Keep paper text fully usable without
  images and do not download every candidate's full-size figures.
- Validate image URLs and update CSP deliberately. Verify ordinary browser and
  multicore-isolation modes; publisher embedding restrictions must fall back
  gracefully rather than break a card or a brew. Explain external image loading
  in the image preference without adding a permission prompt for every paper.
- Carry the metadata through native and browser digests and saved reading. Image
  availability never changes relevance scores, source prestige or selection.

The subsequent “go ahead” authorized the scoped startup and feed-image work
on `codex/browser-compute-prototype`; implementation and verification follow.
Branch integration remains separate and has not been performed.

### Startup and feed-image implementation — 2026-10-16

- First-time readers can enter topics during runtime preparation, then explicitly
  transfer the draft into empty Setup rows. Existing profiles are not replaced.
- Ranking exports, numerical imports and the pipeline no longer initialize just
  to open ordinary pages. Brew/training packages load before job creation and
  support retry. An immediate Stop is retained during this preparation.
- Three isolated Chromium samples of the previous artifact measured median
  runtime readiness 9,154 ms; the final artifact measured median 6,886 ms
  (6,854–6,960 ms). The first topic input was available in approximately 116 ms. These local
  fixture measurements are not a promise for other devices or network speeds.
  A subsequent run with the final image filter and concurrent test load measured
  median 7,918 ms and topic entry at 159 ms; expect local variation, not a fixed
  seven-second startup guarantee.
- Article-feed images survive ingestion/storage and appear as optional previews
  in native/browser digests and saved papers. Generic images remain labelled
  previews. Publisher images load only when opened, without credentials/referrer;
  failed previews retain their source links. Saved reading copies store links.
  Image availability has no effect on ranking or selection.
- Normal and COEP-isolated Chromium verified success and failed-image fallback
  with mocked publisher images. This is not physical multicore/GPU verification.
- Evidence and exact command/results inventory are retained in
  `.full-review/startup-figures-2026-10-16/verification.json`. All 2,361 current
  Python cases were covered exactly once in four passing batches: 2,358 passed
  and 3 skipped. All 126 Node tests and 45 complete browser acceptance checks
  passed; the final optional image filter also passed its focused tests and
  normal/isolated browser verifier. No live brew/provider call, push,
  deployment or branch integration is part of this work.

Remaining: real publisher-feed availability/embedding policies, real phones and
other browser engines, and further reduction of Python/FastAPI startup. JATS
retrieval and a renderer rewrite are not included in this implementation.
