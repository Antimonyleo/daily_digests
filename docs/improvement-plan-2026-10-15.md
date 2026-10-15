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
