# Comprehensive review and plan (2026-10-16)

Status: original audit with verified implementation follow-up below. Release
commits follow this review; GitHub Actions records publication status. It supplements
`docs/improvement-plan-2026-10-15.md` (Codex's plan) and `docs/companion-guide.md`.

Evidence labels: **V** = reproduced by running code or on the live site;
**R** = read from code, not reproduced; **S** = reported by a reviewer and spot-checked.

Scope of this pass: live site `https://antimonyleo.github.io/daily_digests/` at
commit `f9ffe25` (Chrome 152, headless; desktop, 390 px, 320 px, dark mode), a
fictional fixture backup, axe-core 4.10 accessibility scan, a read-only security audit,
and a read-only pipeline/CI audit with scratch experiments. Uncommitted work by Codex
(Guide dialog, linked backup file) was not evaluated; the live site does not have it.

## Corrections to earlier statements

- The security reviewer claimed production flips between two codebases. **Wrong.**
  `origin/main`'s workflow checks out `ref: codex/browser-compute-prototype`, so
  scheduled builds use the branch code (V). The real problem is workflow drift (E1).
- Navigation measured 150–450 ms on the small fixture profile (V). That does not
  explain the reported 1–3 s delays, which probably depend on profile size (not
  reproduced).

- A5 overstated the missing action: the empty page already had a Brew form. The
  last saved digest needed a clearer link.
- C7 needs updating: OpenAlex removed its polite pool in February 2026 and ignores
  `mailto` ([official deprecations](https://help.openalex.org/api/deprecations/)).
  NCBI identification parameters still apply.
- C12 and C13 were partly addressed before this review: percentage help disclaims
  reserved slots, and cards name matched topics, phrases and supporting excerpts.
- The large-profile navigation check already exists at
  `.full-review/navigation-reconciliation-2026-10-16/large-15/report.json`:
  57.19 MiB SQLite, 20,018 items, 4,000 votes and 12,000 embeddings. Settings median
  was 265 ms and Back 319 ms over five rounds. These are synthetic desktop results,
  not timings from the reporter's profile.

## Issue register

### A. First run and everyday use (live site)

| ID | Sev | Finding | Ev |
|---|---|---|---|
| A1 | High | On an 800 px-high window the setup page shows an onboarding box, an example selector and a label with nothing under it. The first interest field sits under the sticky Save bar, partly hidden. "Load selected example" wraps onto its own line. | V |
| A2 | Medium | The settings page for a returning reader repeats first-run text ("Start with a few research interests… No account is needed", "Loading an example is available only while interests are empty"). | V |
| A3 | Medium | Jargon: "Current brew / Previous brew / Time machine", "How are we feeling today? Energetic / Usual / Tired" (these set digest size), "reading desk". | V |
| A4 | Low | The feedback panel shows "0 helpful ratings… 0 useful of 0 rated picks… 0 unrated" before any rating. | V |
| A5 | Medium | With no digest for today, the empty card has no Brew button and the last digest is hidden behind small date chips. | V |
| A6 | Low | Storage wording is inconsistent: "this computer" vs "this browser" vs "this browser profile". | V |
| A7 | Low | Restore uses a native `confirm()` and no progress text. Headless tools dismiss it, so the import looks like it silently does nothing. | V |
| A8 | Medium | Axe finds duplicated `<main>` landmarks on the Saved reading page and one serious contrast failure on the digest page. | V |
| A9 | Medium | Startup is about 9 s. Stage labels are better than before. | V |
| A10 | Medium | Reported slow navigation (Settings ~1 s, back ~3 s) is unexplained on a realistic large profile. | not reproduced |
| A11 | Low | The floating "Display" button overlaps content at the right edge. | V |

### B. Data safety

| ID | Sev | Finding | Ev |
|---|---|---|---|
| B1 | Medium | A normal restore keeps no rollback copy. With a linked auto-copy, a later edit overwrites the only copy of the replaced desk (`runtime.py:415-427`, `app.mjs:900`). | R |
| B2 | Medium | Closing the tab mid-brew loses ingest and embedding work; the notice says it "will reuse saved calculations" (`worker.mjs:319, 346-368`). | R |
| B3 | Low | Recovery copies (including credentials) accumulate without limit (`worker.mjs:176-181`). | R |
| B4 | Low | Auto linked copy turns itself off after a browser restart (`app.mjs:674-682`). | R |
| B5 | Low | A failed IndexedDB write after a linked-file write permanently blocks later writes (`linked-backup.mjs:94`). | R |
| B6 | Low | File sizes are read before the size check (`app.mjs:1035, 1114`). | R |

### C. Retrieval and ranking correctness

| ID | Sev | Finding | Ev |
|---|---|---|---|
| C1 | High | PubMed dates use the print `PubDate`, not `ArticleDate`. E-published papers get a future or month-start date and are dropped before ranking (`ingest/pubmed.py:374`; no `ArticleDate` in the file). | V, S |
| C2 | High | `filter_english` runs langdetect on short title-only text and drops English technical items ("DNA origami nanostructures…" → `da`, "Perovskite solar cells" → `sv`). 3 of 15 titles dropped; no test; no drop count in the audit (`dedupe.py:579-599`). | V, S |
| C3 | High | arXiv fetches at most 50–400 newest items with no pagination and no "limited" signal. Health shows "Responded". | V, S |
| C4 | Medium | One failing OpenAlex term aborts the whole source and discards earlier terms; 4xx responses are retried 3 times (`openalex.py:215-218, 300-305`). PubMed tolerates per-term failure. | V |
| C5 | Medium | `downweight=["AI"]` matches as a substring ("certain", "main"); the same bug produces false "Why read: connects to your interest in AI" text (`ranker.py:526,690`, `summarize.py:65-70`). | V |
| C6 | Medium | Source-health history is the last 7 runs, not 7 days, so a failure disappears after 7 re-brews (`health.py:70`). | V |
| C7 | Medium | Browser requests lose the polite-pool email: it lives in `User-Agent`, which browsers strip (`network.py:18,89`). | R |
| C8 | Medium | Email vote labels (`+R3`) can point at a different paper after a same-day preview re-brew (`pipeline.py:1870`, `store.py:1653-1667`). | R |
| C9 | Low | OpenAlex recall caps: 20 per term (newest, not relevance), 120 per venue group. | R |
| C10 | Low | Author resolution takes the top search hit only (`openalex.py:522`). | R |
| C11 | Low | Profile-load failure silently falls back to generic queries that still pass the personalization gate (`_terms.py:35-45`). | V |
| C12 | Low | Topic percentages are ordering-only (≈0.045 score between 20 % and 80 %); documented, but the UI promises more than that. | V |
| C13 | Low | "Why shown" text is threshold boilerplate ("Reliable source", "Strong topic match"). Partly offset by the "Shown for <topic>" line. | R |
| C14 | Low | Cross-day dedupe windows differ (14/7/30 days) so a paper can reappear through a second source. | R |

### D. Summaries

| ID | Sev | Finding | Ev |
|---|---|---|---|
| D1 | Medium | Extractive summary leaks section labels ("METHODS: …"), omits the conclusion, has no length cap (7.5k chars seen), splits at "Fig. 2", and uses the bare title when there is no abstract (`summarize.py:73,159,177-206`). | V |
| D2 | Medium | The model prompt demands numbers from an empty abstract with no "no abstract" instruction, so hallucination is likely; non-string fields are stored as `str(dict)` (`summarize.py:226,296`). | V (parsing), R (model behaviour) |
| D3 | Low | Codex CLI isolation is a feature denylist; `view_image`, `hooks` etc. stay enabled. Claude's path is locked down. | R, unverified |

### E. Deploy, CI and tests

| ID | Sev | Finding | Ev |
|---|---|---|---|
| E1 | High | `main` and the branch have diverged (main 19 ahead, branch 46 ahead). Scheduled runs use main's workflow file, which checks out the branch by name; the branch's copy uses `github.sha`. If the workflow reaches main before `browser/` does, the 12 h cron builds native-only source. Nothing pins this. | V, S |
| E2 | Medium | Refresh counts as a success with only 50 % of sources fresh; no issue opens (`collect_sources.py:127,162`). | R |
| E3 | Medium | No `timeout-minutes` on any job; the data refresh is coupled to the full test suite and model download. | R |
| E4 | Medium | GitHub disables cron workflows after 60 days of inactivity; the failure-issue path cannot fire for a run that never starts. | R |
| E5 | Low | Pyodide pin `pydantic-settings==2.8.1` vs tested `2.14.2`; no dependabot or audit step. | V |
| E6 | Low | Node tests and browser checks run only on push, not on PRs. Fixed sleeps in several verify scripts; `--test-isolation=none`. | R |

Behaviours with no test: real `filter_english`; PubMed `ArticleDate`; arXiv cap; OpenAlex
mid-list failure; extractive summary edge cases; substring matching; 7-day health;
mid-brew tab close; workflow ref/triggers; email label stability.

### F. Companion

| ID | Sev | Finding | Ev |
|---|---|---|---|
| F1 | High | Setup is not discoverable: the form is inside "Data & compute", the instructions are a paragraph, and Chrome's local-network block gives only "Failed to fetch". Not yet fixed in the live app. | V |
| F2 | Medium | A scheduled Companion brew erases the browser's saved credentials on the next "Restore companion state" (`companion.py:245`, `runtime.py:310`). | V |
| F3 | Medium | Each scheduled brew leaves a plaintext workspace (`store/native-reader-*`, including pushed credentials) that is never deleted (`companion.py:188`). | V |

### G. Security hardening

| ID | Sev | Finding | Ev |
|---|---|---|---|
| G1 | Medium | The site shares the origin `antimonyleo.github.io` with sibling Pages sites. Any script there can read IndexedDB (API keys, Companion token) and call the Companion. Latent today. | R |
| G2 | Low | The iframe sandbox has `allow-scripts allow-same-origin`, so it is not a real boundary; the CSP allows `'unsafe-inline'`, `img-src https:`, postMessage target `'*'`. Output escaping is sound today (no XSS path found). Chrome logs a console warning about it. | R, V (warning) |

Checked and found sound: Companion token comparison, Origin check, body caps, no
command injection, backup path validation, import rollback, credential stripping in
exports, SHA-pinned actions.

### H. Not verified

Real GPU; Safari, Firefox, Windows, macOS; the Chrome local-network prompt in a normal
(headed) window; real Claude, Codex and API providers; email and scheduler; a large
profile's navigation speed; Codex's uncommitted Guide and linked-backup work.

## Plan

Order is by user impact and risk. Each phase ends with its acceptance checks. Ownership
follows `AGENTS.md`: one writer per file set.

### Phase 0 — decisions (owner)

1. **Deploy branch.** Merge `main` and the branch, or make the branch the default
   branch. Until then freeze workflow edits. (E1)
2. **Companion position.** Keep it in the footer menu, or move it under Guide/Data &
   compute. Recommended: keep it only under Data & compute, with a one-line pointer from
   the Guide, and recommend the local app for Claude/Codex summaries.
3. **Digest-size wording.** Replace the mood labels with "Short / Standard / Full".
4. **Dedicated origin** for the app, or accept G1 and warn in Companion setup.

### Phase 1 — silent data loss and wrong results (correctness first)

1. C1 PubMed `ArticleDate`: prefer the electronic date, fall back to history, set
   `None` for month-only or future dates. Tests: print date in the future; month-only.
2. C2 `filter_english`: need ≥ 200 characters or a high-confidence `detect_langs`
   result; otherwise keep; count drops in the funnel audit. Test with the 15 titles.
3. C3 arXiv: paginate until older than the window; otherwise emit `truncated` and show
   "Limited results". Fix compound categories.
4. C4 OpenAlex: per-term failure handling, no 4xx retries, fail only if all core terms
   fail.
5. C5 word-boundary matching for `downweight` and `_matched_interests`.
6. C6 health keyed by day.
7. F2 and F3: carry `credentialsIncluded` through the scheduled rewrite; delete the
   workspace in `finally`.
8. B1 keep a pre-restore recovery copy (last N, with pruning, B3); pause linked auto-copy
   after a restore from another source.
9. B2 checkpoint after ingest and during embedding, or correct the message.

Accept: new tests for each item fail on the current code and pass after the fix; full
pytest and `node --test` stay green.

### Phase 2 — first impression (browser UI)

1. A1 Setup: put the interest field first; reduce the onboarding box to one line plus a
   Guide link; make the Save bar a normal block (or sticky outside the scroll area);
   keep "Load example" on one row. Check at 1280×800, 390 and 320 px.
2. A2 separate first-run and returning copy; A6 one wording ("this browser").
3. A3 rename nav and digest-size labels; A4 hide zero-state feedback stats.
4. A5 empty-today card gets a Brew button and shows the last digest by default.
5. A8 one `<main>` per document; fix the one contrast failure; A11 move "Display".
6. A7 replace the `confirm()` with an in-page dialog and a progress line.
7. A9 keep stage labels; measure again after any change.
8. A10 build a large synthetic profile (≥ 40 MB), time Settings and Back, and profile
   the long task (snapshot, iframe `srcdoc`, template render).

Accept: screenshot checks at three widths in light and dark; axe zero violations on
setup, digest, saved, settings; typed-topic carry-over still works.

### Phase 3 — Companion that people can set up

1. F1 numbered 3-step panel with a copyable command, token location and "Test
   connection". Detect Chrome's local-network block and say what to allow. Show the form
   only in the Companion section. Say what leaves the computer.
2. Reuse `docs/companion-guide.md` for the in-app text; link it.
3. Test the Chrome prompt in a headed window; check Firefox and Safari behaviour.
4. G1: add the shared-origin warning to the panel.

Accept: a person who has never seen it connects in ≤ 5 minutes (observed test).

### Phase 4 — summary and relevance quality

1. D1 extractive summary: strip labels, prefer results/conclusion, cap ≈ 400 chars,
   fix "Fig." splitting, say so when no abstract exists.
2. D2 no-abstract flag in the prompt; validate the three-field output.
3. C7 send `mailto`/`tool` as query parameters; C9 relevance-sorted OpenAlex pages.
4. C13 name the matched topic and phrase in "why shown"; C12 reword the percentage help
   ("guides order; it does not reserve slots").
5. C8 bind email votes to the sent digest's `run_id`.

### Phase 5 — deploy and CI hardening

1. E1 per Phase 0, then add a test pinning checkout ref and triggers, and a guard that
   `browser/build.py` exists.
2. E2 warn below ≈ 80 % fresh sources; E3 `timeout-minutes`; decouple the data refresh
   from the full test suite; E4 add a staleness check that does not depend on the cron.
3. E5 align the Pyodide pin or document the mismatch; add dependabot and an audit step;
   E6 run node tests on PRs and replace fixed sleeps with waits for state.

### Phase 6 — hardening and housekeeping

G2 per-render CSP nonce, narrower `img-src`, `postMessage` to the exact origin. B4–B6
small fixes. C10, C11, C14 as time allows. D3 switch Codex to a no-tools profile if one
exists. Decide the fate of `docs/evaluation-2026-09-11.md`.

### Phase 7 — evidence on real systems

A real GPU, Safari, Firefox, Windows, macOS, headed Chrome with the local-network prompt,
one real Claude or Codex summary, and the scheduler. These are manual checks; do not
claim them from headless runs.

## Suggested sequence

Phase 0 decisions → Phase 1 (retrieval correctness is invisible to users and silently
lowers digest quality) in parallel with Phase 2 (first impression) because they touch
different files → Phase 3 → Phases 4–5 → Phase 6 → Phase 7.

## Implementation follow-up

The owner selected execution of the revised plan and the recommended product
choices. They subsequently chose to keep GitHub Pages because no separate domain
is available. No production data, live providers or email services were used.

### Decisions and integration

- Merge native and browser history together. Local merge `e795385` brings
  `origin/main` into `codex/browser-compute-prototype`; the historical merge adds
  documentation and native regression tests. Its production files match the
  browser branch. Guide and review fixes are committed separately for release. The dirty
  local `main` checkout was not changed. Nothing has been pushed or deployed.
- Keep Companion under Data & compute, with a Guide pointer. Recommend the local
  app for signed-in CLI summaries.
- Use Short / Standard / Full. Internal values stay `minimal/usual/full` so saved
  choices and links continue to work.
- Keep Pages for now. Explain the shared-origin credential risk in Companion
  setup. A project path and a different browser database name do not isolate
  credentials from other scripts on the same origin.
- Leave `docs/evaluation-2026-09-11.md` untracked and untouched.

### Correctness and data safety

| Issues | Implemented behavior |
|---|---|
| C1 | Prefer complete, valid electronic publication dates, then publication history and print dates. Unknown days and future dates stay unknown. Normal enrichment can repair earlier guessed dates on the next requested ingest. |
| C2 | Keep ambiguous short technical titles. Reject only confident non-English long text. Record language drops in progress and the candidate audit. Short non-English titles can also pass this conservative filter. |
| C3 | Paginate arXiv politely within 400 items/eight pages, normalize compound categories and disclose capped or incomplete coverage. This does not promise exhaustive arXiv retrieval. |
| C4, C9–C11 | Retain successful OpenAlex terms after another fails; retry transient errors; fail when all core terms fail. Sort personalized searches by relevance within the date window. Require an unambiguous author-name match and disclose unresolved names. Unreadable profiles fail visibly. |
| C5 | Use whole terms and phrases for penalties and literal interest explanations; “AI” no longer matches “certain” or “main.” |
| C6 | Retain every source check for seven UTC calendar days. Daily aggregation was rejected after tests showed it discarded query diagnostics. Repeated brews no longer erase earlier failures. |
| C7 | Send NCBI `tool` and configured `email` parameters. Existing Crossref `mailto` remains; no OpenAlex polite-pool claim. |
| C8, C14 | Store immutable confirmed-email labels by edition. Replies need the sent edition tag; older untagged replies remain unresolved instead of being applied to a later preview. Preserve sent items during pruning and exact-item suppression. The distinct 30/14/7-day policies remain. |
| B1, B3 | Successful restores atomically retain the previous committed desk, capped at three recovery copies. Numeric, monotonic ordering protects the new copy if the clock moves backwards. Storage failure leaves the original desk intact. |
| B2 | Correct the interruption notice: unsaved brew calculations may run again. Periodic mid-brew resume is not implemented or promised. |
| B4–B6 | Preserve automatic-copy intent through temporary permission failures, recover after a committed file write followed by metadata failure, reject oversized files before reading them, and serialize file selection against copying. External restores pause automatic copies; rejected restores recover the previous preference. |
| F2, F3 | Scheduled state preserves credential-omission metadata. Cleanup removes only the workspace created by that invocation, including failure paths. Existing old workspaces are preserved for a separate recovery audit. |
| D1, D2 | Remove structured abstract labels, preserve figure references, prefer stated outcomes, cap extracted findings at 400 characters and disclose missing abstracts. Reject non-string or incomplete provider summaries and retain fallback provenance. |

### Interface and Companion

A1–A4, A6–A8 and A11 have focused fixes: editable interests first, examples and
compact storage help below, returning-reader copy, plain labels, meaningful feedback
states, an in-page restore confirmation with progress, one main landmark and Display
controls in normal flow. A5 now gives an explicit latest-saved-digest link while
keeping Today’s digest on today. Axe found a second date-label contrast issue during
verification; that was fixed too.

Companion setup now has numbered steps, a copyable command using the current
origin, Connect and test, and useful connection-failure advice. A fetch failure does
not prove Chrome blocked local-network access, so the message lists the likely
causes without claiming to detect the permission state. The five-minute newcomer
setup target and headed permission prompt still need observation.

A9 remains a runtime cost on a first full startup. Three final-build samples took
8.96–9.22 seconds; the topic draft accepted input after 226 ms. The topic draft and
cached reading view provide earlier interaction; this work does not replace Python
or promise that full initialization is instant. A10's earlier large synthetic navigation benchmark is retained above. The final
storage build also passed a 40 MiB consistency check: saved-view refreshes took
133–304 ms without starting Python; a route returned in 99 ms while its checkpoint
was held. These desktop timings were collected during other verification work;
performance on the reporter's actual profile remains unknown.

### CI and security

- Add bounded job timeouts, an event-SHA source-tree guard and Node checks on PRs.
  Preserve release browser acceptance and minimum collection gates. Emit a visible
  build warning below 80% fresh coverage without pretending a partial refresh failed
  its existing 50% publication gate.
- Replace fixed synchronization delays in Guide, saved-view and tab-handoff checks
  with waits for committed state, a running save or the actual handoff status. Slow
  fixture responses and bounded timeout checks remain intentional.
- Keep full release regression checks on scheduled builds for now. Splitting source
  refresh into a fast publication path needs a tested-commit/artifact gate; simply
  skipping tests could publish code whose earlier checks failed. That optimization
  is deferred rather than weakening this release.
- The existing on-open source-age check does not depend on cron running. A disabled
  GitHub schedule still cannot create its own failure issue. An external uptime
  monitor is not configured; repository owners must also watch Actions.
- Keep the Pyodide-compatible settings pin: both settings versions declare
  `pydantic>=2.7`, and the compiled Pyodide lock supplies 2.10.6. Add weekly dependency
  update configuration and a non-blocking Node advisory report. The current npm
  audit reports six advisories in installed dependencies, including native packages
  excluded from the static allowlist. They are not accurately described as all
  development dependencies. A major inference-library upgrade needs separate
  compatibility testing.
- Render trusted frame scripts with per-document nonces, restrict figure image
  origins to the current document and authenticate messages with exact origins.
  The same-origin iframe remains part of the app, not a security boundary.
- Local Codex help does not establish a universal no-tools mode. The guide now
  distinguishes Claude's disabled tools from Codex's read-only sandbox and selected
  disabled features.

### Verification and remaining evidence

Exact commands and reports are retained under
`.full-review/claude-review-2026-10-16/`. Final full-suite and browser acceptance
results are recorded in `verification.json` and `verification.md` there. Earlier attempts are retained too:
one browser fixture crossed UTC midnight, and the first health rollup failed its
diagnostic-preservation tests. Neither attempt is counted as a passing release run.

The first complete viewport audit passed all 18 route/theme/width combinations
(setup, digest and saved; light/dark; 1280, 390 and 320 px), with zero axe violations,
one main landmark and no horizontal overflow. Guide and backup checks passed with
real origin-private file handles; the native file chooser was replaced for testing.
A byte-equality mismatch in an earlier browser run did not reproduce in the exact-build
replay. Investigation separately proved a cold SQLite WAL defect: opening or closing
a restored database could change its bytes outside the tracker. The fix invalidates
private database/WAL/journal caches on physical connection lifecycle changes and
explicitly closes both migration-backup handles. A real cached-encoder regression
failed before this fix and passes afterward; pooled read-only routes remain clean.
The final-build acceptance run passed all 45 checks, including 159 byte-for-byte
route checkpoints. Guide/backup (9), saved-view (5), tab-handoff (8), startup/figures
(4) and viewport/accessibility (18) checks also passed against the final build. This does not establish the exact cause of the
original one-off mismatch.

The final copy review also corrected the empty-data confirmation: it now says
“Start fresh,” while ordinary restores retain “Restore.” Cancel remains the default.
The new regression covers both actions.

Final build: `a4dc938aa8738cee`. The final full suite passed **2,540 tests**, with
three skips (Hatchling unavailable and two Windows-only PowerShell checks). All
144 browser Node tests and five prototype Node tests passed. Ruff F/B checks and
`git diff --check` passed. The build inventory contains 81 files, no private paths
or credential-pattern candidates, and no source-hash drift. The optional GPU model
(133,093,490 bytes) is the only file over 100 MiB.

The full test commands, reports, artifact inventory and prior failed attempts are
retained in `.full-review/claude-review-2026-10-16/verification.md` and adjacent
files. The historical merge and subsequent Guide/review fixes are separate release
commits. `docs/evaluation-2026-09-11.md` remains untracked and unchanged.

Still unverified: live Pages behavior after publication, real hardware GPU and
multicore performance, Safari/Firefox, physical phones, Windows/macOS launchers,
headed Chrome permissions, native file pickers, cross-browser OS-file concurrency,
live providers, real email and scheduled production brews. Offline retrieval fixtures
show the intended mechanics; they do not establish live personalized recall or
research usefulness for every reader.
