# DailyDigest UI integration

The visual baseline is revision 8 of the previously reviewed refinement demo,
carried into the real application. Later A/B/C/D studies used an older production
layout and are not the visual authority. The reader's subsequent request for
slightly more visible context is retained: finding, relevance and caveat remain
separate when supplied.

Release acceptance requires all of the following in the actual application:

- The original green ambient background, Figtree, readable type and consistent
  controls on Digest, Saved and Settings.
- The named/time-based greeting, Standard/Brew at the left and Saved/Settings at
  the right. A 48px Brew button; date selection shares the section navigation.
- Section navigation follows the reading position; digest search combines with
  topic, source and rating filters and preserves their URL state.
- Source/type/date metadata, supplied finding/relevance/caveat and linked
  commentaries survive rendering. Copy summary works or provides selectable text
  when clipboard access is denied. Images remain opt-in and deferred.
- Settings presents compact topic names, inline percentages and one active topic
  editor; advanced groups disclose their current values. The full real form,
  identity metadata, untouched weights, validation and provider binding survive.
- Pip uses the original cup, facts/jokes and leaf sequence. Its note remains
  visible during pointer/touch/keyboard movement, and the bubble stays within the
  viewport. No repeated Tea break/Move/Add leaf/Close text controls.
- Cached reading retains the same type/background and article hierarchy without
  waking Python or enabling article scripts. Its disposable projection may be
  regenerated from the authoritative saved desk; user data is retained.
- Actual desktop/mobile screenshots are compared with the prior baseline;
  functional tests alone do not establish visual preservation.

This work changes presentation and interaction. Ranking/retrieval model source
and quality thresholds stay at the reviewed release. All state-changing checks
use isolated fictional stores and mocked external services.

Implemented in the production Jinja templates, browser reading projection and
native routes. The fixture-only demos are excluded from the release archive.
Figtree is bundled as a small inline WOFF2 with its OFL license; font loading
makes no external request. Cached reading keeps its script-free security policy.

Saved removal now keeps a durable Undo entry in place, with counters changing
only after the request succeeds. Settings drafts receive navigation protection. Returning from Settings also
restores the About disclosure before restoring the reading position.
The restored Pip on Settings/Saved previews the real note deck without updating
the rotation ledger; the reading route retains its existing recording behavior.

The source freeze verifies 43 ranking, retrieval and publication modules against
the previous release. Their source hashes match. The built archive contains 108
allowlisted source files and no reader data. The public snapshot covers all 71
configured sources (70 refreshed, one retained valid snapshot).

Verification evidence is retained locally under
`.full-review/ui-production-integration-2026-10-23/` and the focused UI reports.
The final release checks and their results are recorded below before commit.

Design review covered Digest, Saved and Settings at 1440, 390 and 320 pixels in
light and dark mode. The mechanical template scan reported one intentional
exception: the original Pip steam is drawn with a short curved green border.
It is illustration geometry, not an accented card. Parsed-page and interaction
checks use the rendered Chrome tests rather than treating a template scan as
accessibility evidence.

The integrated production templates and these acceptance checks are now the
baseline for later UI work. Revision 8 documents the origin of the design;
future prototypes should start from the integrated templates. The CI reading
check asserts the actual loaded Figtree face, gradient background, text hierarchy,
48px Brew and Saved/Settings order at all three widths. On a narrow 320px screen,
Saved/Settings share a second row aligned to the right. Legacy summaries without
a Key finding field retain their full original text without repeated context.

Final candidate browser acceptance: 51/51 scenarios passed, plus 5/5 semantic
bridge checks, with no page errors. Tests confirmed actual Brew and interrupted
reload recovery, raw priority preservation, draft navigation rejection/acceptance,
Saved durable Remove/Undo, and About reading-position restoration.
The mocked local-model Brew took about 3.0 seconds; this small acceptance fixture
is functional evidence, not a benchmark for a researcher's full workload.
Physical GPU execution was unavailable on this host; visible CPU fallback passed.

Native focused verification: 186 affected tests passed; the final web module
passed 87 tests after the legacy-summary fix. Two older percentage helper tests
needed new extraction boundaries and the intended raw-weight preservation
expectation; their full personalization module passed 23 tests after adaptation.
Browser unit checks passed 259 tests, and the compute prototype unit checks passed 5.
Lint and dependency audit passed. The exact committed revision must pass the full
CI suite and release browser gates before Pages deploys it.

A final compact-view regression was found during source review: the new cup layout
overrode the saved hide preference with equal CSS specificity. The visibility
rule now follows presentation rules. The real Pip browser suite additionally
checks computed hiding, zero layout rectangles, reload persistence and restoration
in compact mode; it is included in Pages CI alongside reading and Saved checks.

The complete local Python run finished with 3093 passes, 4 environment skips and
the 2 obsolete percentage-helper failures described above. Both failures were
then rerun successfully (2 passed); the full 23-test personalization module also
passed. The four skips were the optional wheel builder and Windows-only checks.
The final UI fixes and helper adaptations landed while that run was in progress,
so it is not recorded as an unchanged-source full-suite pass. Exact-final CI is
the release authority, including native Windows/macOS checks.

All 13 release browser suites completed successfully after adapting the remaining
Settings entry paths and promoted-rationale assertions. This includes 20 complete
daily Brew scenarios, cached reading, large-checkpoint startup, two-tab handoff,
CPU startup, optional figures, reading fidelity, 9 Pip cases, 18 Settings cases,
and the reading/Saved page checks. The shared Pip counter now uses the configured
digest day on Settings/Saved as on Digest, rather than browser UTC; its regression
and the full Settings interface module passed 31 tests. Final packaged-source
verification found 108 matches and no changes in the 43 algorithm modules.

The first publication attempt passed exact-commit native/unit CI but was blocked
by a real mobile Pip click failure. A short note could remain over the leaf after
responsive movement. The reduced-motion reset unintentionally gave positioning
containers a tiny transition, and bubble fitting observed only note dimensions.
The follow-up scopes `transition-property: none` to Pip's positioning root and
bubble, and observes cup/leaf sizes too. Authored descendant animations remain.

The deterministic regression uses the failed supplied joke and longest bank
note, real visible-cup grips and active-drag assertions. Both font-ready and fresh
DOMContentLoaded runs pass 10/10 cases, including 16 touch positions and 144 leaf hit
points per run, with real leaf clicks. No forced clicks or arbitrary waits were
added. The native interaction module passes 39 tests. Full final CI remains a
mandatory deployment gate.

The next exact-commit CI run passed Pip and daily reading, then exposed an early
startup navigation race: visible header buttons could be clicked while CPU
preparation still awaited, before their modal handlers were bound. An automatic
CPU startup reload could also discard that click. Backup, Guide and Data & compute
now acknowledge the latest request immediately and replay it after handlers bind;
only the section name is retained in tab storage when that reload is necessary.
It is cleared after use. Held-CPU and actual-reload tests verify all three entries,
latest-action coalescing and one-time consumption. All 12 Guide/backup cases and
all 259 browser unit tests passed locally after the fix.

The remaining CI keyboard failure was reproduced with CI's Chrome 155.0.8059.39;
local Chrome 152 passed the same 51 scenarios. Enter opened Balance priorities, but
the native queued `toggle` event had not yet added its visibility class when the
test made an immediate `isVisible` check. The event arrived 15.8 ms later and the
control displayed correctly. The verifier now waits for actual visibility before
the unchanged assertion, with the same 30-second deadline. All 51 scenarios remain.
The exact 155 mobile sequence passed 20 consecutive runs after this test-only fix;
product behavior was unchanged. Diagnostics now include browser version, named
assertions, stack and focus state so any future CI failure is directly locatable.

The integrated production UI deployed successfully at `713edb6`: all Pages and
cross-platform checks passed, including 3095 Python tests (5 documented skips),
259 browser unit tests and all release interaction suites. Live verification
matched 108 packaged sources and the frontend to the commit, with no private data.

Final live screenshot inspection then caught a mobile editing defect that
overflow checks alone missed: the topic input was capped at 90 px by a legacy
descendant selector intended for the priority field. Restricting that cap to a
direct child preserves the inline percentage field while allowing the topic name
to fill its grid column. Six additional real-template Chrome cases cover empty
and existing-topic editors at 1280, 390 and 320 px. Before the fix, all four mobile
cases failed the column-width assertion (90 px versus 196/126 px available);
afterward all six passed, alongside the original 18 Settings scenarios. The
affected reading-page suite also passed all 12 checks, including dark mode.

Two CI attempts at `f0f240e` passed every preceding release gate, but the first
diagnostic startup screenshot timed out before the 51-scenario suite could run.
The same revision passed all 51 scenarios and all 5 semantic checks locally on
Chrome 155. A separate capture probe reproduced a screenshot protocol failure
once in ten attempts; all ten applications still reached Settings without page
errors. CPU isolation can automatically reload the page before Worker creation,
so capturing before that boundary races document navigation. The verifier now
waits for its already-instrumented actual app Worker before the existing visible
cup check and screenshot. It retains the 30-second timeout, full-page capture,
font waiting, reduced-motion assertions and all 51 scenarios. Ten fresh Chrome
155 processes passed this gate with screenshots taken before Python readiness
(180–1347 ms per capture). No application code changed for this test correction.
