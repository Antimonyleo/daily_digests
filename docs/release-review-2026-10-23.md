# DailyDigest release review — 2026-10-23

Status: local release acceptance passed; publication proceeds through the GitHub
Pages workflow and its independent release checks. This report contains no reader profile, ratings or
private study responses.

## Release scope and comparison

The published baseline is commit `be7a3e1911ca95bf12a871f7cf2ed6d2db418849`,
build `d12b4fed5952b566`, at
<https://antimonyleo.github.io/daily_digests/>. The candidate retains the existing
reading desk, gradient, Pip, greetings, local summaries and private browser state.

Publication requires the candidate to preserve existing workflows, correct the
confirmed defects below, pass the full regression and real-browser checks, and
build with a currently valid public source snapshot. The deployed build identity
must then match the successful GitHub Pages artifact. Live paid providers, inbox
delivery and universal hardware/browser performance are outside the fixture
verification; their user-facing configuration and failure paths are checked.

## Confirmed defects and fixes

| Area | Defect | Change |
| --- | --- | --- |
| CI bootstrap | New Pyodide tests ran before installing their locked dependency, or without installation | Install browser dependencies before tests in both workflows |
| Reading preservation | Richer feed text could clear a previously published summary before a cancelled or failed Brew | Invalidate reuse signatures while retaining readable text/provenance until successful publication |
| Retractions | Metadata-only retractions retained confident summaries and ordinary research labels | Clear obsolete claims and display explicit retraction/withdrawal status, retaining records, bookmarks and votes |
| Settings priorities | Adding interests could replace saved priorities with equal shares; rounding and mode changes could drift | Preserve custom proportions and make equal-share distribution consistent across add/remove/reset |
| Settings validation | A collapsed priority editor hid the invalid field and prevented Save without a useful focus target | Reveal the editor synchronously before native validation focuses the field |
| Feedback durability | Some controls acknowledged pending browser writes as saved; Saved Remove could reload prematurely | Keep pending states truthful and complete them only after a durable checkpoint |
| Calibration | A completed training job could fail without a visible explanation | Show the failure, retain the previous calibration and allow retry |
| Followed researchers | A surname-particle heuristic combined two separate authors | Require stronger evidence before joining an inverted name |
| Commentary lookup | Specific commentary genres skipped available directed relationship lookup | Use canonical publication genres within the existing request/time budget |
| Private exports | Reader-response JSON could enter Git staging or Docker context | Exclude the export filename family while preserving local files |
| Release coverage | The complete daily-use matrix was absent from clean CI and defaulted to local historical fixtures | Run it with generated isolated fixtures; add the Settings interaction gate |

## Feature coverage

The release matrix covers initial setup and examples, topic weights, research
context and followed researchers, optional spelling/provider configuration,
section controls, all reading modes, search and filtering, ratings/reasons,
bookmarks, opportunity/calendar controls, article details and optional figures,
publication labels and same/next-day companion links, Pip, display and keyboard
controls, cached reading, startup and computing choices, cancellation/retry,
backups/recovery, linked local copies, tab handoff, Companion configuration and
mocked explicit email/state-transfer actions. External services are intercepted
with fictional data; runtime, ranking, rendering and browser storage run normally.

Focused additions check custom priority preservation and hidden invalid inputs at
1280, 390 and 320 px, pending feedback/checkpoint failures, and the remaining name,
Saved pagination and calibration controls.

## Final verification and publication

The local release candidate is build `45597caa11ca874f`; all 103 packaged source
hashes match the reviewed working tree. The previous site's 89 packaged source
hashes all match the baseline commit above.

Completed checks:

- Full Python regression: 3,090 passed, four skipped, three dependency
  deprecation warnings, 1,247 seconds. All 376 frozen production/test/configuration
  inputs were unchanged throughout the run.
- 250 browser JavaScript unit tests and five prototype tests passed.
- Full browser acceptance: 51/51; semantic runtime/native parity: 5/5.
- Daily-use matrix: 20/20, including Standard/Short/Full Brew, interruption,
  source failures and same/next-day publication companions.
- Ten further browser suites passed: cached shell, guide/backup, saved reading,
  storage startup, tab handoff, CPU startup, figures, reading fidelity, Pip and
  Settings priorities. Feedback controls also passed 16 focused click cases.
- The final real-model CPU replay covered 505 candidate decisions. Its existing
  positive-sample, eligibility and topic-score guards passed. These are reused
  development/provisional evaluations, not an independent accuracy estimate.
- Ruff `F,B`, patch whitespace and npm dependency audit passed; the audit reported
  zero vulnerabilities.
- A real wheel was built separately and checked: all 17 templates and default
  sources match, and no private data or reader responses are included. This
  covers the optional wheel-builder test skipped in the locked development
  environment; the other three skips require Windows.

The first browser runs exposed stale verifier assumptions about disclosure
placement, conservative publication labels, cached-reading schema and project
subpath storage. Those assertions were corrected to the current contracts and
the complete affected suites rerun; no failing run was treated as a pass.

On this machine over localhost, three cold-start samples reached the reading desk
in 4.98–5.23 seconds; the topic input accepted typing after 152 ms. A 40 MiB saved
desk reappeared in 239–347 ms without starting Python. These measurements exclude
internet download latency and are not promises for every CPU or network.

Public source collection passed with 71 usable sources,
including 69 freshly collected and two valid retained records. Original
collection times are preserved. The previous scheduled deployment had correctly
stopped when arXiv coverage was missing; the release does not relax that gate.

The candidate passes the release boundary: the baseline's demonstrated author
false matches are corrected, the confirmed reliability and presentation defects
above are covered by regressions, and existing acceptance and ranking guards
remain green. This establishes an improvement for the verified workflows; it
does not establish universal recommendation accuracy or hardware performance.

Deployment must use the successful `main` workflow artifact. Its fresh snapshot
can produce a different build ID from the local candidate; the live manifest's
103 source hashes must match the committed sources, and its build ID must match
the uploaded Pages artifact before publication is reported as verified.

Private `data/`, credentials, reader-study exports and local review/demo artifacts
are excluded from the commit and public site. Optional provider and hardware
limits remain documented in [the browser guide](../browser/README.md).
