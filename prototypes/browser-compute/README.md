# Browser computation prototype

This branch tests whether browser ONNX inference and Python/WebAssembly can
preserve DailyDigest's ranking behavior without forwarding computation to a
server. It is an isolated experiment, not a replacement application.

The static server supplies files only. Embeddings, quality adjustments, topic
gates, identity deduplication and balanced research selection run in a dedicated
browser worker. The ranking functions are extracted from the current repository
without translating their algorithms. Small model/ORM type adapters supply
in-memory records and explicitly frozen settings.

The dataset contains all 72 independently authored fictional scope cases plus
four venue/editorial/publication-version variants. Eight example core interests
and two lower-weight context interests are public development inputs. No private
profile, production database, reading history, votes or embedding cache is used.
The fixture labels do not participate in scoring and are not reader judgments.

## Run

From the repository root:

```bash
cd prototypes/browser-compute
npm ci --ignore-scripts
npm run prepare-fixture
npm start
```

Open <http://127.0.0.1:8766> and press **Brew fixture in this browser**. Preparation
downloads public model/runtime assets once and computes the separate native
reference. `--ignore-scripts` avoids irrelevant Node/CUDA installation hooks;
the prototype uses browser CPU/WebAssembly, not Node inference.

Use Node 24, as in CI. The lockfile overrides the native-only `sharp` dependency
to 0.35.5 to pick up its patched image libraries while retaining the tested
Transformers.js inference version. The `global-agent` 4.1.3 override removes its
retired logging dependencies while retaining the bootstrap API used by the native
installer. Static publication excludes `sharp` and `onnxruntime-node`. CI blocks
moderate or higher dependency advisories before publication. Do not apply
`npm audit fix --force` without rerunning inference parity checks.

Generated model assets and dependencies are ignored by Git. They must be
prepared before serving; copying only the tracked HTML/JavaScript is insufficient.
This prototype has not been published to GitHub Pages. Relative asset URLs are
used so a future static deployment can live under a project subpath.

## What the check establishes

The browser and native reference use the same BGE-small-en-v1.5 graph and
tokenizer. Both use CLS pooling, 512-token truncation, L2 normalization, identical
query prefixes and length-sorted batches of four. Model/source hashes are included
in the downloadable report. No model inference API is called.

The BGE adapter preserves the final BERT separator when truncating, matching the
native tokenizer. The pinned Transformers.js tokenizer otherwise slices that
token off overlong inputs. The model still reads at most 512 tokens: a contribution
after that boundary can be lost in both runtimes; this prototype does not claim
to solve that existing semantic limitation.

The page reports two separate comparisons:

* **Policy portability:** reuse identical reference vectors through the browser's
  Python/NumPy runtime and compare with independently imported native production
  functions. Require matching decisions/order and score differences <= 1e-5.
* **Inference portability:** calculate new embeddings in the browser and repeat
  ranking. Require embedding cosine >= .9999, score differences <= .002, and
  identical candidate decisions and selected order.

Five controlled-vector stress cases cover low-impact/DOI handling, editorial and
off-topic rejection, sparse pools, renamed publication identity, Unicode targets,
and lower-weight context. These test policy execution, not semantic quality.

All state is in memory. Reset terminates the worker and clears the experiment.
The native vectors are diagnostic references: the ordinary browser brew uses
newly computed browser vectors, not cached native results.

## Verification

```bash
uv run pytest tests/test_browser_compute_prototype.py -q
cd prototypes/browser-compute
npm test
cd ../..
node prototypes/browser-compute/verify-browser.mjs /absolute/new-report.json
```

The browser check uses installed Google Chrome and blocks every request except
static GETs to localhost:8766. It records numerical parity, network evidence,
desktop/mobile-layout screenshots and timing. Device CPU measurements are not
equivalent to actual phone benchmarks.

## Boundaries

This does not yet implement live feeds/CORS, browser persistence, synchronized
profiles, scheduled background brewing, production learned feedback, facet
attribution/priority ordering, negative-interest centroids, enrichment,
cross-day history, opportunity eligibility, or email. Fixture papers are undated,
so freshness is neutral. The full serving pipeline is not claimed to be ported.
Parity on this fixture does not establish improved relevance or scientific
quality; existing semantic errors and tokenizer truncation limits can remain.

The next decision is to measure real mobile hardware and a larger frozen pool,
then verify source-by-source browser retrieval. Production remains the Python
application at its original localhost address.

## Full DailyDigest launch target

The target is a deployable DailyDigest application with the existing local
feature set. This fixture experiment is evidence for the browser runtime, not
the launch deliverable. Continue development on `codex/browser-compute-prototype`.

Reuse the current Python rules and existing interface wherever practical. Adapt
network access, storage, embedding execution and job progress to the browser;
do not introduce a second independently maintained ranking algorithm. The local
application and its private defaults remain supported.

| Existing capability | Launch requirement |
| --- | --- |
| Setup, weighted interests, negative interests, settings and optional sections | Editable browser configuration; preserve current validation and privacy defaults. |
| Research, funding, events, industry, AI, clinical/regulatory and world streams | Reuse retrieval queries/parsers; verify browser access for each enabled source and preserve degraded-coverage handling. |
| Ranking, venue quality, topic alignment, freshness, novelty and selection | Carry over the enabled production stages, including feedback, facet priorities and negative-interest handling; compare with the native pipeline. Experimental disabled models remain disabled. |
| Preprint/journal identity and fourteen-day cooldown | Preserve within-brew selection, same-day upgrades and retained-history behavior. |
| Full/usual/minimal sessions, explanations and extractive summaries | Preserve reading modes, recommendation evidence and the default local summarizer. |
| Ratings, reasons, training, saved reading, known items and Time Machine | Persist in browser storage with recovery and explicit import/export; protect retained items during pruning. |
| Brew progress, reconnect, cancellation and last successful digest | Worker-driven progress and safe publication; preserve the previous digest when research coverage is degraded. |
| Calendar files, display controls, accessibility, mobile layout and Pip | Transfer the existing interactions and export behavior. |
| Model/summary caches, retention, diagnostics, evaluation and backups | Adapt persistence and maintenance; verify incremental brewing and recoverability. |
| Optional online summaries, email delivery and inbound email feedback | Retain in the feature inventory; determine provider access and credential handling before implementation. Server secrets must never enter a public build. |
| Local CLI/Ollama summarizers and unattended scheduled brewing | Require a separately configured local companion or remote service; static Pages alone cannot execute local commands or guarantee work with the browser closed. |

Cross-device access to the website does not synchronize private state. A shared
state service, if chosen, requires explicit configuration; it must not silently
upload local profiles, votes or history. Source endpoints that reject browser
requests also need an explicitly chosen alternate delivery route. These are
architecture decisions to resolve, not features to silently omit or mark done.

Implement in this order:

1. Audit runtime dependencies and test browser retrieval for the configured
   source types. Record direct-access failures and decide their delivery route.
2. Transfer configuration, durable storage and the complete enabled ranking and
   summary path. Verify native/browser behavior on dated pools with feedback,
   history, version duplicates and all digest sections.
3. Connect the existing setup, digest, saved-reading and feedback interactions
   to the browser implementation. Verify reload/recovery and failed-brew behavior.
4. Resolve the service-dependent features and optional synchronization without
   changing the private browser/local defaults.
5. Build a minimal static artifact, test a project subpath and actual mobile
   hardware, and prepare the Pages workflow. Commit, push and publication remain
   separate user-authorized actions.

A full launch requires a feature-by-feature result with supporting checks.
Fixture parity alone does not establish complete pipeline parity, good research
alignment, adequate retrieval coverage or deployability of every local feature.
