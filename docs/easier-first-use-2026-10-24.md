# Easier first use and Settings edits

The first-use path is now: enter a research topic and press Brew. It saves the
profile and submits the existing Standard (`usual`) brew once. Returning readers
use Save changes and return to reading without starting another brew. A missing
or unknown action keeps the legacy `/run` destination; no supplied URL is used.

Topic edits have one form-level save. Per-topic Done and Cancel controls are
removed; Enter folds an editor, Escape restores its draft, and Remove retains
Undo. Single-topic priority controls are hidden while the actual priority is
preserved. Multiple topics retain inline percentages and balancing. Removing an
uncommitted addition restores untouched original relative weights, unless the
reader explicitly changed priority. Undo restores that addition and keyboard
focus. Invalid single-topic priorities reveal themselves before native focus.

Examples and research context start collapsed. The first Brew button follows
the topic editor, before optional Settings groups. Getting-started shortcuts fit
one compact row. The original gradient, Figtree typography, greetings, Pip,
reading layout and existing advanced controls remain available.

No source selection, ranking, model, summary or publication-linkage algorithm
changed. All 43 modules in the earlier algorithm baseline have identical hashes.
No personal profile, votes, credentials, history or embedding cache was edited.

Validation uses fictional profiles and intercepted external services. Settings
checks cover 1280, 390 and 320 px, one-submit behavior, untouched legacy weights,
explicit edits, Undo focus, hidden-field recovery and automatic PubMed spelling.
The phone first-use transport probe verifies one Standard start request and
intentionally returns a controlled error; it does not claim a completed brew.
The remaining daily checks run the real pipeline against fictional feeds and
verify repeat brews, changed context, retained feedback, degraded coverage and
associated commentary. Local verification:


- Settings: 39 Chrome checks passed. The Undo focus cases failed before their
  fixes, then passed. No external services were contacted.
- Full browser acceptance: 51 checks passed; no JavaScript errors. This frozen
  candidate preceded only the final removed-placeholder reference fix, which
  the expanded Settings suite and final CI exercise separately.
- Daily release: 20 checks passed; reading layout: 12; backup/guide: 15;
  two-tab handoff: 8; semantic bridge: 5; browser unit tests: 259.
- The exact final shipping build passed desktop/390/320 px checks, including
  loaded Figtree, original gradient, no horizontal overflow, Pip facts/leaf,
  and Backup/Guide/Data dialogs. Its local cold start measured 4.36 seconds;
  that is one local observation, not a general device/network promise.
- Full native `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLM_API_KEY="" .venv/bin/python -m pytest -q` completed successfully: 3,098 passed and 4 skipped.
- Ruff `--select F,B src tests`, JavaScript syntax checks on changed verifiers,
  and `git diff --check` passed.

Build output excludes private data and includes all 108 allowlisted source
files. The final workflow reruns the complete browser release gate and native
checks against the committed source before release verification.
