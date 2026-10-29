# Development and maintenance

[Project overview](../README.md) · [Browser build and verification](../browser/README.md)


## Branches and dependency maintenance

- `main` is the tested, published branch. Only `main` can deploy the website.
- `dev` holds ongoing work and receives the same checks before integration.
- Use one temporary feature or maintenance branch when isolation is needed;
  delete it after integration. Merged temporary branches are deleted automatically.
- Keep `dev` after merging. Its branch protection prevents deletion and force pushes.

Review dependencies monthly, and promptly when a security alert arrives. Prepare
one scoped batch on `dev` (or a temporary maintenance branch), check native and
browser compatibility, run the relevant checks below, then integrate into `main`.
Major runtime upgrades may need separate batches; do not combine unrelated
breaking changes just to reduce branch count. No recurring job is installed by
this policy.

Routine Dependabot version PRs are disabled with `open-pull-requests-limit: 0`
for each ecosystem. This is separate from security alerts and automatic security
PRs ([GitHub documentation](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference)).
Security alerts are enabled; automatic security-update PRs remain disabled so
fixes can be triaged in the same maintenance flow. Alerts must still be reviewed.

The nine old dependency proposals (#2–#10) were closed without merging. Their
commits and diffs remain available through those PRs and a verified local Git
bundle in `.full-review/branch-consolidation-2026-10-29/`. No dependency versions
were changed by this consolidation.

## Checks and releases

Run the automated tests with:

```bash
uv sync --frozen --group dev
uv run pytest
uv run ruff check src tests
```

To publish an update to every installation, add an entry at the top of
`src/dailydigest/release_notes.json` with a higher `version` (`YYYY.MM.DD`, or
`YYYY.MM.DD.N` for a second release that day) and up to eight short,
reader-facing notes, then merge to `main`. Local installations offer the update
once the entry is on `main`; the website shows the notes after its next deploy.
Fellowships, awards and flagship meetings that no feed or API publishes live in
`src/dailydigest/curated/calls.yaml` and `meetings.yaml`. `meetings.yaml` holds
only organisers whose sites refuse automated clients (Gordon Research
Conferences, MRS, ACS, RSC Faraday Discussions, AIChE and Cell Press); once an
organiser's own listing becomes readable, move it to an adapter and drop its
entries here. Each entry names its
official page and the date it was checked (`verified_on`). A `next_deadline`
is shown only when the official page states it; deadlines seen only in search
results are ignored, and meetings dated from search results are labelled
"Verify source". Refresh both files about once a month: many rounds roll over
within months, and some programmes close or change (for example Canada's
Banting and Vanier awards were replaced in 2026). `applicant_regions` uses
lowercase ISO country codes, `eu`, or `international` for schemes open to
applicants from any country.

Keep `DailyDigest-Windows.bat` unchanged: Windows reads a running batch file
from disk, so the updater must never rewrite it (a test pins its contents).

DailyDigest is released under the [MIT License](../LICENSE).


Settings also offer optional example profiles, per-interest search details and
spelling suggestions, followed authors, representative works and background
interests. Test API connection performs an explicit small request without saving
settings or sending research context. Browser users need the companion for local
CLI providers; native users run those installed commands locally.

Unknown venue evidence is distinguished from observed low impact. Both retain
bounded selection and strong raw-topic requirements; a positive allowance permits
one candidate in small digests while explicit zero allowances remain exclusions.
Exact journal aliases avoid accidental downgrades of known venues. CS preprints
use the same publication-status policy as other fields. Source freshness and
limited searches remain visible; funding/event coverage is regionally limited.
