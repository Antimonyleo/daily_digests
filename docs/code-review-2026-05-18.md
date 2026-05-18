# DailyDigest Code Review — 2026-05-18

## 1. Executive Summary

A full-codebase review of DailyDigest (ranking, ingest, pipeline, store, config, web, and tests)
was conducted against the `dev` branch as of 2026-05-18. The most critical findings are a
**wrong stage ordering in the pipeline** (quality gate fires before dedup, causing journal papers
to vanish), a **train/infer profile mismatch** in the voting dataset that corrupts logistic
regression learning, and an **unbounded SQLite IN-list** in `prune()` that will produce an
`OperationalError` as the vote table grows. A further 35 actionable issues — ranging from silent
failure swallowing, feature-engineering improvements, and test-coverage gaps — are catalogued
below with file:line references and concrete fixes.

---

## 2. Ranking Algorithm — Bugs

### Bug 1 — Train/infer LR profile mismatch
**Severity:** HIGH
**Location:** `src/dailydigest/votes.py:919` / `src/dailydigest/rank/ranker.py`

**Problem:** `build_vote_dataset()` in `votes.py` calls `build_profile_matrix` (the static,
non-Rocchio version) to compute cosine features for training. However, `pipeline.py` calls
`build_profile_matrix_with_rocchio` at inference time. The LR model therefore learns cosine
similarities computed against one profile vector but is scored against a different (Rocchio-
updated) vector. On every re-train the features shift, making the learned coefficients
meaningless.

**Fix:** Pass the Rocchio-updated profile into `build_vote_dataset` so train and infer use the
same embedding:

```python
# votes.py — build_vote_dataset signature change
def build_vote_dataset(store, profile_vec):   # accept caller-supplied vec
    ...
    cosine = cos_sim(item_emb, profile_vec)   # use it here

# pipeline.py — pass Rocchio profile when retraining
profile_vec = build_profile_matrix_with_rocchio(store, settings)
dataset = build_vote_dataset(store, profile_vec)
```

---

### Bug 2 — Negative-interest penalty scale mismatch in hybrid LR mode
**Severity:** HIGH
**Location:** `src/dailydigest/rank/ranker.py`

**Problem:** The penalty `0.28 * neg_sim` is subtracted from a raw cosine score (range roughly
−1 to 1). In hybrid LR mode the pipeline first normalises scores to [0, 1] and *then* applies
quality adjustments; the penalty therefore operates on the wrong magnitude and is effectively
10–30× too small after normalisation.

**Fix:** Apply the negative-interest penalty **before** normalisation, inside the quality-
adjustment step that still operates on raw scores:

```python
# Inside _apply_quality_adjustments, before score normalisation
score -= 0.28 * neg_sim   # penalty while still on cosine scale
score = np.clip(score, -1, 1)
# ... normalise to [0,1] only afterwards
```

---

### Bug 3 — arXiv + preprint double-stacking penalty
**Severity:** HIGH
**Location:** `src/dailydigest/rank/source_quality.py`

**Problem:** CS arXiv items receive both `preprint_smooth(−0.15)` and `arxiv_smooth(−0.18)`,
for a combined −0.33 penalty. This is double-counting the same "preprint" signal, systematically
burying the arXiv CS items that many users specifically want.

**Fix:** Take the maximum (least punitive) of the two penalties rather than summing:

```python
# source_quality.py
preprint_pen = preprint_smooth(item)
arxiv_pen = arxiv_smooth(item)
penalty = max(preprint_pen, arxiv_pen)   # not preprint_pen + arxiv_pen
```

---

### Bug 4 — Freshness penalty uses `fetched_at` as fallback
**Severity:** MED
**Location:** `src/dailydigest/rank/ranker.py`

**Problem:** When `published_at` is `None` the code falls back to `fetched_at` to compute the
freshness penalty. An item published 30 days ago but ingested today therefore gets a *zero* age
penalty, appearing artificially fresh and jumping ahead of genuinely new content.

**Fix:** Treat a missing `published_at` as unknown and return a neutral penalty of 0, not a
zero-age penalty:

```python
def _freshness_penalty(item, now):
    if item.published_at is None:
        return 0.0   # unknown — no bonus, no penalty
    age_days = (now - item.published_at).total_seconds() / 86400
    ...
```

---

### Bug 5 — Dead code: `_apply_downweight` and `_apply_quality_adjustments` never called
**Severity:** LOW
**Location:** `src/dailydigest/rank/ranker.py`

**Problem:** Both `_apply_downweight` and `_apply_quality_adjustments` are defined but are
never invoked in the ranking hot path. Any logic inside them silently has no effect.

**Fix:** Either wire them into the ranking pipeline at the appropriate step, or delete them if
they have been superseded. Remove rather than leave dead code that creates confusion.

---

### Bug 6 — Exceptional preprint threshold broken in LR hybrid mode
**Severity:** MED
**Location:** `src/dailydigest/rank/ranker.py`

**Problem:** The logic that promotes an "exceptional preprint" checks `score < 0.90`. In LR
hybrid mode, scores are normalised to [0, 1], so the maximum possible score is 1.0 — the
threshold is never met and no exceptional preprint is ever promoted.

**Fix:** Use a percentile-based threshold that is invariant to score scale:

```python
threshold = np.percentile(scores, 90)
exceptional_mask = (scores >= threshold) & is_preprint_mask
```

---

### Bug 7 — Zero profile vector fails silently
**Severity:** HIGH
**Location:** `src/dailydigest/rank/ranker.py` / `src/dailydigest/rank/profile.py`

**Problem:** If the profile vector is all-zeros (e.g., the profile YAML is empty or embedding
fails), every cosine similarity is 0. The ranker silently selects arbitrary items based on tie-
breaking order, with no warning emitted to the user or logs.

**Fix:** Add an explicit guard after profile construction:

```python
if np.linalg.norm(profile_vec) < 1e-8:
    logger.error(
        "Profile vector is zero — rankings will be arbitrary. "
        "Check config/profile.yaml and embedding output."
    )
```

---

## 3. Ranking Algorithm — Quality Improvements

### Item 8 — Prestige bonus asymmetric
**Severity:** MED

The maximum prestige *bonus* is `+0.039` (for Nature), while the low-prestige *penalty* reaches
`−0.18`. This means the algorithm punishes low-quality sources far more than it rewards high-
quality ones, introducing a negativity bias.

**Recommendation:** Symmetrise around the neutral midpoint:

```python
prestige_adjustment = 0.18 * max(prestige - 0.50, 0)   # bonus only above neutral
```

---

### Item 9 — Correlated LR features
**Severity:** MED

`is_preprint ≈ bucket_score` and `is_hq_journal ≈ prestige` are near-collinear pairs. Including
both inflates variance of LR coefficients and makes feature attributions uninterpretable.

**Recommendation:**
- Drop `is_preprint`; keep `bucket_score` (continuous, more informative).
- Drop `is_hq_journal`; keep `prestige` (continuous).
- Add interaction features that capture joint signal: `cosine * bucket_score`,
  `cosine * prestige`, `cosine * age_norm`.

---

### Item 10 — Rocchio gamma too aggressive
**Severity:** MED

At 30 votes `gamma = 0.45` produces a learned row weight ~15×. A narrow topic cluster of votes
(e.g., three "CRISPR delivery" papers) then dominates the profile vector, drowning the original
broad profile built from the user's bio.

**Recommendation:** Cap gamma at 0.25 regardless of vote count, or apply a logarithmic schedule:

```python
gamma = min(0.25, 0.05 * np.log1p(n_votes))
```

---

### Item 11 — Negative-interest centroid is incoherent
**Severity:** MED

Negative interests are averaged into a single centroid (e.g., mean of {crypto, stock trading,
celebrity, sports}). This centroid sits in a neutral region of the embedding space, so adjacent
topics like "biotech IPO" or "sports medicine" get penalised even though none of the individual
negative keywords apply.

**Recommendation:** Penalise per individual negative interest and take the maximum similarity:

```python
neg_sims = [cos_sim(item_emb, neg_vec) for neg_vec in individual_neg_vecs]
neg_penalty = 0.28 * max(neg_sims) if max(neg_sims) > 0.35 else 0.0
```

---

### Item 12 — Freshness curves not section-aware
**Severity:** MED

A research paper published 4 days ago and a news article published 4 days ago are both
"moderately stale" under the current single freshness curve. But "4-day-old news" is
functionally worthless, while "4-day-old Nature paper" is still highly relevant.

**Recommendation:** Define per-section freshness curves:

| Section | Zero-penalty window | Full-penalty age | <2-day bonus |
|---------|--------------------|--------------------|--------------|
| Research | 3 days | 30 days | +0.05 |
| Industry | 0.5 days | 5 days | — |
| World | 0.5 days | 5 days | — |
| Regulatory | 1 day | 14 days | — |

---

### Item 13 — No MMR re-ranking within sections
**Severity:** LOW

Two closely related papers can both clear the 0.92 cosine-dedup threshold (e.g., two preprints
on the same topic from different groups), and both end up adjacent in the same section of the
digest.

**Recommendation:** After section selection, apply Maximal Marginal Relevance re-ranking:

```python
score_final = 0.7 * score - 0.3 * max_cosine_to_already_picked
```

---

### Item 14 — Section-relative z-score missing (cold-start bias)
**Severity:** LOW

Research abstracts contain dense scientific vocabulary that yields higher raw cosine similarity
to a science-focused profile than short news headlines do. At cold start (cosine-only mode)
Research items therefore dominate the digest even when Industry or Regulatory items are
genuinely relevant.

**Recommendation:** Z-normalise scores within each section before cross-section comparison:

```python
for section, items in section_items.items():
    scores = np.array([i.score for i in items])
    if scores.std() > 1e-6:
        z = (scores - scores.mean()) / scores.std()
        for item, zi in zip(items, z):
            item.score = zi
```

---

### Item 15 — Missing features entirely

The following signals are absent from the feature set and would each provide incremental lift:

- **(a) Author names in embedded text:** prepend `authors: {names}` to the embedded string so
  author reputation is captured by the embedding.
- **(b) Source trust calibration:** track per-source vote-through rate (votes / shown) from vote
  history and use it as a feature.
- **(c) Exploration slot:** reserve 1 item per section drawn from the 60th–80th percentile to
  surface content the ranker hasn't yet learned to promote.
- **(d) Click-through as weak Rocchio signal:** record link opens (in the web UI) as +0.1
  Rocchio weight (weaker than an explicit upvote).
- **(e) Section budget reallocation:** when a section pool has fewer than its quota of items,
  redistribute unused slots to the section with the largest surplus above quota.

---

## 4. Ingest & Pipeline — Bugs

### Bug 16 — Quality gate runs BEFORE dedup (wrong order)
**Severity:** HIGH
**Location:** `src/dailydigest/pipeline.py:464-465`

**Problem:** The quality gate is applied before deduplication. A Nature RSS item with a thin
abstract (e.g., a brief editorial) fails the quality gate and is dropped. The OpenAlex duplicate
of the same paper, which carries the full abstract, then arrives and is deduplicated *away*
because a stub record already exists in the DB from the quality-failed Nature item. The net
result is that the full-abstract version is silently lost.

**Fix:** Reverse the order — deduplicate first, then apply the quality gate:

```python
# pipeline.py
items = dedupe_ranking_candidates(items, store)   # step 1: dedup
items = _quality_gate(items)                       # step 2: quality
```

---

### Bug 17 — arXiv uses HTTP not HTTPS
**Severity:** MED
**Location:** `src/dailydigest/ingest/arxiv.py:65`

**Problem:** `BASE = "http://export.arxiv.org/api/query"` — unencrypted. Responses could be
intercepted or modified; some networks block plain HTTP to academic APIs.

**Fix:**
```python
BASE = "https://export.arxiv.org/api/query"
```

---

### Bug 18 — arXiv OAI ID prefix not stripped
**Severity:** MED
**Location:** `src/dailydigest/ingest/arxiv.py:91`

**Problem:** `entry.id` can be `"oai:arXiv.org:2401.12345v1"`. The code splits on `/abs/` to
extract the ID, which fails on the OAI form, producing a malformed or empty dedup key.

**Fix:**
```python
import re
def _arxiv_id(raw_id: str) -> str:
    raw_id = re.sub(r'^oai:arXiv\.org:', '', raw_id)
    raw_id = re.sub(r'^arXiv:', '', raw_id)
    if '/abs/' in raw_id:
        raw_id = raw_id.rsplit('/abs/', 1)[1]
    return raw_id.split('v')[0]   # strip version suffix
```

---

### Bug 19 — arXiv uses `published_parsed` instead of `updated_parsed`
**Severity:** MED
**Location:** `src/dailydigest/ingest/arxiv.py:108-115`

**Problem:** For v2/v3 revisions, `published_parsed` is the v1 submission date. Using it causes
a revised paper to appear artificially old and accumulate undeserved freshness penalty.

**Fix:** Prefer `updated_parsed` when available:

```python
pub_dt = (
    datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
    if getattr(entry, 'updated_parsed', None)
    else datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
)
```

---

### Bug 20 — PubMed `MedlineDate` not handled; midnight UTC day-boundary error
**Severity:** HIGH
**Location:** `src/dailydigest/ingest/pubmed.py:138-150`

**Problem:** Records using `<MedlineDate>2026 May-Jun</MedlineDate>` have no `<Year>` child
element. The parser returns `pub_dt = None`, causing these items to receive maximum freshness
penalty (treated as ancient). A secondary issue: dates are parsed to midnight UTC, so an item
published early on day D in UTC−5 appears one day old by the time the 08:00 digest runs.

**Fix:**

```python
def _parse_pubmed_date(article_xml):
    year_el = article_xml.find('.//Year')
    if year_el is not None:
        year = int(year_el.text)
    else:
        medline = article_xml.findtext('.//MedlineDate', '')
        m = re.match(r'(\d{4})', medline)
        year = int(m.group(1)) if m else None
    if year is None:
        return None
    month = int(article_xml.findtext('.//Month') or 1)
    day   = int(article_xml.findtext('.//Day')   or 1)
    return datetime(year, month, day, 12, 0, tzinfo=timezone.utc)  # noon UTC anchor
```

---

### Bug 21 — bioRxiv cap of 200 silently drops papers on busy days
**Severity:** MED
**Location:** `src/dailydigest/ingest/biorxiv.py:103`

**Problem:** The 2-day bioRxiv window routinely contains 200–400 preprints. Hard-capping at 200
means roughly half the output may be silently dropped before the ranker even sees them.

**Fix:**

```python
MAX_RESULTS = 500
...
if len(results) >= MAX_RESULTS:
    logger.info("bioRxiv: hit cap of %d — some papers may be missed", MAX_RESULTS)
```

---

### Bug 22 — FDA bare `except` with no logging
**Severity:** MED
**Location:** `src/dailydigest/ingest/fda.py:61`

**Problem:** A bare `except: pass` silently swallows all FDA fetch errors. Health stats still
report `ok=True` for the FDA source even when every request fails.

**Fix:**
```python
except Exception as exc:
    logger.warning("FDA ingest failed: %s", exc)
    return []
```

---

### Bug 23 — ClinicalTrials bare `except` with no logging
**Severity:** MED
**Location:** `src/dailydigest/ingest/clinicaltrials.py:75`

**Problem:** Same pattern as Bug 22 — errors are silently discarded, health stats appear healthy.

**Fix:**
```python
except Exception as exc:
    logger.warning("ClinicalTrials ingest failed: %s", exc)
    return []
```

---

### Bug 24 — Nature/Cell/Science URL→DOI not extracted for cross-source dedup
**Severity:** MED
**Location:** `src/dailydigest/dedupe.py`

**Problem:** Nature RSS items use canonical URLs of the form
`https://www.nature.com/articles/s41586-024-07123-4`. The DOI `10.1038/s41586-024-07123-4` is
embedded in the URL slug. When the same paper arrives via OpenAlex with its DOI as the canonical
key, the two records are never deduplicated.

**Fix:** Add URL→DOI extraction for major publisher URL patterns:

```python
PUBLISHER_URL_DOI = [
    (re.compile(r'nature\.com/articles/(s\d+-\d+-\d+)'), r'10.1038/\1'),
    (re.compile(r'cell\.com/[^/]+/fulltext/(S\S+)'),     lambda m: m.group(1).replace('-', '/')),
    (re.compile(r'science\.org/doi/([^?]+)'),             r'\1'),
]

def url_to_doi(url: str) -> str | None:
    for pattern, repl in PUBLISHER_URL_DOI:
        m = pattern.search(url)
        if m:
            return m.expand(repl) if isinstance(repl, str) else repl(m)
    return None
```

---

### Bug 25 — Title dedup 20-character minimum too strict
**Severity:** MED
**Location:** `src/dailydigest/dedupe.py:261`

**Problem:** Titles shorter than 20 characters (e.g., "Mpox in pregnancy", "mRNA safety") are
skipped by title-similarity dedup entirely, allowing near-duplicates of short-titled items to
both appear in the digest.

**Fix:** Lower the minimum to 12 characters:

```python
if len(title) < 12:
    continue
```

---

### Bug 26 — Missing `User-Agent` header on bioRxiv, FDA, ClinicalTrials
**Severity:** MED
**Location:** `src/dailydigest/ingest/biorxiv.py`, `fda.py`, `clinicaltrials.py`

**Problem:** Requests are sent without a `User-Agent` header. Several APIs (bioRxiv, FDA) block
or rate-limit library default agents.

**Fix:** Add a consistent header to all outbound requests:

```python
HEADERS = {"User-Agent": "dailydigest/0.1 (github.com/user/dailydigest; contact@example.com)"}
response = httpx.get(url, headers=HEADERS, timeout=30)
```

---

### Bug 27 — OpenAlex missing `sort` parameter
**Severity:** LOW
**Location:** `src/dailydigest/ingest/openalex.py`

**Problem:** The default sort order of the OpenAlex API is unspecified and may change. Without
an explicit sort, the items returned may not be the most recent ones.

**Fix:**
```python
params["sort"] = "publication_date:desc"
```

---

### Bug 28 — PubMed `RETMAX=50` too low
**Severity:** MED
**Location:** `src/dailydigest/ingest/pubmed.py`

**Problem:** Busy PubMed queries (e.g., "cancer immunotherapy") can return hundreds of new
papers per day. A cap of 50 means the majority are never seen by the ranker.

**Fix:**
```python
RETMAX = 200
```

---

### Bug 29 — RSS `bozo` flag not logged
**Severity:** LOW
**Location:** `src/dailydigest/ingest/rss.py:113`

**Problem:** `feedparser` sets `feed.bozo = True` when a feed is malformed or contains encoding
errors. The code continues silently, making it impossible to know which feeds are degraded
without a careful audit.

**Fix:**
```python
if feed.bozo:
    logger.info("RSS feed %s is malformed (bozo=True): %s", url, feed.bozo_exception)
```

---

## 5. Store / Config / Web — Bugs

### Bug 30 — `prune()` unbounded `IN` list — crash risk
**Severity:** HIGH
**Location:** `src/dailydigest/store.py:412-428`

**Problem:** `prune()` builds a `NOT IN (all_voted_ids)` clause where `all_voted_ids` is the
complete set of ever-voted item IDs. SQLite has a compile-time variable limit of 32 766. After a
year of daily use (~365 votes minimum) this query will raise `OperationalError: too many SQL
variables` and the prune job will crash silently.

**Fix:** Replace the `IN` list with a correlated subquery:

```sql
DELETE FROM items
WHERE fetched_at < :cutoff
  AND id NOT IN (SELECT DISTINCT item_id FROM votes)
```

```python
stmt = text(
    "DELETE FROM items "
    "WHERE fetched_at < :cutoff "
    "  AND id NOT IN (SELECT DISTINCT item_id FROM votes)"
)
session.execute(stmt, {"cutoff": cutoff})
```

---

### Bug 31 — `_RUN_QUEUES` memory leak in web server
**Severity:** MED
**Location:** `src/dailydigest/web.py:96-98`

**Problem:** When the user triggers a brew from the web UI, a queue is added to
`_RUN_QUEUES[run_id]`. The queue is only removed when an SSE consumer connects and drains it.
If the client navigates away or the SSE connection is never established, the queue (and all
pipeline events buffered in it) accumulates indefinitely in the process's memory.

**Fix:** Clean up the queue in a `finally` block inside the brew thread:

```python
def _brew_thread(run_id, ...):
    try:
        _run_pipeline(run_id, ...)
    finally:
        _RUN_QUEUES.pop(run_id, None)
```

---

### Bug 32 — Invalid timezone silently becomes UTC
**Severity:** MED
**Location:** `src/dailydigest/config.py:47-59`

**Problem:** If `USER_TZ` is set to an invalid value (e.g., `"US/Eaastern"` typo), `zoneinfo`
raises `ZoneInfoNotFoundError`, which is caught and silently falls back to UTC. The digest then
runs at 08:00 UTC instead of 08:00 local time — potentially hours off — with no indication that
anything is wrong.

**Fix:** Escalate the error to `logging.error` and surface it in the `/health` endpoint:

```python
except ZoneInfoNotFoundError:
    logger.error("USER_TZ=%r is invalid — falling back to UTC. Fix your .env!", raw_tz)
    self._tz_error = f"Invalid USER_TZ={raw_tz!r}"
    return ZoneInfo("UTC")
```

---

### Bug 33 — `load_profile` / `load_sources` bypass `get_settings()` LRU cache
**Severity:** LOW
**Location:** `src/dailydigest/config.py:79, 98`

**Problem:** Both functions instantiate a fresh `Settings()` object and re-read the `.env` file
on every call instead of using the already-cached singleton from `get_settings()`. In a long-
running process this causes repeated disk I/O and environment re-parsing on every ingest cycle.

**Fix:**
```python
def load_profile() -> Profile:
    s = get_settings()
    return Profile.from_yaml(s.profile_path)

def load_sources() -> list[SourceSpec]:
    s = get_settings()
    return SourceSpec.from_yaml(s.sources_path)
```

---

### Bug 34 — `init_db()` probes DDL on every call
**Severity:** LOW
**Location:** `src/dailydigest/store.py`

**Problem:** `init_db()` calls `Base.metadata.create_all(engine)` on every invocation. Even
when all tables already exist, SQLAlchemy still issues a `PRAGMA table_list` round-trip and
introspects every table, adding ~10 ms of overhead per pipeline run.

**Fix:** Guard with a module-level flag:

```python
_INITIALIZED = False

def init_db():
    global _INITIALIZED
    if _INITIALIZED:
        return
    Base.metadata.create_all(engine)
    _INITIALIZED = True
```

---

### Bug 35 — `get_vote_reasons` N+1 file reads
**Severity:** MED
**Location:** `src/dailydigest/web.py:307-310`

**Problem:** For each of the ~30 items on the homepage, `get_vote_reasons` reads the vote-
reasons file from disk individually. A single homepage render therefore issues ~30 sequential
disk reads.

**Fix:** Load the full reasons dictionary once per request and pass it down:

```python
all_reasons = load_all_vote_reasons()   # one read
for item in items:
    item.reason = all_reasons.get(item.id)
```

---

### Bug 36 — IMAP env vars bypass pydantic-settings `.env` loading
**Severity:** LOW
**Location:** `src/dailydigest/config.py` / `src/dailydigest/ingest/inbound.py`

**Problem:** `inbound.py` reads `IMAP_HOST`, `IMAP_PORT`, `IMAP_USER`, `IMAP_PASSWORD`, and
`IMAP_MAILBOX` directly from `os.environ`, bypassing the `Settings` class. Values set in `.env`
are therefore not picked up unless the shell has already exported them.

**Fix:** Add the fields to `Settings`:

```python
class Settings(BaseSettings):
    ...
    imap_host: str = ""
    imap_port: int = 993
    imap_user: str = ""
    imap_password: SecretStr = SecretStr("")
    imap_mailbox: str = "INBOX"
```

---

### Bug 37 — `Item.section` not validated against known sections
**Severity:** MED
**Location:** `src/dailydigest/models.py` / `src/dailydigest/config.py`

**Problem:** A typo in `sources.yaml` (e.g., `section: reseach`) produces `Item` records with
an unrecognised section string. These items are silently excluded from the email template
(because the Jinja loop iterates a hard-coded section list), making entire source categories
disappear from the digest with no error.

**Fix:** Add a `field_validator` on `Item.section` and `SourceSpec.section`:

```python
KNOWN_SECTIONS = {"research", "industry", "regulatory", "world"}

@field_validator("section")
@classmethod
def _validate_section(cls, v: str) -> str:
    if v not in KNOWN_SECTIONS:
        raise ValueError(f"Unknown section {v!r}. Must be one of {KNOWN_SECTIONS}")
    return v
```

---

### Bug 38 — GitHub Actions cron removed; digest never runs automatically
**Severity:** HIGH
**Location:** `.github/workflows/digest.yml`

**Problem:** The workflow is currently `workflow_dispatch` only — the `schedule: cron:` trigger
has been removed. The digest therefore only runs when manually triggered; automated morning
delivery is entirely broken.

**Fix:** Restore the hourly schedule:

```yaml
on:
  schedule:
    - cron: '0 * * * *'
  workflow_dispatch:
```

---

## 6. Test Coverage Gaps & False Confidence

### Weak assertions (tests that pass regardless of correctness)

- **`test_dedupe.py`:** Assertions use substring matching — `"Example.Com" in result OR
  "example.com" in result` — which always evaluates to `True`. The test provides zero signal
  about whether deduplication actually removed the duplicate. Replace with exact-equality checks
  on the returned item count and canonical URL.

- **`test_summarize.py`:** `_build_prompt` is called twice; the first call's return value is
  discarded and only the second is asserted. The first call is a dead assertion — any bug
  introduced between the two calls is invisible to the test.

- **`test_ranker.py`:** The test is written to accept *either* LR mode *or* cosine mode. If LR
  loading silently fails and falls back to cosine, the test still passes. This means a broken LR
  pipeline could go undetected for weeks.

### Missing test coverage

| Missing test | Why it matters |
|---|---|
| `RSSSource.fetch` with a real (mocked) feed | RSS is the highest-volume ingest path; zero fetch-level tests |
| `ArxivSource.fetch` with OAI-form entry IDs | Bug 18 would not have been caught |
| `PubMedSource.fetch` with `MedlineDate` records | Bug 20 would not have been caught |
| Negative-interest end-to-end effect | Verify that a known negative-interest item scores below a neutral item |
| `prune()` with > 32 766 voted IDs | Bug 30 would not have been caught until production |
| `/run/stream` SSE heartbeat | Confirms the stream stays alive between events |
| "Shown-but-unvoted item resurfaces" design intent | `exclude_previously_shown` has no test that locks the intended behaviour |

---

## 7. Priority Matrix

| # | Area | Severity | Effort | User-Facing Impact |
|---|------|----------|--------|--------------------|
| 16 | Pipeline | HIGH | 15 min | Nature/journal papers disappearing from digest |
| 38 | Automation | HIGH | 5 min | Automated morning delivery entirely broken |
| 1 | Ranking | HIGH | 30 min | LR learning garbage features from wrong profile |
| 20 | Ingest | HIGH | 30 min | PubMed dates wrong → stale freshness penalty |
| 3 | Ranking | HIGH | 15 min | CS/arXiv papers systematically over-penalised |
| 30 | Store | HIGH | 15 min | Crash risk in prune() as vote table grows |
| 7 | Ranking | HIGH | 10 min | Zero profile vector → silent arbitrary selection |
| 17 | Ingest | MED | 1 min | arXiv reliability on restricted networks |
| 18 | Ingest | MED | 10 min | arXiv cross-source dedup failure |
| 21 | Ingest | MED | 5 min | bioRxiv coverage gap on busy days |
| 22–23 | Ingest | MED | 5 min | Silent FDA/ClinicalTrials failure, false health stats |
| 24 | Ingest | MED | 30 min | ~30% cross-source dedup gap for Nature/Cell/Science |
| 25 | Ingest | MED | 1 min | Short-title dedup gap |
| 8 | Ranking | MED | 10 min | Prestige reward-penalty asymmetry |
| 9 | Ranking | MED | 1 hr | Correlated LR features — noisy coefficients |
| 10 | Ranking | MED | 1 min | Rocchio runaway on narrow vote clusters |
| 11 | Ranking | MED | 1 hr | Negative-interest centroid penalises adjacent topics |
| 12 | Ranking | MED | 1 hr | News/research freshness mismatch |
| 31 | Web | MED | 15 min | Memory leak on abandoned brew sessions |
| 32 | Config | MED | 10 min | Timezone typo silently runs digest at wrong hour |
| 35 | Web | MED | 10 min | N+1 disk reads on every homepage render |
| 37 | Models | MED | 15 min | Section typo silently drops entire source category |
| 13 | Ranking | LOW | 2 hr | Near-duplicate papers in same section |
| 14 | Ranking | LOW | 1 hr | Cold-start bias toward research items |
| 5 | Ranking | LOW | 5 min | Dead code obscures actual pipeline logic |
| 33 | Config | LOW | 5 min | Repeated .env re-reads in long-running process |
| 34 | Store | LOW | 5 min | DDL round-trip on every init_db() call |
| 36 | Config | LOW | 15 min | IMAP env vars bypass pydantic-settings |
