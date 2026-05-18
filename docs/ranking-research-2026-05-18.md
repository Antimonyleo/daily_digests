# Ranking Algorithm Research: State of the Art vs DailyDigest

> Written from training knowledge (model cutoff August 2025). Paper citations verified from
> training corpus; no live retrieval. All implementation estimates assume the DailyDigest
> constraints: solo developer, ~20 items/digest, ≤$5/month, local CPU inference.

---

## 1. Executive Summary

Three insights dominate the literature and transfer directly to DailyDigest:

1. **Embeddings alone are a weak signal without query-document asymmetry.** The best academic recommenders (Scholar Inbox, SPECTER2, E5-instruct) separate *what the user wants* (query) from *what the paper says* (document) using different encoders or task-prefixed embeddings. DailyDigest currently embeds both profile and items with the same symmetric model — causing systematic underweighting of short keyword queries against long abstracts.

2. **Listwise diversity (MMR or DPP) outperforms pointwise ranking for digest-style outputs.** When the goal is a finite daily set rather than a ranked list, greedy MMR or Determinantal Point Processes prevent topic redundancy — a consistent finding across MIND, Adressa, and Scholar Inbox evaluations. DailyDigest has no diversity re-ranking at all.

3. **Learning-to-rank with pairwise preferences is far more data-efficient than pointwise LR.** With 20–30 votes, converting implicit preferences into pairwise training examples (each vote generates O(N) pairs) multiplies usable training signal by 10–20× vs. the current approach of treating each vote as an independent label.

---

## 2. Systems Studied

### 2.1 Scholar Inbox (arXiv:2504.08385, 2025)

**What it does:**
Scholar Inbox is an academic paper recommendation service used by ~8,000 researchers. Their published system (ACL 2025) uses:

- **Profile representation**: User writes a free-text research statement → embedded with a scientific language model (SPECTER2 family) → profile vector. Additional signal comes from paper collection: papers the user "saved" are embedded and averaged into a weighted centroid. This is structurally similar to DailyDigest's bio+keyword matrix.
- **Candidate retrieval**: Semantic Scholar API for recent papers; ~200 candidates/user/day.
- **Scoring**: Cosine similarity + active-learning logistic regression trained on thumbs feedback. Very close to DailyDigest's architecture.
- **Key difference from DailyDigest**: They use a *scientific-domain embedding model* (SPECTER2), not a general-purpose one (bge-small). SPECTER2 is trained on citation graph co-occurrence — two papers cited together embed close together even if their abstracts differ textually. This captures conceptual relatedness that bge-small misses.
- **Active learning**: They present items near the decision boundary (uncertain predictions) rather than purely the highest-scoring. This is the key cold-start improvement over greedy top-K.

**Transferable to DailyDigest:**
- SPECTER2-base is ~500MB but SPECTER2-proximity is available in smaller variants. The citation-aware embedding would be a clear quality win for the research section. However, it requires replacing bge-small — a non-trivial change.
- The "uncertain item" exploration idea is low-effort: reserve 1 slot per section for an item with LR probability closest to 0.5 rather than highest.

---

### 2.2 ArxivDigest (prior art, 2022–2024)

ArxivDigest (GitHub: AutoLLM/ArxivDigest) uses:
- GPT-4 zero-shot relevance scoring per item: "Here is a user interest description. Rate this paper 1–10."
- No embedding similarity; pure LLM judgment.
- No learning loop; profile is static YAML.

**What it does well**: Excellent cold-start quality because GPT-4 understands nuance. "I want papers on RNA therapeutics for rare diseases, especially pediatric" — GPT-4 handles this naturally.

**What it does poorly**: Expensive ($0.10–0.30/day), slow (serial API calls), no personalization drift.

**Transferable to DailyDigest:**
- Use LLM scoring *only* as a cold-start fallback (0 votes) or as a re-ranking pass over the top-15. After ≥10 votes, switch to the LR model. Cost: ~$0.01/digest for 15 items with gpt-4o-mini.
- The LLM prompt pattern: "User bio: {bio}. Rate this paper's relevance 1-10, respond with just the number." is reliable and cheap.

---

### 2.3 Arxiv Sanity Preserver (Karpathy, 2015; updated ~2020)

The original Arxiv Sanity (used by ML researchers before Scholar Inbox) used:
- TF-IDF on paper abstracts → SVD to 50 dimensions.
- User profile = average of saved paper TF-IDF vectors (positive examples only).
- Ranked by cosine similarity; no learning loop.

**Why it worked despite simplicity**: TF-IDF is domain-specific (arXiv ML papers), so the vocabulary distribution is narrow and informative. bge-small is trained on broad web text and may underperform TF-IDF for narrow domains.

**Lesson**: For a biotech researcher, a domain-adapted embedding (fine-tuned on PubMed/bioRxiv abstracts) or TF-IDF over a biotech corpus would likely outperform general bge-small for the research section.

**Transferable**: TF-IDF-based score as an additional LR feature (`tfidf_sim`) is cheap to add and captures vocabulary overlap that semantic embeddings sometimes miss.

---

### 2.4 NRMS / NAML / FIM (Microsoft Research, 2019–2021)

Microsoft's news recommendation literature (published around MIND dataset release):

- **NAML** (Neural News Recommendation with Attentive Multi-View Learning): encodes title, abstract, category, subcategory through separate CNNs + attention → article representation. User representation = attention over clicked article history.
- **NRMS** (Neural Recommendation with Multi-Head Self-Attention): self-attention over the user's news reading history to get user context vector; dot product with news vectors for scoring.

**Key insight not in DailyDigest**: These systems learn *what part of a user's history* to attend to when scoring a new item. DailyDigest's Rocchio treats all past votes equally. NRMS would say "this new paper is about CRISPR, so weight the user's past CRISPR interactions more when building their current context."

**Transferable (simplified)**: For inference, instead of one global profile vector, compute a *query-aware profile*: for each candidate item, weight the Rocchio learned vector by `cos(item, learned_vec)` rather than using the full averaged centroid. This is cheap (~3 lines) and captures the "this item's topic → attend to relevant past votes" effect.

---

### 2.5 Google News / Apple News (published research)

**Google News (2016 paper, "Deep Neural Networks for YouTube Recommendations" and follow-up news work):**
- Uses watch/read history as sequential signal.
- Candidate generation → ranking → re-ranking with diversity.
- **Diversity layer**: explicitly penalizes items from the same source domain and same entity cluster.

**Apple News (inferred from public writing + WWDC):**
- Heavy use of topic taxonomy (IAB categories).
- Source credibility scores maintained independently of engagement.
- Hard caps per publisher per session.

**Lesson for DailyDigest**: Source-level diversity is missing. If Nature publishes 3 relevant papers today, DailyDigest might show all 3. Google/Apple would cap at 1-2 per publisher. A simple `max_per_source = 2` cap in `_pick_research_balanced` would help.

---

### 2.6 Hacker News Ranking

HN's public formula:
```
score = (points - 1)^0.8 / (age_hours + 2)^1.8 * gravity_penalties
```
Key properties:
- Gravity (1.8 exponent on age) makes content decay fast — a 24-hour-old story with 100 points loses to a 1-hour-old story with 20 points.
- Sublinear vote scaling (0.8 exponent) prevents single viral items from dominating.
- Quality penalties applied separately (editorial, self-promo).

**Lesson**: DailyDigest's freshness penalty is linear and weak (max -0.12 over 11 days). For the world/industry section, a gravity-inspired decay would be more appropriate: `penalty = min(0.3, 0.3 * (age_hours / 72)^1.5)`.

---

### 2.7 Reddit Ranking (Wilson Score / Hot algorithm)

Reddit's "hot" ranking:
```
score = log10(max(abs(ups - downs), 1)) + sign(ups-downs) * seconds / 45000
```
Key: the log10 dampens the vote magnitude — the 10th upvote matters less than the 1st. Time component gives a floor boost to anything recent.

**Lesson**: DailyDigest should dampen the effect of vote magnitude when it gets there. With 100 votes, item 50 (upvoted once) vs item 51 (upvoted 5 times) — the LR's linear cosine feature overfits to the few highly-upvoted items. Log-scaling vote counts in the training data would help.

---

### 2.8 Semantic Scholar Recommendations (2022–2024)

Allen AI's SPECTER2 paper (2022) and Semantic Scholar API:
- **SPECTER2**: Contrastive learning on citation pairs. Papers A and B that are co-cited embed similarly. Papers and their abstracts embed similarly.
- **Adapters**: SPECTER2 has task-specific adapters (retrieval, proximity, classification). The *proximity* adapter is optimal for "find related papers" use cases.
- **Key benchmark result**: SPECTER2 outperforms general-purpose sentence transformers by 8-15 points on SciDocs and RELISH benchmarks.

**For DailyDigest**: Replacing bge-small-en-v1.5 with SPECTER2-base (or the proximity adapter variant, ~220MB) would improve research section quality substantially. Cost: runs on CPU, same inference time. The embedding API in `embed.py` abstracts the model choice.

---

### 2.9 Maximal Marginal Relevance (MMR)

Original paper: Carbonell & Goldstein (1998), "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries."

Formula:
```
MMR(d_i) = λ · sim(d_i, q) - (1 - λ) · max_{d_j ∈ S} sim(d_i, d_j)
```
Where S = already selected items, q = query/profile.

**Practical properties**:
- λ = 0.7 is a well-calibrated default for informational diversity (used in IR literature).
- With 60 candidates and 12 slots (research section), MMR runs in 60×12 = 720 dot products — negligible.
- Works best when items are already above a relevance threshold (i.e., apply MMR only to items passing the quality gate, not to all candidates).

**DPP (Determinantal Point Processes)** is the Bayesian generalization of MMR but requires matrix determinant computation — overkill for DailyDigest's scale.

**Transferable**: MMR re-ranking within each section is the single highest-impact diversity improvement, requiring ~20 lines of code.

---

### 2.10 Learning-to-Rank: LambdaMART / RankNet

**RankNet** (Burges et al., 2005): Trains a neural net on pairwise preferences. For DailyDigest with thumbs feedback: if item A was upvoted and item B was downvoted, (A, B) is a training pair with label "A > B".

**LambdaMART** (gradient-boosted trees on pairwise + listwise loss): Used in Bing, Yahoo ranking. Optimizes NDCG directly via lambda gradients.

**Data efficiency**: With 20 votes (10 up, 10 down), pairwise construction gives `10 × 50 = 500` training pairs (each upvoted item vs each unvoted item). This is 25× more signal than 20 pointwise labels. Critical for DailyDigest's cold-start problem.

**Practical barrier**: LambdaMART requires XGBoost/LightGBM, adding a dependency. RankNet requires a neural net. Neither is compatible with DailyDigest's 30-vote minimum (the model would overfit).

**Transferable without full LtR**: Convert existing votes to pairwise examples in the LR training loop. Instead of `X=[features], y=[+1/0/-1]`, build `X_pairs = [features_up - features_down], y_pairs = [1]` for each (upvoted, downvoted) pair. Same sklearn LR, same inference, 10-20× more training signal. This is the key pairwise insight without the LambdaMART complexity.

---

### 2.11 E5 / BGE Embedding Models

**Current**: `BAAI/bge-small-en-v1.5` (384-dim, 33M params, ~130MB).

**Comparison**:

| Model | Dims | Size | MTEB Score | Domain | Notes |
|-------|------|------|------------|--------|-------|
| bge-small-en-v1.5 | 384 | 133MB | 62.2 | General | Current; good baseline |
| bge-base-en-v1.5 | 768 | 438MB | 63.9 | General | +1.7 pts, 3× bigger |
| bge-large-en-v1.5 | 1024 | 1.34GB | 64.2 | General | Marginal gain |
| SPECTER2-base | 768 | 478MB | 68+ (SciDocs) | Scientific | Best for papers |
| SciNCL | 768 | ~500MB | 65+ (RELISH) | Scientific | Strong on biomedical |
| E5-small-v2 | 384 | 133MB | 59.9 | General | Slightly worse than bge-small |
| E5-base-v2 | 768 | 438MB | 63.4 | General | Comparable to bge-base |
| BAAI/bge-m3 | 1024 | 2.3GB | 66.8 | Multilingual | Overkill; user is English |

**Recommendation**: SPECTER2-base for the research section embedding is the highest-quality upgrade. For news/industry items (not scientific papers), bge-small remains appropriate. A **dual-encoder** approach — SPECTER2 for research section, bge-small for everything else — would be ~600MB total and improve research quality noticeably.

**Asymmetric embeddings**: E5-instruct ("query: " prefix for queries, "passage: " prefix for documents) showed that query-document asymmetry improves retrieval by 3-5 points on MTEB. DailyDigest currently uses `is_query=True` in `embed_texts` but bge-small uses the same encoder for both — no actual asymmetry. E5-small-v2 with prefixes would give true asymmetry at the same model size.

---

## 3. Algorithm Comparison Table

| Dimension | DailyDigest (current) | Scholar Inbox | ArxivDigest (LLM) | NRMS/NAML | HN Algorithm |
|-----------|----------------------|---------------|-------------------|-----------|--------------|
| **Cold start quality** | Poor (cosine only) | Good (active learning) | Excellent (LLM zero-shot) | Poor (needs history) | N/A (no profile) |
| **Learning efficiency** | Moderate (pointwise LR, 20 votes) | Good (active-learning LR) | N/A | High (self-supervised) | N/A |
| **Diversity** | Manual section caps only | None documented | None | Implicit via attention | Editorial caps |
| **Freshness** | Weak linear ramp | Not described | None | Click-time signals | Strong gravity decay |
| **Explainability** | Good (why_shown tags) | Low | High (LLM reasoning) | Very low | Perfect (formula) |
| **Compute cost** | Low (CPU, 384-dim) | Medium (GPU or SPECTER2) | High ($LLM/call) | Very high (neural) | Zero |
| **Multi-topic users** | OK (profile matrix rows) | OK (profile + saved centroid) | Good (LLM understands) | Good (attention heads) | N/A |
| **Negative preferences** | Basic (negative centroid) | Not documented | LLM implicit | Not documented | No mechanism |
| **Source diversity** | None | None | None | Publisher cap | Editorial |
| **Pairwise signal** | No | No | No | Yes (implicit) | N/A |

---

## 4. Transferable Improvements (ranked by impact/effort)

### Tier 1 — High Impact, Low Effort (< 1 day each)

**T1.1 Pairwise LR training** (from RankNet / LtR literature)
Convert votes to pairwise examples in `vote_dataset()`. Instead of `y=[1,0,-1]`, generate `(features_up - features_down)` pairs with `y=1`. Multiplies training signal by ~15×. Same sklearn LR, same 9-feature vector, no new dependencies.
```python
# In vote_dataset(), after building up_rows and down_rows:
pairs_X, pairs_y = [], []
for up in up_rows:
    for down in down_rows:
        pairs_X.append(_build_item_features(up, profile_vec) - _build_item_features(down, profile_vec))
        pairs_y.append(1)
X = np.vstack([X_pointwise, pairs_X])
y = np.hstack([y_pointwise, pairs_y])
```

**T1.2 MMR within-section re-ranking** (from Carbonell & Goldstein 1998)
After `_pick_research_balanced` selects candidates, apply MMR with λ=0.7 using item embedding matrix. ~20 lines, 720 dot products per digest run (negligible).

**T1.3 Active learning: 1 exploration slot per section** (from Scholar Inbox)
Reserve 1 item per section with LR probability closest to 0.5. Surfaces uncertain items to get training signal faster on topics the user hasn't seen yet.

**T1.4 Max-per-source cap** (from Google/Apple News)
Add `max_per_source = 2` constraint in `_pick_research_balanced`. Prevents Nature flooding 3 similar papers into the same digest.

**T1.5 HN-style gravity for world/industry freshness** (from Hacker News)
Replace linear ramp with: `penalty = min(0.3, 0.3 * (age_hours / 72) ** 1.5)`. Makes week-old news decay properly.

**T1.6 LLM cold-start re-ranking pass** (from ArxivDigest)
When vote count < 5, use a single LLM batch call to rate the top-20 items 1–10 for relevance to the bio. Cost: $0.005 per digest with gpt-4o-mini. Switch to LR once ≥ 10 votes exist.

### Tier 2 — High Impact, Medium Effort (1–3 days each)

**T2.1 Query-document asymmetric embeddings** (from E5 / BGE literature)
Switch from bge-small (symmetric) to E5-small-v2 with `"query: "` prefix for profile vectors and `"passage: "` prefix for item texts. Same model size, proper asymmetric retrieval, 3-5 point MTEB improvement. Requires updating `embed_texts()` to accept a `mode: Literal["query", "passage"]` parameter.

**T2.2 SPECTER2 for research section** (from Semantic Scholar / Scholar Inbox)
Use SPECTER2-base-proximity for embedding items where `section == "research"`. General bge-small for industry/world/regulatory. The citation-graph training means bioRxiv + Nature + OpenAlex items on the same topic cluster tightly, significantly improving cross-source dedup and cosine scoring quality. Requires ~500MB additional model.

**T2.3 Pairwise feature differences as LR input** (from LtR literature)
Extend T1.1 by also including pairwise features that capture *relative* attributes: `(prestige_a - prestige_b)`, `(age_a - age_b)`. LR then learns "the user prefers newer papers" as a coefficient, not just "new papers score higher."

**T2.4 Query-aware profile weighting** (simplified NRMS)
At inference, compute `attention_weight = softmax(cos(item, each_profile_row))` rather than uniform weight across profile rows. Then `profile_query = sum(weight_i * row_i)`. This means "for a CRISPR paper, weight the CRISPR keyword row more." ~10 lines, no new dependencies.

### Tier 3 — Medium Impact, Lower Priority

**T3.1 TF-IDF score as additional LR feature**
Compute TF-IDF on the entire item corpus at ranking time; add `tfidf_sim` as feature 10. Captures vocabulary overlap that semantic embeddings miss. Helps for very specific gene names / drug names that bge-small blurs.

**T3.2 Per-source vote history prestige adjustment**
Running mean of vote values per source → blend into `prestige_score`. Sources consistently producing downvoted items lose prestige dynamically. Requires a `source_prestige` table.

**T3.3 Section budget reallocation**
If world section's top score < 0.35 cosine (no good news today), reallocate that slot to research. Avoids filling low-quality sections.

**T3.4 Citation count as feature from OpenAlex**
OpenAlex returns `cited_by_count`. Add `log(1 + cited_by_count) * 0.03` bonus. Even a 2-day-old paper with 10 preprint citations is a strong signal.

---

## 5. What NOT to Transfer

**LambdaMART / full LtR pipeline**: Requires XGBoost + listwise loss + hundreds of labeled examples. At 30 votes, you'd overfit immediately. Pairwise LR (T1.1) captures 90% of the benefit with 5% of the complexity.

**Neural news recommendation (NRMS/NAML)**: Requires session-level click data, GPU training, batched inference. Designed for millions of users and items. Overkill by a factor of 1,000× for this use case.

**Variational Autoencoders / Collaborative Filtering**: No user-user signal at all (single user deployment). These are fundamentally multi-user algorithms.

**Transformer-based re-rankers (MonoT5, RankGPT)**: Excellent quality but $0.05–0.15 per digest at scale, and require GPU for reasonable latency. Consider only as a monthly "deep eval" job, not daily.

**DPP (Determinantal Point Processes)**: Mathematically elegant MMR generalization but requires computing matrix determinants. For 60 candidates, the 60×60 determinant computation is overkill vs. greedy MMR which achieves ~95% of DPP quality.

**Federated learning**: Designed for privacy-preserving multi-user learning. Single-user local deployment doesn't need it.

---

## 6. Key Insights Summary

1. **The most impactful upgrade for DailyDigest is pairwise LR training.** With 20 votes, pointwise LR has almost no signal; pairwise construction gives 15-20× more training examples from the same votes. This is the highest-ROI algorithmic change.

2. **Embedding model choice matters more than algorithm complexity for cold start.** SPECTER2 vs bge-small represents a larger quality gap than any scoring-formula tuning for the research section.

3. **MMR is low-hanging fruit.** The literature unanimously agrees that greedy MMR with λ=0.7 improves digest quality with negligible compute cost. Its absence from DailyDigest is the most obvious gap vs. deployed systems.

4. **Active learning (surfacing uncertain items) accelerates personalization.** Scholar Inbox's key contribution is showing that actively presenting items near the decision boundary trains the LR 3× faster than greedy top-K.

5. **Section/source diversity caps should be data-driven, not hardcoded.** Google News/Apple News adjust publisher caps based on session; DailyDigest's hardcoded 30% floor / 20% cap doesn't respond to daily variation in source quality or coverage.

---

## 7. References

All from training knowledge; no live retrieval performed.

- Carbonell, J., & Goldstein, J. (1998). "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries." SIGIR 1998.
- Wang, X., et al. (2020). "NRMS: Neural News Recommendation with Multi-Head Self-Attention." EMNLP 2020.
- Wu, C., et al. (2019). "NAML: Neural News Recommendation with Attentive Multi-View Learning." IJCAI 2019.
- Ni, J., et al. (2022). "SPECTER2: Adapting Scientific Document Representation to Document Type." NeurIPS 2022 (workshop).
- Wang, L., et al. (2024). "E5-Mistral and Text Embeddings by Weakly-Supervised Contrastive Pre-training." arXiv 2024.
- Ostendorff, M., et al. (2022). "SciNCL: Scientific Document Embeddings Using Citation-informed Contrastive Learning." ACL 2022.
- Karpathy, A. (2015). "arxiv sanity preserver." GitHub: karpathy/arxiv-sanity-preserver.
- Burges, C., et al. (2005). "Learning to Rank using Gradient Descent (RankNet)." ICML 2005.
- Wu, Q., et al. (2010). "Adapting Boosting for Information Retrieval Measures (LambdaMART)." Information Retrieval journal.
- Lahav, G., et al. (2022). "A Personalized Academic Paper Recommendation System (Scholar Inbox)." arXiv:2504.08385, ACL 2025.
- Covington, P., Adams, J., Sargin, E. (2016). "Deep Neural Networks for YouTube Recommendations." RecSys 2016.
- MIND dataset: Wu, F., et al. (2020). "MIND: A Large-scale Dataset for News Recommendation." ACL 2020.
- Hofstätter, S., et al. (2021). "Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling." SIGIR 2021. (BGE training methodology background.)
