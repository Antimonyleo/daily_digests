"""Explicit local contribution reranking after existing eligibility checks.

The continuous signal changes final scores, never article eligibility or source
facts. Models require explicit local paths and a matching reader interest profile.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

import numpy as np

from .scope import fingerprint


SCORE_MEANING = 'Relative contribution ranking signal; not probability, confidence, or reader usefulness'
MIN_ABSTRACT_LENGTH = 40
_BIBLIOGRAPHIC_STUB = re.compile(
    r'^[^.!?]+,\s*(?:Volume\s+\d+,\s*Issue\s+[^.!?]+|Ahead\s+of\s+Print)[.!?]?$', re.I)


def interest_signature(profile):
    if hasattr(profile, 'model_dump'):
        profile = profile.model_dump()
    interests = {key: profile.get(key, default) for key, default in (
        ('bio', ''), ('keywords', []), ('canonical_facets', {}),
        ('negative_interests', []), ('context_keywords', []),
    )}
    # Empty defaults preserve existing unweighted artifacts. Populated fields
    # affect positive topic geometry and must invalidate a stale fitted model.
    for key in ('seed_works', 'interest_weights', 'facet_weights'):
        if profile.get(key):
            interests[key] = profile[key]
    return fingerprint(interests)


def has_abstract(row):
    text = ' '.join((getattr(row, 'abstract', '') or '').split())
    return len(text) >= MIN_ABSTRACT_LENGTH and not _BIBLIOGRAPHIC_STUB.fullmatch(text)


def apply_contribution_scores(scored, features, learned_scores, *, weight=.7, model_fingerprint=''):
    """Validate the entire batch, then update final score features and tuple order."""
    weight = float(weight)
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Contribution weight must be finite and within [0, 1]')
    if weight == 0:
        return list(scored)
    updates, result = [], []
    for row, original in scored:
        original = float(original)
        if not math.isfinite(original):
            raise ValueError('Original ranking scores must be finite')
        score = original
        if row.section == 'research' and has_abstract(row):
            iid = int(row.id)
            if iid not in learned_scores:
                raise ValueError(f'Missing contribution score for article {iid}')
            learned = float(learned_scores[iid])
            if not math.isfinite(learned) or not 0 <= learned <= 1:
                raise ValueError('Contribution scores must be finite and within [0, 1]')
            score = weight * learned + (1 - weight) * original
            updates.append((iid, score, {'original_final_score': original,
                                        'rank_signal': learned, 'weight': weight,
                                        'model_fingerprint': model_fingerprint,
                                        'score_meaning': SCORE_MEANING}))
        result.append((row, score))
    for iid, score, provenance in updates:
        features.setdefault(iid, {}).update(final_score=score, rank_score=score, contribution_rerank=provenance)
    return sorted(result, key=lambda pair: pair[1], reverse=True)


class LocalContributionReranker:
    """Load a numeric ranker and verify its explicitly provisioned encoder."""

    def __init__(self, artifact_path, model_dir, profile):
        from .contribution_ranker import ContributionRanker
        from .contribution_claims import FEATURE_PROTOCOL
        from .cross_encoder import LocalCrossEncoder
        from .pair_scope import pair_queries

        artifact = json.loads(Path(artifact_path).expanduser().read_text(encoding='utf-8'))
        if (artifact.get('schema_version') != 1
                or artifact.get('kind') != 'local_contribution_reranker'
                or artifact.get('fingerprint') != fingerprint({
                    k: v for k, v in artifact.items() if k != 'fingerprint'})):
            raise ValueError('Contribution artifact identity or schema mismatch')
        self.feature_protocol = artifact.get('feature_protocol')
        if 'feature_protocol' in artifact and self.feature_protocol != FEATURE_PROTOCOL:
            raise ValueError('Unsupported contribution feature protocol')
        if artifact.get('profile_interest_signature') != interest_signature(profile):
            raise ValueError('Contribution artifact does not match the reader interest profile')
        self.queries = pair_queries(artifact['definitions'])
        if self.queries != artifact.get('queries'):
            raise ValueError('Contribution artifact query definitions differ')
        self.ranker = ContributionRanker.from_artifact(artifact['ranker'])
        # Validate column identity even when there are no eligible articles.
        self.ranker.predict_scores(np.empty((0, len(self.queries))), queries=self.queries)
        expected = artifact['encoder_provenance']
        self.encoder = LocalCrossEncoder(model_dir, **{
            key: expected[key] for key in ('max_length', 'batch_size', 'threads')})
        if self.encoder.provenance != expected:
            raise ValueError('Local encoder assets/runtime differ from the fitted contribution artifact')
        self.fingerprint = artifact['fingerprint']

    def rerank(self, scored, features, *, weight=.7):
        from .contribution_claims import contribution_text

        rows = [row for row, _ in scored if row.section == 'research' and has_abstract(row)]
        # Bound the temporary pair list to one document; ONNX bounds actual batches.
        logits = [self.encoder.score_pairs([
            (query['text'], contribution_text(row.title or '', row.abstract or '')
             if self.feature_protocol else (row.title or '') + '. ' + (row.abstract or ''))
            for query in self.queries]) for row in rows]
        matrix = np.asarray(logits, dtype=float).reshape(-1, len(self.queries))
        values = self.ranker.predict_scores(matrix, queries=self.queries)
        learned = {int(row.id): float(score) for row, score in zip(rows, values, strict=True)}
        result = apply_contribution_scores(scored, features, learned, weight=weight,
                                           model_fingerprint=self.fingerprint)
        if self.feature_protocol and weight > 0:
            for row in rows:
                features[int(row.id)]['contribution_rerank']['feature_protocol'] = dict(self.feature_protocol)
        return result
