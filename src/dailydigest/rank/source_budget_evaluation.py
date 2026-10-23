"""Offline paired trial of the unknown-journal frequency budget.

Scores, input order, eligibility, near-duplicate decisions and source facts are
shared across arms. Only the unknown-journal cap changes. Reused provisional
fixture labels cannot authorize a serving-policy change.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

from .cold_start_evaluation import compare_slates, evaluate_first_slate
from .cross_field_evaluation import load_fixture, replay_offline


TRIAL_PLAN = {
    "schema_version": 1,
    "baseline": "unchanged serving unknown-journal frequency budget",
    "candidate": "unknown-journal cap equals existing dynamic research section cap",
    "fixed": ["captured candidate pools", "scores and topic gates", "quality/retraction gates",
              "within-day deduplication", "source diversity", "known-low-impact budget",
              "unknown and low-impact relevance floor", "preprint budget",
              "dynamic section sizes", "final-score floor", "reading mode"],
    "analysis": "all eight fixture profiles; both full slate and first five; no tuning",
    "labels": "reused provisional judgments; unjudged remains unknown",
    "release_approved": False,
    "promotion_requirement": "independent reader evidence and retrieval-recall evaluation",
}


def _label_counts(corpus, decisions):
    labels = {(row['profile_id'], row['url']): row['label'] for row in corpus['cases']}
    output = {}
    for entry in corpus['profiles']:
        picked = [row for row in decisions if row['profile_id'] == entry['id'] and row['selected']]
        counts = Counter(labels.get((row['profile_id'], row['url']), 'unjudged') for row in picked)
        output[entry['id']] = {'slate_length': len(picked), **{
            label: counts[label] for label in ('aligned', 'off_task', 'uncertain', 'unjudged')}}
    return output


def evaluate_source_budget_trial(corpus, baseline, candidate):
    """Reject a confounded comparison; distinguish unknown labels from negatives."""
    left = {(row['profile_id'], row['url']): row for row in baseline}
    right = {(row['profile_id'], row['url']): row for row in candidate}
    if len(left) != len(baseline) or len(right) != len(candidate) or left.keys() != right.keys():
        raise ValueError('Paired trial requires identical unique candidate populations')
    for key, row in left.items():
        for field in ('eligible', 'features', 'stages', 'embedding_provenance'):
            if row.get(field) != right[key].get(field):
                raise ValueError(f'Source budget trial changed upstream {field}')
        before_trace = row.get('selection_trace')
        after_trace = right[key].get('selection_trace')
        if not isinstance(before_trace, dict) or not isinstance(after_trace, dict):
            raise ValueError('Source budget trial requires complete selection_trace')
        override = 'experimental_unknown_journal_cap'
        if before_trace.get(override) is not None:
            raise ValueError('Baseline must use the default unknown journal cap')
        budget = before_trace.get('section_budgets')
        cap = after_trace.get(override)
        if not isinstance(budget, dict):
            raise ValueError('Source budget trial requires section budgets')
        research_cap = budget.get('research', 0)
        if (type(cap) is not int or cap < 0 or type(research_cap) is not int
                or research_cap < 0 or cap != research_cap):
            raise ValueError('Candidate unknown journal cap must equal the research section budget')
        # Only the declared intervention and measured wall time may differ.
        # Profile text, all fixed controls, dedupe decisions and section caps
        # remain part of the comparison, including future trace fields.
        observational = {override, 'elapsed_seconds'}
        before_fixed = {k: v for k, v in before_trace.items() if k not in observational}
        after_fixed = {k: v for k, v in after_trace.items() if k not in observational}
        if before_fixed != after_fixed:
            raise ValueError('Source budget trial changed fixed selection_trace')
    return {
        'plan': TRIAL_PLAN,
        'baseline': _label_counts(corpus, baseline),
        'candidate': _label_counts(corpus, candidate),
        'baseline_first_session': evaluate_first_slate(corpus, baseline),
        'candidate_first_session': evaluate_first_slate(corpus, candidate),
        'comparison': compare_slates(corpus, baseline, candidate),
        'release_approved': False,
    }


def replay_source_budget_trial(corpus, *, state, model_cache):
    """Run once in an isolated child process, with the replay's network prohibition."""
    from . import cross_field_evaluation

    original_selection = cross_field_evaluation._select_captured_research
    records = []

    def paired_picker(scored, caps, **kwargs):
        from .ranker import pick_top_per_section as original_picker
        from .snapshot import capture_selection_facts

        source_facts = kwargs.get('selection_facts')
        if source_facts is None:
            source_facts = capture_selection_facts([row for row, _ in scored])
        started = time.perf_counter()
        baseline = original_picker(scored, caps, **kwargs)
        baseline_seconds = time.perf_counter() - started
        started = time.perf_counter()
        candidate = original_picker(scored, caps, **kwargs,
                                    experimental_unknown_journal_cap=int(caps.get('research', 0)))
        from ..pipeline import _restore_research_order
        candidate = _restore_research_order(candidate, scored)
        candidate_seconds = time.perf_counter() - started
        records.append({
            'candidate_positions': {row.url: index for index, (row, _) in enumerate(candidate, 1)},
            'section_budgets': caps,
            'source_facts': {row.url: source_facts[row.id] for row, _ in scored},
            'baseline_selection_seconds': baseline_seconds,
            'candidate_selection_seconds': candidate_seconds,
        })
        return baseline

    def paired_selection(*args, **kwargs):
        # Import serving modules only after replay has installed isolated settings.
        from .. import pipeline

        with patch.object(pipeline, 'pick_top_per_section', paired_picker):
            return original_selection(*args, **kwargs)

    started = time.perf_counter()
    with patch.object(cross_field_evaluation, '_select_captured_research', paired_selection):
        baseline = replay_offline(corpus, state=state, model_cache=model_cache)
    if len(records) != len(corpus['profiles']):
        raise ValueError('Expected one selection per profile; trial assumptions no longer hold')
    # replay_offline explicitly uses exploration_slots=0 and full reading mode.
    # Assert those assumptions before treating the captured candidate as final.
    if any(row['selection_trace']['exploration_slots'] != 0
           or row['selection_trace']['reading_mode'] != 'full' for row in baseline):
        raise ValueError('Source trial requires no exploration and full reading mode')
    by_profile = {entry['id']: record for entry, record in zip(corpus['profiles'], records, strict=True)}
    for row in baseline:
        row['selection_trace']['experimental_unknown_journal_cap'] = None
    candidate = deepcopy(baseline)
    for row in candidate:
        record = by_profile[row['profile_id']]
        row['selected_position'] = record['candidate_positions'].get(row['url'])
        row['selected'] = row['selected_position'] is not None
        row['selection_trace']['experimental_unknown_journal_cap'] = record['section_budgets'].get('research', 0)
    report = evaluate_source_budget_trial(corpus, baseline, candidate)
    report.update(elapsed_seconds=time.perf_counter() - started, selection_trials=by_profile,
                  runtime_scope='one cached-model offline replay plus two picker calls per profile; '
                                'selection timings are single observations, not speedup evidence')
    return baseline, candidate, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--model-cache', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture = args.fixture.resolve()
    corpus = load_fixture(fixture)
    source_root = Path(__file__).resolve().parent
    source_paths = [source_root / name for name in (
        'ranker.py', 'source_budget_evaluation.py', 'cross_field_evaluation.py',
        'cold_start_evaluation.py', 'source_quality.py', 'snapshot.py', 'profile.py',
        'embed.py', 'embedding_cache.py', 'contribution_claims.py')]
    source_paths.extend(source_root.parent / name for name in (
        'pipeline.py', 'dedupe.py', 'config.py', 'models.py'))
    from .cross_field_evaluation import _source_configuration

    provenance = {'profile_input': 'provided',
                  'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
                  'code_sha256': {str(path.relative_to(source_root.parent)): hashlib.sha256(
                      path.read_bytes()).hexdigest() for path in source_paths},
                  'source_configuration_sha256': hashlib.sha256(
                      _source_configuration().read_bytes()).hexdigest()}
    # Written before observing outcomes so the single candidate policy is fixed.
    (output / 'plan.json').write_text(json.dumps({**TRIAL_PLAN, **provenance}, indent=2) + '\n')
    baseline, candidate, report = replay_source_budget_trial(
        corpus, state=args.state, model_cache=args.model_cache)
    for name, decisions in [('baseline', baseline), ('candidate', candidate)]:
        (output / f'{name}.json').write_text(json.dumps({**provenance, 'decisions': decisions}, indent=2) + '\n')
    (output / 'report.json').write_text(json.dumps({**report, **provenance}, indent=2) + '\n')


if __name__ == '__main__':
    main()
