"""Prepare an offline blinded reader study; never infer that people took it.

Public packets contain no arm assignments, prior labels, or scores. The separate
analyst manifest freezes selection membership and population denominators.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .cold_start_evaluation import evaluate_first_slate, review_packet

LABELS = ('useful', 'transferable', 'off_task', 'unsure')
ROLES = ('undergraduate', 'graduate', 'technician', 'faculty', 'industry', 'other')


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def prepare_study(corpus: dict, arms: dict[str, list[dict]], *, missed_per_profile: int = 3,
                  profile_input: str = 'provided', include_all: bool = False) -> tuple[dict, dict]:
    """Take every arm's first five plus a label-independent sample of omissions.

    Input profiles must be the same across arms. Different input experiments need
    separate studies; otherwise readers would judge against an ambiguous goal.
    """
    if type(include_all) is not bool:
        raise ValueError('include_all must be boolean')
    if not arms or type(missed_per_profile) is not int or missed_per_profile < 0:
        raise ValueError('Arms and a nonnegative integer omission sample are required')
    profile_ids = [entry['id'] for entry in corpus['profiles']]
    if len(profile_ids) != len(set(profile_ids)) or any(
            len(entry['pool']) != len({item['url'] for item in entry['pool']})
            for entry in corpus['profiles']):
        raise ValueError('Study profiles and within-profile candidate identities must be unique')
    # Validate all decisions but never use corpus judgments in sample selection.
    unlabeled = {**corpus, 'cases': []}
    for rows in arms.values():
        evaluate_first_slate(unlabeled, rows)
    cases = review_packet(unlabeled, profile_input=profile_input)['cases']
    by_pair = {(c['profile_id'], c['article']['url']): c for c in cases}
    included = set()
    populations = {}
    arm_slates = {}
    for name, rows in arms.items():
        arm_slates[name] = {}
        for profile in corpus['profiles']:
            pid = profile['id']
            selected = sorted((r for r in rows if r['profile_id'] == pid and r['selected']),
                              key=lambda r: r['selected_position'])
            ids = [by_pair[pid, r['url']]['id'] for r in selected]
            arm_slates[name][pid] = ids
            included.update(ids[:5])
    for profile in corpus['profiles']:
        pid = profile['id']
        selected_any = {cid for slates in arm_slates.values() for cid in slates[pid]}
        omitted = [c for c in cases if c['profile_id'] == pid and c['id'] not in selected_any]
        # Stable hash order, independent of previous labels and original pool rank.
        included.update(c['id'] for c in omitted[:missed_per_profile])
        populations[pid] = {'candidate_count': len(profile['pool']),
                            'outside_all_slates': len(omitted),
                            'omissions_sampled': len(omitted) if include_all else min(len(omitted), missed_per_profile)}
    if include_all:
        included.update(c['id'] for c in cases)
    public_cases = [{k: c[k] for k in ('id', 'profile_id', 'intent', 'topic_intents',
                                      'supplied_profile', 'article')}
                    for c in cases if c['id'] in included]
    if not public_cases:
        raise ValueError('Study needs at least one selected or sampled candidate')
    packet = {'schema_version': 1, 'status': 'awaiting_readers', 'profile_input': profile_input,
              'labels': list(LABELS), 'roles': list(ROLES), 'cases': public_cases}
    manifest = {'schema_version': 1,
                'corpus_without_judgments_sha256': _hash(unlabeled),
                'decisions_sha256': {name: _hash(rows) for name, rows in arms.items()},
                'arm_slates': arm_slates, 'populations': populations,
                'sample_policy': ('Every captured candidate' if include_all else
                                  'Union of first five; hash sample outside all full slates'),
                'evidence': corpus.get('evaluation_provenance',
                                       'Reused development corpus; human responses pending; not an untouched holdout'),
                'release_approved': False}
    packet['manifest_sha256'] = _hash(manifest)
    packet['study_id'] = _hash(packet)
    manifest['study_id'] = packet['study_id']
    return packet, manifest


def analyze_responses(packet: dict, manifest: dict, responses: list[dict]) -> dict:
    """Keep each reader's utility judgments and uncertainty separate.

    Roles and independence are self-reported. Skips and disagreement never turn
    into negative labels. This does not measure unobserved retrieval misses.
    """
    unsigned = {k: v for k, v in packet.items() if k != 'study_id'}
    if packet.get('study_id') != _hash(unsigned) or manifest.get('study_id') != packet['study_id']:
        raise ValueError('Study identity mismatch')
    if packet.get('manifest_sha256') != _hash({k: v for k, v in manifest.items() if k != 'study_id'}):
        raise ValueError('Analyst manifest identity mismatch')
    cases = {c['id']: c for c in packet['cases']}
    readers, results, ratings = set(), [], {}
    for response in responses:
        reader = response.get('reader', '').strip()
        if (response.get('schema_version') != 1 or response.get('study_id') != packet['study_id']
                or not reader or reader in readers or response.get('role') not in ROLES
                or type(response.get('independent')) is not bool):
            raise ValueError('Invalid, duplicate, or mismatched reader submission')
        readers.add(reader)
        judgments = {}
        for row in response.get('judgments', []):
            cid, label = row.get('id'), row.get('label')
            if cid not in cases or cid in judgments or label not in LABELS:
                raise ValueError('Unknown, duplicate, or invalid judgment')
            judgments[cid] = label
            ratings.setdefault(cid, []).append(label)
        profile_results = {}
        for pid, population in manifest['populations'].items():
            profile_results[pid] = {}
            for arm, slates in manifest['arm_slates'].items():
                top = slates[pid][:5]
                counts = Counter(judgments.get(cid, 'unjudged') for cid in top)
                positives = counts['useful']
                unknown = counts['unsure'] + counts['unjudged']
                profile_results[pid][arm] = {
                    'first_five_counts': {key: counts[key] for key in (*LABELS, 'unjudged')},
                    'selected_count': len(slates[pid]),
                    'direct_usefulness_bounds': ([positives / len(top), (positives + unknown) / len(top)]
                                                 if top else [None, None]),
                    'sampled_useful_outside_slate': sum(
                        label == 'useful' and cases[cid]['profile_id'] == pid and cid not in slates[pid]
                        for cid, label in judgments.items()),
                    'screen_target': ('met_for_this_reader' if len(top) == 5 and positives >= 4 and not unknown
                                      and all(judgments.get(cid) in {'useful', 'transferable'}
                                              for cid in top[:3]) else 'not_demonstrated'),
                    'population': population,
                }
        results.append({'reader': reader, 'role': response['role'],
                        'independent_self_report': response['independent'],
                        'judged_count': len(judgments), 'profiles': profile_results})
    return {'status': 'responses_received' if readers else 'awaiting_readers',
            'reader_count': len(readers), 'readers': results,
            'disagreement_count': sum(len(set(labels)) > 1 for labels in ratings.values()),
            'release_approved': False,
            'limits': ['Self-reported readers and expertise; independence unverified',
                       manifest['evidence'], 'No unobserved retrieval recall or retention evidence']}


def render_study(packet: dict) -> str:
    template = Path(__file__).resolve().parents[3] / 'templates/reader_study_offline.html.j2'
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    env = Environment(loader=FileSystemLoader(template.parent), autoescape=select_autoescape(['j2']))
    return env.get_template(template.name).render(packet=packet)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--fixture', type=Path, required=True)
    prepare.add_argument('--arm', action='append', required=True, help='NAME=report.json')
    prepare.add_argument('--output', type=Path, required=True, help='New output directory')
    prepare.add_argument('--profile-input', default='provided')
    prepare.add_argument('--missed-per-profile', type=int, default=3)
    prepare.add_argument('--all-candidates', action='store_true', help='Rate the entire captured pool')
    analyze = sub.add_parser('analyze')
    analyze.add_argument('--study', type=Path, required=True)
    analyze.add_argument('--response', type=Path, action='append', default=[])
    analyze.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        corpus = json.loads(args.fixture.read_text())
        arms = {}
        for value in args.arm:
            name, path = value.split('=', 1)
            report = json.loads(Path(path).read_text())
            if not name or name in arms or report.get('profile_input') != args.profile_input:
                raise ValueError('Distinct arm names and identical profile input required')
            expected = hashlib.sha256(args.fixture.read_bytes()).hexdigest()
            if report.get('fixture_sha256') != expected:
                raise ValueError('Arm fixture hash mismatch')
            arms[name] = report['decisions']
        packet, manifest = prepare_study(corpus, arms, missed_per_profile=args.missed_per_profile,
                                         profile_input=args.profile_input, include_all=args.all_candidates)
        args.output.mkdir(parents=True, exist_ok=False)
        for name, value in [('packet', packet), ('analyst-manifest', manifest)]:
            (args.output / f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2))
        (args.output / 'index.html').write_text(render_study(packet))
    else:
        packet = json.loads((args.study / 'packet.json').read_text())
        manifest = json.loads((args.study / 'analyst-manifest.json').read_text())
        responses = [json.loads(path.read_text()) for path in args.response]
        report = analyze_responses(packet, manifest, responses)
        with args.output.open('x') as output:
            output.write(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
