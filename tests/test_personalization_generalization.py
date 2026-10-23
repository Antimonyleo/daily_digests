from __future__ import annotations

import json

import numpy as np

from dailydigest.models import Profile, SourceSpec


TOPICS = [
    ('DNA nanotechnology', 5), ('RNA nanotechnology', 5),
    ('nucleic acid delivery', 15), ('mRNA therapeutics', 10),
    ('gene therapy', 10), ('nucleic acid therapeutics', 15),
    ('Target protein degradation', 15), ('Induce proximity', 15),
    ('Drug development', 10),
]


def test_all_nine_interests_are_valid_and_overview_does_not_hide_selected_topics(monkeypatch):
    from dailydigest import web

    parsed, errors = web._parse_ranked_topics('\n'.join(f'{name} | {weight}' for name, weight in TOPICS))
    assert not errors and parsed == TOPICS and sum(weight for _, weight in parsed) == 100
    monkeypatch.setattr(web.health, 'latest_snapshot', lambda: [])
    sections = [{'entries': [{'ranking': {'primary_facet': name}} for name, _ in TOPICS]}]
    assert {row['title'] for row in web._digest_overview(sections)['topic_mix']} == {name for name, _ in TOPICS}


def test_aliases_and_description_are_used_for_retrieval_and_serving_relevance(monkeypatch):
    from dailydigest import config
    from dailydigest.ingest import _terms
    from dailydigest.rank import profile as ranking

    profile = Profile(bio='', keywords=['Target protein degradation'], canonical_facets={
        'Target protein degradation': {'aliases': ['targeted protein degradation', 'PROTAC'],
                                      'description': 'Design degraders that recruit E3 ligases.'},
    })
    monkeypatch.setattr(config, 'load_profile', lambda: profile)
    assert _terms.profile_search_terms() == ['Target protein degradation', 'targeted protein degradation', 'PROTAC']
    calls = []
    monkeypatch.setattr(ranking, 'embed_texts', lambda texts, **kw: calls.append(texts) or np.ones((len(texts), 3)))
    ranking.build_profile_matrix(profile)
    assert len(calls[0]) == 1  # synonyms do not create extra scoring votes
    assert all(text in calls[0][0] for text in ['PROTAC', 'E3 ligases', 'Target protein degradation'])


def test_pubmed_reserves_query_and_item_budget_for_every_core_interest(monkeypatch):
    from dailydigest import config
    from dailydigest.ingest.pubmed import PubMedSource

    profile = Profile(bio='', keywords=[name for name, _ in TOPICS])
    monkeypatch.setattr(config, 'load_profile', lambda: profile)
    calls = []
    monkeypatch.setattr(PubMedSource, '_esearch', lambda self, client, query: calls.append((query, self.RETMAX)) or [])
    monkeypatch.setattr('dailydigest.ingest.pubmed.time.sleep', lambda *_: None)
    source = PubMedSource()
    assert source.fetch(SourceSpec(name='PubMed', kind='pubmed', profile_driven=True)) == []
    primary = [(query, cap) for query, cap in calls if ' NOT ' not in query]
    assert len(primary) == 9 and len(calls) <= 18
    assert sum(cap for _, cap in primary) == source.MAX_ITEMS
    assert all(sum(name in query for query, _ in primary) == 1 for name, _ in TOPICS)
    assert [d['interest'] for d in source.last_fetch_diagnostics['queries'] if d['search_stage'] == 'phrase'] == sorted(profile.keywords, key=str.casefold)


def test_topic_details_reject_unknown_interest_and_excessive_aliases():
    from dailydigest import web

    _, errors = web._parse_topic_details(json.dumps({'unknown': {'aliases': ['x']}}), [('known', 1)])
    assert errors
    _, errors = web._parse_topic_details(json.dumps({'known': {'aliases': ['a', 'b', 'c', 'd']}}), [('known', 1)])
    assert errors


def test_interest_coverage_never_calls_zero_selection_zero_supply(monkeypatch):
    from dailydigest import web

    profile = {'keywords': [name for name, _ in TOPICS]}
    monkeypatch.setattr(web, '_profile_data', lambda **_: profile)
    rows = web._interest_coverage([], [])
    assert len(rows) == 9
    assert all(row['selected'] is None and row['candidates'] is None for row in rows)
    assert all('not recorded' in row['explanation'].lower() for row in rows)


def test_interest_audit_distinguishes_scored_matches_from_search_supply():
    from dailydigest.pipeline import _interest_coverage_audit

    rows = _interest_coverage_audit({
        'profile': {'keywords': ['Target protein degradation', 'Induce proximity']},
        'source_health': [{'diagnostics': {'queries': [
            {'interest': 'Target protein degradation', 'truncated': True},
            {'interest': 'Induce proximity', 'failure': 'timeout'},
        ]}}],
        'candidates': [
            {'features': {'primary_facet': 'Target protein degradation'}, 'eligible': True},
            {'features': {'primary_facet': 'Target protein degradation'}, 'first_rejected_stage': 'after_topic_gate'},
            {'features': {}, 'first_rejected_stage': 'after_quality_gate'},
        ],
    })
    assert rows[0]['candidates'] == 2 and rows[0]['eligible'] == 1
    assert rows[0]['queries'] == ['limited']
    assert rows[0]['rejections'] == {'selection budget': 1, 'after_topic_gate': 1}
    assert rows[1]['candidates'] == 0 and rows[1]['queries'] == ['failed']
    assert 'not all retrieved papers' in rows[1]['explanation']


def test_setup_saves_optional_details_and_legacy_weights(tmp_path, monkeypatch):
    import asyncio
    from urllib.parse import urlencode
    from starlette.requests import Request
    import yaml
    from dailydigest import web

    path = tmp_path / 'profile.yaml'
    monkeypatch.setattr(web, '_get_profile_path', lambda: path)
    monkeypatch.setattr(web, '_ENV_PATH', tmp_path / '.env')
    body = urlencode({'_csrf_token': web._CSRF_TOKEN, 'topics': 'Target protein degradation | 75\nInduce proximity | 25',
                      'topic_details': json.dumps({'Target protein degradation': {'aliases': ['PROTAC'], 'description': 'Validated degrader design methods.'}}),
                      'llm_backend': 'extractive', 'user_tz': 'UTC'}).encode()
    async def receive():
        return {'type': 'http.request', 'body': body, 'more_body': False}
    request = Request({'type': 'http', 'method': 'POST', 'path': '/setup', 'query_string': b'', 'headers': [(b'host', b'testserver'), (b'content-type', b'application/x-www-form-urlencoded')], 'client': ('testclient', 1), 'server': ('testserver', 80), 'scheme': 'http'}, receive)
    assert asyncio.run(web.setup_post(request)).status_code == 303
    saved = yaml.safe_load(path.read_text())
    assert saved['keywords'] == ['Target protein degradation', 'Induce proximity']
    assert saved['canonical_facets']['Target protein degradation']['aliases'] == ['PROTAC']
    assert saved['canonical_facets']['Target protein degradation']['description'] == 'Validated degrader design methods.'
    assert saved['canonical_facets']['Target protein degradation']['priority'] == 75
    defaults = web._load_existing_form_defaults()
    assert json.loads(defaults['topic_details'])['Target protein degradation']['aliases'] == ['PROTAC']


def test_ingest_cancellation_stops_before_next_source(monkeypatch):
    from dailydigest import pipeline

    fetched = []
    monkeypatch.setattr(pipeline, 'init_db', lambda: None)
    monkeypatch.setattr(pipeline, 'load_sources', lambda: [SourceSpec(name=name, kind='rss', url='https://example.test/feed') for name in ['first', 'second']])
    class Source:
        def fetch(self, spec, **kwargs):
            fetched.append(spec.name)
            return []
    monkeypatch.setattr(pipeline, 'dispatch_source', lambda spec: Source())
    import pytest
    with pytest.raises(pipeline.BrewCancelled):
        pipeline.ingest_all(should_cancel=lambda: bool(fetched))
    assert fetched == ['first']


def test_optional_context_keeps_a_bounded_share_after_core_searches(monkeypatch):
    from dailydigest import config
    from dailydigest.ingest.pubmed import PubMedSource

    profile = Profile(bio='', keywords=['specialist one', 'specialist two'], context_keywords=['industry context'])
    monkeypatch.setattr(config, 'load_profile', lambda: profile)
    calls = []
    monkeypatch.setattr(PubMedSource, '_esearch', lambda self, client, query: calls.append((query, self.RETMAX)) or [])
    monkeypatch.setattr('dailydigest.ingest.pubmed.time.sleep', lambda *_: None)
    PubMedSource().fetch(SourceSpec(name='PubMed', kind='pubmed', profile_driven=True))
    primary = [(query, cap) for query, cap in calls if ' NOT ' not in query]
    assert len(primary) == 3 and len(calls) <= 6 and sum(cap for _, cap in primary) == 100
    assert 'industry context' in primary[-1][0] and primary[-1][1] == 5


def test_plain_profiles_retain_identical_semantic_inputs():
    from dailydigest.rank.profile import _profile_parts_with_weights

    profile = Profile(bio='', keywords=[name for name, _ in TOPICS])
    parts, weights = _profile_parts_with_weights(profile)
    assert parts == profile.keywords and weights == [1.0] * 9


def test_explicit_facet_details_work_with_distinct_legacy_keywords(monkeypatch):
    from dailydigest.rank import profile as ranking

    profile = Profile(bio='', keywords=['specific legacy retrieval phrase'], canonical_facets={
        'Target protein degradation': {'aliases': ['PROTAC'], 'description': 'E3 ligase degrader design.'},
    })
    parts, _ = ranking._profile_parts_with_weights(profile)
    assert parts[0] == 'specific legacy retrieval phrase'
    assert len(parts) == 2 and 'PROTAC' in parts[1] and 'E3 ligase' in parts[1]


def test_percentage_editor_uses_exact_largest_remainders_with_positive_floor():
    import shutil
    import subprocess
    from pathlib import Path
    import pytest

    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required to check enhanced percentage math')
    source = (Path(__file__).parents[1] / 'templates/setup.html.j2').read_text()
    function = source.split('  function percentages(', 1)[1].split('  let baselineFormState', 1)[0]
    script = 'function percentages(' + function
    script += '\nconsole.log(JSON.stringify([[1,3],[1,1,1],[0,0],[0.001,999.999],[]].map(weights => percentages(weights))));'
    result = subprocess.run([node, '-e', script], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [[25, 75], [34, 33, 33], [50, 50], [1, 99], []]


def test_all_personalized_provider_failures_stop_generic_feed_replacement(monkeypatch):
    _personalized_coverage_case(monkeypatch, 'failed', must_stop=True)


def test_one_personalized_provider_can_be_healthy_empty(monkeypatch):
    _personalized_coverage_case(monkeypatch, 'empty', must_stop=False)


def test_one_personalized_provider_failure_does_not_block_a_healthy_peer(monkeypatch):
    _personalized_coverage_case(monkeypatch, 'healthy', must_stop=False)


def _personalized_coverage_case(monkeypatch, second_state, *, must_stop):
    from types import SimpleNamespace
    import pytest
    from dailydigest import pipeline

    specs = [SourceSpec(name='Personal OpenAlex', kind='openalex', profile_driven=True),
             SourceSpec(name='Personal PubMed', kind='pubmed', profile_driven=True),
             SourceSpec(name='Generic publisher', kind='rss', url='https://publisher.test/rss'),
             SourceSpec(name='Generic preprints', kind='arxiv', query='all:science')]
    writes = []
    class Source:
        def fetch(self, spec, **kwargs):
            if spec.name == 'Personal OpenAlex' or (spec.name == 'Personal PubMed' and second_state == 'failed'):
                raise RuntimeError('provider unavailable')
            if spec.name == 'Personal PubMed' and second_state == 'empty':
                return []
            return [SimpleNamespace(url=f'https://example.test/{spec.name}/{index}', section='research') for index in range(12)]
    monkeypatch.setattr(pipeline, 'init_db', lambda: None)
    monkeypatch.setattr(pipeline, 'load_sources', lambda: specs)
    monkeypatch.setattr(pipeline, 'dispatch_source', lambda spec: Source())
    monkeypatch.setattr(pipeline.health, 'record', lambda *_: None)
    monkeypatch.setattr(pipeline, 'dedupe_by_url', lambda rows: rows)
    monkeypatch.setattr(pipeline, 'filter_english', lambda rows: rows)
    monkeypatch.setattr(pipeline, 'upsert_items', lambda rows: writes.append(len(rows)) or len(rows))
    if must_stop:
        with pytest.raises(RuntimeError, match='Personalized research source coverage is degraded'):
            pipeline.ingest_all()
        assert writes == []
    else:
        assert pipeline.ingest_all() >= 24
        assert len(writes) == 1


def test_openalex_alias_rich_interest_cannot_take_another_interests_budget(monkeypatch):
    from collections import Counter
    from dailydigest import config
    from dailydigest.ingest import openalex

    names = [name for name, _ in TOPICS]
    profile = Profile(bio='', keywords=names, canonical_facets={
        'Target protein degradation': {'aliases': ['targeted protein degradation', 'PROTAC', 'degraders']},
    })
    monkeypatch.setattr(config, 'load_profile', lambda: profile)
    def fetch(_url, params, _headers):
        return {'results': [{'id': f"https://openalex.org/{params['search.title_abstract_keywords']}-{i}", 'title': f"{params['search.title_abstract_keywords']}-{i}"} for i in range(int(params['per-page']))], 'meta': {'count': 999, 'next_cursor': 'more'}}
    monkeypatch.setattr(openalex, '_get_json', fetch)
    source = openalex.OpenAlexSource()
    assert len(source.fetch(SourceSpec(name='OpenAlex', kind='openalex', profile_driven=True))) == 180
    budgets = Counter()
    for query in source.last_fetch_diagnostics['queries']:
        budgets[query['interest']] += query['fetched_count']
    assert budgets == {name: 20 for name in names}
    assert len(source.last_fetch_diagnostics['queries']) == 9


def test_percentage_editor_submits_displayed_weights_and_blocks_invalid_totals():
    import shutil
    import subprocess
    from pathlib import Path
    import pytest

    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required to verify enhanced editor serialization')
    source = (Path(__file__).parents[1] / 'templates/setup.html.j2').read_text()
    functions = 'function serialize() {' + source.split('  function serialize() {', 1)[1].split('  function finishEditing(', 1)[0]
    submit = "topics.form.addEventListener('submit', event => {" + source.split("  topics.form.addEventListener('submit', event => {", 1)[1].split('  const exampleButton', 1)[0]
    script = '''
const assert = require('node:assert/strict');
function row(name, raw, displayed) {
  const fields = {'[data-interest-name]': {value:name}, '[data-interest-weight]':{value:displayed, setCustomValidity(message) { this.error = message; }}, '[data-interest-aliases]':{value:''}, '[data-interest-description]':{value:''}};
  return {dataset:{rawWeight:raw}, querySelector: key => fields[key]};
}
const children = [row('A','0.1','1'), row('B','100','99')];
const rows = {children, querySelectorAll: key => children.map(row => row.querySelector(key))};
const topics = {value:''}, details = {value:''};
const controls = new Map();
const document = {getElementById(id) {if (!controls.has(id)) controls.set(id, {}); return controls.get(id);}};
let baselineFormState = null, weightsTouched = false, loadedCount = children.length, activeRow = null;
let onSubmit, reported = 0;
topics.form = {
  addEventListener(type, callback) {assert.equal(type, 'submit'); onSubmit = callback;},
  checkValidity() {return children.every(row => !row.querySelector('[data-interest-weight]').error);},
  reportValidity() {reported++;}
};
''' + functions + submit + '''
serialize(); assert.equal(topics.value, 'A | 0.1\\nB | 100');
assert.equal(controls.get('topic-weight-mode').value, 'relative');
children[0].querySelector('[data-interest-description]').value = 'New reader detail';
serialize(); assert.equal(topics.value, 'A | 0.1\\nB | 100');
assert.equal(JSON.parse(details.value).A.description, 'New reader detail');
children[0].querySelector('[data-interest-name]').value = 'Renamed';
children[0].dataset.correctedFrom = 'A';
serialize(); assert.equal(topics.value, 'Renamed | 0.1\\nB | 100');
assert.equal(JSON.parse(details.value).Renamed.previous_name, 'A');
const weight = children[0].querySelector('[data-interest-weight]');
assert.equal(weight.error, '');
weightsTouched = true;
weight.value = '10'; children[1].querySelector('[data-interest-weight]').value = '90';
serialize(); assert.equal(topics.value, 'Renamed | 10\\nB | 90');
assert.equal(controls.get('topic-weight-mode').value, 'percentage');
assert.equal(weight.error, '');
weight.value = '15';
let prevented = false;
onSubmit({preventDefault() {prevented = true;}});
assert.match(weight.error, /totaling 100%/);
assert.equal(prevented, true); assert.equal(reported, 1);
weight.value = '10'; prevented = false;
onSubmit({preventDefault() {prevented = true;}});
assert.equal(weight.error, ''); assert.equal(prevented, false);
assert.equal(topics.value, 'Renamed | 10\\nB | 90');
console.log('ok');
'''
    result = subprocess.run([node, '-e', script], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == 'ok'


def test_openalex_overlapping_aliases_do_not_shrink_unique_supply(monkeypatch):
    from dailydigest import config
    from dailydigest.ingest import openalex

    profile = Profile(bio='', keywords=['protein degradation'], canonical_facets={
        'protein degradation': {'aliases': ['targeted protein degradation', 'PROTAC', 'degraders']},
    })
    monkeypatch.setattr(config, 'load_profile', lambda: profile)
    requests = []
    def fetch(_url, params, _headers):
        requests.append(dict(params))
        # All synonymous queries find the same papers, as real indexes do.
        return {'results': [{'id': f'https://openalex.org/work-{i}', 'title': f'Paper {i}'} for i in range(int(params['per-page']))], 'meta': {'count': 100, 'next_cursor': 'more'}}
    monkeypatch.setattr(openalex, '_get_json', fetch)
    source = openalex.OpenAlexSource()
    assert len(source.fetch(SourceSpec(name='OpenAlex', kind='openalex', profile_driven=True))) == 20
    assert len(requests) == 2
    assert requests[0]['search.title_abstract_keywords'] == 'protein degradation'
    assert requests[1]['search.title_abstract_keywords'] == '"targeted protein degradation" OR "PROTAC" OR "degraders"'
    queries = source.last_fetch_diagnostics['queries']
    assert all(query['interest'] == 'protein degradation' for query in queries)
    assert [query['query_role'] for query in queries] == ['core', 'alias']
    assert [query['unique_yield'] for query in queries] == [20, 0]
    assert sum(query['fetched_count'] for query in queries) == 40
    assert sum(query['fetched_count'] for query in queries) <= source.PROFILE_MAX_ITEMS


def test_setup_preserves_canonical_priority_fallback_by_legacy_alias(tmp_path, monkeypatch):
    from dailydigest import web
    from dailydigest.rank.profile import _normalized_priorities
    import yaml

    path = tmp_path / 'profile.yaml'
    profile = {'bio': '', 'keywords': ['A', 'B'],
               'canonical_facets': {'A': {'priority': None, 'aliases': ['original A']}, 'B': {'priority': None}},
               'topic_priorities': {'ORIGINAL A': 4, 'b': 1}}
    path.write_text(yaml.safe_dump(profile))
    monkeypatch.setattr(web, '_get_profile_path', lambda: path)
    original = _normalized_priorities(Profile.model_validate(profile))
    form = web._load_existing_form_defaults()
    topics, errors = web._parse_ranked_topics(form['topics'])
    assert not errors and topics == [('A', 4), ('B', 1)]
    rebuilt = Profile(bio='', keywords=['A', 'B'], canonical_facets={name: {'priority': weight} for name, weight in topics})
    assert _normalized_priorities(rebuilt) == original


def test_pubmed_successful_context_cannot_hide_all_core_failures(monkeypatch):
    from dailydigest import config
    from dailydigest.ingest.pubmed import PubMedSource
    import pytest

    profile = Profile(bio='', keywords=['specialist research'], context_keywords=['context news'])
    monkeypatch.setattr(config, 'load_profile', lambda: profile)
    monkeypatch.setattr('dailydigest.ingest.pubmed.time.sleep', lambda *_: None)
    def search(self, client, query):
        if 'specialist research' in query:
            raise RuntimeError('core provider search failed')
        return []
    monkeypatch.setattr(PubMedSource, '_esearch', search)
    source = PubMedSource()
    with pytest.raises(RuntimeError, match='all core profile searches failed'):
        source.fetch(SourceSpec(name='PubMed', kind='pubmed', profile_driven=True))
    assert source.last_fetch_diagnostics['queries'][-1]['interest'] == 'context news'
    assert source.last_fetch_diagnostics['queries'][-1]['exhausted']


def test_pubmed_context_only_profile_can_be_healthy_empty(monkeypatch):
    from dailydigest import config
    from dailydigest.ingest.pubmed import PubMedSource

    monkeypatch.setattr(config, 'load_profile', lambda: Profile(bio='', context_keywords=['context news']))
    monkeypatch.setattr(PubMedSource, '_esearch', lambda *_: [])
    assert PubMedSource().fetch(SourceSpec(name='PubMed', kind='pubmed', profile_driven=True)) == []


def test_unlisted_canonical_priority_keeps_effective_half_on_unrelated_save(tmp_path, monkeypatch):
    import yaml
    from dailydigest import web
    from dailydigest.rank.profile import _normalized_priorities

    original = {'bio': '', 'keywords': ['A', 'B'], 'canonical_facets': {'A': {'priority': 18}, 'B': {}}}
    path = tmp_path / 'profile.yaml'
    path.write_text(yaml.safe_dump(original))
    monkeypatch.setattr(web, '_get_profile_path', lambda: path)
    topics, errors = web._parse_ranked_topics(web._load_existing_form_defaults()['topics'])
    assert not errors and topics == [('A', 18), ('B', 9)]
    before = _normalized_priorities(Profile.model_validate(original))
    after = _normalized_priorities(Profile(bio='', keywords=['A', 'B'], canonical_facets={name: {'priority': priority} for name, priority in topics}))
    assert {name: before.get(name, .5) for name in ['A', 'B']} == after


def test_cancellation_inside_final_request_is_reported_as_stopped(monkeypatch):
    from dailydigest import pipeline
    import pytest

    cancelled, writes = [False], []
    class Source:
        def fetch(self, spec, **kwargs):
            cancelled[0] = True
            raise RuntimeError('adapter wrapped cancellation during the final request')
    monkeypatch.setattr(pipeline, 'init_db', lambda: None)
    monkeypatch.setattr(pipeline, 'load_sources', lambda: [SourceSpec(name='last', kind='rss', url='https://example.test/feed')])
    monkeypatch.setattr(pipeline, 'dispatch_source', lambda _: Source())
    monkeypatch.setattr(pipeline.health, 'record', lambda _: writes.append('health'))
    monkeypatch.setattr(pipeline, 'upsert_items', lambda _: writes.append('items'))
    with pytest.raises(pipeline.BrewCancelled, match='cancelled during source collection'):
        pipeline.ingest_all(should_cancel=lambda: cancelled[0])
    assert writes == []
