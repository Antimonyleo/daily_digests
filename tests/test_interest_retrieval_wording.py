"""Offline retrieval fixtures; the index matcher is not a live PubMed emulator.

Relevance scores below are fixed reviewer labels, so these tests exercise the
unchanged production gates, not the embedding model's semantic accuracy.
"""
from __future__ import annotations

import json
import re
import sys

import pytest
from types import SimpleNamespace
from xml.sax.saxutils import escape


from dailydigest.models import Profile, SourceSpec

TOPICS = [
    ('DNA nanotechnology', 5), ('RNA nanotechnology', 5),
    ('nucleic acid delivery', 15), ('mRNA therapeutics', 10),
    ('gene therapy', 10), ('nucleic acid therapeutics', 15),
    ('Target protein degradation', 15), ('Induce proximity', 15),
    ('Drug development', 10),
]
CORPUS = {
    '1': ('PROTAC design recruits E3 ubiquitin ligases',
          'We developed targeted protein degradation with bifunctional PROTAC degraders and validated selective removal of a target protein.', .88),
    '2': ('Molecular glues reprogram induced proximity',
          'Molecular glue design induced selective protein proximity and recruited an E3 ligase; a transferable design method was experimentally validated.', .86),
    '3': ('Natural signaling pathways in distant organisms',
          'Target protein degradation and induce proximity are mentioned only as background; this study describes unrelated cellular signaling.', .20),
    '4': ('Sponsored: Target protein degradation products',
          'Induce proximity products and target protein degradation products are advertised without a new experimental method.', .90),
}


def fixture_matches(query: str, text: str, *, openalex: bool = False) -> bool:
    """Evaluate the bounded Boolean/tiab subset our compiler produces."""
    if openalex:
        raw = re.findall(r'"[^"]*"|[\w-]+|[()]', query)
        tokens = []
        for token in raw:
            if (tokens and tokens[-1] not in ('AND', 'OR', '(')
                    and token not in ('AND', 'OR', ')')):
                tokens.append('AND')
            tokens.append(token if token in ('AND', 'OR', '(', ')') else token + '[tiab]')
    else:
        tokens = re.findall(r'"[^"]*"\[tiab\]|[\w-]+\*?\[tiab\]|AND|OR|NOT|[()]', query)
    position = 0
    words = set(re.findall(r'[\w-]+', text.casefold()))

    def atom():
        nonlocal position
        token = tokens[position]
        position += 1
        if token == '(':
            value = disjunction()
            assert tokens[position] == ')'
            position += 1
            return value
        if token == 'NOT':
            return not atom()
        value = token.removesuffix('[tiab]')
        return (value[1:-1].casefold() in text.casefold()
                if value.startswith('"') else any(word.startswith(value[:-1].casefold()) for word in words)
                if value.endswith('*') else value.casefold() in words)

    def conjunction():
        nonlocal position
        value = atom()
        while position < len(tokens) and tokens[position] in ('AND', 'NOT'):
            if tokens[position] == 'AND':
                position += 1
            right = atom()
            value = value and right
        return value

    def disjunction():
        nonlocal position
        value = conjunction()
        while position < len(tokens) and tokens[position] == 'OR':
            position += 1
            right = conjunction()
            value = value or right
        return value

    result = disjunction()
    assert position == len(tokens)
    return result


def install_fixture(monkeypatch, profile):
    from dailydigest import config
    from dailydigest.ingest.pubmed import PubMedSource

    monkeypatch.setattr(config, 'load_profile', lambda: profile)
    monkeypatch.setattr('dailydigest.ingest.pubmed.time.sleep', lambda *_: None)
    calls = []

    def search(self, client, query):
        calls.append((query, self.RETMAX))
        ids = [key for key, (title, abstract, _) in CORPUS.items()
               if fixture_matches(query, title + ' ' + abstract)]
        self.last_fetch_diagnostics['queries'][-1]['total_available'] = len(ids)
        return ids[:self.RETMAX]

    def fetch(self, client, ids):
        return '<PubmedArticleSet>' + ''.join(
            '<PubmedArticle><MedlineCitation><PMID>' + key + '</PMID><Article>'
            '<ArticleTitle>' + escape(CORPUS[key][0]) + '</ArticleTitle>'
            '<Abstract><AbstractText>' + escape(CORPUS[key][1]) + '</AbstractText></Abstract>'
            '<Journal><Title>Nature Methods</Title></Journal></Article></MedlineCitation></PubmedArticle>'
            for key in ids) + '</PubmedArticleSet>'

    monkeypatch.setattr(PubMedSource, '_esearch', search)
    monkeypatch.setattr(PubMedSource, '_efetch', fetch)
    return calls


def test_nine_topic_wording_reaches_the_unchanged_quality_and_relevance_gates(monkeypatch, record_testsuite_property):
    from dailydigest.ingest.pubmed import PubMedSource
    from dailydigest import pipeline
    from dailydigest.rank.ranker import pick_top_per_section

    profile = Profile(bio='', keywords=[name for name, _ in TOPICS],
                      topic_priorities=dict(TOPICS))
    calls = install_fixture(monkeypatch, profile)
    source = PubMedSource()
    items = source.fetch(SourceSpec(name='PubMed', kind='pubmed', profile_driven=True))
    primary = [d for d in source.last_fetch_diagnostics['queries'] if d['search_stage'] == 'phrase']
    assert len(primary) == 9 and len(calls) <= 18
    assert sum(cap for query, cap in calls if ' NOT ' not in query) == 100
    assert sum(d['fetched_count'] for d in source.last_fetch_diagnostics['queries']) <= 100
    # The pre-fix exact phrases miss both research contributions in this corpus.
    before = {key for key, (title, abstract, _) in CORPUS.items()
              if any(fixture_matches(f'"{name}"[tiab]', title + ' ' + abstract)
                     for name, _ in TOPICS)}
    assert before == {'3', '4'}
    retrieved = {item.external_id.rsplit(':', 1)[-1] for item in items}
    assert retrieved == {'1', '2', '3', '4'}
    rows = [SimpleNamespace(id=int(key), source='Nature Methods', section='research',
                            title=CORPUS[key][0], abstract=CORPUS[key][1],
                            external_id=key, url=f'https://example.test/{key}', metadata_json='{}')
            for key in sorted(retrieved)]
    quality, drops = pipeline._quality_gate(rows)
    assert {row.id for row in quality} == {1, 2, 3}
    assert drops[0]['item_id'] == 4
    features = {int(key): {'topic_score': value[2]} for key, value in CORPUS.items()}
    settings = SimpleNamespace(adaptive_section_sizes=True, min_topic_relevance=.65)
    monkeypatch.setattr(pipeline, 'get_settings', lambda: settings)
    relevant = pipeline._filter_off_topic([(row, features[row.id]['topic_score']) for row in quality], features)
    assert {row.id for row, _ in relevant} == {1, 2}
    selected = pick_top_per_section(relevant, {'research': 5}, score_features=features)
    assert {row.id for row, _ in selected} == {1, 2}
    before_quality, _ = pipeline._quality_gate([row for row in rows if str(row.id) in before])
    before_relevant = pipeline._filter_off_topic(
        [(row, features[row.id]['topic_score']) for row in before_quality], features)
    before_selected = pick_top_per_section(before_relevant, {'research': 5}, score_features=features)
    assert [row.id for row in before_quality] == [3]
    assert before_relevant == before_selected == []
    by_interest = {name: [d['query'] for d in source.last_fetch_diagnostics['queries'] if d['interest'] == name] for name, _ in TOPICS}
    assert any(fixture_matches(query, CORPUS['1'][1]) for query in by_interest['Target protein degradation'])
    assert any(fixture_matches(query, CORPUS['2'][1]) for query in by_interest['Induce proximity'])
    record_testsuite_property('stage_trace', json.dumps({
        'limitation': 'Offline fixture index; relevance scores are fixed reviewer labels, not model outputs.',
        'queries': [{'interest': name, 'before': f'"{name}"[tiab]', 'after': by_interest[name],
                     'before_ids': [key for key, (title, abstract, _) in CORPUS.items()
                                    if fixture_matches(f'"{name}"[tiab]', title + ' ' + abstract)],
                     'after_ids': [key for key, (title, abstract, _) in CORPUS.items()
                                   if any(fixture_matches(query, title + ' ' + abstract) for query in by_interest[name])]}
                    for name, _ in TOPICS],
        'before': {'retrieved': sorted(before), 'quality': [r.id for r in before_quality],
                   'relevance': [], 'selected': []},
        'after': {'retrieved': sorted(retrieved), 'quality': [r.id for r in quality],
                  'relevance': [r.id for r, _ in relevant], 'selected': [r.id for r, _ in selected]},
    }))


def test_pubmed_adequate_exact_aliases_do_not_trigger_loose_expansion(monkeypatch):
    from dailydigest.ingest.pubmed import PubMedSource

    profile = Profile(bio='', keywords=['Target protein degradation'], canonical_facets={
        'Target protein degradation': {'aliases': ['molecular glue', 'PROTAC']},
    })
    calls = install_fixture(monkeypatch, profile)
    PubMedSource().fetch(SourceSpec(name='PubMed', kind='pubmed', profile_driven=True))
    assert len(calls) == 1 and calls[0][1] == 100
    query = calls[0][0]
    assert 'AND' not in query and '*' not in query
    assert '"molecular glue"[tiab]' in query and '"PROTAC"[tiab]' in query
    assert not fixture_matches(query, 'molecular measurements of adhesive glue')
    assert not fixture_matches(query, 'a protein target without any degradation')


def test_openalex_alias_query_preserves_original_nonadjacent_recall(monkeypatch):
    from dailydigest import config
    from dailydigest.ingest import openalex

    outputs = []
    queries = []
    for aliases in ([], ['PROTAC']):
        profile = Profile(bio='', keywords=['protein degradation'], canonical_facets={
            'protein degradation': {'aliases': aliases},
        })
        monkeypatch.setattr(config, 'load_profile', lambda profile=profile: profile)
        def fetch(_url, params, _headers):
            query = params['search.title_abstract_keywords']
            queries.append(query)
            # Fixture has both words, but not the phrase or the alias.
            matches = fixture_matches(query, 'Protein regulation through selective degradation', openalex=True)
            return {'meta': {'count': int(matches)}, 'results':
                    [{'id': 'https://openalex.org/W1', 'title': 'Protein regulation through selective degradation'}]
                    if matches else []}
        monkeypatch.setattr(openalex, '_get_json', fetch)
        outputs.append([item.external_id for item in openalex.OpenAlexSource().fetch(
            SourceSpec(name='OpenAlex', kind='openalex', profile_driven=True))])
    assert outputs[0] and outputs[1] == outputs[0]
    # The original unquoted word search gets its own protected page; the
    # optional explicit alias runs afterwards without narrowing that recall.
    assert queries == ['protein degradation', 'protein degradation', '"PROTAC"']


@pytest.mark.parametrize('precise_count', [0, 2, 3, 14])
def test_precise_interests_keep_their_budget_ahead_of_newer_loose_matches(monkeypatch, precise_count):
    from dailydigest.ingest.pubmed import PubMedSource

    corpus = {}
    for offset, term in enumerate(('Drug development', 'gene therapy')):
        # Insertion order models the newest-first index: loose matches are newer.
        loose = ('A drug affects development of tissue' if offset == 0 else
                 'A gene changes inflammation during supportive therapy')
        for index in range(30):
            corpus[str(offset * 100 + index + 1)] = (loose, 'Words occur separately, without the requested contribution.', .2)
        for index in range(precise_count):
            corpus[str(offset * 100 + index + 51)] = (term + ' methods validated', 'Older precise contribution.', .9)
    monkeypatch.setattr(sys.modules[__name__], 'CORPUS', corpus)
    profile = Profile(bio='', keywords=[name for name, _ in TOPICS], topic_priorities=dict(TOPICS))
    calls = install_fixture(monkeypatch, profile)
    source = PubMedSource()
    items = source.fetch(SourceSpec(name='PubMed', kind='pubmed', profile_driven=True))
    ids = [int(item.external_id.rsplit(':', 1)[-1]) for item in items]
    for offset, term in enumerate(('Drug development', 'gene therapy')):
        diagnostics = [d for d in source.last_fetch_diagnostics['queries'] if d['interest'] == term]
        primary = diagnostics[0]
        cap = next(cap for query, cap in calls if query == primary['query'])
        selected = [key for key in ids if offset * 100 < key < (offset + 1) * 100]
        precise = [offset * 100 + index + 51 for index in range(min(precise_count, cap))]
        assert selected[:len(precise)] == precise
        assert ' AND ' not in primary['query'], 'Exact phrases/aliases must have first access to the reserved slots'
        assert len(selected) <= cap
        assert sum(d['fetched_count'] for d in diagnostics) <= cap
        if precise_count >= 3:
            assert len(diagnostics) == 1 and selected == precise
        else:
            assert len(diagnostics) == 2 and diagnostics[1]['search_stage'] == 'word_fallback'
            assert 'NOT' in diagnostics[1]['query']
            assert len(selected) == cap
    assert len(calls) <= 2 * len(TOPICS)
    assert sum(d['fetched_count'] for d in source.last_fetch_diagnostics['queries']) <= source.MAX_ITEMS


@pytest.mark.parametrize('interest,abstract', [
    ('protein engineer', 'We engineered a new protein assembly.'),
    ('nucleic deliver', 'We present delivery of nucleic cargo.'),
    ('Target protein degradation', 'Selective targeted removal uses protein degradation.'),
    ('Induce proximity', 'We induced selective molecular proximity.'),
])
def test_sparse_phrase_search_uses_generic_prefixes_without_word_variant_table(monkeypatch, interest, abstract):
    from dailydigest.ingest.pubmed import PubMedSource
    monkeypatch.setattr(sys.modules[__name__], 'CORPUS', {'1': ('New contribution', abstract, .9)})
    calls = install_fixture(monkeypatch, Profile(bio='', keywords=[interest]))
    source = PubMedSource()
    assert len(source.fetch(SourceSpec(name='PubMed', kind='pubmed', profile_driven=True))) == 1
    assert len(calls) == 2
    assert '*' not in calls[0][0] and '*' in calls[1][0]
    diagnostics = source.last_fetch_diagnostics['queries']
    assert [d['search_stage'] for d in diagnostics] == ['phrase', 'word_fallback']
    assert sum(d['fetched_count'] for d in diagnostics) <= source.MAX_ITEMS


def test_optional_word_fallback_failure_keeps_precise_results_and_diagnostics(monkeypatch):
    from dailydigest.ingest.pubmed import PubMedSource
    monkeypatch.setattr(sys.modules[__name__], 'CORPUS', {
        str(i): ('Gene therapy contribution', 'Precise result.', .9) for i in (1, 2)
    })
    install_fixture(monkeypatch, Profile(bio='', keywords=['gene therapy']))
    original = PubMedSource._esearch
    def search(self, client, query):
        if ' NOT ' in query:
            raise RuntimeError('fixture fallback failure')
        return original(self, client, query)
    monkeypatch.setattr(PubMedSource, '_esearch', search)
    source = PubMedSource()
    assert len(source.fetch(SourceSpec(name='PubMed', kind='pubmed', profile_driven=True))) == 2
    diagnostic = source.last_fetch_diagnostics['queries'][-1]
    assert diagnostic['search_stage'] == 'word_fallback' and diagnostic['failure']


def test_parse_losses_do_not_allow_fallback_to_exceed_raw_reserved_budget(monkeypatch):
    from dailydigest.ingest.pubmed import PubMedSource
    monkeypatch.setattr(sys.modules[__name__], 'CORPUS', {
        str(i): ('Gene therapy contribution', 'Precise result.', .9) for i in (1, 2, 3, 4)
    })
    calls = install_fixture(monkeypatch, Profile(bio='', keywords=['gene therapy']))
    parse = PubMedSource._parse_article
    def parse_some(self, article, spec):
        return parse(self, article, spec) if int(article.findtext('.//PMID')) < 3 else None
    monkeypatch.setattr(PubMedSource, '_parse_article', parse_some)
    source = PubMedSource()
    source.MAX_ITEMS = 4
    assert len(source.fetch(SourceSpec(name='PubMed', kind='pubmed', profile_driven=True))) == 2
    assert len(calls) == 1
    assert source.last_fetch_diagnostics['queries'][0]['fetched_count'] == 4
