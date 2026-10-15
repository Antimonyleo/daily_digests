"""Offline retrieval fixtures; the index matcher is not a live PubMed emulator.

Relevance scores below are fixed reviewer labels, so these tests exercise the
unchanged production gates, not the embedding model's semantic accuracy.
"""
from __future__ import annotations

import json
import re
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
        tokens = re.findall(r'"[^"]*"\[tiab\]|[\w-]+\[tiab\]|AND|OR|[()]', query)
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
        value = token.removesuffix('[tiab]')
        return (value[1:-1].casefold() in text.casefold()
                if value.startswith('"') else value.casefold() in words)

    def conjunction():
        nonlocal position
        value = atom()
        while position < len(tokens) and tokens[position] == 'AND':
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
    assert len(calls) == 9 and sum(cap for _, cap in calls) == 100
    assert len(source.last_fetch_diagnostics['queries']) == 9
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
    by_interest = {d['interest']: d['query'] for d in source.last_fetch_diagnostics['queries']}
    assert fixture_matches(by_interest['Target protein degradation'], CORPUS['1'][1])
    assert fixture_matches(by_interest['Induce proximity'], CORPUS['2'][1])
    record_testsuite_property('stage_trace', json.dumps({
        'limitation': 'Offline fixture index; relevance scores are fixed reviewer labels, not model outputs.',
        'queries': [{'interest': name, 'before': f'"{name}"[tiab]', 'after': by_interest[name],
                     'before_ids': [key for key, (title, abstract, _) in CORPUS.items()
                                    if fixture_matches(f'"{name}"[tiab]', title + ' ' + abstract)],
                     'after_ids': [key for key, (title, abstract, _) in CORPUS.items()
                                   if fixture_matches(by_interest[name], title + ' ' + abstract)]}
                    for name, _ in TOPICS],
        'before': {'retrieved': sorted(before), 'quality': [r.id for r in before_quality],
                   'relevance': [], 'selected': []},
        'after': {'retrieved': sorted(retrieved), 'quality': [r.id for r in quality],
                  'relevance': [r.id for r, _ in relevant], 'selected': [r.id for r, _ in selected]},
    }))


def test_pubmed_aliases_remain_exact_but_original_allows_bounded_variants(monkeypatch):
    from dailydigest.ingest.pubmed import PubMedSource

    profile = Profile(bio='', keywords=['Target protein degradation'], canonical_facets={
        'Target protein degradation': {'aliases': ['molecular glue', 'PROTAC']},
    })
    calls = install_fixture(monkeypatch, profile)
    PubMedSource().fetch(SourceSpec(name='PubMed', kind='pubmed', profile_driven=True))
    assert len(calls) == 1 and calls[0][1] == 100
    query = calls[0][0]
    assert 'targeted[tiab]' in query and 'target[tiab]' in query
    assert 'AND' in query
    assert '"molecular glue"[tiab]' in query and '"PROTAC"[tiab]' in query
    assert fixture_matches(query, 'selective protein loss by targeted degradation')
    assert not fixture_matches(query, 'molecular measurements of adhesive glue')
    # Retrieval is broader; downstream relevance still decides usefulness.
    assert fixture_matches(query, 'a protein target without any degradation')


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
            query = params['search']
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
    assert queries[1] == '((protein degradation) OR "PROTAC")'
