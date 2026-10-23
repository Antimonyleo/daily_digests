"""Online publication verification remains explicit and preserves trustworthy evidence."""
import asyncio
from html.parser import HTMLParser
from types import SimpleNamespace

import pytest

from dailydigest.rank.source_quality import research_publication_metadata
from tests.test_setup_interface import _render_settings
from tests.test_web import _request


class EnrichmentInputs(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.inputs = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'input' and attrs.get('name') == 'publication_enrichment':
            self.inputs.append(attrs)


@pytest.mark.parametrize('enabled', [False, True])
def test_settings_checkbox_reflects_configuration(monkeypatch, tmp_path, enabled):
    from dailydigest import web

    monkeypatch.setattr(web, 'SETTINGS', web.SETTINGS.model_copy(update={
        'publication_enrichment': enabled}))
    html = _render_settings(monkeypatch, tmp_path)
    fields = EnrichmentInputs(html).inputs
    assert [(field['type'], field['value']) for field in fields] == [
        ('hidden', 'false'), ('checkbox', 'true')]
    assert ('checked' in fields[1]) is enabled
    assert 'Verify article types online' in html
    assert 'Europe PMC and Crossref' in html


@pytest.mark.parametrize('initial,posted,expected', [
    (False, ['false', 'true'], True),
    (True, ['false'], False),
    (True, [], True),
    (False, [], False),
])
def test_setup_persists_explicit_checkbox_and_preserves_omitted_setting(
    monkeypatch, tmp_path, initial, posted, expected,
):
    from dailydigest import web

    profile = tmp_path / 'profile.yaml'
    profile.write_text('keywords: [DNA origami]\n')
    env = tmp_path / '.env'
    env.write_text('PUBLICATION_ENRICHMENT=' + str(initial).lower() + '\n')
    monkeypatch.setattr(web, '_get_profile_path', lambda: profile)
    monkeypatch.setattr(web, '_get_opportunity_profile_path', lambda: tmp_path / 'opportunities.yaml')
    monkeypatch.setattr(web, '_ENV_PATH', env)
    monkeypatch.setattr(web, 'SETTINGS', web.SETTINGS.model_copy(update={
        'publication_enrichment': initial, 'llm_backend': 'extractive'}))
    form = [('_csrf_token', web._CSRF_TOKEN), ('topics', 'DNA origami'),
            ('llm_backend', 'extractive'), ('user_tz', 'UTC')]
    form.extend(('publication_enrichment', value) for value in posted)
    response = asyncio.run(web.setup_post(_request('POST', '/setup', form=form)))
    assert response.status_code == 303
    assert web._read_env_file(env)['PUBLICATION_ENRICHMENT'] == str(expected).lower()
    assert web.SETTINGS.publication_enrichment is expected
    assert web._load_existing_form_defaults()['publication_enrichment'] == str(expected).lower()


def row_with_evidence(evidence):
    return SimpleNamespace(section='research', title='DNA origami lattice assembly',
                           metadata={'publication_types': ['Research Article'],
                                     'publication_type_evidence': [evidence]})


def trusted_evidence():
    return {'field': 'europepmc_pub_type', 'source': 'Europe PMC',
            'value': 'research-article', 'url': 'https://europepmc.org/article/MED/12345'}


def test_research_genre_preserves_raw_europepmc_provenance():
    evidence = trusted_evidence()
    result = research_publication_metadata(row_with_evidence(evidence))
    assert result['publication_types'] == ['research article']
    assert result['publication_type_evidence'] == [evidence]


@pytest.mark.parametrize('change', [
    {'value': 'Review'},
    {'field': 'crossref_type'},
    {'field': {}},
    {'field': []},
    {'source': 'Other publisher'},
    {'url': 'https://europepmc.org.evil.example/article/MED/12345'},
    {'url': 'https://evil.example/article/MED/12345'},
    {'url': None},
])
def test_mismatched_provider_or_genre_does_not_preserve_false_evidence(change):
    evidence = trusted_evidence() | change
    result = research_publication_metadata(row_with_evidence(evidence))
    assert result['publication_types'] == ['research article']
    assert 'publication_type_evidence' not in result
