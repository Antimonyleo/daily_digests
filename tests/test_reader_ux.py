from starlette.requests import Request


def request(path):
    return Request({'type': 'http', 'method': 'GET', 'path': path, 'query_string': b'', 'headers': [], 'scheme': 'http', 'server': ('localhost', 80)})


def test_first_setup_is_research_only_and_has_no_dead_back_link(monkeypatch, tmp_path):
    from dailydigest import web
    monkeypatch.setattr(web, '_get_profile_path', lambda: tmp_path / 'absent.yaml')
    form = web._load_existing_form_defaults()
    assert all(form[key] == 'false' for key in form if key.startswith('include_'))
    html = web.setup_get(request('/setup')).body.decode()
    assert 'Back to today’s digest' not in html
    assert '<main' in html and 'Skip to content' in html
    assert 'id="priority-options"' in html


def test_run_requires_an_explicit_start_and_consistent_compute_notice():
    from dailydigest import web
    html = web.run_get(request('/run')).body.decode()
    assert 'heat, fan noise and battery use' in html
    assert '20 minutes or more' in html
    assert 'a minute or more' not in html
    assert html.index('heat, fan noise and battery use') < html.index('id="brew-button"')
    assert '<main' in html


def test_learning_line_does_not_claim_ready_when_legacy_weights_are_missing(monkeypatch):
    from dailydigest import config, votes
    from dailydigest.rank import ranker
    monkeypatch.setattr(votes, 'vote_counts', lambda: {'signed': 30, 'good': 15, 'bad': 15, 'neutral': 0, 'total': 30})
    monkeypatch.setattr(config, 'resolve_scoring_mode', lambda _settings: 'hybrid_lr')
    monkeypatch.setattr(ranker, 'get_lr_ranker', lambda: None)
    status = votes.lr_training_status()
    assert status['training_status'] == 'model_missing'
    assert 'unavailable' in status['learning_message']
    assert '0 to unlock' not in status['learning_message']


def test_first_setup_error_does_not_reintroduce_a_dead_back_link(monkeypatch, tmp_path):
    import asyncio
    from dailydigest import web
    from tests.test_web import _request
    monkeypatch.setattr(web, '_get_profile_path', lambda: tmp_path / 'absent.yaml')
    monkeypatch.setattr(web, '_ENV_PATH', tmp_path / '.env')
    response = asyncio.run(web.setup_post(_request('POST', '/setup', form={
        '_csrf_token': web._CSRF_TOKEN, 'topics': 'RNA delivery | 40',
        'topic_weight_mode': 'percentage', 'llm_backend': 'extractive',
    })))
    assert response.status_code == 400
    assert 'Back to today’s digest' not in response.body.decode()
    assert not (tmp_path / 'absent.yaml').exists()
    assert not (tmp_path / '.env').exists()
