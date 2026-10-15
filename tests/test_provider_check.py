"""Explicit API checks use fictional providers and never reader context."""
import httpx
import pytest


@pytest.mark.parametrize('backend', ['api', 'anthropic'])
def test_explicit_check_is_one_small_request_without_reader_context(monkeypatch, backend):
    from dailydigest.provider_check import check_api_connection
    calls = []

    def post(client, url, **kwargs):
        calls.append((url, kwargs, client.timeout.read, client.follow_redirects))
        body = ({'content': [{'type': 'text', 'text': 'OK'}]} if backend == 'anthropic'
                else {'choices': [{'message': {'content': 'OK'}}]})
        return httpx.Response(200, json=body, request=httpx.Request('POST', url))

    monkeypatch.setattr(httpx.Client, 'post', post)
    assert check_api_connection(backend, 'https://provider.example/v1', 'fictional-key', 'fixture')['ok']
    assert len(calls) == 1
    url, kwargs, timeout, redirects = calls[0]
    assert url.endswith('/messages' if backend == 'anthropic' else '/chat/completions')
    assert timeout == 10 and not redirects
    assert kwargs['json']['max_tokens'] == 8
    assert kwargs['json']['messages'] == [{'role': 'user', 'content': 'Reply OK.'}]


@pytest.mark.parametrize('response', [401, 429, 503, 'timeout', 'bad-json', 'empty'])
def test_check_errors_are_redacted_and_do_not_retry(monkeypatch, response):
    from dailydigest.provider_check import check_api_connection
    calls = []

    def post(client, url, **kwargs):
        calls.append(True)
        if response == 'timeout':
            raise httpx.ReadTimeout('secret-key private-url')
        if isinstance(response, int):
            return httpx.Response(response, text='secret-key', request=httpx.Request('POST', url))
        return httpx.Response(200, text='bad-json' if response == 'bad-json' else '{}',
                              request=httpx.Request('POST', url))

    monkeypatch.setattr(httpx.Client, 'post', post)
    result = check_api_connection('api', 'https://provider.example/v1', 'secret-key', 'fixture')
    assert result['ok'] is False and 'secret-key' not in result['message']
    assert 'private-url' not in result['message'] and len(calls) == 1


@pytest.mark.parametrize('values', [
    ('extractive', 'https://provider.example', 'key', 'fixture'),
    ('api', 'file:///private', 'key', 'fixture'),
    ('api', 'https://user:key@provider.example', 'key', 'fixture'),
    ('api', 'https://provider.example', '', 'fixture'),
    ('api', 'https://provider.example', 'key', ''),
    ('api', 'https://provider.example', 'key\ninjection', 'fixture'),
    ('api', 'https://provider.example:bad/v1', 'key', 'fixture'),
    ('api', 'https://provider.example:99999/v1', 'key', 'fixture'),
])
def test_invalid_configuration_never_contacts_service(monkeypatch, values):
    from dailydigest.provider_check import check_api_connection
    monkeypatch.setattr(httpx.Client, 'post', lambda *a, **k: pytest.fail('Unexpected network'))
    assert check_api_connection(*values)['ok'] is False


def test_httpx_url_validation_failure_is_a_redacted_result(monkeypatch):
    from dailydigest.provider_check import check_api_connection
    def invalid(*args, **kwargs):
        raise httpx.InvalidURL('secret-key')
    monkeypatch.setattr(httpx.Client, 'post', invalid)
    result = check_api_connection('api', 'https://provider.example/v1', 'secret-key', 'fixture')
    assert result['ok'] is False and 'secret-key' not in result['message']


@pytest.mark.parametrize('backend,suffix', [('api', '/chat/completions'), ('anthropic', '/messages')])
def test_saved_key_binding_normalizes_effective_endpoint(backend, suffix):
    from dailydigest.provider_check import api_provider_binding

    expected = (backend, 'https', 'provider.example', 443, '/v1' + suffix)
    assert api_provider_binding(backend, 'https://provider.example/v1') == expected
    assert api_provider_binding(backend, 'https://PROVIDER.example:443/v1/') == expected
    assert api_provider_binding(backend, 'https://provider.example:8443/v1') != expected
    assert api_provider_binding(backend, 'https://provider.example/other') != expected


@pytest.mark.parametrize('base_url', [
    'https://user:key@provider.example/v1', 'https://@provider.example/v1',
    'https://provider.example/v1?', 'https://provider.example/v1#',
    'https://provider.example:bad/v1', 'file:///private',
])
def test_invalid_api_url_cannot_identify_a_saved_credential_binding(base_url):
    from dailydigest.provider_check import api_provider_binding

    assert api_provider_binding('api', base_url) is None
