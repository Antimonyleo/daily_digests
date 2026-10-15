"""A reader-initiated, minimal API request that never reads private state."""
from urllib.parse import urlsplit

import httpx


def api_provider_binding(backend: str, base_url: str) -> tuple | None:
    """Identify the API and effective endpoint allowed to reuse a saved key."""
    if backend not in {'api', 'anthropic'}:
        return None
    base_url = str(base_url or '').strip()
    if any(character in base_url for character in '\r\n\0?#'):
        return None
    try:
        url = urlsplit(base_url)
        if (url.scheme not in {'https', 'http'} or not url.hostname
                or url.username is not None or url.password is not None):
            return None
        port = url.port
    except ValueError:
        return None
    if port is None:
        port = 443 if url.scheme == 'https' else 80
    suffix = '/messages' if backend == 'anthropic' else '/chat/completions'
    return backend, url.scheme, url.hostname, port, url.path.rstrip('/') + suffix


def check_api_connection(backend: str, base_url: str, api_key: str, model: str) -> dict:
    """Check supplied settings without saving them or sending reader context."""
    values = [str(value or '').strip() for value in (backend, base_url, api_key, model)]
    backend, base_url, api_key, model = values
    if (backend not in {'api', 'anthropic'} or not all(values)
            or any(len(value) > 4096 or any(c in value for c in '\r\n\0') for value in values)):
        return {'ok': False, 'message': 'Choose an API provider and enter its URL, key and model first.'}
    try:
        if api_provider_binding(backend, base_url) is None:
            raise ValueError('Invalid URL')
        headers = {'Content-Type': 'application/json'}
        body = {'model': model, 'max_tokens': 8,
                'messages': [{'role': 'user', 'content': 'Reply OK.'}]}
        if backend == 'anthropic':
            headers.update({'x-api-key': api_key, 'anthropic-version': '2023-06-01'})
            suffix = '/messages'
        else:
            headers['Authorization'] = f'Bearer {api_key}'
            suffix = '/chat/completions'
        with httpx.Client(timeout=10.0, follow_redirects=False) as client:
            response = client.post(base_url.rstrip('/') + suffix, headers=headers, json=body)
            response.raise_for_status()
            if response.is_redirect:
                return {'ok': False, 'message': 'The endpoint redirected. Enter the final API base URL.'}
            result = response.json()
        text = (''.join(block.get('text', '') for block in result.get('content', [])
                        if block.get('type') == 'text') if backend == 'anthropic'
                else result['choices'][0]['message']['content'])
        if not isinstance(text, str) or not text.strip():
            raise ValueError('No text')
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        message = ('Authentication was rejected. Check the key and provider.' if code in {401, 403}
                   else 'The provider rate limit was reached. Try again later.' if code == 429
                   else f'The provider returned HTTP {code}. Check the URL and model.')
        return {'ok': False, 'message': message}
    except (httpx.HTTPError, httpx.InvalidURL, ValueError, KeyError, IndexError, TypeError, AttributeError, RuntimeError):
        return {'ok': False, 'message': 'The API did not return a usable reply. Check the URL, model, network and browser access.'}
    return {'ok': True, 'message': 'API connection verified with a small test reply. Settings have not been saved.'}
