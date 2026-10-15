"""Optional public workflow alert; never include profiles, logs or credentials."""
import json
import os
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def main():
    endpoint = os.environ.get('DAILYDIGEST_ALERT_WEBHOOK', '').strip()
    if not endpoint:
        print('::warning::Refresh failed. Configure DAILYDIGEST_ALERT_WEBHOOK for an optional alert, or enable repository Actions notifications. The published snapshot will continue aging.')
        return 0
    try:
        parsed = urlsplit(endpoint)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('Invalid alert endpoint')
        run_url = ('https://github.com/' + os.environ.get('GITHUB_REPOSITORY', '')
                   + '/actions/runs/' + os.environ.get('GITHUB_RUN_ID', ''))
        body = json.dumps({'text': 'DailyDigest publication or source refresh failed. '
                           'Check the published source age and workflow: ' + run_url}).encode()
        request = Request(endpoint, data=body, headers={'Content-Type': 'application/json'}, method='POST')
        with urlopen(request, timeout=10):
            pass
    except (OSError, ValueError):
        print('::warning::The optional failure alert could not be delivered. Check Actions and the alert configuration.')
        return 1
    print('Optional public workflow failure alert delivered.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
