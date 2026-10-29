"""Optional public workflow alert; never include profiles, logs or credentials."""
import json
import os
import re
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ISSUE_TITLE = 'DailyDigest source refresh needs attention'
ISSUE_MARKER = '<!-- dailydigest-recurring-refresh-alert-v1 -->'
ACCESS_TITLE = 'DailyDigest browser source access changed'
ACCESS_MARKER = '<!-- dailydigest-source-access-alert-v1 -->'


def _refresh_body(run_url):
    return ('A scheduled or manual source refresh failed. '
            'The last successfully published site remains available, but its source records continue aging. '
            'Records older than 48 hours are not used as fresh snapshot coverage.\n\n'
            f'[Inspect the failed workflow]({run_url})\n\n'
            'This issue is reused on repeated failures and closed after a successful strict refresh.')


def _access_body(run_url):
    # Only public source names and outcomes from the audit summary; no logs.
    summary = os.environ.get('DAILYDIGEST_ACCESS_SUMMARY', '')
    lines = [line for line in summary.splitlines() if line.startswith('- ')][:40]
    return ('The daily browser audit could not read one or more sources that browser brews fetch '
            'directly. Those sources will fail in browser brews until fixed; a section whose every '
            'provider fails stops the brew and keeps the saved digest.\n\n'
            + ('\n'.join(lines) + '\n\n' if lines else '')
            + f'[Inspect the audit run]({run_url})\n\n'
            'This issue is reused while access stays broken and closed after a clean audit.')


def report_issue(title=ISSUE_TITLE, marker=ISSUE_MARKER, body=_refresh_body):
    """Maintain one public operational issue; never attach logs or reader state."""
    repository = os.environ.get('GITHUB_REPOSITORY', '')
    token = os.environ.get('GITHUB_TOKEN', '')
    result = os.environ.get('DAILYDIGEST_REFRESH_RESULT', '')
    run_id = os.environ.get('GITHUB_RUN_ID', '')
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', repository) or not token or not run_id.isdigit() or result not in {'success', 'failure'}:
        print('::warning::Recurring refresh issue is not configured. Check the workflow token and public run metadata.')
        return 1
    endpoint = f'https://api.github.com/repos/{repository}/issues'

    def api(url, method='GET', payload=None):
        request = Request(url, method=method,
                          data=json.dumps(payload).encode() if payload is not None else None,
                          headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json',
                                   'X-GitHub-Api-Version': '2022-11-28', 'Content-Type': 'application/json'})
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read())

    try:
        page, existing = 1, None
        while existing is None:
            issues = api(f'{endpoint}?state=all&sort=updated&direction=desc&per_page=100&page={page}')
            if not isinstance(issues, list):
                raise ValueError('Invalid GitHub response')
            existing = next((row for row in issues if 'pull_request' not in row
                             and (row.get('user') or {}).get('login') == 'github-actions[bot]'
                             and (row.get('user') or {}).get('type') == 'Bot'
                             and row.get('title') == title and marker in (row.get('body') or '')), None)
            if existing or len(issues) < 100:
                break
            page += 1
        run_url = f'https://github.com/{repository}/actions/runs/{run_id}'
        if result == 'success':
            if existing and existing.get('state') == 'open':
                api(f'{endpoint}/{existing["number"]}', 'PATCH', {
                    'state': 'closed', 'body': f'{marker}\n\nThe check recovered.\n\n[Successful workflow]({run_url})'})
            print('Recurring issue checked after a successful run.')
            return 0
        payload = {'title': title, 'body': f'{marker}\n\n{body(run_url)}'}
        if existing:
            api(f'{endpoint}/{existing["number"]}', 'PATCH', {**payload, 'state': 'open'})
        else:
            api(endpoint, 'POST', payload)
        print('Recurring issue updated.')
        return 0
    except (OSError, ValueError, KeyError, TypeError):
        print('::warning::The recurring refresh issue could not be updated. Check Actions and repository issue permissions.')
        return 1


def main():
    if os.environ.get('DAILYDIGEST_ALERT_MODE') == 'github-issue':
        return report_issue()
    if os.environ.get('DAILYDIGEST_ALERT_MODE') == 'source-access-issue':
        return report_issue(ACCESS_TITLE, ACCESS_MARKER, _access_body)
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
