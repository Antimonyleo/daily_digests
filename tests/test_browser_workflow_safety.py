"""Release failures must surface without granting build jobs deploy privileges."""
from pathlib import Path
import re

import yaml


def test_release_permissions_checks_and_failure_reporting():
    workflow = yaml.load((Path(__file__).resolve().parents[1]/'.github/workflows/browser-pages.yml').read_text(), Loader=yaml.BaseLoader)
    assert workflow['permissions'] == {'contents': 'read'}
    jobs = workflow['jobs']
    assert jobs['deploy']['needs'] == 'build'
    assert jobs['deploy']['permissions'] == {'pages': 'write', 'id-token': 'write'}
    assert 'permissions' not in jobs['build']
    assert jobs['notify-failure']['needs'] == ['build', 'deploy']
    condition = jobs['notify-failure']['if']
    assert 'always()' in condition and "needs.build.result == 'failure'" in condition
    assert "needs.deploy.result == 'failure'" in condition
    assert any('pytest' in step.get('run', '') for step in jobs['build']['steps'])
    for job in jobs.values():
        for step in job['steps']:
            if 'uses' in step:
                assert re.search(r'@[0-9a-f]{40}$', step['uses'])


def test_release_exercises_both_browser_kernels_before_upload():
    workflow = yaml.load((Path(__file__).resolve().parents[1]/'.github/workflows/browser-pages.yml').read_text(), Loader=yaml.BaseLoader)
    steps = workflow['jobs']['build']['steps']
    runs = '\n'.join(step.get('run', '') for step in steps)
    assert 'prototypes/browser-compute/browser.test.mjs' in runs
    assert 'node browser/verify-browser.mjs' in runs
    assert 'node browser/verify-semantic.mjs' in runs
    acceptance = next(i for i, step in enumerate(steps) if 'node browser/verify-semantic.mjs' in step.get('run', ''))
    upload = next(i for i, step in enumerate(steps) if step.get('uses', '').startswith('actions/upload-pages-artifact@'))
    assert acceptance < upload
    # Both kernels must use the served project subpath, including the matcher
    # whose standalone default differs from the full browser acceptance tool.
    assert steps[acceptance].get("env", {}).get("DAILYDIGEST_BROWSER_URL") == "http://127.0.0.1:8767/daily_digests/"


def test_recurring_issue_is_scoped_to_strict_refreshes_and_its_own_permissions():
    workflow = yaml.load((Path(__file__).resolve().parents[1]/'.github/workflows/browser-pages.yml').read_text(), Loader=yaml.BaseLoader)
    job = workflow['jobs']['refresh-status']
    assert job['needs'] == ['build', 'deploy']
    assert job['permissions'] == {'contents': 'read', 'issues': 'write'}
    assert "github.event_name == 'schedule'" in job['if']
    assert "github.event_name == 'workflow_dispatch'" in job['if']
    assert "needs.deploy.result == 'success'" in job['if']
    report = next(step for step in job['steps'] if 'report_failure.py' in step.get('run', ''))
    assert report['env']['DAILYDIGEST_ALERT_MODE'] == 'github-issue'
    assert 'secrets.GITHUB_TOKEN' in report['env']['GITHUB_TOKEN']
    assert 'needs.build.result' in report['env']['DAILYDIGEST_REFRESH_RESULT']


def test_public_comparison_cache_is_restored_before_static_build():
    workflow = yaml.load((Path(__file__).resolve().parents[1]/'.github/workflows/browser-pages.yml').read_text(), Loader=yaml.BaseLoader)
    steps = workflow['jobs']['build']['steps']
    restore = next(i for i, step in enumerate(steps) if 'fetch_public_embeddings.py' in step.get('run', ''))
    build = next(i for i, step in enumerate(steps) if 'python browser/build.py' in step.get('run', ''))
    assert restore < build


def test_every_ci_job_has_a_bounded_timeout():
    root = Path(__file__).resolve().parents[1]
    for name in ('browser-pages.yml', 'ci.yml'):
        workflow = yaml.load((root / '.github/workflows' / name).read_text(), Loader=yaml.BaseLoader)
        for job in workflow['jobs'].values():
            assert 1 <= int(job['timeout-minutes']) <= 60


def test_pages_uses_event_commit_and_checks_browser_tree_before_bootstrap():
    workflow = yaml.load((Path(__file__).resolve().parents[1] / '.github/workflows/browser-pages.yml').read_text(), Loader=yaml.BaseLoader)
    assert {'push', 'schedule', 'workflow_dispatch'} <= workflow['on'].keys()
    assert 'pull_request' not in workflow['on']
    steps = workflow['jobs']['build']['steps']
    checkout = next(i for i, step in enumerate(steps) if step.get('uses', '').startswith('actions/checkout@'))
    assert steps[checkout]['with']['ref'] == '${{ github.sha }}'
    guard = next(i for i, step in enumerate(steps) if 'test -f browser/build.py' in step.get('run', ''))
    bootstrap = next(i for i, step in enumerate(steps) if 'uv sync' in step.get('run', ''))
    assert checkout < guard < bootstrap
    assert any('verify-guide-backup.mjs' in step.get('run', '') for step in steps)


def test_pull_requests_run_browser_unit_checks_without_publishing():
    workflow = yaml.load((Path(__file__).resolve().parents[1] / '.github/workflows/ci.yml').read_text(), Loader=yaml.BaseLoader)
    assert 'pull_request' in workflow['on']
    job = workflow['jobs']['browser-unit']
    assert 'if' not in job
    assert not job.get('permissions', {}).get('pages')
    steps = job['steps']
    assert any(step.get('with', {}).get('node-version') == '24' for step in steps)
    runs = '\n'.join(step.get('run', '') for step in steps)
    assert 'node --test --test-isolation=none browser/*.test.mjs' in runs
    assert 'prototypes/browser-compute/browser.test.mjs' in runs
    assert 'collect_sources.py' not in runs
