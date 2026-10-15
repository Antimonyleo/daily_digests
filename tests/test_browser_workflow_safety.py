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
