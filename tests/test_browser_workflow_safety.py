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
