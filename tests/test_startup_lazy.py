"""Opening the reading desk must not initialize the brew pipeline."""
import os
from pathlib import Path
import subprocess
import sys


def test_setup_opens_without_importing_brew_pipeline(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'src'
    script = '''
import importlib.abc
import sys
class NoBrew(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'dailydigest.pipeline', 'numpy', 'dailydigest.rank.embedding_cache'}:
            raise AssertionError('Reading desk imported compute dependency: ' + fullname)
sys.meta_path.insert(0, NoBrew())
from dailydigest import web
from starlette.requests import Request
from jinja2 import FileSystemLoader
from pathlib import Path
web.templates.env.loader = FileSystemLoader(Path(web.__file__).resolve().parents[2] / 'templates')
request = Request({'type': 'http', 'method': 'GET', 'path': '/setup',
                   'headers': [], 'query_string': b''})
response = web.setup_get(request)
assert response.status_code == 200
assert b'research' in response.body.lower()
assert 'dailydigest.pipeline' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path,
                            env={**os.environ, 'PYTHONPATH': str(source),
                                 'DD_APP_ROOT': str(tmp_path), 'PROFILE_PATH': 'data/profile.yaml', 'OPPORTUNITY_PROFILE_PATH': 'data/opportunities.yaml', 'DB_PATH': str(tmp_path / 'data/digest.db')},
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
