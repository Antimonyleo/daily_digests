import assert from 'node:assert/strict';
import {test} from 'node:test';
import {loadPyodide} from '../prototypes/browser-compute/node_modules/pyodide/pyodide.mjs';
import {compileImportedPython} from './compile-python.mjs';

test('target-interpreter bytecode skips compilation but validates edited source', async () => {
  const py = await loadPyodide();
  py.FS.writeFile('/lib/python3.13/site-packages/startup_dependency.py', new TextEncoder().encode('VALUE = 4\n'));
  py.runPython(`import zipfile
with zipfile.ZipFile('/fixture.zip', 'w') as archive:
    archive.writestr('src/fixture.py', 'VALUE = 17\\n')
    archive.writestr('templates/page.j2', 'Untouched template')
`);
  const result = compileImportedPython(py, py.FS.readFile('/fixture.zip'), ['fixture', 'startup_dependency']);
  assert.equal(result.report.invalidation, 'checked-hash');
  py.FS.writeFile('/verified.zip', result.app);
  py.FS.writeFile('/deps.zip', result.bytecode);
  py.runPython(`
import builtins, struct
with zipfile.ZipFile('/verified.zip') as archive:
    assert archive.read('src/fixture.py') == b'VALUE = 17\\n'
    assert archive.read('templates/page.j2') == b'Untouched template'
    caches = [name for name in archive.namelist() if name.endswith('.pyc')]
    assert len(caches) == 1
    assert struct.unpack('<I', archive.read(caches[0])[4:8])[0] == 3
with zipfile.ZipFile('/deps.zip') as archive:
    assert len(archive.namelist()) == 1, 'The target sysconfig may use a double-slash prefix'
    assert all(name.startswith('lib/') and name.endswith('.pyc') for name in archive.namelist())
sys.modules.pop('fixture')
original_compile = builtins.compile
def unexpected_compile(*args, **kwargs):
    raise AssertionError('Source was compiled despite a valid bytecode cache')
builtins.compile = unexpected_compile
try:
    assert importlib.import_module('fixture').VALUE == 17
finally:
    builtins.compile = original_compile
# Same byte length: mtime/size-only invalidation must not accept stale code.
pathlib.Path('/app/src/fixture.py').write_text('VALUE = 23\\n')
sys.modules.pop('fixture')
assert importlib.import_module('fixture').VALUE == 23
`);
});
