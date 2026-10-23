// Build-time only: use the deployed interpreter, never the host Python ABI.
import {readFile, writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

export function compileImportedPython(py, archive, imports) {
  py.FS.writeFile('/source.zip', archive);
  py.globals.set('startup_imports', JSON.stringify(imports));
  const report = py.runPython(`
import importlib, importlib.util, json, os, pathlib, py_compile, sys, sysconfig, zipfile
with zipfile.ZipFile('/source.zip') as source:
    source_names = set(source.namelist())
    if any(name.startswith('/') or '..' in pathlib.PurePosixPath(name).parts for name in source_names):
        raise ValueError('Unsafe application archive path')
    source.extractall('/app')
os.chdir('/app')
sys.path[:0] = ['/app/src', '/app/browser', '/app/site-packages']
for name in json.loads(startup_imports):
    importlib.import_module(name)
purelib = pathlib.Path(sysconfig.get_path('purelib')).resolve()
files = {getattr(module, '__file__', '') for module in list(sys.modules.values())}
caches = []
for name in sorted(name for name in files if isinstance(name, str)):
    path = pathlib.Path(name)
    if path.suffix != '.py' or not path.is_file():
        continue
    if path.is_relative_to('/app'):
        if str(path.relative_to('/app')) not in source_names:
            continue
    elif not path.is_relative_to(purelib):
        continue
    cache = py_compile.compile(str(path), doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH)
    caches.append(pathlib.Path(cache))
with zipfile.ZipFile('/compiled-app.zip', 'w', zipfile.ZIP_DEFLATED) as output, zipfile.ZipFile('/source.zip') as source:
    for info in source.infolist():
        output.writestr(info, source.read(info.filename))
    for path in caches:
        if path.is_relative_to('/app'):
            output.write(path, str(path.relative_to('/app')))
with zipfile.ZipFile('/startup-bytecode.zip', 'w', zipfile.ZIP_DEFLATED) as output:
    for path in caches:
        if path.is_relative_to(purelib):
            output.write(path, str(path).lstrip('/'))
json.dumps({'modules': len(caches), 'python': sys.version.split()[0],
           'invalidation': 'checked-hash'})
`);
  return {app: py.FS.readFile('/compiled-app.zip'), bytecode: py.FS.readFile('/startup-bytecode.zip'),
    report: JSON.parse(report)};
}

async function main() {
  const site = resolve(process.argv[2]);
  const runtime = resolve(site, 'node_modules/pyodide');
  const {loadPyodide} = await import(pathToFileURL(resolve(runtime, 'pyodide.mjs')).href);
  const py = await loadPyodide({indexURL: runtime + '/'});
  const dependencies = JSON.parse(await readFile(resolve(site, 'assets/dependencies.json'), 'utf8'));
  await py.loadPackage(dependencies.pyodide.filter(name => name !== 'micropip'));
  // Import public code in the empty WASM filesystem. No reader data or host
  // environment is mounted, and no routes, feeds or model providers are run.
  const compiled = compileImportedPython(py, await readFile(resolve(site, 'assets/app.zip')),
    ['network', 'filelock', 'dailydigest.web', 'dailydigest.summarize',
      'dailydigest.rank.embed', 'dailydigest.rank.semantic_alignment']);
  await writeFile(resolve(site, 'assets/app.zip'), compiled.app);
  await writeFile(resolve(site, 'assets/startup-bytecode.zip'), compiled.bytecode);
  console.log(JSON.stringify(compiled.report));
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await main();
