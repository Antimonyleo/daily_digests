"""Bounded public-only browser-vector publication; no native encoder or reader state."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess

import numpy as np

MODEL_FILES = ('config.json', 'tokenizer.json', 'tokenizer_config.json',
               'special_tokens_map.json', 'onnx/model.onnx')
MAX_DOCUMENTS = 10000
MAX_BYTES = 16_000_000
DIMENSION = 384


def run_precompute(command, *, cwd, timeout=1200):
    """Bound the owned Node+Chromium tree, not just its launcher process."""
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=os.name != 'nt',
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        if os.name == 'nt':
            try:
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               capture_output=True, timeout=10, check=False)
            finally:
                if process.poll() is None:
                    process.kill()
        else:
            try:
                # Let Node close Playwright's separately owned Chromium group.
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        process.communicate(timeout=10)
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def embedding_signature(model_root, compute, bert, lock):
    """Bind the precise CPU export, tokenizer, pooling and runtime version."""
    digest = hashlib.sha256(b'public-browser-cpu-float32-v1\0')
    for name in MODEL_FILES:
        digest.update(name.encode() + b'\0' + hashlib.sha256((Path(model_root) / name).read_bytes()).digest())
    for name, path in [('compute', compute), ('bert', bert), ('lock', lock)]:
        digest.update(name.encode() + b'\0' + hashlib.sha256(Path(path).read_bytes()).digest())
    return 'browser-onnx:' + digest.hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def public_texts(snapshot, *, max_documents=MAX_DOCUMENTS):
    """Only select exact public item title+abstract text; never profile fields.

    The validated source snapshot already enforces its bounded date window.
    Newest records win if an exceptionally large snapshot exceeds our cap.
    """
    if snapshot.get('public_only') is not True or not isinstance(snapshot.get('sources'), dict):
        raise ValueError('Only a public source snapshot can be embedded')
    candidates = []
    for source in snapshot['sources'].values():
        if source.get('status') != 'ok':
            continue
        for item in source.get('items', []):
            title, abstract = (item.get('title') or '').strip(), (item.get('abstract') or '').strip()
            text = f'{title}. {abstract}' if abstract else title
            if text:
                candidates.append((str(item.get('published_at') or ''), text))
    seen = {}
    for _published, text in sorted(candidates, reverse=True):
        seen.setdefault(text_hash(text), text)
    return list(seen.values())[:max(0, min(max_documents, MAX_DOCUMENTS, MAX_BYTES // (DIMENSION * 4)))]


def _checked_matrix(vectors, count):
    values = np.asarray(vectors, dtype='<f4')
    if (values.shape != (count, DIMENSION) or not np.isfinite(values).all()
            or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=.0005)):
        raise ValueError('Invalid public comparison matrix')
    return values


def read_vectors(path, signature):
    """Discard absent, stale-model or corrupt public caches instead of mixing."""
    try:
        path = Path(path)
        manifest = json.loads(path.read_text())
        hashes = manifest['hashes']
        if (manifest.get('version') != 1 or manifest.get('public_only') is not True
                or manifest.get('signature') != signature or manifest.get('backend') != 'browser-onnx'
                or manifest.get('dimension') != DIMENSION or manifest.get('dtype') != 'float32-le'
                or not isinstance(hashes, list) or len(hashes) > MAX_DOCUMENTS
                or len(set(hashes)) != len(hashes)
                or any(not re.fullmatch(r'[a-f0-9]{64}', value) for value in hashes)
                or manifest.get('bytes') != len(hashes) * DIMENSION * 4
                or manifest['bytes'] > MAX_BYTES
                or not re.fullmatch(r'public-embeddings-[a-zA-Z0-9-]+\.f32', manifest['file'])):
            return {}
        raw = (path.parent / manifest['file']).read_bytes()
        if len(raw) != manifest['bytes'] or hashlib.sha256(raw).hexdigest() != manifest.get('sha256'):
            return {}
        matrix = _checked_matrix(np.frombuffer(raw, dtype='<f4').reshape(-1, DIMENSION), len(hashes))
        return dict(zip(hashes, matrix, strict=True))
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {}


def publish_vectors(output, snapshot, signature, infer, *, previous=None, max_documents=MAX_DOCUMENTS):
    """Compute only missing documents; a failed optional encoder leaves honest gaps."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    texts = public_texts(snapshot, max_documents=max_documents)
    available = read_vectors(previous, signature) if previous else {}
    selected = {text_hash(text): text for text in texts}
    available = {key: value for key, value in available.items() if key in selected}
    reused = len(available)
    missing = sorted(((key, text) for key, text in selected.items() if key not in available),
                     key=lambda pair: len(pair[1]))
    warning = ''
    if missing:
        try:
            result = infer([text for _, text in missing])
            partial = isinstance(result, dict) and result.get('partial') is True
            values = _checked_matrix(result['vectors'] if partial else result,
                                     len(result['vectors']) if partial else len(missing))
            if len(values) > len(missing):
                raise ValueError('Too many public comparison vectors')
            if partial:
                warning = str(result.get('warning') or 'Public comparison precompute was interrupted')[:500]
            available.update((key, values[index]) for index, (key, _) in enumerate(missing[:len(values)]))
        except Exception as exc:  # Optional acceleration must not block code hotfixes.
            warning = f'{type(exc).__name__}: {exc}'[:500]
    hashes = sorted(available)
    raw = np.asarray([available[key] for key in hashes], dtype='<f4').tobytes()
    checksum = hashlib.sha256(raw).hexdigest()
    file = f'public-embeddings-{checksum[:20]}.f32'
    (output / file).write_bytes(raw)
    manifest = {'version': 1, 'public_only': True, 'signature': signature,
                'backend': 'browser-onnx', 'dimension': DIMENSION, 'dtype': 'float32-le',
                'hashes': hashes, 'bytes': len(raw), 'sha256': checksum, 'file': file}
    temporary = output / 'public-embeddings.json.part'
    temporary.write_text(json.dumps(manifest, separators=(',', ':')) + '\n')
    temporary.replace(output / 'public-embeddings.json')
    return {'documents': len(texts), 'published': len(hashes), 'reused': reused,
            'computed': len(hashes) - reused, 'missing': len(texts) - len(hashes),
            'bytes': len(raw), 'complete': len(hashes) == len(texts), 'warning': warning,
            'signature': signature, 'manifest': 'assets/public-embeddings.json'}
