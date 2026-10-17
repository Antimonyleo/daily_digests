"""Restore optional public comparison assets; never fetch reader state."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen

from public_embeddings import MAX_BYTES, read_vectors


def restore_public_cache(destination: Path, base: str) -> bool:
    destination = Path(destination)
    incoming = destination / 'public-embeddings.incoming.json'
    try:
        with urlopen(base.rstrip('/') + '/assets/public-embeddings.json', timeout=20) as response:
            encoded = response.read(2_000_001)
        if len(encoded) > 2_000_000:
            raise ValueError('Manifest exceeds public cache budget')
        manifest = json.loads(encoded)
        if not isinstance(manifest, dict):
            raise ValueError('Invalid public manifest root')
        filename = manifest.get('file', '')
        checksum = manifest.get('sha256', '')
        if (not re.fullmatch(r'public-embeddings-[a-zA-Z0-9-]+\.f32', filename)
                or not isinstance(checksum, str) or not re.fullmatch(r'[a-f0-9]{64}', checksum)
                or filename != f'public-embeddings-{checksum[:20]}.f32'
                or type(manifest.get('bytes')) is not int or not 0 <= manifest['bytes'] <= MAX_BYTES
                or not isinstance(manifest.get('hashes'), list)
                or not isinstance(manifest.get('signature'), str)):
            raise ValueError('Invalid public manifest')
        with urlopen(base.rstrip('/') + '/assets/' + filename, timeout=20) as response:
            raw = response.read(MAX_BYTES + 1)
        if len(raw) != manifest['bytes'] or hashlib.sha256(raw).hexdigest() != manifest.get('sha256'):
            raise ValueError('Public comparison checksum mismatch')
        destination.mkdir(parents=True, exist_ok=True)
        vector = destination / filename
        vector.write_bytes(raw)
        incoming.write_bytes(encoded)
        if len(read_vectors(incoming, manifest['signature'])) != len(manifest['hashes']):
            raise ValueError('Invalid public vectors')
        # read_vectors returns an empty mapping for invalid empty manifests too.
        if not manifest['hashes']:
            raise ValueError('No public vectors to reuse')
        incoming.replace(destination / 'public-embeddings.json')
        print(f'Restored {len(manifest["hashes"])} public comparisons; build checks the current model signature.')
        return True
    except (OSError, ValueError, KeyError, TypeError):
        print('::warning::Previous public comparisons unavailable; the build can compute missing vectors or use local browser fallback.')
        return False
    finally:
        try:
            incoming.unlink(missing_ok=True)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=Path('browser/.build-cache/public-embeddings'))
    parser.add_argument('--site', default='https://antimonyleo.github.io/daily_digests')
    args = parser.parse_args()
    restore_public_cache(args.destination, args.site)


if __name__ == '__main__':
    main()
