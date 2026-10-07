"""Check distributed files without requiring EEG, PyTorch, or the server."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(f'Path outside repository: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'File differs from preparation snapshot: {name}')
    print(f'PASS: {len(manifest)} distributed files verified. This is not a training replication.')


if __name__ == '__main__':
    main()
