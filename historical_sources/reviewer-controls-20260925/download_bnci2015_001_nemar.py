"""Fetch and SHA-256 verify the 24 A/B source MAT files from NEMAR v1.0.2."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

BASE = 'https://data.nemar.org/nm000140/v1.0.2/sourcedata/'
DEST = Path('/mnt/helongfei/database/raw/bnci2015_001_nemar_v1.0.2')
DEST.mkdir(parents=True, exist_ok=True)
provenance = json.loads(subprocess.check_output([
    'curl', '-4', '-fLsS', '--connect-timeout', '10', '--max-time', '45',
    BASE + 'sourcedata_provenance.json'], timeout=50))
(DEST / 'sourcedata_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
expected = {f'S{subject:02d}{phase}.mat' for subject in range(1, 13) for phase in 'AB'}
files = [item for item in provenance['files'] if item['file'] in expected]
if len(files) != 24 or {item['file'] for item in files} != expected:
    raise RuntimeError('NEMAR provenance does not contain exactly 24 A/B files')


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


for index, item in enumerate(files, 1):
    name = item['file']
    path = DEST / name
    if path.exists() and path.stat().st_size == item['bytes'] and digest(path) == item['sha256']:
        print(f'{index:02d}/24 verified-existing {name}', flush=True)
        continue
    part = DEST / (name + '.part')
    for attempt in range(1, 5):
        try:
            subprocess.run([
                'curl', '-4', '-fLsS', '--connect-timeout', '10',
                '--max-time', '600', '--output', str(part), BASE + name],
                check=True, timeout=610)
            if part.stat().st_size != item['bytes'] or digest(part) != item['sha256']:
                raise RuntimeError(f'Checksum/size mismatch for {name}')
            os.replace(part, path)
            print(f'{index:02d}/24 downloaded-verified {name} {item["bytes"]}', flush=True)
            break
        except Exception as exc:
            print(f'{index:02d}/24 retry {attempt} {name}: {exc}', flush=True)
            if attempt == 4:
                raise
            time.sleep(5 * attempt)
print('ALL_24_VERIFIED', flush=True)
