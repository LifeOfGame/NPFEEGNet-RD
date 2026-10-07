"""Inspect Zhou2020 source-session metadata without opening target-session entries."""
import json
import socket
import sys
import shutil
from pathlib import Path

import numpy as np

ROOT = Path('/mnt/helongfei/NPFEEGNet-RD-zhou2020-20260927')
sys.path.insert(0, str(ROOT / 'vendor'))
from remotezip import RemoteZip

_getaddrinfo = socket.getaddrinfo
def _ipv4_only(*args, **kwargs):
    return [item for item in _getaddrinfo(*args, **kwargs) if item[0] == socket.AF_INET]
socket.getaddrinfo = _ipv4_only


def main():
    url = 'https://zenodo.org/records/18988317/files/S01.zip?download=1'
    with RemoteZip(url, initial_buffer_size=65536) as archive:
        files = [{'name': x.filename, 'file_size': x.file_size,
                  'compress_size': x.compress_size} for x in archive.infolist()]
        source = [x for x in files if 'session_1/' in x['name'] and x['name'].endswith('.npz')]
        print('archive_entries', len(files), 'source_npz', len(source), flush=True)
        print('first_source', source[0] if source else None, flush=True)
        if not source:
            raise RuntimeError('No session_1 NPZ in archive')
        local = ROOT / 'source_probe_run_01.npz'
        with archive.open(source[0]['name']) as fp, local.open('wb') as dst:
            shutil.copyfileobj(fp, dst)
        with np.load(local, allow_pickle=True) as npz:
                metadata = {'keys': list(npz.files)}
                for key in npz.files:
                    val = npz[key]
                    if key in ('signal', 'MarkOnSignal'):
                        metadata[key] = {'shape': list(val.shape), 'dtype': str(val.dtype)}
                        if key == 'MarkOnSignal':
                            metadata[key]['first_20'] = val[:20].tolist()
                            metadata[key]['unique_codes'] = np.unique(val[:, 1], return_counts=True)[0].tolist()
                    else:
                        metadata[key] = {'shape': list(val.shape), 'dtype': str(val.dtype),
                                         'preview': val.flat[:50].tolist()}
        report = {'subject': 1, 'source_url': url, 'session_1_files': source, 'first_run_metadata': metadata}
        (ROOT / 'results' / 'source_probe.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
        print(json.dumps(metadata, indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()
