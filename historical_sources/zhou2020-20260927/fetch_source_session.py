"""Fetch only Zhou2020 session_1 members, verifying ZIP CRCs and SHA-256."""
import concurrent.futures
import hashlib
import json
import shutil
import socket
import sys
import time
from pathlib import Path

ROOT = Path('/mnt/helongfei/NPFEEGNet-RD-zhou2020-20260927')
RAW = Path('/mnt/helongfei/database/raw/zhou2020_zenodo_v1')
sys.path.insert(0, str(ROOT / 'vendor'))
from remotezip import RemoteZip

_getaddrinfo = socket.getaddrinfo
def _ipv4_only(*args, **kwargs):
    return [item for item in _getaddrinfo(*args, **kwargs) if item[0] == socket.AF_INET]
socket.getaddrinfo = _ipv4_only


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as fp:
        for block in iter(lambda: fp.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch_subject(subject):
    name = f'S{subject:02d}'
    url = f'https://zenodo.org/records/18988317/files/{name}.zip?download=1'
    folder = RAW / name / 'session_1'
    folder.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / 'results' / f'{name}_session_1_source.json'
    for attempt in range(1, 6):
        try:
            files = []
            with RemoteZip(url, initial_buffer_size=65536) as archive:
                members = [info for info in archive.infolist()
                           if info.filename.startswith('session_1/') and info.filename.endswith('.npz')]
                if not 3 <= len(members) <= 6:
                    raise RuntimeError(f'{name}: expected 3-6 session_1 runs; found {len(members)}')
                for info in sorted(members, key=lambda x: x.filename):
                    dst = folder / Path(info.filename).name
                    if not dst.exists() or dst.stat().st_size != info.file_size:
                        temp = dst.with_suffix('.npz.part')
                        with archive.open(info.filename) as src, temp.open('wb') as out:
                            shutil.copyfileobj(src, out, 1024 * 1024)
                        if temp.stat().st_size != info.file_size:
                            raise RuntimeError(f'{name}: size mismatch for {info.filename}')
                        temp.replace(dst)
                    files.append({'source_zip_member': info.filename, 'zip_crc32': f'{info.CRC:08x}',
                                  'uncompressed_bytes': info.file_size, 'compressed_bytes': info.compress_size,
                                  'local_path': str(dst), 'sha256': sha256(dst)})
            report = {'subject': subject, 'archive_url': url, 'source_session': 'session_1 only',
                      'files': files, 'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            print(f'{name} COMPLETE {sum(x["compressed_bytes"] for x in files) / 1e6:.0f} MB compressed', flush=True)
            return report
        except Exception as error:
            print(f'{name} attempt {attempt}/5 failed: {error!r}', flush=True)
            if attempt == 5:
                raise
            time.sleep(10 * attempt)


def main():
    lock = ROOT / 'results' / 'protocol_lock.json'
    amendment = ROOT / 'results' / 'source_only_addendum.json'
    if not lock.is_file() or not amendment.is_file():
        raise RuntimeError('Protocol and source-only amendment must exist before extraction')
    RAW.mkdir(parents=True, exist_ok=True)
    status = ROOT / 'results' / 'fetch_status.txt'
    status.write_text('source_session_1_download_running\n')
    started = time.monotonic()
    missing = [i for i in range(1, 21) if not (ROOT / 'results' / f'S{i:02d}_session_1_source.json').is_file()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch_subject, i): i for i in missing}
        for future in concurrent.futures.as_completed(futures):
            future.result()
    if any(not (ROOT / 'results' / f'S{i:02d}_session_1_source.json').is_file() for i in range(1, 21)):
        raise RuntimeError('At least one source subject is missing')
    status.write_text('source_session_1_download_complete\n')
    print(f'ALL 20 SOURCE SUBJECTS COMPLETE in {(time.monotonic()-started)/60:.1f} minutes', flush=True)


if __name__ == '__main__':
    main()
