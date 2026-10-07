"""Release Zhou2020 session_2 only after source-only training is frozen."""
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
    with Path(path).open('rb') as fp:
        for block in iter(lambda: fp.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def check_gate():
    path = ROOT / 'results' / 'final_unlock_lock.json'
    lock = json.loads(path.read_text())
    if lock['status'] != 'ready_for_session2' or lock['dataset'] != 'Zhou2020':
        raise RuntimeError('Target lock is absent or invalid')
    for item in lock['artifacts']:
        if sha256(item['path']) != item['sha256']:
            raise RuntimeError(f'Locked artifact changed: {item["path"]}')
    return sha256(path)

def fetch_subject(subject, unlock_sha):
    name = f'S{subject:02d}'
    url = f'https://zenodo.org/records/18988317/files/{name}.zip?download=1'
    folder = RAW / name / 'session_2'
    folder.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / 'results' / f'{name}_session_2_target.json'
    for attempt in range(1, 6):
        try:
            files = []
            with RemoteZip(url, initial_buffer_size=65536) as archive:
                members = [info for info in archive.infolist()
                           if info.filename.startswith('session_2/') and info.filename.endswith('.npz')]
                if not 1 <= len(members) <= 6:
                    raise RuntimeError(f'{name}: expected 1-6 target runs; found {len(members)}')
                for info in sorted(members, key=lambda x: x.filename):
                    dst = folder / Path(info.filename).name
                    if not dst.exists() or dst.stat().st_size != info.file_size:
                        temp = dst.with_suffix('.npz.part')
                        with archive.open(info.filename) as src, temp.open('wb') as out:
                            shutil.copyfileobj(src, out, 1024 * 1024)
                        if temp.stat().st_size != info.file_size:
                            raise RuntimeError(f'{name}: size mismatch {info.filename}')
                        temp.replace(dst)
                    files.append({'source_zip_member': info.filename, 'zip_crc32': f'{info.CRC:08x}',
                                  'uncompressed_bytes': info.file_size, 'compressed_bytes': info.compress_size,
                                  'local_path': str(dst), 'sha256': sha256(dst)})
            report = {'subject': subject, 'archive_url': url, 'target_session': 'session_2',
                      'unlock_sha256': unlock_sha, 'files': files,
                      'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            print(f'{name} TARGET COMPLETE', flush=True)
            return
        except Exception as error:
            print(f'{name} target attempt {attempt}/5 failed: {error!r}', flush=True)
            if attempt == 5:
                raise
            time.sleep(10 * attempt)

def main():
    unlock_sha = check_gate()
    (ROOT / 'results' / 'target_fetch_started_at.txt').write_text(
        time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + '\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_subject, subject, unlock_sha) for subject in range(1, 21)]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    (ROOT / 'results' / 'target_fetch_status.txt').write_text('complete\n')
    print('ALL TARGET SESSION_2 RUNS FETCHED AFTER LOCK', flush=True)

if __name__ == '__main__':
    main()
