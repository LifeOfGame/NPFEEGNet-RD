"""Unlock and preprocess BNCI2015-001 B only after audited A-only epoch maps."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

ROOT = Path('/mnt/helongfei/NPFEEGNet-RD-reviewer-controls-20260925')
RAW = Path('/mnt/helongfei/database/raw/bnci2015_001_nemar_v1.0.2')
OUT = ROOT / 'data/bnci2015_001'
LOCK = ROOT / 'results/bnci2015_001_protocol_lock.json'


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


lock = json.loads(LOCK.read_text())
if lock.get('status') != 'ready_for_session2' or len(lock.get('selection', {})) != 3:
    raise RuntimeError('Complete A-only protocol lock required before B processing')
for artifact in lock['artifacts']:
    path = Path(artifact['path'])
    if digest(path) != artifact['sha256']:
        raise RuntimeError(f'Locked artifact changed: {path}')

report = {'dataset': 'BNCI2015-001', 'session': 'B', 'unlock': str(LOCK),
          'window': '[0,4) s from 1-based trial event marker',
          'sampling_rate_input': 512, 'sampling_rate_output': 250, 'subjects': []}
for subject in range(1, 13):
    name = f'S{subject:02d}B.mat'
    path = RAW / name
    if digest(path) != lock['session_b_sha256'][name]:
        raise RuntimeError(f'Source B checksum mismatch: {path}')
    run_array = np.atleast_1d(loadmat(path, struct_as_record=False, squeeze_me=True)['data'])
    if len(run_array) != 1:
        raise ValueError(f'{name}: unexpected run count')
    run = run_array[0]
    signal = np.asarray(run.X)
    labels = np.asarray(run.y).ravel().astype(np.int64)
    onsets = np.asarray(run.trial).ravel().astype(np.int64) - 1
    if int(run.fs) != 512 or signal.shape[1] != 13 or len(labels) != 200 or len(onsets) != 200:
        raise ValueError(f'{name}: unexpected shape, sample rate or trial count')
    if set(labels.tolist()) != {1, 2} or np.any(onsets < 0) or np.any(onsets + 2048 > len(signal)):
        raise ValueError(f'{name}: invalid labels or epoch bounds')
    epochs = np.stack([signal[start:start + 2048].T for start in onsets])
    epochs = resample_poly(epochs, 125, 256, axis=-1).astype(np.float32)
    if epochs.shape != (200, 13, 1000) or not np.isfinite(epochs).all():
        raise ValueError(f'{name}: invalid epochs')
    stem = OUT / f'N{subject:02d}E'
    np.save(f'{stem}_data.npy', epochs)
    np.save(f'{stem}_label.npy', labels)
    report['subjects'].append({'subject': subject, 'source_file': name,
                               'sha256': lock['session_b_sha256'][name],
                               'trials': 200,
                               'class_counts': np.bincount(labels, minlength=3)[1:].tolist(),
                               'epoch_shape': list(epochs.shape)})
    print(f'{subject:02d}/12 B processed after lock', flush=True)
(OUT / 'session_b_preprocess_report.json').write_text(json.dumps(report, indent=2) + '\n')
print('ALL_12_SESSION_B_PROCESSED_AFTER_LOCK', flush=True)
