"""BNCI2015-001 Session A only: cue-relative 0-4 s, 250 Hz, fixed 5 folds."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
from sklearn.model_selection import StratifiedKFold

RAW = Path('/mnt/helongfei/database/raw/bnci2015_001_nemar_v1.0.2')
OUT = Path('/mnt/helongfei/NPFEEGNet-RD-reviewer-controls-20260925/data/bnci2015_001')
OUT.mkdir(parents=True, exist_ok=True)
WINDOW_SAMPLES = 2048  # 4 s at 512 Hz, start at the 1-based trial marker
CHANNELS = ['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CPz', 'CP4']


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


provenance = json.loads((RAW / 'sourcedata_provenance.json').read_text())
expected = {item['file']: item for item in provenance['files']}
report = {'dataset': 'BNCI2015-001', 'source': 'NEMAR nm000140 v1.0.2 sourcedata',
          'session': 'A only', 'sampling_rate_input': 512, 'sampling_rate_output': 250,
          'window': '[0,4) s from 1-based trial event marker',
          'channels': CHANNELS, 'folds': 'StratifiedKFold(5, shuffle=True, random_state=20260926)',
          'subjects': []}

for subject in range(1, 13):
    name = f'S{subject:02d}A.mat'
    path = RAW / name
    assert path.stat().st_size == expected[name]['bytes']
    assert digest(path) == expected[name]['sha256']
    mat = loadmat(path, struct_as_record=False, squeeze_me=True)
    data = np.atleast_1d(mat['data'])
    if len(data) != 1:
        raise ValueError(f'{name}: expected one continuous run, got {len(data)}')
    run = data[0]
    signal = np.asarray(run.X)
    labels = np.asarray(run.y).ravel().astype(np.int64)
    onsets = np.asarray(run.trial).ravel().astype(np.int64) - 1
    if int(run.fs) != 512 or signal.shape[1] != 13 or len(labels) != 200 or len(onsets) != 200:
        raise ValueError(f'{name}: unexpected shape, sfreq or count')
    if set(labels.tolist()) != {1, 2} or np.any(onsets < 0) or np.any(onsets + WINDOW_SAMPLES > len(signal)):
        raise ValueError(f'{name}: invalid labels or window bounds')
    epochs = np.stack([signal[start:start + WINDOW_SAMPLES].T for start in onsets])
    epochs = resample_poly(epochs, 125, 256, axis=-1).astype(np.float32)
    if epochs.shape != (200, 13, 1000) or not np.isfinite(epochs).all():
        raise ValueError(f'{name}: invalid processed epochs')
    groups = np.empty(200, dtype=np.int64)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260926)
    for fold, (_, validation) in enumerate(splitter.split(np.arange(200), labels)):
        groups[validation] = fold
    if sorted(np.bincount(groups).tolist()) != [40] * 5:
        raise ValueError(f'{name}: unexpected fold sizes')
    for fold in range(5):
        if sorted(np.bincount(labels[groups == fold] - 1).tolist()) != [20, 20]:
            raise ValueError(f'{name}: nonstratified fold')
    stem = OUT / f'N{subject:02d}T'
    np.save(f'{stem}_data.npy', epochs)
    np.save(f'{stem}_label.npy', labels)
    np.save(f'{stem}_group.npy', groups)
    info = {'subject': subject, 'source_file': name, 'sha256': expected[name]['sha256'],
            'trials': 200, 'class_counts': np.bincount(labels, minlength=3)[1:].tolist(),
            'epoch_shape': list(epochs.shape), 'fold_counts': np.bincount(groups).tolist(),
            'first_onsets_zero_based': onsets[:5].tolist(),
            'signal_min_uv': float(epochs.min()), 'signal_max_uv': float(epochs.max())}
    report['subjects'].append(info)
    print(f'{subject:02d}/12 A verified and processed', flush=True)

(OUT / 'session_a_preprocess_report.json').write_text(json.dumps(report, indent=2) + '\n')
print('ALL_12_SESSION_A_PROCESSED', flush=True)
