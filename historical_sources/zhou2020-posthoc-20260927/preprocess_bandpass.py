"""Post-hoc 8-30 Hz, 0.5-4.5 s Zhou2020 arrays for sanity controls."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

ROOT = Path('/mnt/helongfei/NPFEEGNet-RD-zhou2020-posthoc-20260927')
RAW = Path('/mnt/helongfei/database/raw/zhou2020_zenodo_v1')
LABELS = {769: 1, 770: 2, 771: 3, 780: 4}
SOS = butter(4, (8, 30), btype='bandpass', fs=250, output='sos')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as fp:
        for block in iter(lambda: fp.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def convert_run(path, n_channels):
    with np.load(path, allow_pickle=False) as archive:
        signal = archive['signal']
        events = archive['MarkOnSignal']
        sfreq = int(archive['SampleRate'][0])
    if sfreq != 500 or signal.shape[1] < n_channels or not np.isfinite(signal[:, :n_channels]).all():
        raise ValueError(f'Invalid source run: {path}')
    code = events[:, 1].astype(int)
    starts = events[np.isin(code, list(LABELS)), 0].astype(int)
    labels = events[np.isin(code, list(LABELS)), 1].astype(int)
    ends = events[code == 800, 0].astype(int)
    if not 35 <= len(starts) <= 85:
        raise ValueError(f'Unexpected class-event count {len(starts)} in {path}')
    down = resample_poly(signal[:, :n_channels], 1, 2, axis=0)
    filtered = sosfiltfilt(SOS, down, axis=0)
    trials, y, exclusions = [], [], []
    for i, (start, label) in enumerate(zip(starts, labels)):
        next_class = starts[i + 1] if i + 1 < len(starts) else signal.shape[0]
        possible_ends = ends[(ends > start) & (ends < next_class)]
        if len(possible_ends) == 0 or start + 2250 > int(possible_ends[0]) or start + 2250 > signal.shape[0]:
            exclusions.append(i)
            continue
        first = int(round(start / 2)) + 125
        trial = filtered[first:first + 1000, :].T
        if trial.shape != (n_channels, 1000):
            raise RuntimeError(f'Invalid epoch shape: {path}')
        trials.append(trial.astype(np.float32))
        y.append(LABELS[int(label)])
    return trials, y, {'path': str(path), 'sha256': digest(path), 'class_events': len(starts),
                       'kept': len(y), 'excluded_indices': exclusions}

def main():
    (ROOT / 'data').mkdir(parents=True, exist_ok=True)
    for session in (1, 2):
        report = {'session': session, 'subjects': [], 'preprocessing': 'resample 250 Hz; IIR zero-phase 8-30 Hz; class [0.5,4.5) s'}
        for subject in range(1, 21):
            n_channels = 41 if subject <= 12 else 26
            runs = sorted((RAW / f'S{subject:02d}' / f'session_{session}').glob('run_*.npz'))
            if not runs:
                raise RuntimeError(f'No source files for subject {subject}, session {session}')
            trials, labels, groups, run_reports = [], [], [], []
            for group, path in enumerate(runs):
                x, y, r = convert_run(path, n_channels)
                trials.extend(x)
                labels.extend(y)
                groups.extend([group] * len(y))
                run_reports.append(r)
            x = np.stack(trials)
            y = np.asarray(labels, dtype=np.int64)
            g = np.asarray(groups, dtype=np.int64)
            if x.shape[1:] != (n_channels, 1000) or not np.isfinite(x).all() or np.any(np.bincount(y, minlength=5)[1:] < 10):
                raise RuntimeError(f'Invalid arrays for subject {subject}, session {session}')
            stem = ROOT / 'data' / f'Z{subject:02d}{"T" if session == 1 else "E"}'
            np.save(f'{stem}_data.npy', x)
            np.save(f'{stem}_label.npy', y)
            np.save(f'{stem}_group.npy', g)
            report['subjects'].append({'subject': subject, 'shape': list(x.shape),
                                       'class_counts': np.bincount(y, minlength=5)[1:].tolist(),
                                       'run_counts': np.bincount(g, minlength=len(runs)).tolist(),
                                       'runs': run_reports})
            print(f'Session {session}, subject {subject}: {len(x)} trials', flush=True)
        (ROOT / 'results').mkdir(exist_ok=True)
        (ROOT / 'results' / f'session_{session}_preprocess_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('BOTH SESSIONS POST-HOC PREPROCESSED', flush=True)

if __name__ == '__main__':
    main()
