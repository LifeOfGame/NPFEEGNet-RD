"""Convert locked Zhou2020 source or gated target runs to model arrays."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT = Path('/mnt/helongfei/NPFEEGNet-RD-zhou2020-20260927')
RAW = Path('/mnt/helongfei/database/raw/zhou2020_zenodo_v1')
LABELS = {769: 1, 770: 2, 771: 3, 780: 4}
N_INPUT = 2000
N_OUTPUT = 1000


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as fp:
        for block in iter(lambda: fp.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_target_gate():
    lock_path = ROOT / 'results' / 'final_unlock_lock.json'
    lock = json.loads(lock_path.read_text())
    if lock['status'] != 'ready_for_session2' or lock['dataset'] != 'Zhou2020':
        raise RuntimeError('Target unlock status invalid')
    for item in lock['artifacts']:
        path = Path(item['path'])
        if digest(path) != item['sha256']:
            raise RuntimeError(f'Locked artifact changed: {path}')
    return digest(lock_path)


def convert_run(path, n_channels):
    with np.load(path, allow_pickle=False) as archive:
        signal = archive['signal']
        events = archive['MarkOnSignal']
        sfreq = int(archive['SampleRate'][0])
    if sfreq != 500 or signal.ndim != 2 or signal.shape[1] < n_channels:
        raise ValueError(f'Unexpected source shape or sampling rate: {path}')
    if not np.isfinite(signal[:, :n_channels]).all():
        raise ValueError(f'Non-finite EEG values: {path}')
    if events.ndim != 2 or events.shape[1] != 2:
        raise ValueError(f'Invalid events: {path}')
    code = events[:, 1].astype(int)
    starts = events[np.isin(code, list(LABELS)), 0].astype(int)
    labels = events[np.isin(code, list(LABELS)), 1].astype(int)
    ends = events[code == 800, 0].astype(int)
    if len(starts) < 35 or len(starts) > 85:
        raise ValueError(f'Unexpected class-event count {len(starts)}: {path}')
    # Original NPZ signal is in microvolts. Resample the continuous run, then
    # slice trials to avoid per-epoch padding artifacts.
    resampled = resample_poly(signal[:, :n_channels], 1, 2, axis=0)
    trials, y, exclusions = [], [], []
    for i, (start, label) in enumerate(zip(starts, labels)):
        next_class = starts[i + 1] if i + 1 < len(starts) else signal.shape[0]
        subsequent_ends = ends[(ends > start) & (ends < next_class)]
        if len(subsequent_ends) == 0:
            exclusions.append({'trial_index': i, 'reason': 'missing_end_marker'})
            continue
        end = int(subsequent_ends[0])
        if start < 0 or start + N_INPUT > end or start + N_INPUT > signal.shape[0]:
            exclusions.append({'trial_index': i, 'reason': 'short_or_out_of_bounds',
                               'available_samples': int(end - start)})
            continue
        first = int(round(start / 2))
        trial = resampled[first:first + N_OUTPUT, :].T
        if trial.shape != (n_channels, N_OUTPUT):
            raise RuntimeError(f'Bad resampled trial shape: {path}')
        trials.append(trial.astype(np.float32))
        y.append(LABELS[int(label)])
    return trials, y, {'path': str(path), 'sha256': digest(path), 'n_class_events': len(starts),
                       'class_counts_before': {str(c): int((labels == c).sum()) for c in LABELS},
                       'kept': len(y), 'excluded': exclusions,
                       'raw_median_abs_uv': float(np.median(np.abs(signal[:, :n_channels]))),
                       'raw_q99_abs_uv': float(np.quantile(np.abs(signal[:, :n_channels]), .99))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=int, choices=[1, 2], required=True)
    parser.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 21)))
    args = parser.parse_args()
    if args.session == 2:
        unlock_sha = check_target_gate()
    else:
        if not (ROOT / 'results' / 'protocol_lock.json').is_file():
            raise RuntimeError('Missing protocol lock')
        unlock_sha = None
    out = ROOT / 'data'
    out.mkdir(exist_ok=True)
    report = {'dataset': 'Zhou2020', 'session': args.session, 'source_units': 'microvolts',
              'input_sampling_rate_hz': 500, 'model_sampling_rate_hz': 250,
              'window': 'class-event [0,4) seconds', 'target_unlock_sha256': unlock_sha,
              'subjects': []}
    for subject in args.subjects:
        n_channels = 41 if subject <= 12 else 26
        folder = RAW / f'S{subject:02d}' / f'session_{args.session}'
        files = sorted(folder.glob('run_*.npz'))
        minimum_runs = 3 if args.session == 1 else 1
        if not minimum_runs <= len(files) <= 6:
            raise RuntimeError(f'Subject {subject}: expected {minimum_runs}-6 runs in {folder}; found {len(files)}')
        trials, labels, groups, run_reports = [], [], [], []
        for run, path in enumerate(files):
            x, y, run_report = convert_run(path, n_channels)
            trials.extend(x)
            labels.extend(y)
            groups.extend([run] * len(y))
            run_reports.append(run_report)
        x = np.stack(trials)
        y = np.asarray(labels, dtype=np.int64)
        g = np.asarray(groups, dtype=np.int64)
        minimum_trials = 90 if args.session == 1 else 30
        if x.shape[1:] != (n_channels, N_OUTPUT) or len(x) < minimum_trials or not np.isfinite(x).all():
            raise RuntimeError(f'Subject {subject}: invalid processed array')
        minimum_per_class = 20 if args.session == 1 else 5
        if np.any(np.bincount(y, minlength=5)[1:] < minimum_per_class):
            raise RuntimeError(f'Subject {subject}: insufficient class trials')
        stem = out / f'Z{subject:02d}{"T" if args.session == 1 else "E"}'
        np.save(f'{stem}_data.npy', x)
        np.save(f'{stem}_label.npy', y)
        np.save(f'{stem}_group.npy', g)
        report['subjects'].append({'subject': subject, 'shape': list(x.shape),
                                   'class_counts': np.bincount(y, minlength=5)[1:].tolist(),
                                   'run_counts': np.bincount(g, minlength=len(files)).tolist(),
                                   'runs': run_reports})
        print(f'Subject {subject}/20 session_{args.session}: {len(x)} trials, {n_channels} channels', flush=True)
    report_path = out / f'session_{args.session}{"_smoke" if len(args.subjects) != 20 else ""}_report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'ALL SESSION_{args.session} PROCESSED', flush=True)


if __name__ == '__main__':
    main()
