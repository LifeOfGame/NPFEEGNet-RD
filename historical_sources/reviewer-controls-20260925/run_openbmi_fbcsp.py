"""Fixed 9-band FBCSP-style classical OpenBMI S1 -> S2 audit.

The frequency bank and four selected CSP pairs are set before S2 evaluation.
Only S1 labels enter CSP, mutual information, and LDA fitting.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, cohen_kappa_score

DATA = Path('/mnt/helongfei/database/processed/openbmi')
OUT = Path('/mnt/helongfei/NPFEEGNet-RD-reviewer-controls-20260925/results')
BANDS = [(4 + 4 * i, 8 + 4 * i) for i in range(9)]


def load(subject, suffix):
    stem = DATA / f'O{subject:02d}{suffix}'
    x = np.load(f'{stem}_data.npy').astype(np.float64)
    y = np.load(f'{stem}_label.npy').ravel()
    if x.shape != (200, 20, 1000) or set(y.tolist()) != {1, 2}:
        raise ValueError(f'Unexpected data or labels: {stem}')
    return x, y


def covariance(x):
    centered = x - x.mean(axis=-1, keepdims=True)
    cov = centered @ centered.transpose(0, 2, 1)
    trace = np.trace(cov, axis1=1, axis2=2)
    return cov / trace[:, None, None]


def csp_filter(x, y):
    cov = covariance(x)
    c1 = cov[y == 1].mean(axis=0)
    c2 = cov[y == 2].mean(axis=0)
    reg = 1e-6 * np.trace(c1 + c2) / c1.shape[0]
    _, vectors = eigh(c1, c1 + c2 + reg * np.eye(c1.shape[0]))
    return vectors[:, np.r_[0:2, -2:0]].T


def features(x, spatial_filter):
    projected = spatial_filter @ x
    variance = np.var(projected, axis=-1)
    variance /= variance.sum(axis=1, keepdims=True)
    return np.log(np.clip(variance, 1e-12, None))


def run(subject):
    train_x, train_y = load(subject, 'T')
    test_x, test_y = load(subject, 'E')
    train_features, test_features = [], []
    for low, high in BANDS:
        sos = butter(4, [low, high], btype='bandpass', fs=250, output='sos')
        train_filtered = sosfiltfilt(sos, train_x, axis=-1)
        test_filtered = sosfiltfilt(sos, test_x, axis=-1)
        spatial_filter = csp_filter(train_filtered, train_y)
        train_features.append(features(train_filtered, spatial_filter))
        test_features.append(features(test_filtered, spatial_filter))
    train_features = np.concatenate(train_features, axis=1)
    test_features = np.concatenate(test_features, axis=1)
    # MIBIF-style pair selection: four spatial-filter pairs, ranked by the
    # stronger member of each pair. MI is fitted on S1 only.
    mi = mutual_info_classif(train_features, train_y, discrete_features=False,
                              n_neighbors=3, random_state=20260925)
    spatial_pairs = [(4*b, 4*b+3) for b in range(9)] + [(4*b+1, 4*b+2) for b in range(9)]
    pair_scores = [max(mi[left], mi[right]) for left, right in spatial_pairs]
    chosen_pairs = sorted(np.argsort(pair_scores)[-4:].tolist())
    chosen = [index for pair in chosen_pairs for index in spatial_pairs[pair]]
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(train_features[:, chosen], train_y)
    prediction = clf.predict(test_features[:, chosen])
    return {'subject': subject, 'test_accuracy': float(accuracy_score(test_y, prediction)),
            'test_kappa': float(cohen_kappa_score(test_y, prediction)),
            'selected_pairs': json.dumps(chosen_pairs),
            'selected_bands': json.dumps([BANDS[pair % 9] for pair in chosen_pairs])}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / 'openbmi_fbcsp_subjects.csv'
    rows = []
    for subject in range(1, 55):
        rows.append(run(subject))
        with destination.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'{subject:02d}/54 {rows[-1]["test_accuracy"]:.3f}', flush=True)
    report = {
        'status': 'post-confirmatory fixed classical FBCSP-style audit',
        'protocol': 'OpenBMI 54 subjects; S1 train and S2 test; 200 trials each',
        'method': '9 non-overlapping 4-Hz bands (4-40 Hz), 4th-order zero-phase trial-wise Butterworth; CSP 2+2 per band; S1-only mutual-information selection of four CSP pairs; shrinkage LDA',
        'subject_macro_accuracy_pct': 100 * float(np.mean([r['test_accuracy'] for r in rows])),
        'subject_macro_kappa': float(np.mean([r['test_kappa'] for r in rows])),
        'limitations': 'FBCSP-style reference, not author-exact MIBIF/NBPW; trial-wise filtering may have boundary effects. No S2 hyperparameter selection.'
    }
    (OUT / 'openbmi_fbcsp_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
