"""Author-inspired, post-hoc three-class CSP-SVM sanity check."""
import argparse
import csv
import json
from pathlib import Path

import mne
import numpy as np
from mne.decoding import CSP
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path('/mnt/helongfei/NPFEEGNet-RD-zhou2020-posthoc-20260927')
mne.set_log_level('ERROR')

def model():
    return make_pipeline(CSP(n_components=6, reg='ledoit_wolf', log=True, norm_trace=False),
                         StandardScaler(), SVC(kernel='rbf', C=0.8))

def load(subject, split):
    stem = ROOT / 'data' / f'Z{subject:02d}{split}'
    x = np.load(f'{stem}_data.npy')
    y = np.load(f'{stem}_label.npy')
    keep = y <= 3
    return x[keep], y[keep]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 21)))
    args = parser.parse_args()
    rows = []
    for subject in args.subjects:
        x_train, y_train = load(subject, 'T')
        x_test, y_test = load(subject, 'E')
        if set(np.unique(y_train)) != {1, 2, 3} or set(np.unique(y_test)) != {1, 2, 3}:
            raise RuntimeError(f'Subject {subject}: missing active MI class')
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=20260927)
        scores = cross_val_score(model(), x_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
        fitted = model().fit(x_train, y_train)
        crossday = float(fitted.score(x_test, y_test))
        row = {'subject': subject, 'source_trials': len(y_train), 'target_trials': len(y_test),
               'source_within_session_10fold_accuracy': float(scores.mean()),
               'source_within_session_10fold_sd': float(scores.std(ddof=1)),
               'session1_to_session2_accuracy': crossday}
        rows.append(row)
        print(f'Subject {subject}/20: within={scores.mean():.3f}, crossday={crossday:.3f}', flush=True)
    label = 'all' if len(args.subjects) == 20 else 'smoke'
    with (ROOT / 'results' / f'csp_{label}_subjects.csv').open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {'status': 'complete', 'role': 'post-hoc author-inspired sanity check, not original locked result',
              'classes': ['left_hand', 'right_hand', 'feet'], 'subjects': len(args.subjects),
              'source_within_session_10fold_accuracy_percent': float(100 * np.mean([r['source_within_session_10fold_accuracy'] for r in rows])),
              'session1_to_session2_accuracy_percent': float(100 * np.mean([r['session1_to_session2_accuracy'] for r in rows])),
              'method': '8-30Hz IIR, [0.5,4.5)s, six CSP components, StandardScaler, RBF SVM C=0.8; no ICA/EEMD-CCA',
              'subject_csv': str(ROOT / 'results' / f'csp_{label}_subjects.csv')}
    (ROOT / 'results' / f'csp_{label}_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
