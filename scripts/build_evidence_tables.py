"""Portable descriptive reconstruction of complete studies, not GPU partitions.

Uses exact rational accuracies for ties, subject-first aggregation, and pooled
calibration. No confidence intervals or p-values are silently recomputed.
"""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_predictions import calibration, summarize

ROOT = Path(__file__).resolve().parents[1]


def combine(parts, expected_subjects):
    """parts: (cohort namespace, arrays, validated run summaries)."""
    runs, labels, probabilities = {}, [], []
    for namespace, arrays, summaries in parts:
        for row in summaries:
            subject = f'{namespace}:{row["subject"]}'
            key = (subject, row['seed'])
            if key in runs:
                raise ValueError(f'Duplicate subject/seed across partitions: {key}')
            runs[key] = dict(row, subject=subject)
        labels.append(arrays['labels'])
        probabilities.append(arrays['probabilities'])
    subjects = sorted({s for s, _ in runs})
    if len(subjects) != expected_subjects:
        raise ValueError(f'Expected {expected_subjects} subjects; got {len(subjects)}')
    means = {}
    for subject in subjects:
        rows = [r for (s, _), r in runs.items() if s == subject]
        if {r['seed'] for r in rows} != {0, 1, 2}:
            raise ValueError(f'Incomplete three-seed study for {subject}')
        means[subject] = sum((Fraction(r['correct'], r['trials']) for r in rows), Fraction()) / 3
    nll, ece = calibration(np.concatenate(labels), np.concatenate(probabilities))
    accuracies = np.array([float(x) for x in means.values()])
    return dict(subjects=len(subjects), runs=len(runs), trials=sum(len(y) for y in labels),
                accuracy_pct=float(accuracies.mean()*100), subject_sd_pct=float(accuracies.std(ddof=1)*100),
                pooled_nll=nll, pooled_ece10_pct=ece*100), means


def paired(left, right):
    if left.keys() != right.keys():
        raise ValueError('Paired subjects differ')
    d = [left[s]-right[s] for s in sorted(left)]
    return dict(subjects=len(d), difference_pp=float(sum(d)/len(d)*100),
                wins=sum(x>0 for x in d), ties=sum(x==0 for x in d), losses=sum(x<0 for x in d))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args=parser.parse_args()
    spec=json.loads((ROOT/'evidence/studies.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'evidence/predictions/manifest.json').read_text(encoding='utf-8'))['groups']
    cache={}
    for study in spec['studies'].values():
        for part in study['parts']:
            group=part['group']
            if group in cache: continue
            entry=manifest[group]
            path=ROOT/'evidence/predictions'/entry['file']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
                raise ValueError(f'Prediction hash mismatch: {group}')
            summary, rows=summarize(path)
            if summary['runs']!=entry['runs'] or summary['trials']!=entry['trials']:
                raise ValueError(f'Manifest count mismatch: {group}')
            with np.load(path,allow_pickle=False) as f:
                cache[group]=({k:f[k] for k in ('labels','probabilities')},rows)
    metrics, subject_rows, means=[],[],{}
    for key,study in spec['studies'].items():
        parts=[(p['namespace'],*cache[p['group']]) for p in study['parts']]
        summary, means[key]=combine(parts,study['subjects'])
        metrics.append(dict(study=key,evidence_role=study['role'],**summary))
        subject_rows.extend(dict(study=key,subject=s,accuracy_pct=float(a*100),
                                 accuracy_numerator=a.numerator,accuracy_denominator=a.denominator)
                            for s,a in means[key].items())
    contrasts=[dict(contrast=c['name'],**paired(means[c['left']],means[c['right']])) for c in spec['contrasts']]
    args.output.mkdir(parents=True,exist_ok=False)
    write_csv(args.output/'study_metrics.csv',metrics)
    write_csv(args.output/'subject_metrics.csv',subject_rows)
    write_csv(args.output/'paired_descriptive.csv',contrasts)
    print(f'PASS: {len(metrics)} complete study arms, {len(contrasts)} descriptive contrasts')


if __name__=='__main__':
    main()
