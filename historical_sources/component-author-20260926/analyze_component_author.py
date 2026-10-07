"""Verify all 162 final rows and compare subject-mean OpenBMI accuracies."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path('/mnt/helongfei/NPFEEGNet-RD-component-author-20260926')
RD = Path('/mnt/helongfei/NPFEEGNet-RD-unified-repro-20260918/output/openbmi_final_npf_raw_demean_3seed/NPFEEGNet/2026-08-31--14-52/summary.csv')
EXPECTED = {(s, z) for s in range(1, 55) for z in range(3)}


def read_grid(path):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    keys = [(int(r['subject']), int(r['seed'])) for r in rows]
    if len(rows) != 162 or set(keys) != EXPECTED or len(set(keys)) != 162:
        raise RuntimeError(f'Incomplete 54x3 final grid: {path}')
    if not all('test_accuracy' in r and r['test_accuracy'] for r in rows):
        raise RuntimeError(f'Missing test accuracy: {path}')
    return {(int(r['subject']), int(r['seed'])): float(r['test_accuracy']) for r in rows}


def summary(data):
    by_subject = np.array([np.mean([data[s, z] for z in range(3)]) for s in range(1, 55)])
    return by_subject


def paired(a, b, rng):
    diff = a - b
    draws = rng.integers(0, 54, size=(100000, 54))
    boot = diff[draws].mean(axis=1)
    signs = rng.choice([-1.0, 1.0], size=(200000, 54))
    null = (signs * diff).mean(axis=1)
    p = (np.count_nonzero(np.abs(null) >= abs(diff.mean())) + 1) / (len(null) + 1)
    return {'mean_pp': float(100 * diff.mean()),
            'subject_bootstrap_95_ci_pp': [float(x) for x in 100 * np.quantile(boot, [0.025, 0.975])],
            'sign_flip_monte_carlo_p': float(p),
            'wins_ties_losses': [int(np.sum(diff > 0)), int(np.sum(diff == 0)), int(np.sum(diff < 0))]}


def main():
    if (ROOT / 'results/final_status.txt').read_text().strip() != 'complete':
        raise RuntimeError('Final training is not complete')
    lock = json.loads((ROOT / 'results/session2_lock.json').read_text())
    if lock['status'] != 'ready_for_session2':
        raise RuntimeError('No S2 lock')
    grids = {'npfeegnet_rd': read_grid(RD)}
    source_paths = {'npfeegnet_rd': str(RD)}
    for name in ('spectral_only', 'no_spectral_prior', 'fbcnet_author_training'):
        config = ROOT / 'config' / f'{name}_final.yaml'
        import yaml
        payload = yaml.safe_load(config.read_text())
        paths = list((Path(payload['out_folder']) / payload['network']).glob('*/summary.csv'))
        if len(paths) != 1:
            raise RuntimeError(f'{name}: expected one final summary, found {paths}')
        grids[name] = read_grid(paths[0])
        source_paths[name] = str(paths[0])
    means = {name: summary(data) for name, data in grids.items()}
    with (ROOT / 'results/component_author_subjects.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['subject', *means])
        for idx in range(54):
            writer.writerow([idx + 1, *[float(arr[idx]) for arr in means.values()]])
    rng = np.random.default_rng(20260926)
    report = {
        'status': 'complete', 'analysis_role': 'post-confirmatory',
        'subjects': 54, 'seeds': 3,
        'accuracy_subject_macro': {name: float(arr.mean()) for name, arr in means.items()},
        'contrasts': {
            'rd_minus_spectral_only': paired(means['npfeegnet_rd'], means['spectral_only'], rng),
            'rd_minus_no_spectral_prior': paired(means['npfeegnet_rd'], means['no_spectral_prior'], rng),
            'rd_minus_fbcnet_author_training': paired(means['npfeegnet_rd'], means['fbcnet_author_training'], rng),
        },
        'source_summaries': source_paths,
    }
    (ROOT / 'results/component_author_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
