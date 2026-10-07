"""Analyze full 54-subject continuous-filtering sensitivity results."""

import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
EXPECTED = {(s, z) for s in range(1, 55) for z in range(3)}


def read(path, variant=None):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if variant is not None:
        rows = [row for row in rows if row['variant'] == variant]
    pairs = [(int(row['subject']), int(row['seed'])) for row in rows]
    if len(pairs) != 162 or len(set(pairs)) != 162 or set(pairs) != EXPECTED:
        raise RuntimeError(f'Incomplete subject/seed grid: {path}')
    key = 'accuracy' if variant is not None else 'test_accuracy'
    mapping = {(int(row['subject']), int(row['seed'])): float(row[key]) for row in rows}
    return np.asarray([np.mean([mapping[s, z] for z in range(3)]) for s in range(1, 55)])


def contrast(a, b, seed):
    difference = 100 * (a - b)
    # Each seed scores 200 trials; after averaging 3 seeds, 1 count = 1/6 pp.
    # Comparisons (including gain differences) lie on this same integer lattice.
    count_difference = np.rint(difference * 6).astype(np.int64)
    if not np.allclose(difference, count_difference / 6, atol=1e-10, rtol=0):
        raise ValueError('Accuracy differences are not on the expected 600-prediction lattice')
    rng = np.random.default_rng(seed)
    boot = difference[rng.integers(0, 54, size=(100000, 54))].mean(axis=1)
    return {
        'difference_pp': float(difference.mean()),
        'subject_bootstrap_95_ci_pp': np.quantile(boot, [0.025, 0.975]).tolist(),
        'wins_ties_losses': [int((count_difference > 0).sum()), int((count_difference == 0).sum()), int((count_difference < 0).sum())],
    }


def locate_data_dir():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, help='Directory containing run CSVs and frozen epoch maps')
    args = parser.parse_args()
    if args.data_dir is not None:
        chosen = args.data_dir.resolve()
    elif (HERE.parent / 'bp830' / 'bp830_rd_final_summary.csv').is_file():
        chosen = HERE.parent / 'bp830'
    else:
        chosen = HERE
    if not (chosen / 'bp830_rd_final_summary.csv').is_file():
        raise FileNotFoundError(f'No BP8-30 results in {chosen}; provide --data-dir PATH')
    return chosen

def main():
    data_dir = locate_data_dir()
    source = data_dir / 'openbmi_final_3seed_rows.csv'
    hp_rd = read(data_dir / 'bp830_rd_final_summary.csv')
    hp_no = read(data_dir / 'bp830_nodemean_final_summary.csv')
    original_rd = read(source, 'raw_demean')
    original_no = read(source, 'nodemean')
    result = {
        'analysis_status': 'post-confirmatory 8-30-Hz continuous-filter sensitivity',
        'subjects': 54, 'seeds_per_subject': 3,
        'tie_rule': 'Exact integer correct-count differences across 3 x 200 predictions; float lattice validation atol=1e-10 pp, rtol=0. Mean and bootstrap computations unchanged.',
        'subject_macro_accuracy_pct': {
            'BP830-RD': float(100 * hp_rd.mean()),
            'BP830-NoDemean': float(100 * hp_no.mean()),
            'Original-RD': float(100 * original_rd.mean()),
            'Original-NoDemean': float(100 * original_no.mean()),
        },
        'bp830_rd_minus_bp830_no': contrast(hp_rd, hp_no, 20260925),
        'original_rd_minus_original_no': contrast(original_rd, original_no, 20260928),
        'bp830_rd_gain_minus_original_rd_gain': contrast(hp_rd - hp_no, original_rd - original_no, 20260929),
        'bp830_rd_minus_original_rd': contrast(hp_rd, original_rd, 20260926),
        'bp830_no_minus_original_no': contrast(hp_no, original_no, 20260927),
        'original_rd_minus_bp830_no': contrast(original_rd, hp_no, 20260930),
        'interpretation_limit': 'Cutoff, continuous phase filtering, and per-variant S1 epoch selection differ from the earlier epoched 1-Hz control; all contrasts are post-confirmatory.',
    }
    (data_dir / 'bp830_report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
