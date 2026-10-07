"""Check completed bandpass runs against separately frozen S1 epoch maps."""

import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPECTED = {(subject, seed) for subject in range(1, 55) for seed in range(3)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    report = {}
    for variant in ('bp830_rd', 'bp830_nodemean'):
        summary = data_dir / f'{variant}_final_summary.csv'
        selection = data_dir / f'{variant}_selection_summary.csv'
        frozen = data_dir / f'{variant}_selected_epochs.json'
        mapping = json.loads(frozen.read_text(encoding='utf-8'))
        with summary.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        with selection.open(newline='', encoding='utf-8') as stream:
            selection_rows = list(csv.DictReader(stream))
        selection_keys = [(int(row['subject']), int(row['seed'])) for row in selection_rows]
        if len(selection_rows) != 162 or len(set(selection_keys)) != 162:
            raise RuntimeError(f'Incomplete or duplicate selection grid: {variant}')
        selected = {(int(row['subject']), int(row['seed'])): int(row['best_val_epoch']) for row in selection_rows}
        keys = [(int(row['subject']), int(row['seed'])) for row in rows]
        if len(rows) != 162 or len(set(keys)) != 162 or set(keys) != EXPECTED or set(selected) != EXPECTED:
            raise RuntimeError(f'Incomplete or duplicate grid: {variant}')
        for row in rows:
            key = int(row['subject']), int(row['seed'])
            epoch = int(row['retrain_epochs'])
            frozen_epoch = int(mapping['selected_epochs'][str(key[0])][str(key[1])])
            if epoch != selected[key] or epoch != frozen_epoch:
                raise RuntimeError(f'Selected/retrain epoch mismatch: {variant} {key}')
            if row['selection_strategy'] != 'preselected_group_cv':
                raise RuntimeError(f'Wrong final selection strategy: {variant} {key}')
            if not row['selected_epoch_source'].endswith(f'{variant}_selected_epochs.json'):
                raise RuntimeError(f'Wrong selected epoch source: {variant} {key}')
            value = float(row['test_accuracy'])
            if not 0 <= value <= 1:
                raise RuntimeError(f'Invalid test accuracy: {variant} {key}')
        if mapping['selection_summary_sha256'] != sha(selection):
            raise RuntimeError(f'Selection summary hash mismatch: {variant}')
        report[variant] = {
            'subjects': 54, 'seeds': 3, 'rows_verified': len(rows),
            'selection_summary_sha256': sha(selection),
            'frozen_map_sha256': sha(frozen),
            'final_summary_sha256': sha(summary),
        }
    target = data_dir / 'bp830_verification.json'
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
