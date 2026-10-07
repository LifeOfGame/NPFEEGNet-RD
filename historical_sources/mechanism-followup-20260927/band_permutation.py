from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path('/mnt/helongfei/NPFEEGNet-RD-unified-repro-20260918')
sys.path.insert(0, str(ROOT))
from model.NPF_EEGNet import NPFEEGNet

RUN_ROOT = ROOT / 'output/openbmi_final_npf_raw_demean_3seed/NPFEEGNet'
OUT = Path('/mnt/helongfei/NPFEEGNet-RD-mechanism-followup-20260927/results')
BANDS = ['4-8', '8-13', '13-20', '20-30', '30-40']
REPEATS = 20


def nll(logits, labels):
    return float(torch.nn.functional.cross_entropy(logits, labels).item())


def ci(values, seed):
    rng = np.random.default_rng(seed)
    x = np.asarray(values, float)
    means = x[rng.integers(0, len(x), (100000, len(x)))].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    runs = sorted(RUN_ROOT.glob('*/sub*/seed*'))
    assert len(runs) == 162, len(runs)
    rows = []
    for index, run in enumerate(runs, 1):
        sub = int(run.parent.name[3:])
        seed = int(run.name[4:])
        with (run / 'config.yaml').open(encoding='utf8') as f:
            config = yaml.safe_load(f)
        model = NPFEEGNet(**config['network_args']).eval()
        model.load_state_dict(torch.load(run / 'model.pth', map_location='cpu', weights_only=True))
        with np.load(run / 'test_auxiliary.npz') as z:
            bands = torch.from_numpy(z['band_features']).float()
            raw = torch.from_numpy(z['raw_logits']).float()
            labels = torch.from_numpy(z['labels']).long()
            saved = torch.from_numpy(z['probabilities']).float()
        assert bands.shape == (200, 5, 64)
        with torch.no_grad():
            strength = model.residual_max_strength * torch.sigmoid(model.residual_strength_logit)
            full = raw + strength * model.spectral_classifier(bands.mean(dim=1))
            assert (full.softmax(1) - saved).abs().max().item() < 1e-5
            base_nll = nll(full, labels)
            base_acc = float((full.argmax(1) == labels).float().mean())
            for band in range(5):
                harms = []
                accuracy_harms = []
                for repeat in range(REPEATS):
                    rng = np.random.default_rng(20260927 + sub * 10000 + seed * 1000 + band * 100 + repeat)
                    order = torch.from_numpy(rng.permutation(200).copy()).long()
                    permuted_mean = bands.mean(1) + (bands[order, band] - bands[:, band]) / 5
                    logits = raw + strength * model.spectral_classifier(permuted_mean)
                    harms.append(nll(logits, labels) - base_nll)
                    accuracy_harms.append(base_acc - float((logits.argmax(1) == labels).float().mean()))
                rows.append(dict(subject=sub, seed=seed, band=BANDS[band], nll_harm=float(np.mean(harms)),
                                 accuracy_harm_pp=100 * float(np.mean(accuracy_harms))))
        if index % 27 == 0:
            print(f'{index}/162', flush=True)
    with (OUT / 'band_permutation_runs.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    report = {}
    for band in BANDS:
        subject_values = []
        for sub in range(1, 55):
            subject_values.append(np.mean([r['nll_harm'] for r in rows if r['band'] == band and r['subject'] == sub]))
        report[band] = {'nll_harm': float(np.mean(subject_values)), 'subject_bootstrap_95_ci': ci(subject_values, 20260927 + BANDS.index(band))}
    (OUT / 'band_permutation_report.json').write_text(json.dumps({'status':'complete','analysis_role':'post-confirmatory sensitivity',
        'subjects':54,'seeds':3,'permutations_per_band_per_run':REPEATS,'method':'shuffle one band embedding across 200 Session-2 trials within each subject/seed; preserve empirical band-embedding marginal, raw logits and trained weights',
        'nll_harm_by_band':report}, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
