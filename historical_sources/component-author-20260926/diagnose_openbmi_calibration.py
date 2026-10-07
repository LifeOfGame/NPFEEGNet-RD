"""Descriptive S2 probability audit; never selects a calibration parameter."""

import json
from pathlib import Path

import numpy as np


RD = Path('/mnt/helongfei/NPFEEGNet-RD-unified-repro-20260918/output/openbmi_final_npf_raw_demean_3seed/NPFEEGNet/2026-08-31--14-52')
FBC = Path('/mnt/helongfei/NPFEEGNet-RD-openbmi-fbcnet-audit-20260922/output')
OUT = Path('/mnt/helongfei/NPFEEGNet-RD-component-author-20260926/results/openbmi_rd_fbcnet_calibration_diagnostic.json')


def load(paths):
    labels, probs = [], []
    if len(paths) != 162:
        raise RuntimeError(f'Expected 162 prediction files, got {len(paths)}')
    for path in paths:
        with np.load(path) as z:
            labels.append(z['labels'].astype(int))
            probs.append(z['probabilities'].astype(float))
    labels = np.concatenate(labels)
    probs = np.concatenate(probs)
    if labels.shape != (32400,) or probs.shape != (32400, 2):
        raise RuntimeError('Unexpected prediction shape')
    return labels, probs


def summarize(y, p):
    predicted = p.argmax(axis=1)
    correct = predicted == y
    conf = p.max(axis=1)
    ptrue = p[np.arange(len(y)), y]
    # Match the manuscript's pooled NLL implementation exactly.
    nll = -np.log(np.clip(ptrue, 1e-12, 1))
    bins = []
    for k in range(10):
        lo, hi = k / 10, (k + 1) / 10
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            bins.append({'range': [lo, hi], 'n': int(mask.sum()),
                         'accuracy': float(correct[mask].mean()),
                         'mean_confidence': float(conf[mask].mean()),
                         'confidence_minus_accuracy': float(conf[mask].mean() - correct[mask].mean())})
    worst = np.argsort(nll)[-int(0.01 * len(nll)):]
    return {'accuracy': float(correct.mean()), 'nll': float(nll.mean()),
            'mean_confidence': float(conf.mean()),
            'overall_overconfidence': float(conf.mean() - correct.mean()),
            'wrong_prediction_mean_confidence': float(conf[~correct].mean()),
            'correct_prediction_mean_confidence': float(conf[correct].mean()),
            'wrong_prediction_mean_nll': float(nll[~correct].mean()),
            'correct_prediction_mean_nll': float(nll[correct].mean()),
            'true_class_probability_below_1e-12': int(np.sum(ptrue < 1e-12)),
            'nll_from_worst_1pct_fraction': float(nll[worst].sum() / nll.sum()),
            'confidence_bins': bins}


def main():
    rd_paths = sorted(RD.glob('sub*/seed*/test_auxiliary.npz'))
    fbc_paths = sorted(FBC.glob('final_gpu*/FBCNet/*/sub*/seed*/test_auxiliary.npz'))
    report = {'role': 'descriptive posthoc S2 audit; no parameter fitted',
              'rd': summarize(*load(rd_paths)), 'fbcnet': summarize(*load(fbc_paths))}
    OUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
