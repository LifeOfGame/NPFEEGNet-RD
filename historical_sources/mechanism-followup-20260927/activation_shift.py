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

DATA = Path('/mnt/helongfei/database/processed/openbmi')
OUTPUT = ROOT / 'output'
OUT = Path('/mnt/helongfei/NPFEEGNet-RD-mechanism-followup-20260927/results')
VARIANTS = {'RD':'openbmi_final_npf_raw_demean_3seed', 'NoDemean':'openbmi_final_npf_nodemean_3seed'}
LAYERS = ['raw_input', 'temporal_conv', 'temporal_bn', 'spatial_conv', 'spatial_bn']


def session_vectors(model, array, device):
    captured = {name: [] for name in LAYERS}
    with torch.no_grad():
        for start in range(0, len(array), 16):
            x = torch.as_tensor(np.asarray(array[start:start+16]).copy(), device=device, dtype=torch.float32)
            if x.ndim == 3:
                x = x[:, None]
            assert x.ndim == 4 and x.shape[1] == 1, x.shape
            x = model._standardize(x)
            stages = [x]
            for layer in list(model.raw_features)[:4]:
                x = layer(x)
                stages.append(x)
            for name, tensor in zip(LAYERS, stages):
                captured[name].append(tensor.mean(dim=-1).flatten(start_dim=1).cpu().numpy())
    return {name: np.concatenate(chunks) for name, chunks in captured.items()}


def shift(a, b):
    assert a.shape == b.shape and a.shape[0] == 200, (a.shape,b.shape)
    mu_a, mu_b = a.mean(0), b.mean(0)
    variance = (a.var(0, ddof=1) + b.var(0, ddof=1)) / 2
    standardized = (mu_a - mu_b) / np.sqrt(variance + 1e-8)
    return float(np.sqrt(np.mean(standardized ** 2)))


def ci(x, seed):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (100000, len(x)))].mean(1)
    return np.quantile(means, [.025, .975]).tolist()


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rows = []
    for variant, directory in VARIANTS.items():
        runs = sorted((OUTPUT / directory / 'NPFEEGNet').glob('*/sub*/seed*'))
        assert len(runs) == 162, (variant,len(runs))
        for count, run in enumerate(runs, 1):
            subject = int(run.parent.name[3:]); seed = int(run.name[4:])
            with (run / 'config.yaml').open(encoding='utf8') as f:
                config = yaml.safe_load(f)
            model = NPFEEGNet(**config['network_args']).to(device).eval()
            model.load_state_dict(torch.load(run / 'model.pth', map_location=device, weights_only=True))
            s1 = np.load(DATA / f'O{subject:02d}T_data.npy', mmap_mode='r')
            s2 = np.load(DATA / f'O{subject:02d}E_data.npy', mmap_mode='r')
            v1 = session_vectors(model, s1, device)
            v2 = session_vectors(model, s2, device)
            for layer in LAYERS:
                rows.append({'variant':variant,'subject':subject,'seed':seed,'layer':layer,
                             'session_centroid_standardized_rms':shift(v1[layer],v2[layer])})
            del model
            if count % 27 == 0:
                print(f'{variant} {count}/162', flush=True)
    with (OUT / 'activation_shift_runs.csv').open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    report={}
    for layer in LAYERS:
        paired=[]; rd=[]; no=[]
        for sub in range(1,55):
            r=np.mean([x['session_centroid_standardized_rms'] for x in rows if x['layer']==layer and x['subject']==sub and x['variant']=='RD'])
            n=np.mean([x['session_centroid_standardized_rms'] for x in rows if x['layer']==layer and x['subject']==sub and x['variant']=='NoDemean'])
            paired.append(r-n);rd.append(r);no.append(n)
        report[layer]={'rd_mean':float(np.mean(rd)),'nodemean_mean':float(np.mean(no)),
                       'rd_minus_nodemean':float(np.mean(paired)),'subject_bootstrap_95_ci':ci(paired,20260927+LAYERS.index(layer))}
    (OUT / 'activation_shift_report.json').write_text(json.dumps({'status':'complete','analysis_role':'post-confirmatory mechanism diagnostic',
        'subjects':54,'seeds':3,'metric':'RMS standardized Session-1 versus Session-2 centroid distance on trial-level time-averaged feature maps; each feature standardized by pooled within-session trial SD',
        'note':'Separate trained RD and NoDemean checkpoints are compared; distance is descriptive and does not identify the causal mediator of accuracy gain.',
        'layers':report},indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
