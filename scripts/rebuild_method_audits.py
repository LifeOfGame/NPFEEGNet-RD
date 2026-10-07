# Portable adaptation of the preserved methods-audit source; statistical formulas and RNG retained.
"""Descriptive post-hoc collapse sensitivity and exact soft-response audit."""
import csv
import hashlib
import json
import argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True, help='New audit output directory')
parser.add_argument('--plots', action='store_true', help='Also render Fig. S3; requires matplotlib')
args = parser.parse_args()
HERE = args.output.resolve()
HERE.mkdir(parents=True, exist_ok=False)
SOURCE = ROOT / 'evidence/bp830/openbmi_final_3seed_rows.csv'
rows = list(csv.DictReader(SOURCE.open(encoding='utf-8')))
keyed = {(r['variant'], int(r['subject']), int(r['seed'])): r for r in rows}
assert len(keyed) == len(rows) == 324
out = []
excluded = []
for subject in range(1, 55):
    all_d, kept_d = [], []
    for seed in range(3):
        rd, nd = [keyed[v, subject, seed] for v in ('raw_demean', 'nodemean')]
        delta = (float(rd['accuracy']) - float(nd['accuracy'])) * 100
        collapse = [min(float(r['class0_recall']), float(r['class1_recall'])) < .10 for r in (rd, nd)]
        all_d.append(delta)
        if any(collapse):
            excluded.append(dict(subject=subject, seed=seed, rd_collapse=collapse[0], nodemean_collapse=collapse[1]))
        else:
            kept_d.append(delta)
    out.append(dict(subject=subject, all_seeds_gain_pp=float(np.mean(all_d)), retained_seed_pairs=len(kept_d), noncollapse_gain_pp=float(np.mean(kept_d)) if kept_d else None))

def stats(values):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260928)
    means = x[rng.integers(0, len(x), (100000, len(x)))].mean(axis=1)
    return dict(n_subjects=len(x), mean_pp=float(x.mean()), median_pp=float(np.median(x)), ci95_pp=np.percentile(means,[2.5,97.5]).tolist(), wins=int((x>1e-9).sum()),ties=int((abs(x)<=1e-9).sum()),losses=int((x < -1e-9).sum()))

report = dict(analysis_role='post-hoc descriptive; collapse-based exclusion conditions on target outcomes and does not replace primary analysis',source=SOURCE.relative_to(ROOT).as_posix(),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),threshold='minimum class recall < 0.10 in either paired run',all_subjects=stats([r['all_seeds_gain_pp'] for r in out]),paired_noncollapse=stats([r['noncollapse_gain_pp'] for r in out if r['noncollapse_gain_pp'] is not None]),excluded_pairs=excluded,retained_pairs=sum(r['retained_seed_pairs'] for r in out))
(HERE/'noncollapse_report.json').write_text(json.dumps(report,indent=2)+'\n')
with (HERE/'noncollapse_subjects.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

bands=np.array([[4,8],[8,13],[13,20],[20,30],[30,40]],dtype=float)
f=np.fft.rfftfreq(1000,d=1/250)
sigmoid=lambda x: 1/(1+np.exp(-x))
h=sigmoid(f[None,:]-bands[:,0,None])*sigmoid(bands[:,1,None]-f[None,:])
norm=np.sqrt(.25*np.sum(h*h,axis=1)); hn=h/norm[:,None]
samples=[]
for i,(lo,hi) in enumerate(bands):
    for label,freq in [('DC',0),('1 Hz',1),('lower',lo),('center',(lo+hi)/2),('upper',hi)]:
        response=float(sigmoid(freq-lo)*sigmoid(hi-freq))
        center=float(sigmoid((hi-lo)/2)**2)
        samples.append(dict(band=f'{lo:g}-{hi:g}',location=label,frequency_hz=freq,response=response,normalized_response=response/norm[i],relative_to_center=response/center))
with (HERE/'soft_response_samples.csv').open('w',newline='') as stream:
    w=csv.DictWriter(stream,fieldnames=list(samples[0])); w.writeheader(); w.writerows(samples)
np.savetxt(HERE/'soft_response_grid.csv',np.column_stack([f,hn.T]),delimiter=',',header='frequency_hz,4_8,8_13,13_20,20_30,30_40',comments='')
print(json.dumps(report,indent=2))
if args.plots:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.3),gridspec_kw={'width_ratios':[1.5,1]})
    for i,(lo,hi) in enumerate(bands):
        for ax in axes: ax.plot(f,hn[i],label=f'{lo:g}–{hi:g} Hz',lw=1.8)
    axes[0].set(xlim=(0,50),ylim=(0,None),xlabel='Frequency (Hz)',ylabel='Normalized amplitude response',title='a  Five nominal soft bands')
    axes[1].set(xlim=(0,5),ylim=(1e-8,1),yscale='log',xlabel='Frequency (Hz)',title='b  Low-frequency tails')
    for ax in axes: ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=8,ncol=2)
    fig.tight_layout()
    target=HERE/'FigS3_soft_frequency_responses.png'
    fig.savefig(target,dpi=300,bbox_inches='tight'); fig.savefig(HERE/'soft_frequency_responses.svg',bbox_inches='tight')
    print(json.dumps(report,indent=2)); print('figure',target)
