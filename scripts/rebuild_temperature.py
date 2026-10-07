"""Retrospective reconstruction of source-OOF-fitted temperature scaling.

This reproduces archived artifacts; it does not create a new prospective lock.
The objective only receives S1 probabilities and labels. S2 is loaded after fit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def scale(p,temperature):
    logits=np.log(np.clip(p,1e-30,1))/temperature
    logits-=logits.max(axis=1,keepdims=True)
    p=np.exp(logits)
    return p/p.sum(axis=1,keepdims=True)


def metrics(labels,p):
    confidence=p.max(axis=1)
    correct=p.argmax(axis=1)==labels
    ece=0.
    # Preserve the historical temperature-analysis boundary expression.
    for low in np.linspace(0,.9,10):
        mask=(confidence>low)&(confidence<=low+.1)
        if mask.any():ece+=float(mask.mean())*abs(float(correct[mask].mean()-confidence[mask].mean()))
    return dict(accuracy=float(correct.mean()),nll=float(-np.log(np.clip(p[np.arange(len(labels)),labels],1e-12,1)).mean()),ece=ece)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New output directory')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'evidence/temperature/manifest.json').read_text())
    source=ROOT/'evidence/temperature/s1_oof.npz'
    lock=ROOT/'evidence/historical/temperature-20260926/results/temperature_lock.json'
    if sha(source)!=manifest['sha256'] or sha(lock)!=manifest['temperature_lock_sha256']:
        raise ValueError('Source or historical lock hash changed')
    with np.load(source,allow_pickle=False) as z:
        y=z['labels'].astype(np.int64)
        p=z['probabilities'].astype(np.float64)
        keys=np.column_stack([z['subject'],z['seed'],z['fold']])
    unique,counts=np.unique(keys,axis=0,return_counts=True)
    expected={(s,z,f) for s in range(1,55) for z in range(3) for f in range(2)}
    if set(map(tuple,unique))!=expected or not (counts==100).all():
        raise ValueError('Incomplete source OOF grid')
    if y.shape!=(32400,) or p.shape!=(32400,2) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=2e-6):
        raise ValueError('Invalid source probabilities')
    result=minimize_scalar(lambda logt:metrics(y,scale(p,np.exp(logt)))['nll'],
                           bounds=(np.log(.5),np.log(5)),method='bounded',options={'xatol':1e-8})
    if not result.success:raise RuntimeError(result.message)
    temperature=float(np.exp(result.x))
    old=json.loads(lock.read_text())
    if not np.isclose(temperature,old['temperature'],rtol=0,atol=1e-6):
        raise ValueError('Fitted temperature differs from historical lock')
    args.output.mkdir(parents=True,exist_ok=False)
    fitted=dict(role='retrospective artifact reconstruction',temperature=temperature,
                historical_temperature=old['temperature'],session2_used_for_fitting=False,
                source_sha256=sha(source),s1_before=metrics(y,p),s1_after=metrics(y,scale(p,temperature)))
    (args.output/'source_fit.json').write_text(json.dumps(fitted,indent=2),encoding='utf-8')
    # Target file is deliberately loaded only after fitting and saving the result.
    group='paper-20260917__openbmi_final_npf_raw_demean_3seed'
    entry=json.loads((ROOT/'evidence/predictions/manifest.json').read_text())['groups'][group]
    target=ROOT/'evidence/predictions'/entry['file']
    if sha(target)!=entry['sha256']:raise ValueError('Target artifact hash changed')
    with np.load(target,allow_pickle=False) as z:
        y=z['labels'].astype(np.int64);p=z['probabilities'].astype(np.float64)
    adjusted=scale(p,temperature)
    if not np.array_equal(p.argmax(1),adjusted.argmax(1)):
        raise ValueError('Positive temperature unexpectedly changed class predictions')
    report=dict(**fitted,s2_before=metrics(y,p),s2_after=metrics(y,adjusted),target_sha256=sha(target))
    (args.output/'temperature_reconstruction.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'temperature':temperature,'s2_after':report['s2_after']}))


if __name__=='__main__':main()
