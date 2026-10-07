"""Audit and summarize frozen Zhou2016 3-class Session-0-to-Session-1 runs."""
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

ROOT=Path('/mnt/helongfei/NPFEEGNet-RD-zhou2016-20260927')
MODELS={'rd':'NPFEEGNet','nodemean':'NPFEEGNet','eegnet_demean':'EEGNet'}
EXPECTED={(s,seed) for s in range(1,5) for seed in range(3)}

def contrast(a,b,seed):
    delta=100*(subjects[a][:,0]-subjects[b][:,0])
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,4,size=(100000,4))
    interval=np.quantile(delta[indices].mean(1),[.025,.975]).tolist()
    observed=abs(delta.mean())
    null=[abs(np.mean(delta*np.asarray(signs))) for signs in itertools.product((-1,1),repeat=4)]
    return {'mean_pp':float(delta.mean()),'subject_bootstrap_95_ci_pp':interval,
            'exact_two_sided_sign_flip_p':float(np.mean(np.asarray(null)>=observed-1e-12)),
            'wins_ties_losses':[int((delta>0).sum()),int((delta==0).sum()),int((delta<0).sum())]}

subjects={}; paths={};confusions={}
for name,network in MODELS.items():
    summaries=list((ROOT/'output'/f'zhou2016_{name}_final'/network).glob('*/summary.csv'))
    if len(summaries)!=1:raise RuntimeError(f'{name}: expected one final summary, got {summaries}')
    summary=summaries[0]
    with summary.open(newline='') as f:rows=list(csv.DictReader(f))
    keys=[(int(r['subject']),int(r['seed'])) for r in rows]
    if len(rows)!=12 or set(keys)!=EXPECTED or len(set(keys))!=12:raise RuntimeError(f'{name}: incomplete grid')
    epochs=json.loads((ROOT/'results'/f'zhou2016_{name}_selected_epochs.json').read_text())['selected_epochs']
    subject_runs={s:[] for s in range(1,5)}
    confusion=np.zeros((3,3),dtype=int)
    for row in rows:
        subject=int(row['subject']);seed=int(row['seed'])
        if int(row['retrain_epochs'])!=epochs[str(subject)][str(seed)]:raise RuntimeError('Epoch mismatch')
        with np.load(summary.parent/f'sub{subject}'/f'seed{seed}'/'test_auxiliary.npz') as saved:
            y=saved['labels'];pred=saved['predictions']
        source=np.load(ROOT/'data'/f'Z{subject:02d}E_label.npy')-1
        if not np.array_equal(y,source) or len(y)!=len(source):raise RuntimeError('Target labels mismatch')
        acc=float(np.mean(y==pred))
        if abs(acc-float(row['test_accuracy']))>1e-8:raise RuntimeError('Summary mismatch')
        subject_runs[subject].append((acc,float(balanced_accuracy_score(y,pred))))
        confusion+=confusion_matrix(y,pred,labels=[0,1,2])
    subjects[name]=np.asarray([np.mean(subject_runs[s],axis=0) for s in range(1,5)])
    paths[name]=str(summary);confusions[name]=confusion.tolist()

report={'status':'complete','dataset':'Zhou2016','role':'later independently locked three-class cross-session audit',
        'subjects':4,'seeds':3,'training_session':'0','target_session':'1',
        'models':{name:{'subject_macro_accuracy_pct':float(100*values[:,0].mean()),
                        'subject_macro_balanced_accuracy_pct':float(100*values[:,1].mean()),
                        'pooled_confusion_three_seeds':confusions[name]}
                  for name,values in subjects.items()},
        'contrasts':{'RD_minus_NoDemean':contrast('rd','nodemean',20260927),
                     'RD_minus_EEGNet_Demean':contrast('rd','eegnet_demean',20260928)},
        'summary_paths':paths,
        'interpretation_limit':'Only four subjects; inference is subject-level and exact sign-flip p has coarse resolution. This later dataset was locked after original OpenBMI outcomes were known.'}
(ROOT/'results/final_report.json').write_text(json.dumps(report,indent=2)+'\n')
with (ROOT/'results/final_subjects.csv').open('w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['subject']+[f'{name}_{metric}' for name in MODELS for metric in ('accuracy','balanced_accuracy')])
    for s in range(1,5):w.writerow([s]+[float(v) for name in MODELS for v in subjects[name][s-1]])
print(json.dumps(report,indent=2))
