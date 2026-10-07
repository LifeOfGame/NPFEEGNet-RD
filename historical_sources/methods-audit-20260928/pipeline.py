"""Six frozen arms, complete S1 selection before any new S2 evaluation."""
import concurrent.futures
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/mnt/helongfei/NPFEEGNet-RD-paper-20260917')
DATA=Path('/mnt/helongfei/database/processed/openbmi')
GPU=['GPU-be0de490-67b9-b5c9-7496-b03349297857','GPU-c0136925-949a-5398-6e8b-1f5843db6a11']
ARMS=[f'{b}_{d}' for d in ('rd','nodemean') for b in ('current','fft_reflect','raw_reflect')]
EXPECTED={(s,z) for s in range(1,55) for z in range(3)}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def write(p,x): p.write_text(json.dumps(x,indent=2)+'\n')
def status(stage,**kw): write(ROOT/'status.json',dict(stage=stage,updated_at=now(),**kw))
def prepare():
    for d in ('config','logs','results'): (ROOT/d).mkdir(exist_ok=True)
    lock=ROOT/'design_lock.json'
    if lock.exists(): raise FileExistsError(lock)
    base=yaml.safe_load((SOURCE/'config/openbmi_selection_npf_raw_demean_3seed.yaml').read_text())
    files=[ROOT/p for p in ('boundary_models.py','run_boundary.py','pipeline.py')]
    files += list((SOURCE/'model').glob('*.py'))+[SOURCE/'utils.py',SOURCE/'repro/frozen_engines/openbmi_dd010/train_dcsa_2a.py']
    for sub in range(1,55):
        files += [DATA/f'O{sub:02d}T_{s}.npy' for s in ('data','label','group')]
    for arm in ARMS:
        c=yaml.safe_load(yaml.safe_dump(base)); boundary,dm=arm.rsplit('_',1)
        c.update(data_path=str(DATA),nGPU=0,boundary_variant=boundary,design_lock_file=str(lock),out_folder=str(ROOT/'output'/arm/'selection'),analysis_role='post-confirmatory boundary sensitivity; earlier S2 outcomes known')
        c['network_args']['input_standardization']='demean' if dm=='rd' else 'none'
        p=ROOT/'config'/f'{arm}_selection.yaml'; p.write_text(yaml.safe_dump(c,sort_keys=False)); files.append(p)
    write(lock,dict(created_at=now(),analysis_role='post-confirmatory; not new prospective confirmation',arms=ARMS,subjects=list(range(1,55)),seeds=[0,1,2],fft_padding_samples_each_side=250,raw_padding='first raw temporal convolution only, reflect 31 left / 32 right',selection='two source phases, max 300, patience 40; numpy.rint median ties to even',final='fresh S1 training; one S2 evaluation after all six epoch maps frozen',planned_contrasts=['fft_reflect_rd-current_rd','fft_reflect_nodemean-current_nodemean','raw_reflect_rd-current_rd','raw_reflect_nodemean-current_nodemean','raw-padding x demeaning difference-in-differences'],statistics='seeds averaged within subject; subject paired mean/median, bootstrap CI; exploratory family if tests subsequently reported',sha256={str(p):sha(p) for p in files}))
    status('prepared')
def lane(dm,gpu,phase):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for boundary in ('current','fft_reflect','raw_reflect'):
        arm=f'{boundary}_{dm}'
        with (ROOT/'logs'/f'{arm}_{phase}.log').open('w') as f:
            subprocess.run([sys.executable,'-u',str(ROOT/'run_boundary.py'),'--config',str(ROOT/'config'/f'{arm}_{phase}.yaml')],cwd=SOURCE,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def phase_run(phase):
    status(phase,started_at=now())
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        fs=[ex.submit(lane,dm,g,phase) for dm,g in zip(('rd','nodemean'),GPU)]
        for f in fs: f.result()
def freeze():
    files=[ROOT/'design_lock.json']
    for arm in ARMS:
        c=yaml.safe_load((ROOT/'config'/f'{arm}_selection.yaml').read_text())
        paths=list(Path(c['out_folder']).glob('NPFEEGNet/*/summary.csv')); assert len(paths)==1,paths
        rows=list(csv.DictReader(paths[0].open())); assert len(rows)==162
        assert {(int(r['subject']),int(r['seed'])) for r in rows}==EXPECTED
        assert not any(k.startswith('test_') for k in rows[0])
        for r in rows:
            folds=[int(x) for x in r['cv_fold_epochs'].split(';')]
            assert len(folds)==2 and max(1,int(np.rint(np.median(folds))))==int(r['best_val_epoch'])
        ep=ROOT/'results'/f'{arm}_epochs.json'
        write(ep,dict(selected_epochs={str(s):{str(z):int(next(r['best_val_epoch'] for r in rows if int(r['subject'])==s and int(r['seed'])==z)) for z in range(3)} for s in range(1,55)},official_evaluation_used_for_selection=False))
        c.update(out_folder=str(ROOT/'output'/arm/'final'),selected_epochs_file=str(ep),protocol_lock_file=str(ROOT/'session2_lock.json'),fixed_epoch_training=True,two_stage_training=False,selection_only=False,early_stopping_patience=None,export_aux=True)
        p=ROOT/'config'/f'{arm}_final.yaml';p.write_text(yaml.safe_dump(c,sort_keys=False)); files.extend([ep,p,paths[0]])
    write(ROOT/'session2_lock.json',dict(status='ready_for_session2',created_at=now(),sha256={str(p):sha(p) for p in files}))
def main():
    if '--prepare' in sys.argv: prepare(); return
    try:
        phase_run('selection'); freeze(); phase_run('final'); analyze(); status('complete')
    except Exception as e:
        status('failed',error=repr(e)); raise
def analyze():
    arrays={}; sources={}
    for arm in ARMS:
        paths=list((ROOT/'output'/arm/'final').glob('NPFEEGNet/*/summary.csv')); assert len(paths)==1
        rows=list(csv.DictReader(paths[0].open())); assert len(rows)==162
        grid={(int(r['subject']),int(r['seed'])):float(r['test_accuracy'])*100 for r in rows}; assert set(grid)==EXPECTED
        arrays[arm]=np.array([np.mean([grid[s,z] for z in range(3)]) for s in range(1,55)])
        sources[arm]={'path':str(paths[0]),'sha256':sha(paths[0])}
    rng=np.random.default_rng(20260928)
    def paired(d):
        ci=np.percentile(d[rng.integers(0,54,(100000,54))].mean(axis=1),[2.5,97.5])
        return dict(mean_pp=float(d.mean()),median_pp=float(np.median(d)),ci95_pp=ci.tolist(),wtl=[int((d>1e-9).sum()),int((abs(d)<=1e-9).sum()),int((d < -1e-9).sum())])
    contrasts={f'{b}_{d}-current_{d}':paired(arrays[f'{b}_{d}']-arrays[f'current_{d}']) for b in ('fft_reflect','raw_reflect') for d in ('rd','nodemean')}
    for b in ('current','fft_reflect','raw_reflect'):
        contrasts[f'{b}_demeaning_gain']=paired(arrays[f'{b}_rd']-arrays[f'{b}_nodemean'])
    contrasts['raw_padding_x_demeaning']=paired((arrays['raw_reflect_rd']-arrays['raw_reflect_nodemean'])-(arrays['current_rd']-arrays['current_nodemean']))
    write(ROOT/'results/final_report.json',dict(analysis_role='post-confirmatory exploratory descriptive sensitivity; no multiplicity-controlled significance claims',completed_at=now(),mean_accuracy_percent={a:float(v.mean()) for a,v in arrays.items()},contrasts=contrasts,sources=sources))
    with (ROOT/'results/subject_accuracies.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['subject']+ARMS)
        for i in range(54): w.writerow([i+1]+[arrays[a][i] for a in ARMS])
if __name__=='__main__': main()
