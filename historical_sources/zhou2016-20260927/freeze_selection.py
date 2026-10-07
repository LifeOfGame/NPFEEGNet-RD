"""Verify Session-0-only selection and freeze all epochs before Session-1 extraction."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml

ROOT=Path('/mnt/helongfei/NPFEEGNet-RD-zhou2016-20260927')
PROJECT=Path('/mnt/helongfei/NPFEEGNet-RD-paper-20260917')
RAW=Path('/mnt/helongfei/database/raw/zhou2016_nemar_v1.0.0')
MODELS={'rd':'NPFEEGNet','nodemean':'NPFEEGNet','eegnet_demean':'EEGNet'}
EXPECTED={(subject,seed) for subject in range(1,5) for seed in range(3)}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    artifacts=[]
    def record(path):
        path=Path(path)
        if not path.is_file():raise FileNotFoundError(path)
        artifacts.append({'path':str(path),'sha256':digest(path)})
    for path in (ROOT/'results/protocol_lock.json',ROOT/'results/session_a_addendum.json',ROOT/'preprocess_session_a.py',
                 ROOT/'preprocess_session_b.py',ROOT/'build_selection_configs.py',ROOT/'freeze_selection.py',
                 ROOT/'analyze_final.py',ROOT/'run_pipeline.sh',ROOT/'data/session_a_report.json',
                 PROJECT/'model/NPF_EEGNet.py',PROJECT/'model/openbmi_baselines.py',
                 PROJECT/'repro/frozen_engines/openbmi_dd010/train_dcsa_2a.py',
                 PROJECT/'train_dcsa_2a.py',PROJECT/'train_openbmi_baselines.py'):
        record(path)
    for subject in range(1,5):
        for suffix in ('data','label','group'):
            record(ROOT/'data'/f'Z{subject:02d}T_{suffix}.npy')
    selection={}
    for name,network in MODELS.items():
        config_path=ROOT/'config'/f'zhou2016_{name}_selection.yaml'
        record(config_path)
        summaries=list((ROOT/'output'/f'zhou2016_{name}_selection'/network).glob('*/summary.csv'))
        if len(summaries)!=1:raise RuntimeError(f'{name}: summaries {summaries}')
        summary=summaries[0]
        with summary.open(newline='') as f:rows=list(csv.DictReader(f))
        keys=[(int(r['subject']),int(r['seed'])) for r in rows]
        if len(rows)!=12 or set(keys)!=EXPECTED or len(set(keys))!=12:
            raise RuntimeError(f'{name}: incomplete selection grid')
        epochs={}
        for row in rows:
            subject=int(row['subject']);seed=int(row['seed'])
            if row['selection_strategy']!='group_cv' or int(row['cv_num_folds'])!=2 or row['cv_group_partitions']!='0|1':
                raise RuntimeError(f'{name}: unexpected CV design {subject} {seed}')
            fold_path=summary.parent/f'sub{subject}'/f'seed{seed}'/'selection/fold_summary.csv'
            with fold_path.open(newline='') as f: folds=list(csv.DictReader(f))
            if len(folds)!=2 or [int(x['fold']) for x in folds]!=[0,1]:
                raise RuntimeError(f'{name}: missing fold')
            groups=np.load(ROOT/'data'/f'Z{subject:02d}T_group.npy')
            for fold in folds:
                val=int(fold['fold'])
                if int(fold['validation_samples'])!=int((groups==val).sum()) or int(fold['train_samples'])!=int((groups!=val).sum()):
                    raise RuntimeError(f'{name}: fold count mismatch')
            chosen=max(1,int(np.rint(np.median([int(x['best_val_epoch']) for x in folds]))))
            if chosen!=int(row['best_val_epoch']):raise RuntimeError(f'{name}: chosen epoch mismatch')
            epochs.setdefault(str(subject),{})[str(seed)]=chosen
            record(fold_path)
        record(summary)
        epoch_path=ROOT/'results'/f'zhou2016_{name}_selected_epochs.json'
        epoch_path.write_text(json.dumps({'source_summary':str(summary),'selected_epochs':epochs},indent=2)+'\n')
        record(epoch_path)
        final=yaml.safe_load(config_path.read_text())
        final.update(out_folder=str(ROOT/'output'/f'zhou2016_{name}_final'),selected_epochs_file=str(epoch_path),
                     fixed_epoch_training=True,two_stage_training=False,selection_only=False,export_aux=True,
                     analysis_status='zhou2016_external_session1_final')
        if name!='eegnet_demean': final['protocol_lock_file']=str(ROOT/'results/final_unlock_lock.json')
        final_path=ROOT/'config'/f'zhou2016_{name}_final.yaml'
        final_path.write_text(yaml.safe_dump(final,sort_keys=False))
        record(final_path)
        selection[name]={'epochs':epochs,'summary':str(summary)}
    # The frozen training engine expects this literal status for any held-out
    # evaluation session; here its dataset-specific target is ses-1.
    lock={'status':'ready_for_session2','dataset':'Zhou2016','target_session':'ses-1',
          'source_archive_sha256':digest(RAW/'v1.0.0.zip'),
          'selection':selection,'artifacts':artifacts}
    (ROOT/'results/final_unlock_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    print(f'FROZEN {len(artifacts)} artifacts, all 36 model/subject/seed epochs',flush=True)

if __name__=='__main__':main()
