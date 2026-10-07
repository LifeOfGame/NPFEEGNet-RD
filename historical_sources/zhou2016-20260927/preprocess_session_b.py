"""Unlock and process Zhou2016 Session 1 only after all Session-0 epochs are frozen."""
import csv
import hashlib
import json
import zipfile
from pathlib import Path
import mne
import numpy as np

ROOT=Path('/mnt/helongfei/NPFEEGNet-RD-zhou2016-20260927')
RAW=Path('/mnt/helongfei/database/raw/zhou2016_nemar_v1.0.0')
LOCK=ROOT/'results/final_unlock_lock.json'
LABELS={'feet':1,'left_hand':2,'right_hand':3}
CHANNELS=['Fp1','Fp2','FC3','FCz','FC4','C3','Cz','C4','CP3','CPz','CP4','O1','Oz','O2']

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    lock=json.loads(LOCK.read_text())
    if lock.get('status')!='ready_for_session2' or lock.get('target_session')!='ses-1' or set(lock['selection'])!={'rd','nodemean','eegnet_demean'}:
        raise RuntimeError('Complete frozen selection lock required')
    for item in lock['artifacts']:
        if digest(item['path'])!=item['sha256']:
            raise RuntimeError(f'Locked artifact changed: {item["path"]}')
    archive=RAW/'v1.0.0.zip'
    if digest(archive)!=lock['source_archive_sha256']:
        raise RuntimeError('Source archive changed')
    target=RAW/'source_b'
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names=[n for n in z.namelist() if '/ses-1/' in n and n.startswith(('sub-1/','sub-2/','sub-3/','sub-4/'))]
        if not names:raise RuntimeError('No target session files found')
        z.extractall(target,members=names)
    report={'dataset':'Zhou2016','session':'1','unlock_sha256':digest(LOCK),'subjects':[]}
    for subject in range(1,5):
        trials=[];labels=[];groups=[];files=[]
        for run in (0,1):
            folder=target/f'sub-{subject}/ses-1/eeg'
            stem=f'sub-{subject}_ses-1_task-imagery_run-{run}'
            eeg=folder/f'{stem}_eeg.edf';events=folder/f'{stem}_events.tsv'
            raw=mne.io.read_raw_edf(eeg,preload=False,verbose='ERROR')
            if raw.info['sfreq']!=250 or list(raw.ch_names)!=CHANNELS:
                raise ValueError(f'{eeg}: unexpected channels/rate')
            with events.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f,delimiter='\t'))
            if len(rows)<60 or len(rows)>90:raise ValueError(f'{events}: event count {len(rows)}')
            starts=np.array([int(r['sample']) for r in rows])
            if np.any(starts<0) or np.any(starts+1000>raw.n_times):raise ValueError('Target epoch outside signal')
            if any(abs(float(r['onset'])*250-int(r['sample']))>1 or float(r['duration'])<4 for r in rows):
                raise ValueError('Target onset/duration mismatch')
            y=np.array([LABELS[r['trial_type']] for r in rows],dtype=np.int64)
            if np.any(np.bincount(y,minlength=4)[1:]<15):raise ValueError('Target class count below fixed threshold')
            for start in starts:
                trials.append((raw.get_data(start=int(start),stop=int(start)+1000)*1e6).astype(np.float32))
            labels.extend(y.tolist());groups.extend([run]*len(y))
            files.append({'eeg':str(eeg),'eeg_sha256':digest(eeg),'events':str(events),'events_sha256':digest(events),
                          'events_count':len(rows)})
        x=np.stack(trials);y=np.asarray(labels,dtype=np.int64);g=np.asarray(groups,dtype=np.int64)
        if x.shape[1:]!=(14,1000) or len(x)<120 or not np.isfinite(x).all() or np.any(np.bincount(y,minlength=4)[1:]<30):
            raise ValueError(f'Subject {subject}: invalid target arrays')
        stem=ROOT/'data'/f'Z{subject:02d}E'
        np.save(f'{stem}_data.npy',x);np.save(f'{stem}_label.npy',y);np.save(f'{stem}_group.npy',g)
        report['subjects'].append({'subject':subject,'shape':list(x.shape),'class_counts':np.bincount(y,minlength=4)[1:].tolist(),
                                   'run_counts':np.bincount(g).tolist(),'files':files})
        print(f'Subject {subject}/4 Session 1 processed after lock',flush=True)
    (ROOT/'data/session_b_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ALL_SESSION_B_PROCESSED_AFTER_LOCK',flush=True)

if __name__=='__main__':main()
