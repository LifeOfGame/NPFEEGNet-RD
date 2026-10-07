"""Zhou2016 Session 0 only; target Session 1 stays inside unopened archive."""
import csv
import hashlib
import json
from pathlib import Path

import mne
import numpy as np

ROOT = Path('/mnt/helongfei/NPFEEGNet-RD-zhou2016-20260927')
RAW = Path('/mnt/helongfei/database/raw/zhou2016_nemar_v1.0.0')
SOURCE = RAW / 'source_a'
OUT = ROOT / 'data'
LABELS = {'feet':1, 'left_hand':2, 'right_hand':3}
CHANNELS = ['Fp1','Fp2','FC3','FCz','FC4','C3','Cz','C4','CP3','CPz','CP4','O1','Oz','O2']

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report={'dataset':'Zhou2016','session':'0 only','source_archive_sha256':digest(RAW/'v1.0.0.zip'),
            'window':'cue-relative [0,4) s','sampling_rate_hz':250,'channels':CHANNELS,'subjects':[]}
    for subject in range(1,5):
        trials=[];labels=[];groups=[];files=[]
        for run in (0,1):
            folder=SOURCE/f'sub-{subject}/ses-0/eeg'
            stem=f'sub-{subject}_ses-0_task-imagery_run-{run}'
            eeg=folder/f'{stem}_eeg.edf'; events=folder/f'{stem}_events.tsv'
            raw=mne.io.read_raw_edf(eeg,preload=False,verbose='ERROR')
            if raw.info['sfreq']!=250 or list(raw.ch_names)!=CHANNELS:
                raise ValueError(f'{eeg}: unexpected channels/sampling rate: {raw.ch_names}, {raw.info["sfreq"]}')
            with events.open(encoding='utf-8-sig',newline='') as f:
                event_rows=list(csv.DictReader(f,delimiter='\t'))
            if len(event_rows)<60 or len(event_rows)>90:
                raise ValueError(f'{events}: unexpected event count {len(event_rows)}')
            starts=np.array([int(r['sample']) for r in event_rows])
            if np.any(starts<0) or np.any(starts+1000>raw.n_times):
                raise ValueError(f'{events}: epoch out of bounds')
            if any(abs(float(r['onset'])*250-int(r['sample']))>1 for r in event_rows):
                raise ValueError(f'{events}: onset and sample disagree')
            if any(float(r['duration'])<4 for r in event_rows):
                raise ValueError(f'{events}: event shorter than window')
            y=np.array([LABELS[r['trial_type']] for r in event_rows])
            if np.any(np.bincount(y,minlength=4)[1:]<15):
                raise ValueError(f'{events}: a class has fewer than 15 trials')
            for start in starts:
                # MNE returns volts; all study models take microvolts.
                trial=raw.get_data(start=int(start),stop=int(start)+1000)*1e6
                trials.append(trial.astype(np.float32))
            labels.extend(y.tolist());groups.extend([run]*len(y))
            files.append({'eeg':str(eeg),'eeg_sha256':digest(eeg),'events':str(events),'events_sha256':digest(events),
                          'first_event_samples':starts[:5].tolist(),'n_times':int(raw.n_times)})
        x=np.stack(trials);y=np.asarray(labels,dtype=np.int64);g=np.asarray(groups,dtype=np.int64)
        if x.shape[1:]!=(14,1000) or len(x)<120 or not np.isfinite(x).all() or np.any(np.bincount(y,minlength=4)[1:]<30):
            raise ValueError(f'Subject {subject} invalid output')
        stem=OUT/f'Z{subject:02d}T'
        np.save(f'{stem}_data.npy',x);np.save(f'{stem}_label.npy',y);np.save(f'{stem}_group.npy',g)
        report['subjects'].append({'subject':subject,'shape':list(x.shape),'class_counts':np.bincount(y,minlength=4)[1:].tolist(),
                                   'run_counts':np.bincount(g).tolist(),'min_uv':float(x.min()),'max_uv':float(x.max()),'files':files})
        print(f'Subject {subject}/4 Session 0 verified',flush=True)
    (OUT/'session_a_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ALL_SESSION_A_PROCESSED',flush=True)

if __name__=='__main__':main()
