"""Post-confirmatory fixed-duration sensitivity; preserve all primary artifacts."""
from pathlib import Path
import sys,json,hashlib,csv,os,subprocess,concurrent.futures
from datetime import datetime,timezone
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/mnt/helongfei/NPFEEGNet-RD-paper-20260917')
DATA=Path('/mnt/helongfei/database/processed/openbmi')
GPU=['GPU-be0de490-67b9-b5c9-7496-b03349297857','GPU-c0136925-949a-5398-6e8b-1f5843db6a11']
ARMS=['rd_at_rd','rd_at_no','no_at_rd','no_at_no']
GRID={(s,z) for s in range(1,55) for z in range(3)}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def write(p,x): Path(p).write_text(json.dumps(x,indent=2)+'\n')
def status(stage,**kw): write(ROOT/'status.json',dict(stage=stage,updated_at=now(),**kw))
def verify():
 for p,h in json.loads((ROOT/'design_lock.json').read_text())['sha256'].items():
  assert sha(p)==h,p
def prepare():
 assert not (ROOT/'design_lock.json').exists()
 for d in ['config','results','logs']: (ROOT/d).mkdir(exist_ok=True)
 files=[ROOT/'run.py',ROOT/'primary_rows.csv']
 files+=list((SOURCE/'model').glob('*.py'))+[SOURCE/'utils.py',SOURCE/'repro/frozen_engine_loader.py',SOURCE/'repro/frozen_engines/openbmi_dd010/train_dcsa_2a.py']
 for name in ['rd','no']:
  p=ROOT/f'{name}_epochs.json'; e=json.loads(p.read_text())['selected_epochs']
  assert {(int(s),int(z)) for s,v in e.items() for z in v}==GRID
  files.append(p)
 for arm in ARMS:
  variant,duration=arm.split('_at_')
  src=SOURCE/'config'/f'openbmi_final_npf_{"raw_demean" if variant=="rd" else "nodemean"}_3seed.yaml'
  c=yaml.safe_load(src.read_text());files.append(src)
  c.update(data_path=str(DATA),out_folder=str(ROOT/'output'/arm),selected_epochs_file=str(ROOT/f'{duration}_epochs.json'),protocol_lock_file=str(ROOT/'design_lock.json'),nGPU=0,analysis_role='post-confirmatory common-duration sensitivity; primary S2 results already known')
  p=ROOT/'config'/f'{arm}.yaml';p.write_text(yaml.safe_dump(c,sort_keys=False));files.append(p)
 for sub in range(1,55):
  files += [DATA/f'O{sub:02d}{role}_{suffix}.npy' for role in ['T','E'] for suffix in ['data','label','group']]
 write(ROOT/'design_lock.json',dict(created_at=now(),role='post-confirmatory sensitivity',subjects=54,seeds=[0,1,2],analysis_seed=20260928,bootstrap_resamples=100000,arms=ARMS,common_duration_conditions={'A':'both models at original RD-selected S1 epochs','B':'both models at original NoDemean-selected S1 epochs'},primary_results_untouched=True,contemporary_diagonals='rerun to avoid comparing crossed runs from Linux against historical Windows diagonals only',planned_analyses='subject seed-average paired gain, pointwise percentile CI, exact sign-flip, W/T/L; two sensitivity p-values with Holm adjustment, no new confirmatory endpoint',sha256={str(p):sha(p) for p in set(files)}))
 status('prepared')
def train(arm):
 verify();sys.path.insert(0,str(SOURCE))
 from repro.frozen_engine_loader import load_openbmi_engine
 engine=load_openbmi_engine('common_duration_frozen_engine')
 engine.main(yaml.safe_load((ROOT/'config'/f'{arm}.yaml').read_text()))
def lane(variant,gpu):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 for duration in ['rd','no']:
  arm=f'{variant}_at_{duration}'
  with (ROOT/'logs'/f'{arm}.log').open('w') as f:
   subprocess.run([sys.executable,'-u',str(ROOT/'run.py'),'--arm',arm],cwd=SOURCE,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def read_rows(path,variant=None):
 rows=list(csv.DictReader(Path(path).open()))
 if variant: rows=[r for r in rows if r['variant']==variant]
 assert len(rows)==162 and {(int(r['subject']),int(r['seed'])) for r in rows}==GRID
 key='accuracy' if variant else 'test_accuracy'
 return np.array([np.mean([float(next(r[key] for r in rows if int(r['subject'])==s and int(r['seed'])==z)) for z in range(3)])*100 for s in range(1,55)])
def exact(d):
 # 200 balanced target trials x three seeds: integer total-correct differences.
 counts=np.rint(d*6).astype(int);assert np.allclose(d*6,counts)
 states={0:1}
 for v in counts:
  nxt={}
  for k,n in states.items():
   nxt[k+int(v)]=nxt.get(k+int(v),0)+n;nxt[k-int(v)]=nxt.get(k-int(v),0)+n
  states=nxt
 return sum(n for k,n in states.items() if abs(k)>=abs(counts.sum()))/2**54
def paired(a,b):
 d=a-b;rng=np.random.default_rng(20260928)
 return dict(rd_accuracy_percent=float(a.mean()),nodemean_accuracy_percent=float(b.mean()),gain_pp=float(d.mean()),median_pp=float(np.median(d)),ci95_pp=np.quantile(d[rng.integers(0,54,(100000,54))].mean(axis=1),[.025,.975]).tolist(),wtl=[int((d>1e-9).sum()),int((abs(d)<=1e-9).sum()),int((d<-1e-9).sum())],exact_signflip_p=exact(d))
def analyze():
 arrays={};sources={}
 for arm in ARMS:
  paths=list((ROOT/'output'/arm).glob('NPFEEGNet/*/summary.csv'));assert len(paths)==1
  arrays[arm]=read_rows(paths[0]);sources[arm]=dict(path=str(paths[0]),sha256=sha(paths[0]))
  expected=json.loads((ROOT/f'{arm.split("_at_")[1]}_epochs.json').read_text())['selected_epochs']
  for row in csv.DictReader(paths[0].open()): assert int(row['retrain_epochs'])==int(expected[row['subject']][row['seed']])
 contrasts={name:paired(arrays[f'rd_at_{epoch}'],arrays[f'no_at_{epoch}']) for name,epoch in [('A_RD_duration','rd'),('B_NoDemean_duration','no')]}
 order=sorted(contrasts,key=lambda k:contrasts[k]['exact_signflip_p']);floor=0
 for i,k in enumerate(order):
  floor=max(floor,min(1,(2-i)*contrasts[k]['exact_signflip_p']));contrasts[k]['holm_p_two_sensitivity_conditions']=floor
 primary_rd=read_rows(ROOT/'primary_rows.csv','raw_demean');primary_no=read_rows(ROOT/'primary_rows.csv','nodemean')
 report=dict(role='post-confirmatory sensitivity',completed_at=now(),bootstrap_seed=20260928,subjects=54,seeds=[0,1,2],conditions=contrasts,contemporary_independent_duration_reference=paired(arrays['rd_at_rd'],arrays['no_at_no']),historical_primary_diagonal_reuse={'A':paired(primary_rd,arrays['no_at_rd']),'B':paired(arrays['rd_at_no'],primary_no)},historical_comparison_caveat='historical diagonals and new crossed runs differ in execution environment; contemporary four-arm results are preferred for isolating duration',sources=sources)
 write(ROOT/'results/report.json',report)
 with (ROOT/'results/subject_summary.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['subject']+ARMS+['gain_at_rd','gain_at_no'])
  for i in range(54):w.writerow([i+1]+[arrays[k][i] for k in ARMS]+[arrays['rd_at_rd'][i]-arrays['no_at_rd'][i],arrays['rd_at_no'][i]-arrays['no_at_no'][i]])
 status('complete')
if __name__=='__main__':
 if '--prepare' in sys.argv: prepare()
 elif '--arm' in sys.argv: train(sys.argv[sys.argv.index('--arm')+1])
 elif '--analyze' in sys.argv: analyze()
 else:
  try:
   verify();status('training',started_at=now())
   with concurrent.futures.ThreadPoolExecutor(2) as pool:
    futures=[pool.submit(lane,v,g) for v,g in zip(['rd','no'],GPU)]
    for f in futures:f.result()
   analyze()
  except Exception as e:status('failed',error=repr(e));raise
