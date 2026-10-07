"""Real prepared-data integration check; never overwrites historical experiments.

Runs selection, fresh fitting, test scoring and checkpoint inference through the
public CLIs. Baseline smoke runs do not reproduce their original training recipes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--subject',type=int,default=1)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--full-npf-epochs',type=int,default=300)
    parser.add_argument('--smoke-epochs',type=int,default=2)
    parser.add_argument('--models',nargs='+',choices=['npf','eegnet','fbcnet','atcnet','conformer'],default=['npf','eegnet','fbcnet','atcnet','conformer'])
    args=parser.parse_args()
    if not 1<=args.subject<=54 or min(args.full_npf_epochs,args.smoke_epochs)<1:raise ValueError('Invalid validation settings')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    train=args.data_root.resolve()/f'O{args.subject:02d}T_data.npy'
    target=args.data_root.resolve()/f'O{args.subject:02d}E_data.npy'
    report={'scope':'New maintained-engine integration validation, not historical reproduction','subject':args.subject,'models':{}}
    def run(argv,log):
        with log.open('w',encoding='utf-8') as stream:
            subprocess.run([sys.executable,*map(str,argv)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,PYTHONUNBUFFERED='1'))
    for model in args.models:
        started=time.monotonic();dest=out/model;dest.mkdir()
        epochs=args.full_npf_epochs if model=='npf' else args.smoke_epochs
        config=ROOT/f'configs/openbmi_{model}_maintained.json'
        print(f'START {model}: source-only selection, maximum {epochs} epochs',flush=True)
        run(['train.py','select','--train',train,'--model-config',config,'--label-offset','1','--epochs',epochs,'--patience',min(40,epochs),'--seed','0','--device',args.device,'--output',dest/'selection'],dest/'selection.log')
        run(['train.py','fit','--train',train,'--selection',dest/'selection/selection.json','--test',target,'--device',args.device,'--output',dest/'final'],dest/'fit.log')
        run(['predict.py','--checkpoint',dest/'final/model.pt','--input',target,'--device',args.device,'--output',dest/'predictions.npz'],dest/'predict.log')
        manifest=json.loads((dest/'final/protocol_manifest.json').read_text())
        if manifest['status']!='complete' or manifest['test_evaluations']!=1:raise RuntimeError('Incomplete final manifest')
        report['models'][model]={'role':'full maintained recipe (single subject/seed)' if model=='npf' else 'short integration smoke; not recipe reproduction',
            'max_epochs':epochs,'selected_epoch':manifest['selected_epoch'],'elapsed_seconds':time.monotonic()-started,
            'metrics':json.loads((dest/'final/test_metrics.json').read_text()),'protocol':manifest['protocol'],
            'model_kwargs':manifest['model_kwargs'],'environment':manifest['environment'],
            'source_sha256':manifest['source_sha256'],'checkpoint_sha256':manifest['checkpoint_sha256'],
            'source_data_sha256':list(manifest['training_files'].values()),'test_data_sha256':list(manifest['test_files'].values()),
            'test_evaluations':manifest['test_evaluations'],'prediction_file_sha256':hashlib.sha256((dest/'predictions.npz').read_bytes()).hexdigest()}
        (out/'validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(f'PASS {model}: selected epoch {manifest["selected_epoch"]}; {time.monotonic()-started:.1f}s',flush=True)
    print(f'COMPLETE {out}',flush=True)


if __name__=='__main__':main()
