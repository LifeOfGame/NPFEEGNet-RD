import argparse
import hashlib
import json
from pathlib import Path
import sys
import yaml

SOURCE = Path('/mnt/helongfei/NPFEEGNet-RD-paper-20260917')
sys.path.insert(0,str(SOURCE))
from repro.frozen_engine_loader import load_openbmi_engine
from boundary_models import make_model

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True)
    a=p.parse_args(); c=yaml.safe_load(a.config.read_text())
    lock=Path(c['design_lock_file'])
    for name,digest in json.loads(lock.read_text())['sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest, name
    if c.get('fixed_epoch_training'):
        final=json.loads(Path(c['protocol_lock_file']).read_text())
        assert final['status']=='ready_for_session2'
        for name,digest in final['sha256'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    t=load_openbmi_engine('boundary_sensitivity')
    t.NPFEEGNet=make_model(c['boundary_variant'])
    t.main(c)

if __name__=='__main__': main()
