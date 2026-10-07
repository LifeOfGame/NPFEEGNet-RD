"""Verify pinned vendored bytes and absence of legacy imports in maintained code.

This checks the declared source policy, not ownership or legal sufficiency of
unknown historical files. It never treats missing permission as a license.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN={'utils','data','repro','train_dcsa_2a','model.baselines','model.baseModel','model.CSANet_2b'}


def main():
    provenance=json.loads((ROOT/'baselines/FBCNET_PROVENANCE.json').read_text())
    for path,digest in [(ROOT/provenance['distributed_file'],provenance['source_sha256']),
                        (ROOT/'baselines/FBCNet-MIT.txt',provenance['license_sha256'])]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError(f'Pinned upstream file changed: {path}')
    paths=[ROOT/'train.py',ROOT/'predict.py']
    for folder in ['model','training','baselines']:paths.extend((ROOT/folder).glob('*.py'))
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            modules=[]
            if isinstance(node,ast.Import):modules=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom) and not node.level:modules=[node.module or '']
            if any(m==f or m.startswith(f+'.') for m in modules for f in FORBIDDEN):
                raise ValueError(f'Excluded legacy import in maintained path: {path}:{node.lineno}')
    print(f'PASS: pinned FBCNet source/license bytes; {len(paths)} maintained modules have no declared excluded imports')


if __name__=='__main__':main()
