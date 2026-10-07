from pathlib import Path
import tarfile,json
root=Path(__file__).resolve().parent
files=[root/n for n in ['run.py','design_lock.json','status.json','rd_epochs.json','no_epochs.json','primary_rows.csv','runtime.json','pip-freeze.txt']]
files+=list((root/'config').glob('*.yaml'))+list((root/'results').glob('*'))
for arm in ['rd_at_rd','rd_at_no','no_at_rd','no_at_no']:
 files+=list((root/'output'/arm).glob('NPFEEGNet/*/summary.csv'))
assert json.loads((root/'status.json').read_text())['stage']=='complete'
with tarfile.open(root/'metadata_bundle.tar.gz','w:gz') as t:
 for p in files:
  assert p.is_file(),p
  t.add(p,arcname=p.relative_to(root))
print(len(files),'metadata files archived; checkpoints remain on server')
