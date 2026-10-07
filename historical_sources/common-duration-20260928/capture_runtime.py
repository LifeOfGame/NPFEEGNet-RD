from pathlib import Path
import json,sys,platform,subprocess,importlib.metadata
from datetime import datetime,timezone
import torch
root=Path(__file__).resolve().parent
def version(n):
 try: return importlib.metadata.version(n)
 except importlib.metadata.PackageNotFoundError: return None
r={'captured_at':datetime.now(timezone.utc).isoformat(),'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'packages':{n:version(n) for n in ['numpy','scipy','pandas','scikit-learn','PyYAML']},'note':'Runtime captured during training; original design hashes are retained separately. Null means distribution metadata absent.'}
(root/'runtime.json').write_text(json.dumps(r,indent=2)+'\n')
p=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
(root/'pip-freeze.txt').write_text(p.stdout)
print(json.dumps(r,indent=2))
