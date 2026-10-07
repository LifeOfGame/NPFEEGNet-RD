import json
import sys
from pathlib import Path
import time
import torch
import torch.nn.functional as F
import yaml
SOURCE=Path('/mnt/helongfei/NPFEEGNet-RD-paper-20260917')
sys.path.insert(0,str(SOURCE))
from model.NPF_EEGNet import NPFEEGNet
from boundary_models import make_model
cfg=yaml.safe_load((SOURCE/'config/openbmi_selection_npf_raw_demean_3seed.yaml').read_text())['network_args']
torch.manual_seed(42); original=NPFEEGNet(**cfg).cuda().eval()
x=torch.randn(16,1,20,1000,device='cuda')*20+15
report={}
for mode in ('current','fft_reflect','raw_reflect'):
    torch.manual_seed(42); model=make_model(mode)(**cfg).cuda().eval()
    assert list(model.state_dict())==list(original.state_dict())
    for k,v in model.state_dict().items(): assert torch.equal(v,original.state_dict()[k]),k
    if mode=='current': assert torch.equal(model(x),original(x))
    if mode=='raw_reflect':
        conv=model.raw_features[0]
        expected=F.conv2d(F.pad(x,(31,32,0,0),mode='reflect'),conv.weight)
        assert torch.equal(conv(x),expected)
    filtered=model.filter_bank(x); assert filtered.shape==(16,5,20,1000)
    if mode=='fft_reflect':
        ref=original.filter_bank(F.pad(x.squeeze(1),(250,250),mode='reflect').unsqueeze(1))[...,250:1250]
        assert torch.equal(filtered,ref)
    model.train(); start=time.monotonic()
    for _ in range(10):
        model.zero_grad(); z,aux=model(x,return_aux=True)
        loss=F.cross_entropy(z,torch.zeros(16,device='cuda',dtype=torch.long))+.5*aux['raw_logits'].square().mean()+.25*aux['spectral_logits'].square().mean()
        loss.backward(); assert torch.isfinite(loss)
    torch.cuda.synchronize()
    report[mode]={'seconds_per_microbatch':(time.monotonic()-start)/10,'parameters':sum(p.numel() for p in model.parameters()),'forward_backward_finite':True}
print(json.dumps(report,indent=2))
