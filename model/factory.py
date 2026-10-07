"""Maintained, explicitly sourced model factory; no legacy engine imports.

These interfaces support new validation runs. They are not assertions of
byte-identical historical training or original-paper baseline performance.
"""
import numpy as np
import torch
from torch import nn

from .npf_eegnet_rd import NPFEEGNetRD


class BaselineInterface(nn.Module):
    def __init__(self, core, centering='none'):
        super().__init__()
        if centering not in {'none','demean'}:raise ValueError('Baseline centering must be none or demean')
        self.core=core
        self.centering=centering

    def forward(self,x,return_aux=False):
        if self.centering=='demean':x=x-x.mean(dim=-1,keepdim=True)
        logits=self.core(x)
        return (logits,{}) if return_aux else logits


class IndependentEEGNet(nn.Module):
    """New native-PyTorch assembly of the cited EEGNet topology.

    Uses the project's documented layer dimensions; does not extract the
    unresolved historical mixed-baseline module. Width is explicit.
    """
    def __init__(self,channels,classes,times,width=8):
        super().__init__()
        if width<1 or times<32:raise ValueError('Invalid EEGNet dimensions')
        f=width*2
        self.temporal=nn.Sequential(nn.Conv2d(1,width,(1,64),padding='same',bias=False),nn.BatchNorm2d(width,eps=1e-4))
        self.spatial=nn.Sequential(nn.Conv2d(width,f,(channels,1),groups=width,bias=False),nn.BatchNorm2d(f,eps=1e-4),nn.ELU(),nn.AvgPool2d((1,4)),nn.Dropout(.5))
        self.separable=nn.Sequential(nn.Conv2d(f,f,(1,16),padding='same',groups=f,bias=False),nn.Conv2d(f,f,1,bias=False),nn.BatchNorm2d(f,eps=1e-4),nn.ELU(),nn.AvgPool2d((1,8)),nn.Dropout(.5))
        self.classifier=nn.Linear(f*(times//4//8),classes)

    def forward(self,x):
        return self.classifier(self.separable(self.spatial(self.temporal(x))).flatten(1))


class OfficialFBCAdapter(nn.Module):
    """Pinned MIT FBCNet with a new finite-epoch causal filter adapter.

    Chebyshev-II design follows the pinned author's specified stopband, with
    3 dB passband/30 dB stopband and 2-Hz allowance. FFT convolution implements
    zero-state lfilter over each complete epoch; this is not continuous filtering.
    """
    def __init__(self,channels,classes,times,sampling_rate=250.):
        super().__init__()
        from scipy import signal
        from baselines.vendor_fbcnet import FBCNet
        if sampling_rate<=84 or times%4 or times<8:raise ValueError('FBCNet needs fs>84 and times divisible by 4')
        impulse=np.zeros(times);impulse[0]=1
        responses=[]
        for low in range(4,40,4):
            high=low+4
            wp=np.array([low,high])/(sampling_rate/2)
            ws=np.array([low-2,high+2])/(sampling_rate/2)
            order,_=signal.cheb2ord(wp,ws,3,30)
            b,a=signal.cheby2(order,30,ws,btype='bandpass')
            responses.append(signal.lfilter(b,a,impulse))
        self.register_buffer('impulse',torch.tensor(np.stack(responses),dtype=torch.float64))
        self.times=times
        self.core=FBCNet(nChan=channels,nTime=times,nClass=classes,nBands=9,m=32,strideFactor=4,temporalLayer='LogVarLayer')

    def filter(self,x):
        if x.ndim!=4 or x.shape[1]!=1 or x.shape[-1]!=self.times:raise ValueError('Invalid FBCNet epoch shape')
        nfft=2*self.times
        spectrum=torch.fft.rfft(x[:,0].to(torch.float64),n=nfft)
        kernel=torch.fft.rfft(self.impulse,n=nfft)
        filtered=torch.fft.irfft(spectrum[:,None]*kernel[None,:,None],n=nfft)[...,:self.times]
        return filtered.to(x.dtype)

    def forward(self,x):
        # upstream input: batch, 1, channels, samples, filter-band
        filtered=self.filter(x).permute(0,2,3,1).unsqueeze(1)
        return self.core(filtered)


def build_model(model_kwargs):
    options=dict(model_kwargs)
    name=options.pop('model_name','npf')
    if name=='npf':return NPFEEGNetRD(**options)
    channels=options.pop('num_channels');classes=options.pop('classes');times=options.pop('n_times',1000)
    fs=options.pop('sampling_rate',250.)
    centering=options.pop('centering','demean')
    width=options.pop('width',8)
    if options:raise ValueError(f'Unsupported baseline options: {sorted(options)}')
    if name=='eegnet':core=IndependentEEGNet(channels,classes,times,width)
    elif name=='fbcnet':core=OfficialFBCAdapter(channels,classes,times,fs)
    elif name=='atcnet':
        from baselines.openbmi_atcnet import OpenBMIATCNet
        core=OpenBMIATCNet(channels,classes,n_times=times,input_standardization='none')
    elif name=='conformer':
        from baselines.openbmi_eegconformer import OpenBMIEEGConformer
        core=OpenBMIEEGConformer(channels,classes,n_times=times,input_standardization='none')
    else:raise ValueError(f'Unknown model: {name}')
    return BaselineInterface(core,centering)
