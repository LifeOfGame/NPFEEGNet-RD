import numpy as np
from pathlib import Path

path = Path('/mnt/helongfei/NPFEEGNet-RD-zhou2020-20260927/source_probe_run_01.npz')
with np.load(path, allow_pickle=False) as z:
    x = z['signal']
    ev = z['MarkOnSignal']
    print('signal shape', x.shape, 'median_abs', np.median(np.abs(x[:100000,:41])),
          'percentile_99_abs', np.percentile(np.abs(x[:100000,:41]), 99),
          'maximum_abs', np.max(np.abs(x[:,:41])))
    for code in (769,770,771,780,786,800):
        print('event', code, int((ev[:,1]==code).sum()))
    classes = ev[np.isin(ev[:,1], [769,770,771,780]),0]
    cues = ev[ev[:,1]==786,0]
    ends = ev[ev[:,1]==800,0]
    prior_cue = np.searchsorted(cues, classes, side='right') - 1
    valid = prior_cue >= 0
    print('cue_to_class_samples', np.unique(classes[valid] - cues[prior_cue[valid]], return_counts=True))
    print('class_to_end_sample_range', int(np.min(ends-classes)), int(np.max(ends-classes)))
