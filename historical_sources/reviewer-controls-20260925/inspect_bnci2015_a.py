"""Inspect only first source session metadata before external evaluation."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat

path = Path('/mnt/helongfei/database/raw/bnci2015_001_nemar_v1.0.2/S01A.mat')
mat = loadmat(path, struct_as_record=False, squeeze_me=True)
print('top-level:', [(k, getattr(v, 'shape', None)) for k, v in mat.items() if not k.startswith('_')])
for key, value in mat.items():
    if key.startswith('_'):
        continue
    entries = np.atleast_1d(value)
    for index, entry in enumerate(entries[:3]):
        print('entry', key, index, getattr(entry, '_fieldnames', None))
        for field in getattr(entry, '_fieldnames', []):
            val = getattr(entry, field)
            arr = np.asarray(val)
            print(field, 'shape', arr.shape, 'dtype', arr.dtype,
                  'first', arr.ravel()[:8] if arr.dtype != object else 'object')
