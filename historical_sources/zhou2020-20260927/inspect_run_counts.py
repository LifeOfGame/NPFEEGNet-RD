from pathlib import Path
import numpy as np

root = Path('/mnt/helongfei/database/raw/zhou2020_zenodo_v1')
for subject in (17, 20):
    for path in sorted((root / f'S{subject:02d}' / 'session_1').glob('run_*.npz')):
        with np.load(path, allow_pickle=False) as z:
            events = z['MarkOnSignal']
            codes = events[:, 1]
            counts = [int((codes == c).sum()) for c in (769, 770, 771, 780)]
        print(path, counts, sum(counts), flush=True)
