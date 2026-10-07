# Source-session calibration evidence

`s1_oof.npz` contains 32,400 anonymous source-session predictions from 324
fold outputs (54 subjects x 3 seeds x 2 folds x 100 trials). Only subject,
seed, fold, within-fold export index, labels and probabilities are included.
There are no EEG arrays, features or checkpoints.

Each original NPZ was checked against the historical temperature lock before
extraction. The manifest contains original-file and selected-array hashes.
The original historical NPZ files contain additional arrays and are not
identical to this sanitized archive.

```bash
python -m pip install -r requirements-audit.txt
python scripts/rebuild_temperature.py --output /path/to/new-temperature-audit
```

The reconstruction fits only S1 OOF probabilities, saves the fitted temperature,
then loads the already frozen S2 prediction bundle. It reproduces
T=2.5463403064273793, S2 NLL=0.5553834068753374 and ECE=0.025640330738154304,
without changing predicted classes. These are pooled descriptive metrics.

This is retrospective reconstruction of archived analysis, not a new prospective
lock. Do not refit or select calibration settings using the released S2 labels.
