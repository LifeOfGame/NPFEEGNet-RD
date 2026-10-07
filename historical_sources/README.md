# Historical source reference

These are preserved experiment-specific preprocessing, analysis and wrapper
files, not a portable replacement for the original training workspace. They
retain original paths and imports. Read before executing: some download data,
create files, or invoke training. No such operations are needed for the public
prediction audit in `scripts/audit_predictions.py`.

`manifest.json` identifies the exact original file and SHA256. Configurations
and reports are under `evidence/historical/<study>/`. Copied source bytes have
not been edited to pretend that a new implementation generated old results.
They need their documented data and original dependencies; unresolved legacy
engines and the mixed-provenance baseline module are excluded.

In particular, `build_cue_aligned_2a.py` documents the corrected MAT-trial
relative [2,6) s extraction; older 2a reports remain historical records and
must not replace that corrected analysis. Activation scripts use saved
checkpoints; temperature fitting requires source-session out-of-fold outputs,
which are not in the target-prediction bundle. Do not fit temperature to the
released target labels.
