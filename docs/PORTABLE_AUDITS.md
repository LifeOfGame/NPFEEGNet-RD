# Portable evidence reconstruction

Run from the repository root. Each output directory must be new. These commands
use included anonymous predictions and historical summaries, not EEG or GPUs.

```bash
python -m pip install -r requirements-audit.txt
python scripts/verify_release_files.py
python scripts/build_evidence_tables.py --output /path/to/new-study-tables
python scripts/rebuild_method_audits.py --output /path/to/new-methods-audit --plots
python scripts/rebuild_temperature.py --output /path/to/new-calibration-audit
python evidence/bp830_protocol/verify.py
python evidence/bp830_protocol/analyze.py
```

## Outputs and aggregation

`build_evidence_tables.py` produces complete-study, subject-level and paired
descriptive CSVs for 40 arms and eight specified contrasts. It merges GPU
partitions without duplicate runs, namespaces Zhou2020 cohorts, requires all
three seeds per subject, and computes ties from exact rational correct counts.
NLL/ECE pool trial predictions; accuracy averages seeds within subject then
subjects. It does not average calibration metrics across GPU partitions.

`rebuild_method_audits.py` preserves the archived methods-audit bootstrap seed
20260928 and 100,000 resamples. It recreates non-collapse sensitivity, soft-filter
frequency samples, and optionally Fig. S3. Collapse means minimum class recall
below 0.10 in either paired run. Exclusion conditions on target outcomes and
is strictly post-hoc: 156 retained seed pairs across 53 subjects do not replace
the primary analysis. The all-subject CI here uses this audit's RNG seed and
must not overwrite the original primary CI from another recorded analysis.

`rebuild_temperature.py` reproduces S1-only fitting and then frozen S2 scoring.
The fit agrees with the historical temperature lock within 1e-6. The original
temperature script's ECE boundary expression is retained. See the temperature
README for inputs and the distinction between retrospective verification and
prospective experimental locking.

`evidence/reconstructed/` contains reference outputs generated during packaging.
`docs/MANUSCRIPT_REGISTRY.json` records content hashes for 29 tables/figures in
the inspected BSPC manuscript snapshot. Its scope notes explicitly distinguish
reconstructable quantities, archived reports, conceptual diagrams and remaining
gaps. A source mapping alone does not prove every figure can be regenerated.

## What remains separate

The maintained model/trainer supports new experiments. It is not certified as
the engine that generated the historical runs. Original-engine permission,
mixed-baseline provenance, some checkpoint-dependent mechanisms/confusion
inputs, and the full raw-data-to-result execution path remain unresolved.
No existing manuscript number is changed by these retrospective checks.
