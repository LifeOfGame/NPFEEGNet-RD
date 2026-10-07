# Maintained training and source policy validation

The maintained workflow no longer depends on the excluded historical engine or
mixed-baseline module. This resolves that dependency for **new runs**, not the
missing permission to redistribute old code. See `LICENSE_RESOLUTION.md`.

## What was actually executed

On 2026-10-07, an isolated checkout ran on an idle RTX 2080 Ti using Python
3.12.14, PyTorch 2.11.0+cu126, NumPy 2.4.4 and SciPy 1.17.1. Original experiment
directories were not overwritten. The report is
`evidence/training_validation/validation_report.json`.

- OpenBMI subject 1, seed 0: maintained NPF, source-only group selection with
  maximum 300 epochs and patience 40, selected 44 epochs; fresh full-source
  fitting, one target scoring pass, then independent checkpoint inference.
- EEGNet, official-source FBCNet, ATCNet and EEG Conformer: the same software
  stages completed with a **two-epoch integration budget**. This validates the
  interfaces, not their complete published training recipes or accuracies.
- The portable MAT preprocessor regenerated both sessions. All six exported
  data/label/group NPY files were **byte-identical** to the original prepared
  inputs used in the training check. Session 2 preprocessing required the
  completed source selection record. This exact identity connects the raw-MAT
  reconstruction to the independently executed prepared-data training stages.

No all-subject rerun, statistical-equivalence test against the old engine, or
full numerical replication of all seven datasets is claimed. New check metrics
are software-validation outcomes and do not replace manuscript results.

## Raw MAT to source selection to target evaluation

Obtain the dataset under its own terms. MAT files are not distributed here.
The original phase order, 20 channels, four-second window, integer labels,
microvolt scale and polyphase resampling are preserved. The source phase IDs
are offline training=0 and online feedback=1; both are source-session groups.

```bash
python -m pip install -r requirements-baselines.txt
python scripts/verify_source_policy.py

python scripts/prepare_openbmi.py --mat /data/sess01_subj01_EEG_MI.mat --subject 1 --session 1 --output /work/openbmi
python train.py select --train /work/openbmi/O01T_data.npy --label-offset 1 --model-config configs/openbmi_npf_maintained.json --epochs 300 --patience 40 --seed 0 --device cuda:0 --output /work/npf-selection
python scripts/prepare_openbmi.py --mat /data/sess02_subj01_EEG_MI.mat --subject 1 --session 2 --selection /work/npf-selection/selection.json --output /work/openbmi
python train.py fit --train /work/openbmi/O01T_data.npy --selection /work/npf-selection/selection.json --test /work/openbmi/O01E_data.npy --device cuda:0 --output /work/npf-final
python predict.py --checkpoint /work/npf-final/model.pt --input /work/openbmi/O01E_data.npy --device cuda:0 --output /work/npf-predictions.npz
```

All output paths must be new. For multiple compared conditions, freeze every
condition's source selection before opening target data and pass their records
together to `--selection`. Change `--device` to `cpu` if needed. The current
model configs are maintained interfaces, not automatic conversions of historical
YAML training recipes. Optimizer, scheduling, augmentation, batch size and
selection rules must be matched explicitly before claiming recipe replication.

For a repeat of the prepared-data integration check:

```bash
python scripts/validate_training_flow.py --data-root /data/processed/openbmi --output /work/new-validation --subject 1 --device cuda:0 --full-npf-epochs 300 --smoke-epochs 2
```

FBCNet's wrapper preserves the pinned upstream network. Its new differentiable
FFT frontend is tested against direct zero-state causal `lfilter` over each
finite epoch. It is not the manuscript's continuous high-pass/bandpass control.

## Remaining historical issues

The historical EEG-CSANet engine remains private by the release decision of
2026-10-08. No author contact or permission request is being pursued. The first upstream revision used in the old mixed baseline remains
unknown; the new pinned dependency is not retroactive provenance. Exact old
checkpoint/optimizer equivalence and all-dataset raw preprocessing still require
separate verification. These boundaries remain visible in the release checklist.
