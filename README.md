# NPFEEGNet-RD

Fixed spectral-residual EEGNet with raw-branch trial-wise temporal demeaning.

**Version 0.5.0: maintained implementation and audit evidence.** This version contains the independent maintained model/training/inference path and portable audits of available historical evidence. It excludes the legacy EEG-CSANet-derived training engine and does not certify full numerical reproduction of all manuscript experiments. See [release scope](docs/RELEASE_v0.5.0.md), [citation metadata](CITATION.cff), and the [versioned release](https://github.com/LifeOfGame/NPFEEGNet-RD/releases/tag/v0.5.0).

Version-specific archive DOI: [10.5281/zenodo.23221417](https://doi.org/10.5281/zenodo.23221417). The DOI identifies the immutable `v0.5.0` archive; subsequent repository documentation updates do not change that archive.

## Independent training validation (7 October 2026)

The maintained v2 trainer supports NPF, EEGNet, FBCNet, ATCNet and EEG Conformer
without importing the excluded legacy engine or mixed-baseline module.
FBCNet's official MIT source is now pinned and included with its license.
See [license/source resolution](docs/LICENSE_RESOLUTION.md) and
[raw-MAT-to-training commands and validation limits](docs/TRAINING_VALIDATION.md).

One real OpenBMI subject completed source selection, fresh NPF fitting, target
scoring and checkpoint inference; four baseline interfaces completed short
integration runs. Re-exported raw MAT inputs matched all six historical input
files byte-for-byte. This is not a 54-subject historical numerical reproduction.
Release decision (2026-10-08): the legacy EEG-CSANet-derived engine remains
private and is not part of this distribution. No permission request is being
pursued for this release. No grant or historical numerical equivalence is claimed.

## Portable reconstruction added (7 October 2026)

See [portable audit commands](docs/PORTABLE_AUDITS.md). Complete-study aggregation
now covers 40 arms and eight descriptive contrasts, with exact tie handling and
separate Zhou2020 cohort IDs. Source-OOF temperature fitting and the non-collapse/
soft-response audits are runnable from this checkout. A content-hashed registry
maps 29 tables/figures in the inspected BSPC snapshot to their available sources
and explicit gaps. No manuscript results were changed.

The full CPU suite passed 18 tests; optional plotting and source-only temperature
reconstruction also passed in an isolated server directory. The temperature,
NLL and ECE reproduce the archived report. These are retrospective artifact
checks, not a new prospective experiment or complete training replication.

## Expanded historical evidence (7 October 2026)

- `evidence/predictions/`: 45 groups, 5,346 runs, anonymous labels/probabilities and source hashes.
- `scripts/audit_predictions.py`: portable subject-macro accuracy and pooled NLL/ECE reconstruction with NumPy only.
- `evidence/historical/`: 573 original configurations, locks and result records, including superseded developmental records.
- `historical_sources/`: original experiment-specific preprocessing/analysis/wrapper sources. These retain legacy imports and paths; they are reference material, not a turnkey training pipeline.
- `baselines/`: preserved ATCNet and independently implemented EEG Conformer architecture controls, with provenance notices.
- `evidence/environments/`: recovered common-duration audit environment, not a claim that all historical experiments used it.

```bash
python scripts/audit_predictions.py --output /path/to/new-audit-directory
python -m unittest discover -s tests -p test_prediction_audit.py -v
```

See `docs/EXPERIMENT_MAP.md` before selecting results. No raw EEG, features or weights are included. Remaining original-engine licensing and execution-path gaps are explicit in `docs/REPRODUCTION_COVERAGE.md`.

## Included

- `model/`: fixed nominal 4–8, 8–13, 13–20, 20–30 and 30–40 Hz responses, shared spectral encoder, uniform aggregation, bounded spectral-logit residual and raw-path centering.
- `training/`, `train.py`, `predict.py`: source-session-only epoch selection, fresh fixed-epoch fitting and inference.
- `configs/`: model input dimensions, not complete frozen historical experiment recipes.
- `tests/`: synthetic model-property, training, data-separation and inference checks.
- `evidence/bp830/`: supplied run summaries, selection records and frozen epoch maps for the later BP8–30 control.
- `evidence/bp830_protocol/`: portable validation and paired accuracy analysis.
- `docs/REPRODUCTION_COVERAGE.md`: evidence coverage and remaining release work.

## Install

Use Python 3.12. Install a PyTorch build appropriate for the CPU/CUDA platform, then run `python -m pip install -r requirements-baselines.txt` for all maintained models and tests (`requirements.txt` suffices for NPF-only use). These dependency bounds describe the maintained implementation, not a reconstructed historical environment. Do not replace an existing validated experimental environment merely to satisfy this example.

```bash
python -m unittest discover -s tests -v
python scripts/verify_release_files.py
python evidence/bp830_protocol/verify.py
python evidence/bp830_protocol/analyze.py
```

The evidence scripts locate their data relative to the script and can run from another working directory. They validate 324 final rows and reproduce the corrected within-bandpass W/T/L of **27/6/21**. Integer correct-count differences resolve ties; original means and bootstrap computations are retained. The historical-comparator bootstrap seed in this analysis is not the original primary-analysis seed.

## Train on your own prepared session arrays

Data are not bundled. An NPZ contains `X` (trials × channels × time), `y` and source-session `groups`. Set `--label-offset 0` for zero-based labels or `1` for labels starting at one. Ensure the data and preprocessing match your study. See `configs/README.md` for dimensions.

```bash
python train.py select --train /path/to/session1.npz --model-config configs/openbmi.json --label-offset 0 --seed 0 --output runs/s01_seed0_selection
python train.py fit --train /path/to/session1.npz --selection runs/s01_seed0_selection/selection.json --test /path/to/session2.npz --output runs/s01_seed0_final
python predict.py --checkpoint runs/s01_seed0_final/model.pt --input /path/to/session2.npz --output runs/s01_seed0_predictions.npz
```

The default execution device is CPU. Select a CUDA device explicitly through the CLI when appropriate. Existing output directories are rejected to avoid overwriting experiments. Source group identities, acquisition units, channel order, temporal windows and preprocessing must be established before training. Synthetic tests do not establish paper-result replication.

## Validation

On 7 October 2026, 22 tests passed (21-model/statistics/training suite plus the new preprocessing-guard test) on CPU in an isolated server directory (Python 3.12.14, PyTorch 2.11.0+cu126, NumPy 2.4.4). The 324-row BP8–30 verification and corrected paired analysis also passed. See `docs/VALIDATION_20261007.json` for the initial checks and `docs/VALIDATION_EXPANDED_20261007.json` for the expanded checks. These checks do not constitute rerunning the paper experiments.

## Scientific and licensing scope

Temporal centering is an established preprocessing operation. The paper's positive OpenBMI finding is preprocessing-dependent and is not a claim of universal centering benefit or general architectural superiority. Later filtering and external-data controls must retain their reported limitations.

See `LICENSE_SCOPE.md` and `THIRD_PARTY_NOTICES.md`. Raw EEG, model weights, mixed-provenance legacy training engines and incomplete author/ethics documents are intentionally excluded. A full paper release requires resolving their provenance or documenting how readers can obtain required components lawfully, and validating the resulting execution path.

Repository: https://github.com/LifeOfGame/NPFEEGNet-RD

No Zenodo DOI or formal release tag is claimed. Complete the coverage checklist and confirm author metadata before archiving the paper-associated version.
