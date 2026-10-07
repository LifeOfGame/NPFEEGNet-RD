# NPFEEGNet-RD

Fixed spectral-residual EEGNet with raw-branch trial-wise temporal demeaning.

**Status: repository preparation, version 0.2.0.dev0. This is not the complete manuscript reproducibility release and has no archive DOI yet.** This repository contains an independent maintained model implementation, a new training/inference interface, and the corrected summary-level OpenBMI BP8–30 audit. The new trainer does not certify or regenerate all historical paper results. Do not describe this initial repository as the final paper-associated v1.0.0.

## Included

- `model/`: fixed nominal 4–8, 8–13, 13–20, 20–30 and 30–40 Hz responses, shared spectral encoder, uniform aggregation, bounded spectral-logit residual and raw-path centering.
- `training/`, `train.py`, `predict.py`: source-session-only epoch selection, fresh fixed-epoch fitting and inference.
- `configs/`: model input dimensions, not complete frozen historical experiment recipes.
- `tests/`: synthetic model-property, training, data-separation and inference checks.
- `evidence/bp830/`: supplied run summaries, selection records and frozen epoch maps for the later BP8–30 control.
- `evidence/bp830_protocol/`: portable validation and paired accuracy analysis.
- `docs/REPRODUCTION_COVERAGE.md`: evidence coverage and remaining release work.

## Install

Use Python 3.12. Install a PyTorch build appropriate for the CPU/CUDA platform, then run `python -m pip install -r requirements.txt`. These dependency bounds describe the maintained implementation, not a reconstructed historical environment. Do not replace an existing validated experimental environment merely to satisfy this example.

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

On 7 October 2026, all 10 included tests passed on CPU in an isolated server directory (Python 3.12.14, PyTorch 2.11.0+cu126, NumPy 2.4.4). The 324-row BP8?30 verification and corrected paired analysis also passed. See `docs/VALIDATION_20261007.json`. These checks do not constitute rerunning the paper experiments.

## Scientific and licensing scope

Temporal centering is an established preprocessing operation. The paper's positive OpenBMI finding is preprocessing-dependent and is not a claim of universal centering benefit or general architectural superiority. Later filtering and external-data controls must retain their reported limitations.

See `LICENSE_SCOPE.md` and `THIRD_PARTY_NOTICES.md`. Raw EEG, model weights, mixed-provenance legacy training engines and incomplete author/ethics documents are intentionally excluded. A full paper release requires resolving their provenance or documenting how readers can obtain required components lawfully, and validating the resulting execution path.

Repository: https://github.com/LifeOfGame/NPFEEGNet-RD

No Zenodo DOI or formal release tag is claimed. Complete the coverage checklist and confirm author metadata before archiving the paper-associated version.
