# Sources and dependencies

The EEGNet-style temporal, depthwise spatial and separable-convolution architecture is attributed to V. J. Lawhern et al., EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces, Journal of Neural Engineering 15, 056013 (2018), https://doi.org/10.1088/1741-2552/aace8c. Architectural attribution does not claim EEGNet as a new contribution.

The maintained implementation was copied from the project's independent current model/training workspace; its original file hashes are retained in docs/SOURCE_PROVENANCE.json. It is not a byte-identical release of the historical frozen training engine. PyTorch and NumPy are external dependencies under their own licenses.

BP8–30 results and corrected summary-level verification scripts are taken from the repaired 2026-10-07 submission package. No raw or processed EEG is included. The historical EEG-CSANet-derived engine and mixed-provenance ATCNet/FBCNet/other baseline files were excluded pending file-level permission and provenance review. Their historical preservation does not authorize redistribution.

## Added architecture controls

`baselines/openbmi_atcnet.py` preserves the Apache-2.0 adaptation notice,
Copyright (C) 2022 King Saud University, Saudi Arabia. Pinned reference:
https://github.com/Altaheri/EEG-ATCNet/tree/d8fb0c10abc14b796dfbd317f53f56a68ebe29df
Local changes translate TensorFlow topology to PyTorch, use OpenBMI dimensions,
return logits, and add optional trial/channel demeaning. Full license text is
in `baselines/Apache-2.0.txt`. This is not the authors' original training pipeline.

`baselines/openbmi_eegconformer.py` retains its historical independent paper-based
implementation statement and Song et al. citation (doi:10.1109/TNSRE.2022.3230250).
No GPL upstream source is included. Neither architecture file certifies original
paper accuracy. Source hashes are in `historical_sources/manifest.json`.

Historical experiment-specific wrappers import dependencies that are not
distributed, including the unresolved mixed baseline module. Publishing a wrapper
does not license those dependencies or establish that it runs in this checkout.

## Pinned official FBCNet (new maintained dependency)

`baselines/vendor_fbcnet.py` is a byte-identical copy of
`codes/centralRepo/networks.py` at official commit
`de1bbdd8a54cb1e466830e3d47070e0e56761a37`:
https://github.com/ravikiran-mane/FBCNet/tree/de1bbdd8a54cb1e466830e3d47070e0e56761a37
Copyright (c) 2020 ravikiran-mane. Distributed under MIT; the complete text is
in `baselines/FBCNet-MIT.txt`. Source and license SHA256 are recorded in
`baselines/FBCNET_PROVENANCE.json`. The new filter/input adapter is separate
in `model/factory.py`; its finite-epoch causal response is checked against
SciPy lfilter. This pin is not presented as the unidentified revision used to
develop the historical mixed-baseline module, which remains excluded.

The new EEGNet control is assembled separately from the documented topology;
it does not import or extract the unresolved mixed module. The architecture
continues to be attributed to Lawhern et al. New training is labelled maintained
v2, with no claim to regenerate historical training trajectories.
