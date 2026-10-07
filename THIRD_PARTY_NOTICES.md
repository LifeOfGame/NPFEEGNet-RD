# Sources and dependencies

The EEGNet-style temporal, depthwise spatial and separable-convolution architecture is attributed to V. J. Lawhern et al., EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces, Journal of Neural Engineering 15, 056013 (2018), https://doi.org/10.1088/1741-2552/aace8c. Architectural attribution does not claim EEGNet as a new contribution.

The maintained implementation was copied from the project's independent current model/training workspace; its original file hashes are retained in docs/SOURCE_PROVENANCE.json. It is not a byte-identical release of the historical frozen training engine. PyTorch and NumPy are external dependencies under their own licenses.

BP8–30 results and corrected summary-level verification scripts are taken from the repaired 2026-10-07 submission package. No raw or processed EEG is included. The historical EEG-CSANet-derived engine and mixed-provenance ATCNet/FBCNet/other baseline files were excluded pending file-level permission and provenance review. Their historical preservation does not authorize redistribution.
