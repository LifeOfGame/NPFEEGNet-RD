# Code availability ? proposed manuscript wording

The maintained implementation, experiment configurations and available anonymous
prediction-level evidence are available at https://github.com/LifeOfGame/NPFEEGNet-RD.
The release excludes the legacy EEG-CSANet-derived training engine. Historical
source snapshots and hashes are retained privately for provenance. The public
maintained implementation is a separate training path; its validation includes
an OpenBMI raw-MAT preprocessing check with byte-identical exported inputs,
one-subject NPF source-selection/fitting/inference validation, and short baseline
integration checks. These checks do not establish numerical equivalence to the
historical engine or full reproduction of all reported experiments. Detailed
coverage and remaining gaps are documented in the repository.

This wording does not promise access to excluded code on request. Use a real
release identifier/DOI only once one has been created. No manuscript or PDF was
edited as part of this distribution-scope update.
