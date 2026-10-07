# Reproduction coverage, preparation version 0.4.0.dev0

| Scope | Included | Remaining limitation |
|---|---|---|
| Maintained model/training | Independent code and CPU tests | Not the historical frozen engine |
| Original OpenBMI primary | Run records, anonymous predictions, portable accuracy/NLL/ECE audit | Exact full training path remains dependent on legacy source |
| BP8-30 | 324-run epoch/hash audit, fixed W/T/L, predictions | Full raw-data retraining path not yet validated |
| FullDemean, HP, capacity, components | Configs, reports, available predictions, historical source | Original wrappers retain paths/imports; not turnkey |
| Baselines | ATCNet and paper-based Conformer source, historical records | Mixed EEGNet/FBCNet dependency provenance unresolved; not all recipe predictions exported |
| BCIC/HGD | Historical config/result records; corrected 2a preprocessing reference | Superseded 2a records must not be treated as final; portable rerun not validated |
| BNCI/Zhou | Configs, locks, reports, available predictions and original preprocessing references | Dataset downloads and legacy training dependencies required |
| Mechanism/calibration | Historical analysis code and result records | Source OOF predictions and portable temperature refit included; mechanism checkpoints remain excluded |

The newly generated metrics match original OpenBMI primary accuracy and pooled
NLL. This is evidence reconstruction, not independent experimental replication.
The historical record collection intentionally includes superseded development
work. The content-hashed MANUSCRIPT_REGISTRY.json now maps 29 tables/figures
for the inspected BSPC snapshot, with explicit reconstruction gaps.

## Outstanding for a complete paper-associated release

1. Resolve EEG-CSANet-derived frozen-engine permission and mixed baseline
   provenance, or independently reimplement and validate the missing path.
2. Fill the remaining reconstruction gaps identified in the version-specific
   manuscript registry; validate preprocessing-to-selection-to-final commands
   for every reported experiment.
3. Recover experiment-specific environments where absent; one later environment
   cannot establish earlier versions.
4. Validate the full runnable package on legally obtained data. CPU unit tests
   and prediction audits do not establish training equivalence.
5. Confirm citation author names/order, then create the actual release and
   Zenodo version DOI. No v1.0.0 tag or DOI is claimed here.
