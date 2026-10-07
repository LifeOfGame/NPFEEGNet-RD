# Reproduction coverage, version 0.5.0

| Scope | Included | Remaining limitation |
|---|---|---|
| Maintained model/training | Independent code, 22 tests, real OpenBMI integration | Not the historical frozen engine; only one subject/seed integration checked |
| Original OpenBMI primary | Run records, anonymous predictions, portable accuracy/NLL/ECE audit | New local-MAT-to-training path validated without legacy imports; exact historical numerical replication remains unverified |
| BP8-30 | 324-run epoch/hash audit, fixed W/T/L, predictions | Full raw-data retraining path not yet validated |
| FullDemean, HP, capacity, components | Configs, reports, available predictions, historical source | Original wrappers retain paths/imports; not turnkey |
| Baselines | Independent EEGNet, pinned MIT FBCNet, Apache ATCNet, paper-based Conformer; four real-data short integrations | Historical mixed-module provenance and full author-recipe reruns remain unverified |
| BCIC/HGD | Historical config/result records; corrected 2a preprocessing reference | Superseded 2a records must not be treated as final; portable rerun not validated |
| BNCI/Zhou | Configs, locks, reports, available predictions and original preprocessing references | Dataset downloads and legacy training dependencies required |
| Mechanism/calibration | Historical analysis code and result records | Source OOF predictions and portable temperature refit included; mechanism checkpoints remain excluded |

The newly generated metrics match original OpenBMI primary accuracy and pooled
NLL. This is evidence reconstruction, not independent experimental replication.
The historical record collection intentionally includes superseded development
work. The content-hashed MANUSCRIPT_REGISTRY.json now maps 29 tables/figures
for the inspected BSPC snapshot, with explicit reconstruction gaps.

## Outstanding for a complete paper-associated release

1. Scope fixed on 2026-10-08: legacy engine redistribution is excluded, not a
   pending public deliverable. The maintained path removes that dependency;
   historical numerical equivalence remains unverified.
2. Fill the remaining reconstruction gaps identified in the version-specific
   manuscript registry; validate preprocessing-to-selection-to-final commands
   for every reported experiment.
3. Recover experiment-specific environments where absent; one later environment
   cannot establish earlier versions.
4. Validate the full runnable package on legally obtained data. CPU unit tests
   and prediction audits do not establish training equivalence.
5. Confirm citation author names/order, then create the actual release and
   Zenodo version DOI. No v1.0.0 tag or DOI is claimed here.
