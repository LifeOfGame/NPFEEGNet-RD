# NPFEEGNet-RD v0.5.0

First formal release of the independently maintained implementation and available
audit evidence. This release is scoped to the public contents described below;
it is not a complete numerical reproduction of every manuscript experiment.

## Included

- Maintained NPF, EEGNet, FBCNet, ATCNet and EEG Conformer training/inference interfaces.
- Source-only epoch selection followed by fresh fixed-epoch fitting and target evaluation.
- Portable accuracy, pooled NLL/ECE, temperature-scaling and method-audit scripts.
- 573 historical configuration/lock/result records, including clearly identified superseded records.
- 45 anonymous prediction groups covering 5,346 runs; complete-study aggregation covers 40 arms and eight descriptive contrasts.
- Corrected BP8–30 subject-level win/tie/loss count of 27/6/21 and portable protocol verification.
- Per-file SHA-256 manifest, source provenance and component-specific license notices.

## Validation and limits

- 22 tests passed in the documented isolated environment.
- Six re-exported OpenBMI input files matched the historical prepared inputs byte-for-byte.
- OpenBMI subject 1, seed 0 completed NPF source selection (44 selected epochs), fresh fitting, scoring and checkpoint inference.
- Four baseline interfaces completed two-epoch integration smoke runs; these are not full author-recipe evaluations.
- Historical predictions and portable temperature refitting were audited; these checks do not establish historical-engine equivalence or a full 54-subject training replication.

Raw EEG, trained weights and the EEG-CSANet-derived legacy training engine are
not distributed. The maintained workflow does not import that engine. Project-owned
code is Apache-2.0; third-party components retain their licenses, including MIT
for the pinned official FBCNet source. See LICENSE_SCOPE.md, THIRD_PARTY_NOTICES.md,
docs/TRAINING_VALIDATION.md and docs/REPRODUCTION_COVERAGE.md.

For citations use CITATION.cff and, once assigned, the version-specific Zenodo DOI
shown on the release page. No DOI is claimed before the archive record exists.
