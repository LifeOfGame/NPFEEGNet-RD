# Reproduction coverage and release checklist

| Evidence or operation | Current repository | Remaining requirement |
|---|---|---|
| Maintained NPFEEGNet-RD model | Included with synthetic tests | This is not a new replication of historical runs |
| Maintained source-only selection and fitting | Included | Match/validate against each original frozen recipe before claiming numerical replication |
| OpenBMI BP8–30 54 × 3 × 2 final-run audit | Included: summaries, epoch maps, verify.py/analyze.py | Full raw-data preprocessing/retraining path is not included |
| Original OpenBMI accuracy comparator | Run-level CSV included | Original primary-analysis entry point and prediction-level pooled NLL/ECE reconstruction remain to be released |
| FullDemean, HP0.5/HP1, duration/capacity/component controls | Not included as complete execution pipelines | Frozen configs, legal engine path and table/figure mapping |
| FBCNet, ATCNet, EEGConformer and CSP baselines | Not included | Resolve source licenses and preserve each actual recipe |
| BCIC, HGD, BNCI2015-001 and Zhou external audits | Not included | Dataset preparation, frozen configs, analyses and lawful source dependencies |
| Activation/frequency/calibration analyses | Not included | Analysis scripts and permissible prediction artifacts |

## Before a paper-associated v1.0.0

1. Complete the experiment-to-script/config/result mapping for every reported table and figure.
2. Resolve excluded source provenance: obtain permission where required or provide an independently written alternative with documented validation. Never claim new code is the old frozen engine.
3. Record actual experimental environments separately from current verification environments.
4. Verify from a fresh checkout, including pooled calibration calculations from permitted prediction files.
5. Confirm software copyright, author names/order and citation metadata; add CITATION.cff with actual facts.
6. Publish the repository as appropriate, freeze a reviewed commit in a release, and archive it with Zenodo. Cite its real version DOI.

The package version 0.2.0.dev0 marks preparation only. No formal GitHub release or DOI exists as part of this initial upload.
