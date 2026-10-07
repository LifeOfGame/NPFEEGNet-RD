# Anonymous prediction evidence

45 stored groups contain 5,346 runs and 1,074,447 prediction rows. They include
original OpenBMI comparisons, later filtering/capacity/component controls,
boundary and duration audits, and external dataset runs. These are not 45
independent datasets or 5,346 independent statistical subjects.

Each NPZ contains only anonymous within-dataset subject IDs, seed, within-run
trial index, integer target label, predicted label and class probabilities.
No EEG, learned features or model weights are distributed. The manifest records
the source relative path and hashes of preserved arrays. A trial index is an
export-order index, not an independently verified offline/online phase label.

Run from a checkout with NumPy installed:

```bash
python scripts/audit_predictions.py --output /path/to/new-audit-directory
python -m unittest discover -s tests -p test_prediction_audit.py -v
```

Accuracy averages seeds within each subject and then subjects. NLL and ECE
pool all rows within a group; ECE uses 10 equal-width bins (lo, hi], and NLL
clips selected probabilities at 1e-12. Values are descriptive, with no new
inferential claim. `single_class_prediction_runs` means exactly one predicted
class; it must not silently replace another historical collapse definition.

Some groups are GPU partitions of the same experiment; combine their disjoint
subject/seed runs before computing an overall result. Zhou2020 `s` and `a`
cohorts have overlapping integer IDs: keep cohort prefixes when combining
them. A simple average of group metrics is not the complete-study statistic.
The original primary comparison is the two `paper-20260917__openbmi_final_npf`
groups. Later `methods-audit`, `common-duration` and filtered groups are distinct
controls, not replacements for that locked primary result.

Not every paper experiment has predictions here. Baseline recipe extensions,
temperature source-fold exports and older developmental results require their
own records. See the coverage document; do not infer completeness from size.
