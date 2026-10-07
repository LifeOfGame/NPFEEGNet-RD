# Evidence navigation

All paths below are relative to `evidence/historical/`. Original records are
preserved byte-for-byte and may describe the server's absolute paths. Use the
manifest for source attribution. This index is by experiment, not a definitive
numbered table/figure registry for an unspecified manuscript revision.

| Question | Historical directory / primary record |
|---|---|
| Original OpenBMI RD vs NoDemean | `paper-20260917/results/openbmi_final_3seed_report.json` |
| EEGNet 2x2, matched ATCNet/Conformer | `paper-20260917/results/openbmi_*extension*` and `openbmi_baseline_2x2*` |
| FullDemean / 1-Hz epoch filter | `paper-20260917/results/openbmi_preprocessing_controls*` |
| Corrected BCIC-IV-2a [2,6) s | `baseline-audit-20260922/results/cue_aligned_three_model_report.json` |
| FBCNet filter implementation | `openbmi-fbcnet-audit-20260922/` |
| Stronger ATCNet training recipe | `sentinel-audit-20260923/sentinel/full54_design_lock.json` and corresponding results |
| HP0.5, capacity, FBCSP, BNCI2015-001 | `reviewer-controls-20260925/` |
| Spectral-only/no-prior, stronger FBCNet | `component-author-20260926/results/component_author_report.json` |
| Source-fitted temperature | `temperature-20260926/` |
| Activation and frequency permutation | `mechanism-followup-20260927/` |
| Padding/FFT boundary and statistical audits | `methods-audit-20260928/` |
| Shared training duration | `common-duration-20260928/` |
| Continuous 8-30-Hz control | `bp830-20260929/`; corrected portable analysis is `evidence/bp830_protocol/` |
| External Zhou cohorts | `zhou2016-20260927/`, `zhou2020-20260927/` |
| Later Zhou diagnostic controls | `zhou2020-posthoc-20260927/` (not original independent confirmation) |

The original BP8-30 historical reports retain the floating-point W/T/L error.
Use the corrected portable analysis: **27/6/21**, not 27/5/22. Historical files
are kept unchanged for provenance, not endorsed as corrected final reports.
Likewise, early 2a developmental summaries do not override corrected cue-aligned
results. Retain all evidence-role distinctions in the manuscript.
