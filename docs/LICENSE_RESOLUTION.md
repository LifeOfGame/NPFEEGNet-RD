# Source and license resolution, 7 October 2026

| Component | Evidence / decision | Current executable dependency |
|---|---|---|
| EEG-CSANet-derived historical engine | No explicit license located at checked HEAD `117095fd3bcc3e7ad41b8cf00334807cfc945ccf`; excluded, permission not obtained | None in maintained training |
| Maintained training/data engine | Separately maintained source under repository scope; uses NumPy/PyTorch, not legacy engine imports | `training/`, `train.py`, `predict.py` |
| EEGNet control | New native-PyTorch assembly of cited topology in `model/factory.py`; no extraction from unresolved mixed module | Independent assembly |
| FBCNet control | Official MIT source pinned at `de1bbdd8a54cb1e466830e3d47070e0e56761a37`; cached checkout clean and matched remote HEAD | `baselines/vendor_fbcnet.py` plus new documented filter/input adapter |
| ATCNet control | Existing Apache-2.0 adaptation pinned at `d8fb0c10abc14b796dfbd317f53f56a68ebe29df`, copyright preserved | `baselines/openbmi_atcnet.py` |
| EEG Conformer control | Existing paper-based independent native-PyTorch implementation, provenance statement retained | `baselines/openbmi_eegconformer.py` |
| Historical mixed-baseline module | Historical upstream revision/ownership not fully established; not relicensed or published | None in maintained training |

The new FBCNet dependency is pinned and auditable. This does **not** establish
which upstream revision was used when the older mixed baseline was first
developed. The maintained factory must not be presented as that historical
file or as exact reproduction of original-paper training recipes.

The training protocol is now `npfeegnet-maintained-v2`. Selection records from
other protocols are rejected. Existing maintained-v1 model checkpoints remain
loadable for inference, while new selection/fitting requires the current source
hashes. Adam/AdamW and cosine/constant scheduling are explicit saved settings.
NPF branch auxiliary losses are retained; single-output baselines use ordinary
cross entropy without NPF branch losses.

## Unresolved permission versus removed dependency

Removing a dependency allows the maintained workflow to run without redistributing
that legacy code. It does not retroactively authorize the legacy files or prove
the new trainer numerically equivalent. Historical artifacts remain evidence of
the old runs; new integration results are separately labelled.

Release decision on 2026-10-08: do not redistribute the legacy engine. Preserve
the original privately for historical records, and distribute the maintained
implementation and permitted evidence. No author has been contacted and no
permission request is being pursued for this release. Authorization to distribute
excluded files is not claimed. Remaining reproduction gaps are documented
separately; the private engine is not a required public-release deliverable.

Sources checked:
- https://github.com/Xiangrui-Cai/EEG-CSANet/tree/117095fd3bcc3e7ad41b8cf00334807cfc945ccf
- https://github.com/ravikiran-mane/FBCNet/tree/de1bbdd8a54cb1e466830e3d47070e0e56761a37
- https://github.com/ravikiran-mane/FBCNet/blob/de1bbdd8a54cb1e466830e3d47070e0e56761a37/LICENSE
- https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository
