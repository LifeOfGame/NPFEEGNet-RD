# License scope

The existing Apache License 2.0 is retained for the project's own source and documentation to the extent its copyright holders have the right to license them: model/, training/, train.py, predict.py, the included tests/configurations, and newly authored packaging/verification documentation and scripts. It does not license EEG recordings, participant data, external dependencies or third-party material. Author identities and final publication metadata remain to be confirmed.

The supplied run-level evidence is historical scientific output; preserving it here does not imply a new experiment or a new license for the originating datasets. No raw EEG or checkpoints are distributed.

Legacy EEG-CSANet-derived engines and unresolved mixed-provenance baseline implementations are not included. This license must not be interpreted as a grant to those excluded files. See docs/REPRODUCTION_COVERAGE.md.

Added experiment-specific preprocessing, analysis and wrapper code is covered
only to the extent of project-owned authorship; it conveys no license to imported
legacy dependencies. ATCNet adaptation retains its upstream Apache-2.0 notice.
The independent Conformer implementation retains its paper/source provenance.
Anonymous classifier outputs are scientific evidence, not raw EEG redistribution.
