# Stage-4 acceptance audit

- Development acceptance: `True`
- Final acceptance: `False`

| ID | Scope | Status | Note |
|---|---|---|---|
| A01_PROTOCOL | development | passed | frozen protocol is valid |
| A02_PROTOCOL_REGISTRY | development | passed | protocol-only registry is structurally valid |
| A03_C1_GATE_A | development | passed | C1 validation Gate A passed at matched nominal bits; final H2 remains untested |
| A04_C2_GATE_B_NEGATIVE | development | passed | Gate B investigated with learned, analytic, and preview variants; rejected as a documented negative result |
| A05_C3_GATE_C_NEGATIVE | development | passed | controlled C3 robustness completed; Gate C rejected and no H3 spatial claim is made |
| A06_END_TO_END_DECODER | development | passed | 150-scene decoder-path sweep covers 3 channels x 4 Eb/N0 values; selective G2 saves actual transmitted bits |
| A07_QUALITY_DIAGNOSTICS | development | passed | quality calibration and controlled anomaly diagnostics are recorded |
| A08_VALUE_RANKING_DIAGNOSTICS | development | passed | aggregate MAE, rank correlation, top-k recall, and scheduler effect are recorded |
| A09_FINAL_PREREGISTRATION | development | passed | single-use final evaluation, primary families, multiplicity control, and acceptance thresholds are frozen |
| F01_INDEPENDENT_FINAL_SCENES | final | passed | registered independent final scenes: 200/200 |
| F02_SINGLE_USE_FINAL_EVALUATION | final | blocked | cannot run before an eligible independent catalog is registered |
| F03_H3_SPATIAL_COLLABORATION | claim | waived | no formal H3 claim is made; controlled faults are reported only as mechanism/negative evidence |
| A10_PRE_FINAL_SNAPSHOT | development | passed | executable source, configuration, and frozen checkpoint hashes are recorded with passing regression evidence |
| A11_COMPLEXITY_PROFILE | development | passed | C1/C2 parameter count, linear MACs, checkpoint size, and single-thread CPU latency are recorded |
| A12_MATRIX_COVERAGE | development | passed | baseline, ablation, and sweep coverage has no unaccounted development failures; gate-conditioned waivers are explicit |
| A13_THESIS_MATERIALS | development | passed | generated figures/tables, chapter draft, claim matrix, and locked final template pass integrity and claim-boundary checks |
| A14_FINAL_STATISTICS_ENGINE | development | passed | frozen scene-level bootstrap, paired-randomization, noninferiority, Holm, success/failure, and integrity-rejection branches are verified |
| A15_VERIFIED_RELATED_WORK | development | passed | recent related-work metadata and project-boundary comparisons are verified against primary publisher, institution, CVF, or arXiv pages |

> Development acceptance is not final H2/H3 evidence.
