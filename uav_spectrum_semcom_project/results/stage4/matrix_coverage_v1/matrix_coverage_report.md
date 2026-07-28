# Stage-4 baseline, ablation, and matrix coverage

Development complete: `True`; counts: `{'blocked': 2, 'failed': 0, 'passed': 19, 'waived': 7}`.

| ID | Scope | Status | Note |
|---|---|---|---|
| B01_REPRESENTATION | development | passed | hard semantic, strong PSD, and fixed G1/G2 were compared; G3 cost is present in C2/C3 diagnostics |
| B02_NODE_SELECTION | development | passed | no-upgrade, SNR, confidence, prior value/bit, learned/analytic, and oracle baselines |
| B03_FUSION | development | passed | classical, learned pilot, audited, and reliable-minority variants; negative results retained |
| B04_RETRANSMISSION | development | passed | no/CRC/fixed retransmission are stage-2 controls; ACK and task-conflict G3 are stage-3/4 controls |
| A01_REMOVE_RESOURCE_LOSS | development | passed | detection-only is the paired removal |
| A02_RATE_TERM | development | passed | detection+rate is a frozen negative control |
| A03_CONFIDENCE_FOR_VALUE | development | passed | confidence-G3 is compared directly |
| A04_REDUNDANCY_PENALTY | development | waived | Gate B rejected all dynamic schedulers before a full RAVES redundancy term; no full-method claim is made |
| A05_CVAR_LOSS | development | waived | tail term existed in full-joint negative control; isolated retuning was stopped by D4-003 |
| A06_RELIABLE_MINORITY | development | passed | direct protection and confirmation retransmission were both evaluated and rejected |
| A07_ANOMALY_QUALITY | development | passed | quality calibration and controlled anomaly sensitivity are reported |
| A08_ACK_FALLBACK | development | passed | ACK fallback bit/latency tradeoff is retained from the frozen stage-3 component |
| A09_FIXED_GRANULARITY | development | passed | all-G1 and all-G2 are decoder-path controls |
| A10_FROZEN_DETECTOR | development | passed | C1 uses the frozen detector cache |
| A11_JOINT_FINETUNE | development | waived | full-joint head control failed; detector unfreezing was stopped by D4-003 to avoid validation retuning |
| A12_FULL_RAVES | development | waived | C2/C3 gates failed, so no nonexistent full RAVES method is claimed |
| M01_NODE_COUNT_1_2_4_8 | development | passed | node-count pressure exists; unrelated aggregation remains H1/H4 diagnosis, not H3 |
| M02_RESOURCE_AND_DEMAND_GRID | development | passed | 4/8/16 resources and 1/2/4 contiguous demand were truth-only audited |
| M03_BUDGET_GRID | development | waived | 512-bit Gate A and local C2 budget curves were run; 1/2/4/8-kbit full-method sweep was stopped after Gate B |
| M04_SENSING_SNR_GRID | development | waived | heterogeneous -6/0/6/12 and prior robustness evidence exist; full -12-to-12 full-method sweep stopped after gates |
| M05_REPORTING_CHANNEL_GRID | development | passed | stage4 covers AWGN/Rayleigh/Rician at 0/3/6/9 dB; stage2 supplies the 12-dB reference |
| M06_FAILURE_AND_AGE | development | passed | dropout and stale-report conditions are included |
| M07_ANOMALY_GRID | development | passed | high-confidence wrong, clipping/leakage diagnostics, shift, stale, and dropout are covered |
| M08_RELIABLE_MINORITY_2V2 | development | waived | 1-v-3 family failed Gate C, so 2-v-2 expansion was not used for threshold search |
| M09_FEC_AND_RETRANSMISSION | development | passed | Hamming and LDPC reference paths and 0/1/2 retransmission controls are stage-2 frozen components |
| M10_MODEL_COMPLEXITY | development | passed | parameters, linear MACs, checkpoint bytes, and CPU latency are reported |
| F04_FULL_FINAL_MATRIX | final | blocked | new independent final scenes do not exist; only preregistered C1 and selective-G2 primary families will run |
| F05_REAL_H3_MATRIX | final | blocked | 30+ real multi-receiver scenes and H3 comparison set are unavailable |

> Waived means a preregistered gate stopped that branch; it is not a positive result. Blocked items require independent final data.
