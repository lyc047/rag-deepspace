# Stage-4 quality diagnostics

Validation rows: 600; final holdout accessed: false.

## Calibration

| Metric | Raw | Calibrated |
|---|---:|---:|
| Brier | 0.179472 | 0.000421 |
| ECE | 0.422625 | 0.002756 |

## Controlled anomaly sensitivity

| Fault | Pairwise detection | Score AUC |
|---|---:|---:|
| clipping | 1.0000 | 0.6614 |
| out_of_band_leakage | 1.0000 | 0.6906 |
| noise_instability | 1.0000 | 0.8079 |
| stale_report | 1.0000 | 1.0000 |
| report_failure_risk | 1.0000 | 0.7787 |
| calibration_failure | 1.0000 | 0.6906 |
| high_confidence_miscalibrated | 1.0000 | 0.6806 |

> Calibration uses calibration truth and reports validation aggregates only. Injected faults validate score sensitivity, not real-world anomaly prevalence or H3.
