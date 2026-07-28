# Stage 3 DeepSets Fusion Pilot

Source frames are disjoint across train/validation/test. Scenes are controlled max-hold composites and therefore remain semi-synthetic.

| Method | Clean rate | Regret | Oracle-equivalent | bit/decision |
|---|---:|---:|---:|---:|
| majority | 0.7100 | 0.013941 | 0.4950 | 3374.3 |
| mean | 0.8050 | 0.008464 | 0.6400 | 3374.3 |
| or | 0.7900 | 0.009077 | 0.6300 | 3374.3 |
| snr_weighted | 0.8050 | 0.008464 | 0.6400 | 3374.3 |
| quality_weighted | 0.8000 | 0.008659 | 0.6350 | 3374.3 |
| deepset_resource_fusion | 0.7350 | 0.013131 | 0.5250 | 3374.3 |

The learned method and classical methods consume the same realized node reports. This pilot does not establish real UAV or synchronized-SDR performance.
