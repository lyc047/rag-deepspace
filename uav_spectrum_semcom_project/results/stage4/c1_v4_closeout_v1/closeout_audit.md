# 阶段4收尾审计

- 总状态：`stage4_closeout_complete`
- 通过：`True`

| 项目 | 通过 | 证据 |
|---|---|---|
| immutable_snapshot | True | changed=[] |
| single_use_access | True | 2ac71dcf093ea14e0d06b7454e58c7b508f19291f222c4079f8b2df8f0393d96 |
| registered_independence | True | 280 scenes, 74 clusters, overlap 0 |
| H1_confirmed | True | {'mean_regret_superiority': True, 'cvar_superiority': True, 'actual_bit_superiority': True} |
| H2_confirmed | True | {'mean_regret_superiority': True, 'cvar_superiority': True, 'actual_bits_exactly_equal': True} |
| overall_final | True | c1_v4_single_use_temporal_final_complete |
| paper_assets_integrity | True | sources=True, outputs=True, files=6 |
| authoritative_docs | True | docs=4 |
| claim_boundary | True | Single-use temporal confirmation within the same AERPAW February 2022 fixed-site power-sweep campaign at two sites. It is not cross-dataset evidence, UAV mobility, raw-IQ detection, MIMO, or a measured reporting link. |
