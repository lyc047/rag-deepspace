"""Run scene-by-seed paired bootstrap on frozen Gate A checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
for path in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

from spectrum_semcom.gate_a_analysis import hierarchical_paired_bootstrap
from spectrum_semcom.gate_a_model import VariableRateOccupancyHead
from spectrum_semcom.gate_a_training import load_gate_a_cache, nominal_transmitted_bits
from spectrum_semcom.resource_losses import discrete_occupancy_regret
from spectrum_semcom.reproducibility import sha256_file
from run_stage2_digital_link import build_link_config


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--protocol",type=Path,default=PROJECT_DIR/"configs"/"stage4_protocol.json"); parser.add_argument("--stage2-config",type=Path,default=PROJECT_DIR/"configs"/"stage2_digital_link.json"); parser.add_argument("--training-dir",type=Path,default=PROJECT_DIR/"results"/"stage4"/"gate_a_training_v1"); parser.add_argument("--bootstrap-repetitions",type=int,default=10000); args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text(encoding="utf-8")); settings=protocol["gate_a_controlled_training"]; stage2=json.loads(args.stage2_config.read_text(encoding="utf-8")); link=build_link_config(stage2,6.0)
    cache_meta_path=PROJECT_DIR/settings["cache_result"]; cache_meta=json.loads(cache_meta_path.read_text(encoding="utf-8")); task=cache_meta["task"]
    scene_ids,base,truth,_=load_gate_a_cache(PROJECT_DIR/cache_meta["splits"]["validation"]["cache"])
    methods=settings["methods"]; seeds=settings["training_seeds"]; regrets={method:[] for method in methods}; bits={method:[] for method in methods}
    for method in methods:
        for seed in seeds:
            model=VariableRateOccupancyHead(int(task["n_channels"]),int(settings["hidden_dim"])); checkpoint=args.training_dir/"checkpoints"/f"{method}_seed{seed}.pt"; model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)); model.eval()
            reconstructed,_,application=model.infer(base); regrets[method].append(discrete_occupancy_regret(reconstructed,truth,int(task["demand_channels"]),reduction="none").numpy()); bits[method].append(np.asarray([nominal_transmitted_bits(int(value),link) for value in application]))
    comparisons=[]; baseline="detection_only"
    for method in methods[1:]:
        proposed_regret=np.stack(regrets[method]); base_regret=np.stack(regrets[baseline]); proposed_bits=np.stack(bits[method]); base_bits=np.stack(bits[baseline])
        comparisons.append({"method":method,"regret":hierarchical_paired_bootstrap(proposed_regret,base_regret,repetitions=args.bootstrap_repetitions),"cvar_0_9":hierarchical_paired_bootstrap(proposed_regret,base_regret,statistic="cvar",cvar_alpha=0.9,repetitions=args.bootstrap_repetitions,seed=20260717),"nominal_bits":hierarchical_paired_bootstrap(proposed_bits,base_bits,repetitions=args.bootstrap_repetitions,seed=20260718)})
    resource=next(item for item in comparisons if item["method"]=="detection_plus_resource"); decision="retain_resource_aware_loss_for_C1" if resource["regret"]["ci95_high"]<0 else "do_not_claim_gate_a_success"
    result={"experiment_id":"stage4_gate_a_hierarchical_paired_analysis_v1","training_result_sha256":sha256_file(args.training_dir/"gate_a_training_result.json"),"validation_scene_ids":scene_ids,"validation_scene_count":len(scene_ids),"training_seed_count":len(seeds),"comparisons":comparisons,"decision":decision,"final_holdout_accessed":False,"claim_boundary":"Validation decision only; final H2 remains untested."}
    (args.training_dir/"gate_a_paired_analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# Stage 4 Gate A Paired Hierarchical Analysis","",f"{len(seeds)} training seeds x {len(scene_ids)} paired validation scenes; {args.bootstrap_repetitions} bootstrap repetitions.","","| Method vs detection-only | Regret delta [95% CI] | CVaR0.9 delta [95% CI] | nominal bit delta [95% CI] |","|---|---:|---:|---:|"]
    for item in comparisons:
        r=item["regret"]; c=item["cvar_0_9"]; b=item["nominal_bits"]; lines.append(f"| {item['method']} | {r['difference']:.6f} [{r['ci95_low']:.6f}, {r['ci95_high']:.6f}] | {c['difference']:.6f} [{c['ci95_low']:.6f}, {c['ci95_high']:.6f}] | {b['difference']:.2f} [{b['ci95_low']:.2f}, {b['ci95_high']:.2f}] |")
    lines += ["",f"Decision: `{decision}`.","","This is validation-only evidence; no final holdout was created or accessed."]
    (args.training_dir/"gate_a_paired_analysis.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); print(args.training_dir/"gate_a_paired_analysis.md")


if __name__=="__main__": main()
