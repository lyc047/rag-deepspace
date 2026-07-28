"""Generate train-only C2 labels and validation-only aggregate oracle diagnostics."""

from __future__ import annotations

import argparse,json,sys
from pathlib import Path

import numpy as np

PROJECT_DIR=Path(__file__).resolve().parents[1]
for path in (PROJECT_DIR/"src",PROJECT_DIR/"scripts"):
    if str(path) not in sys.path: sys.path.insert(0,str(path))

from spectrum_semcom.counterfactual_value import evaluate_one_step_oracle, generate_counterfactual_labels
from spectrum_semcom.multigranular_semantics import SemanticQuality
from spectrum_semcom.reproducibility import environment_snapshot,sha256_file,sha256_strings
from spectrum_semcom.stage4_protocol import assert_valid_stage4_protocol
from run_stage2_digital_link import build_link_config

def load(path: Path):
    with np.load(path,allow_pickle=False) as data: return {key:data[key] for key in data.files}

def qualities(row: np.ndarray) -> tuple[SemanticQuality,...]:
    return tuple(SemanticQuality(*[float(value) for value in node]) for node in row)

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--protocol",type=Path,default=PROJECT_DIR/"configs"/"stage4_protocol.json"); parser.add_argument("--stage2-config",type=Path,default=PROJECT_DIR/"configs"/"stage2_digital_link.json"); parser.add_argument("--cache-result",type=Path,default=PROJECT_DIR/"results"/"stage4"/"multinode_cache_v1"/"multinode_cache_result.json"); parser.add_argument("--preview-bits",type=int,choices=(1,2),default=1); parser.add_argument("--out-dir",type=Path,default=PROJECT_DIR/"results"/"stage4"/"counterfactual_labels_v1"); args=parser.parse_args()
    if not args.out_dir.is_absolute(): args.out_dir=PROJECT_DIR/args.out_dir
    protocol=json.loads(args.protocol.read_text(encoding="utf-8")); assert_valid_stage4_protocol(protocol); settings=protocol["counterfactual_value_protocol"]; stage2=json.loads(args.stage2_config.read_text(encoding="utf-8")); meta=json.loads(args.cache_result.read_text(encoding="utf-8")); task=meta["task"]
    links=tuple(build_link_config(stage2,float(value)) for value in settings["reporting_ebn0_db"]); train=load(PROJECT_DIR/meta["splits"]["train"]["cache"]); rows=[]
    for scene_index,scene_id in enumerate(train["scene_ids"].tolist()):
        rows.extend(generate_counterfactual_labels(split_role="train",scene_id=str(scene_id),node_occupancy=train["node_occupancy"][scene_index],truth_occupancy=train["truth_occupancy"][scene_index],qualities=qualities(train["node_quality"][scene_index]),link_configs=links,demand_channels=int(task["demand_channels"]),max_upgraded_nodes=int(settings["maximum_current_upgraded_nodes"]),preview_probability_bits=args.preview_bits))
    arrays={"scene_id":np.asarray([row.scene_id for row in rows]),"current_state":np.asarray([row.current_state for row in rows],dtype=np.int8),"candidate_node":np.asarray([row.candidate_node for row in rows],dtype=np.int8),"target_granularity":np.asarray([row.target_granularity for row in rows],dtype=np.int8),"current_belief":np.asarray([row.current_belief for row in rows],dtype=np.float32),"candidate_current_report":np.asarray([row.candidate_current_report for row in rows],dtype=np.float32),"candidate_report":np.asarray([row.candidate_report for row in rows],dtype=np.float32),"candidate_quality":np.asarray([row.candidate_quality for row in rows],dtype=np.float32),"quality_availability":train["quality_availability"].astype(bool),"baseline_regret":np.asarray([row.baseline_regret for row in rows],dtype=np.float32),"upgraded_regret":np.asarray([row.upgraded_regret for row in rows],dtype=np.float32),"marginal_value":np.asarray([row.marginal_value for row in rows],dtype=np.float32),"application_bits":np.asarray([row.application_bits for row in rows],dtype=np.int32),"nominal_transmitted_bits":np.asarray([row.nominal_transmitted_bits for row in rows],dtype=np.int32),"expected_transmitted_bits":np.asarray([row.expected_transmitted_bits for row in rows],dtype=np.float32),"value_per_expected_bit":np.asarray([row.value_per_expected_bit for row in rows],dtype=np.float32)}
    args.out_dir.mkdir(parents=True,exist_ok=True); label_path=args.out_dir/"train_counterfactual_labels.npz"; np.savez_compressed(label_path,**arrays)
    validation=load(PROJECT_DIR/meta["splits"]["validation"]["cache"]); oracle=[evaluate_one_step_oracle(validation["node_occupancy"][index],validation["truth_occupancy"][index],int(task["demand_channels"]),args.preview_bits) for index in range(len(validation["scene_ids"]))]
    values=arrays["marginal_value"]; result={"experiment_id":"stage4_c2_counterfactual_labels_v1","protocol_sha256":sha256_file(args.protocol),"stage2_config_sha256":sha256_file(args.stage2_config),"multinode_cache_result_sha256":sha256_file(args.cache_result),"label_file":str(label_path.relative_to(PROJECT_DIR)).replace("\\","/"),"label_sha256":sha256_file(label_path),"train_scene_ids_sha256":sha256_strings(str(x) for x in train["scene_ids"]),"train_scene_count":len(train["scene_ids"]),"train_label_count":len(rows),"labels_per_scene":len(rows)/len(train["scene_ids"]),"truth_access_audit":{"saved_supervision_split":"train","calibration_labels_saved":False,"validation_labels_saved":False,"final_holdout_accessed":False,"candidate_high_precision_report_is_supervision_only":True},"deployable_feature_schema":["current_belief","candidate_current_report","available_candidate_quality","current_state","candidate_node","target_granularity","expected_transmitted_bits"],"label_statistics":{"positive_fraction":float(np.mean(values>1e-12)),"zero_fraction":float(np.mean(np.abs(values)<=1e-12)),"negative_fraction":float(np.mean(values< -1e-12)),"mean_marginal_value":float(np.mean(values)),"mean_expected_transmitted_bits":float(np.mean(arrays["expected_transmitted_bits"]))},"validation_one_step_oracle":{"scene_count":len(oracle),"mean_baseline_regret":float(np.mean([x.baseline_regret for x in oracle])),"mean_best_regret":float(np.mean([x.best_upgraded_regret for x in oracle])),"mean_regret_reduction":float(np.mean([x.regret_reduction for x in oracle])),"improved_scene_fraction":float(np.mean([x.regret_reduction>1e-12 for x in oracle]))},"environment":environment_snapshot(["numpy"]),"claim_boundary":"Train-only labels; validation aggregate oracle diagnostic only; no final evidence."}; (args.out_dir/"counterfactual_label_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    result["experiment_id"]=f"stage4_c2_counterfactual_labels_G1q{args.preview_bits}"; result["preview_probability_bits"]=args.preview_bits; (args.out_dir/"counterfactual_label_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# Stage 4 C2 Counterfactual Label Audit","",f"G1 preview probability bits: {args.preview_bits}.","",f"Train scenes: {result['train_scene_count']}; labels: {result['train_label_count']}; labels/scene: {result['labels_per_scene']:.1f}.","",f"Positive/zero/negative label fractions: {result['label_statistics']['positive_fraction']:.4f} / {result['label_statistics']['zero_fraction']:.4f} / {result['label_statistics']['negative_fraction']:.4f}.","",f"Validation one-step oracle: baseline regret {result['validation_one_step_oracle']['mean_baseline_regret']:.6f}, best-upgrade regret {result['validation_one_step_oracle']['mean_best_regret']:.6f}, improved scenes {result['validation_one_step_oracle']['improved_scene_fraction']:.4f}.","","Saved supervision uses train truth only. No calibration/validation labels or final holdout were saved/accessed."]; (args.out_dir/"counterfactual_label_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); print(args.out_dir/"counterfactual_label_report.md")

if __name__=="__main__": main()
