"""Build paid scheduler-only residual previews and train feature labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR=Path(__file__).resolve().parents[1]
for path in (PROJECT_DIR/"src",PROJECT_DIR/"scripts"):
    if str(path) not in sys.path: sys.path.insert(0,str(path))

from run_stage2_digital_link import build_link_config
from spectrum_semcom.counterfactual_value import candidate_message_cost
from spectrum_semcom.digital_link import expected_transmitted_bits_awgn
from spectrum_semcom.multigranular_semantics import SemanticQuality
from spectrum_semcom.reproducibility import environment_snapshot,sha256_file
from spectrum_semcom.scheduler_preview import build_scheduler_preview,decode_scheduler_preview,encode_scheduler_preview
from spectrum_semcom.stage4_protocol import assert_valid_stage4_protocol


SKETCH_AVAILABILITY=np.asarray([True,True,False,True,False,False,False,True,True,False],dtype=bool)


def load(path: Path) -> dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as data: return {key:data[key] for key in data.files}


def build_split(cache: dict[str,np.ndarray],links:tuple,top_k:int,magnitude_bits:int,selected_nodes:set[int] | None=None) -> dict[str,np.ndarray]:
    scenes,nodes,channels=cache["node_occupancy"].shape
    reports=np.empty((scenes,nodes,channels),dtype=np.float32); qualities=np.zeros((scenes,nodes,10),dtype=np.float32); application=np.empty((scenes,nodes),dtype=np.int32); expected=np.empty((scenes,nodes),dtype=np.float32)
    for scene in range(scenes):
        for node in range(nodes):
            original=cache["node_quality"][scene,node]
            if selected_nodes is None or node in selected_nodes:
                preview=build_scheduler_preview(cache["node_occupancy"][scene,node],node,str(cache["scene_ids"][scene]),float(original[0]),float(original[1]),float(original[3]),float(original[7]),top_k,magnitude_bits)
                encoded=encode_scheduler_preview(preview); decoded=decode_scheduler_preview(encoded.bits)
                reports[scene,node]=decoded.scheduler_report; qualities[scene,node,[0,1,3,7,8]]=[decoded.sensing_snr_db,decoded.prediction_confidence,decoded.clipping_ratio,decoded.age_s,float(original[8])]; application[scene,node]=encoded.application_bits; expected[scene,node]=expected_transmitted_bits_awgn(encoded.application_bits,links[node])
            else:
                reports[scene,node]=np.rint(cache["node_occupancy"][scene,node]); qualities[scene,node,[0,8]]=[float(original[0]),float(original[8])]
                semantic_quality=SemanticQuality(*[float(value) for value in original]); application_bits,_,expected_bits=candidate_message_cost(str(cache["scene_ids"][scene]),node,1,reports[scene,node],semantic_quality,links[node],probability_bits=1); application[scene,node]=application_bits; expected[scene,node]=expected_bits
    return {"scene_ids":cache["scene_ids"],"scheduler_report":reports,"scheduler_quality":qualities,"quality_availability":SKETCH_AVAILABILITY,"application_bits":application,"expected_transmitted_bits":expected}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--protocol",type=Path,default=PROJECT_DIR/"configs"/"stage4_protocol.json"); parser.add_argument("--stage2-config",type=Path,default=PROJECT_DIR/"configs"/"stage2_digital_link.json"); parser.add_argument("--cache-result",type=Path,default=PROJECT_DIR/"results"/"stage4"/"multinode_cache_v1"/"multinode_cache_result.json"); parser.add_argument("--base-labels",type=Path,default=PROJECT_DIR/"results"/"stage4"/"counterfactual_labels_v1"/"train_counterfactual_labels.npz"); parser.add_argument("--selected-nodes",type=str); parser.add_argument("--out-dir",type=Path,default=PROJECT_DIR/"results"/"stage4"/"scheduler_preview_v1"); args=parser.parse_args()
    if not args.out_dir.is_absolute(): args.out_dir=PROJECT_DIR/args.out_dir
    protocol=json.loads(args.protocol.read_text(encoding="utf-8")); assert_valid_stage4_protocol(protocol); settings=protocol["scheduler_preview_sketch_protocol"]; stage2=json.loads(args.stage2_config.read_text(encoding="utf-8")); meta=json.loads(args.cache_result.read_text(encoding="utf-8")); links=tuple(build_link_config(stage2,float(value)) for value in protocol["counterfactual_value_protocol"]["reporting_ebn0_db"]); selected_nodes=None if not args.selected_nodes else {int(value) for value in args.selected_nodes.split(",")}; args.out_dir.mkdir(parents=True,exist_ok=True); split_records={}; built={}
    for split in ("train","calibration","validation"):
        cache=load(PROJECT_DIR/meta["splits"][split]["cache"]); arrays=build_split(cache,links,int(settings["sketch_top_k"]),int(settings["residual_magnitude_bits"]),selected_nodes); path=args.out_dir/f"scheduler_preview_{split}.npz"; np.savez_compressed(path,**arrays); built[split]=arrays; split_records[split]={"scene_count":len(arrays["scene_ids"]),"cache":str(path.relative_to(PROJECT_DIR)).replace("\\","/"),"sha256":sha256_file(path),"mean_application_bits_per_node":float(arrays["application_bits"].mean()),"mean_expected_bits_all_nodes":float(arrays["expected_transmitted_bits"].sum(axis=1).mean())}
    labels=load(args.base_labels); train=built["train"]; scene_index={str(scene):index for index,scene in enumerate(train["scene_ids"])}; candidate_report=[]; candidate_quality=[]; preview_cost=[]
    for scene,node in zip(labels["scene_id"],labels["candidate_node"]):
        index=scene_index[str(scene)]; candidate_report.append(train["scheduler_report"][index,int(node)]); candidate_quality.append(train["scheduler_quality"][index,int(node)]); preview_cost.append(train["expected_transmitted_bits"][index,int(node)])
    sketch_labels=dict(labels); sketch_labels["candidate_current_report"]=np.asarray(candidate_report,dtype=np.float32); sketch_labels["candidate_quality"]=np.asarray(candidate_quality,dtype=np.float32); sketch_labels["quality_availability"]=SKETCH_AVAILABILITY; sketch_labels["candidate_scheduler_preview_expected_bits"]=np.asarray(preview_cost,dtype=np.float32)
    label_path=args.out_dir/"train_counterfactual_labels_with_sketch.npz"; np.savez_compressed(label_path,**sketch_labels)
    train_reports=train["scheduler_report"]; pair=[]
    for scene in train_reports:
        pair.extend([float(np.mean(np.abs(scene[left]-scene[right]))) for left in range(scene.shape[0]) for right in range(left+1,scene.shape[0])])
    result={"experiment_id":"stage4_scheduler_preview_top2_v1","protocol_sha256":sha256_file(args.protocol),"base_labels_sha256":sha256_file(args.base_labels),"sketch_labels":str(label_path.relative_to(PROJECT_DIR)).replace("\\","/"),"sketch_labels_sha256":sha256_file(label_path),"splits":split_records,"quality_availability":SKETCH_AVAILABILITY.tolist(),"train_scheduler_report_audit":{"all_nodes_identical_fraction":float(np.mean([len({tuple(row) for row in scene})==1 for scene in train_reports])),"mean_pair_absolute_difference":float(np.mean(pair)),"nonzero_pair_fraction":float(np.mean(np.asarray(pair)>1e-12))},"truth_access_audit":{"truth_read_for_preview_generation":False,"base_train_labels_reused":True,"calibration_validation_labels_saved":False,"final_holdout_accessed":False},"environment":environment_snapshot(["numpy"]),"claim_boundary":"Scheduler preview development artifact; not final H2/H3 evidence."}; (args.out_dir/"scheduler_preview_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    result["selected_nodes"]=None if selected_nodes is None else sorted(selected_nodes); result["experiment_id"]="stage4_scheduler_preview_selective_v1" if selected_nodes is not None else "stage4_scheduler_preview_top2_v1"; (args.out_dir/"scheduler_preview_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# Stage 4 Scheduler Preview Cache","",f"Selected sketch nodes: {result['selected_nodes']}.","",f"Application bits/node: {split_records['train']['mean_application_bits_per_node']:.1f}; expected all-node preview bits: {split_records['train']['mean_expected_bits_all_nodes']:.2f}.","",f"Train all-node-identical scenes: {result['train_scheduler_report_audit']['all_nodes_identical_fraction']:.4f}; nonzero pair fraction: {result['train_scheduler_report_audit']['nonzero_pair_fraction']:.4f}; mean pair difference: {result['train_scheduler_report_audit']['mean_pair_absolute_difference']:.6f}.","","Sketch generation used no truth. Calibration/validation labels and final holdout were not accessed."]; (args.out_dir/"scheduler_preview_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); print(args.out_dir/"scheduler_preview_report.md")


if __name__=="__main__": main()
