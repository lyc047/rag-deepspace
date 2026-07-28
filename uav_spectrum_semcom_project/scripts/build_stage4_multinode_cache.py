"""Build C2 same-scene controlled multi-node caches from the frozen registry."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR=Path(__file__).resolve().parents[1]
for path in (PROJECT_DIR/"src",PROJECT_DIR/"scripts"):
    if str(path) not in sys.path: sys.path.insert(0,str(path))

from spectrum_semcom.gate_a_data import pool_probability_mask, validate_multinode_cache_arrays
from spectrum_semcom.gate_a_model import VariableRateOccupancyHead
from spectrum_semcom.multi_uav_fusion import perturb_correlated_spectrogram
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file, sha256_strings
from spectrum_semcom.stage4_development import validate_development_registry
from spectrum_semcom.stage4_protocol import assert_valid_stage4_protocol
from build_stage4_gate_a_cache import infer_masks, scene_arrays

QUALITY_NAMES=["sensing_snr_db","prediction_confidence","normalized_entropy","clipping_ratio","out_of_band_leakage_ratio","noise_floor_stability_db","peak_to_background_db","age_s","report_success_probability","calibration_error"]
QUALITY_AVAILABILITY=np.asarray([True,True,True,True,False,False,True,True,True,False],dtype=bool)

def quality_vector(image: np.ndarray, occupancy: np.ndarray, snr: float, age: float, ebn0: float) -> np.ndarray:
    p=np.clip(occupancy,1e-7,1-1e-7); entropy=float(np.mean(-(p*np.log2(p)+(1-p)*np.log2(1-p)))); power=np.square(image.astype(np.float64)/255.0); background=float(np.median(power)); peak=float(np.max(power)); peak_db=0.0 if peak<=1e-20 else 10*np.log10(peak/max(background,1e-20)); reliability=float(1/(1+np.exp(-(ebn0-3)/2)))
    return np.asarray([snr,1-entropy,entropy,float(np.mean(image>=254)),0.0,0.0,peak_db,age,reliability,0.0],dtype=np.float32)

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--protocol",type=Path,default=PROJECT_DIR/"configs"/"stage4_protocol.json"); parser.add_argument("--registry",type=Path,default=PROJECT_DIR/"configs"/"stage4_development_registry_v1.json"); parser.add_argument("--root",type=Path,default=PROJECT_DIR/"data"/"raw"/"raddet"/"RadDet40k128HW001Tv2"); parser.add_argument("--detector-checkpoint",type=Path,default=PROJECT_DIR/"results"/"phase1"/"raddet_occupancy_mask.pt"); parser.add_argument("--out-dir",type=Path,default=PROJECT_DIR/"results"/"stage4"/"multinode_cache_v1"); parser.add_argument("--device",default="auto"); args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text(encoding="utf-8")); assert_valid_stage4_protocol(protocol); settings=protocol["counterfactual_value_protocol"]; registry=json.loads(args.registry.read_text(encoding="utf-8")); errors=validate_development_registry(registry)
    if errors: raise ValueError("invalid registry: "+"; ".join(errors))
    task=registry["selected_task"]; n_channels=int(task["n_channels"]); n_nodes=int(settings["node_count"]); device="cuda" if args.device=="auto" and torch.cuda.is_available() else "cpu" if args.device=="auto" else args.device
    from spectrum_semcom.models import TinyOccupancyCNN
    detector=TinyOccupancyCNN(channels=16).to(device); detector.load_state_dict(torch.load(args.detector_checkpoint,map_location=device,weights_only=True)); detector.eval()
    c1=VariableRateOccupancyHead(n_channels,int(protocol["gate_a_controlled_training"]["hidden_dim"])).to(device); c1_path=PROJECT_DIR/settings["canonical_c1_checkpoint"]; c1.load_state_dict(torch.load(c1_path,map_location=device,weights_only=True)); c1.eval(); args.out_dir.mkdir(parents=True,exist_ok=True); records={}
    for split_name,entry in registry["splits"].items():
        base_images=[]; truths=[]; scene_ids=[]
        for scene in entry["scenes"]:
            image,truth=scene_arrays(args.root,scene,n_channels,str(task["channel_axis"])); base_images.append(image); truths.append(truth); scene_ids.append(scene["scene_id"])
        views=[]
        for scene_index,image in enumerate(base_images):
            for node in range(n_nodes):
                rng=np.random.default_rng(int(protocol["seed"])+scene_index*100+node+{"train":0,"calibration":1000000,"validation":2000000}[split_name]); views.append(perturb_correlated_spectrogram(image,float(settings["sensing_snr_db"][node]),rng,float(settings["gain_db"][node]),int(settings["frequency_shift_bins"][node])))
        masks=infer_masks(views,detector,device,32); pooled=torch.as_tensor(np.stack([pool_probability_mask(mask,n_channels,str(task["channel_axis"])) for mask in masks]),dtype=torch.float32,device=device)
        with torch.no_grad(): refined=c1(pooled).refined_occupancy.cpu().numpy().reshape(len(scene_ids),n_nodes,n_channels).astype(np.float32)
        quality=np.empty((len(scene_ids),n_nodes,len(QUALITY_NAMES)),dtype=np.float32)
        for scene_index in range(len(scene_ids)):
            for node in range(n_nodes): quality[scene_index,node]=quality_vector(views[scene_index*n_nodes+node],refined[scene_index,node],float(settings["sensing_snr_db"][node]),float(settings["age_s"][node]),float(settings["reporting_ebn0_db"][node]))
        truth=np.stack(truths).astype(np.float32); cache_errors=validate_multinode_cache_arrays(scene_ids,refined,truth,quality,QUALITY_AVAILABILITY,n_nodes,n_channels)
        if cache_errors: raise ValueError(f"invalid {split_name} cache: "+"; ".join(cache_errors))
        path=args.out_dir/f"multinode_{split_name}.npz"; np.savez_compressed(path,scene_ids=np.asarray(scene_ids),node_occupancy=refined,truth_occupancy=truth,node_quality=quality,quality_names=np.asarray(QUALITY_NAMES),quality_availability=QUALITY_AVAILABILITY)
        records[split_name]={"scenes":len(scene_ids),"cache":str(path.relative_to(PROJECT_DIR)).replace("\\","/"),"sha256":sha256_file(path),"scene_ids_sha256":sha256_strings(scene_ids)}
    result={"experiment_id":"stage4_c2_multinode_cache_v1","protocol_sha256":sha256_file(args.protocol),"registry_sha256":sha256_file(args.registry),"detector_sha256":sha256_file(args.detector_checkpoint),"c1_checkpoint":settings["canonical_c1_checkpoint"],"c1_checkpoint_sha256":sha256_file(c1_path),"task":task,"node_settings":settings,"quality_names":QUALITY_NAMES,"quality_availability":QUALITY_AVAILABILITY.tolist(),"splits":records,"environment":environment_snapshot(["numpy","torch"]),"final_holdout_accessed":False,"claim_boundary":"Controlled same-scene C2/H2 development cache; not real H3 evidence."}; (args.out_dir/"multinode_cache_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(args.out_dir/"multinode_cache_result.json")

if __name__=="__main__": main()
