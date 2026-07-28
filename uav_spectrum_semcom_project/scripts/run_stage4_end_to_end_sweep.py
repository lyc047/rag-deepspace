"""Frozen decoder-path sweep for the surviving classical stage-4 system."""

from __future__ import annotations

import argparse,json,sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_DIR=Path(__file__).resolve().parents[1]
for path in (PROJECT_DIR/"src",PROJECT_DIR/"scripts"):
    if str(path) not in sys.path:sys.path.insert(0,str(path))

from run_stage2_digital_link import build_link_config
from spectrum_semcom.counterfactual_value import discrete_scene_regret,quantized_node_reports
from spectrum_semcom.multigranular_semantics import SemanticMessage,SemanticQuality,transmit_semantic_message
from spectrum_semcom.reproducibility import environment_snapshot,sha256_file
from spectrum_semcom.stage4_protocol import assert_valid_stage4_protocol


def load(path:Path)->dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as data:return {key:data[key] for key in data.files}


def quality(row:np.ndarray)->SemanticQuality:return SemanticQuality(*[float(value) for value in row])


def select_prior_node(qualities:np.ndarray,links:tuple)->int:
    scores=[]
    for node in range(len(qualities)):
        sensing=1/(1+np.exp(-float(qualities[node,0])/4));scores.append(sensing*float(qualities[node,8])/(1+max(0,-float(links[node].ebn0_db))))
    return int(np.argmax(scores))


def run_method(method:str,scene_id:str,occupancy:np.ndarray,qualities:np.ndarray,links:tuple,seed:int)->tuple[np.ndarray,dict[str,float]]:
    reports=quantized_node_reports(occupancy,1);selected=select_prior_node(qualities,links);plan=[]
    if method in {"all_G1","G1_plus_stage3_prior_G2"}:plan.extend((node,1) for node in range(len(occupancy)))
    if method=="G1_plus_stage3_prior_G2":plan.append((selected,2))
    if method=="all_G2":plan.extend((node,2) for node in range(len(occupancy)))
    latest={};bits=0;latency=0;delivered=0;attempted=0
    for message_index,(node,target) in enumerate(plan):
        granularity=f"G{target}";probability_bits=1 if target==1 else 4;message=SemanticMessage(granularity,node,scene_id,reports[target][node],probability_bits,quality=None if target==1 else quality(qualities[node]),evidence=());transmission=transmit_semantic_message(message,links[node],seed+message_index*1009);attempted+=1;bits+=transmission.link.transmitted_bits;latency+=transmission.link.duration_s
        if transmission.decoded is not None:latest[node]=transmission.decoded.occupancy;delivered+=1
    fused=np.mean(np.stack(list(latest.values())),axis=0) if latest else np.full(occupancy.shape[1],.5)
    return fused,{"transmitted_bits":bits,"latency_s":latency,"delivered_reports":delivered,"attempted_reports":attempted}


def summarize(rows:list[dict[str,float]])->dict[str,float]:
    regret=np.asarray([row["regret"] for row in rows]);tail=max(1,int(np.ceil(.1*len(regret))));return {"mean_regret":float(regret.mean()),"cvar_0_9_regret":float(np.sort(regret)[-tail:].mean()),"clean_resource_rate":float(np.mean([row["clean"] for row in rows])),"brier":float(np.mean([row["brier"] for row in rows])),"mean_actual_transmitted_bits":float(np.mean([row["transmitted_bits"] for row in rows])),"mean_latency_s":float(np.mean([row["latency_s"] for row in rows])),"mean_delivered_reports":float(np.mean([row["delivered_reports"] for row in rows])),"report_delivery_ratio":float(sum(row["delivered_reports"] for row in rows)/sum(row["attempted_reports"] for row in rows))}


def paired_bootstrap(rows:list[dict],repetitions:int,seed:int)->dict:
    rng=np.random.default_rng(seed);output={};scene_ids=sorted({row["scene_id"] for row in rows})
    for channel in sorted({row["channel"] for row in rows}):
        for ebn0 in sorted({row["ebn0_db"] for row in rows}):
            key=f"{channel}_{ebn0:g}";output[key]={};by_method={}
            for method in ("all_G1","G1_plus_stage3_prior_G2","all_G2"):
                selected=[row for row in rows if row["channel"]==channel and row["ebn0_db"]==ebn0 and row["method"]==method];by_scene=defaultdict(list)
                for row in selected:by_scene[row["scene_id"]].append(row)
                by_method[method]={metric:np.asarray([np.mean([item[metric] for item in by_scene[scene]]) for scene in scene_ids]) for metric in ("regret","transmitted_bits")}
            proposed=by_method["G1_plus_stage3_prior_G2"]
            for baseline in ("all_G1","all_G2"):
                output[key][baseline]={};indices=rng.integers(0,len(scene_ids),size=(repetitions,len(scene_ids)))
                for metric in ("regret","transmitted_bits"):
                    difference=proposed[metric]-by_method[baseline][metric];samples=difference[indices].mean(axis=1);output[key][baseline][metric]={"difference":float(difference.mean()),"ci95":np.quantile(samples,[.025,.975]).tolist()}
    return output


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--protocol",type=Path,default=PROJECT_DIR/"configs"/"stage4_protocol.json");parser.add_argument("--stage2-config",type=Path,default=PROJECT_DIR/"configs"/"stage2_digital_link.json");parser.add_argument("--cache-result",type=Path,default=PROJECT_DIR/"results"/"stage4"/"multinode_cache_v1"/"multinode_cache_result.json");parser.add_argument("--out-dir",type=Path,default=PROJECT_DIR/"results"/"stage4"/"end_to_end_sweep_v1");args=parser.parse_args();protocol=json.loads(args.protocol.read_text(encoding="utf-8"));assert_valid_stage4_protocol(protocol);settings=protocol["end_to_end_link_sweep"];stage2=json.loads(args.stage2_config.read_text(encoding="utf-8"));meta=json.loads(args.cache_result.read_text(encoding="utf-8"));cache=load(PROJECT_DIR/meta["splits"][settings["split"]]["cache"]);demand=int(meta["task"]["demand_channels"]);rows=[]
    for channel_index,channel in enumerate(settings["channel_models"]):
        for ebn0_index,global_ebn0 in enumerate(settings["global_ebn0_db"]):
            links=tuple(replace(build_link_config(stage2,float(global_ebn0)+float(offset)),channel=channel) for offset in settings["node_relative_ebn0_db"])
            for scene_index,scene_id in enumerate(cache["scene_ids"]):
                truth=cache["truth_occupancy"][scene_index];kernel=np.ones(demand)/demand;truth_blocks=np.convolve(truth,kernel,mode="valid")
                for repeat in range(int(settings["link_repeats_per_scene"])):
                    for method_index,method in enumerate(settings["methods"]):
                        seed=int(protocol["seed"])+channel_index*10_000_000+ebn0_index*1_000_000+scene_index*1000+repeat*10+method_index;fused,link=run_method(method,str(scene_id),cache["node_occupancy"][scene_index],cache["node_quality"][scene_index],links,seed);chosen=int(np.argmin(np.convolve(fused,kernel,mode="valid")));rows.append({"channel":channel,"ebn0_db":float(global_ebn0),"scene_id":str(scene_id),"repeat":repeat,"method":method,"regret":discrete_scene_regret(fused,truth,demand),"clean":float(truth_blocks[chosen]<=.02),"brier":float(np.square(fused-truth).mean()),**link})
    grouped=defaultdict(list)
    for row in rows:grouped[(row["channel"],row["ebn0_db"],row["method"])].append(row)
    summary=[{"channel":key[0],"ebn0_db":key[1],"method":key[2],**summarize(values)} for key,values in grouped.items()];result={"experiment_id":"stage4_end_to_end_decoder_sweep_v1","protocol_sha256":sha256_file(args.protocol),"stage2_config_sha256":sha256_file(args.stage2_config),"cache_result_sha256":sha256_file(args.cache_result),"scene_count":len(cache["scene_ids"]),"link_repeats_per_scene":settings["link_repeats_per_scene"],"summary":summary,"claim_boundary":"validation decoder-path robustness only; no final H2/H3 claim","final_holdout_accessed":False,"environment":environment_snapshot(["numpy"])};args.out_dir.mkdir(parents=True,exist_ok=True);(args.out_dir/"end_to_end_sweep_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    result["paired_scene_bootstrap"]=paired_bootstrap(rows,int(settings["paired_scene_bootstrap_repetitions"]),int(protocol["seed"]));(args.out_dir/"end_to_end_sweep_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# Stage 4 End-to-End Decoder Sweep","","| Channel | Eb/N0 | Method | regret | CVaR | clean | actual bit | latency ms | report delivery |","|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:lines.append(f"| {row['channel']} | {row['ebn0_db']:.0f} | {row['method']} | {row['mean_regret']:.6f} | {row['cvar_0_9_regret']:.6f} | {row['clean_resource_rate']:.4f} | {row['mean_actual_transmitted_bits']:.1f} | {1000*row['mean_latency_s']:.3f} | {row['report_delivery_ratio']:.4f} |")
    lines.extend(["","All task metrics are computed after the actual decoder path. No final holdout was accessed."]);(args.out_dir/"end_to_end_sweep_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(args.out_dir/"end_to_end_sweep_report.md")


if __name__=="__main__":main()
