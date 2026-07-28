"""Evaluate C3 audited reliable-minority protection on frozen failure scenarios."""

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
from spectrum_semcom.counterfactual_value import candidate_message_cost,discrete_scene_regret
from spectrum_semcom.digital_link import expected_transmitted_bits_awgn
from spectrum_semcom.multi_uav_fusion import NodeOccupancyReport,fuse_occupancy
from spectrum_semcom.multigranular_semantics import SemanticQuality
from spectrum_semcom.reliable_minority import AuditedOccupancyReport,audited_quality_mean,confirmed_conflict_retransmission,reliable_minority_fusion
from spectrum_semcom.reproducibility import environment_snapshot,sha256_file
from spectrum_semcom.stage4_protocol import assert_valid_stage4_protocol


def load(path:Path)->dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as data:return {key:data[key] for key in data.files}


def base_reports(occupancy:np.ndarray,quality:np.ndarray)->list[AuditedOccupancyReport]:
    return [AuditedOccupancyReport(node,occupancy[node].astype(np.float64),SemanticQuality(*[float(value) for value in quality[node]])) for node in range(len(occupancy))]


def scenarios(current:list[AuditedOccupancyReport],previous:list[AuditedOccupancyReport])->dict[str,list[AuditedOccupancyReport]]:
    output={"normal":current}
    false_clean=[]; shifted=[]; wrong=[]; stale=[]
    for index,report in enumerate(current):
        if index==3:
            false_clean.append(report);shifted.append(report);wrong.append(report);stale.append(report);continue
        false_clean.append(AuditedOccupancyReport(report.node_id,.05*report.occupancy,replace(report.quality,prediction_confidence=.99,clipping_ratio=.4,calibration_error=.4)))
        shifted.append(AuditedOccupancyReport(report.node_id,np.roll(report.occupancy,2),replace(report.quality,prediction_confidence=.99,calibration_error=.35)))
        wrong.append(AuditedOccupancyReport(report.node_id,1-report.occupancy,replace(report.quality,prediction_confidence=.99,clipping_ratio=.3,calibration_error=.5)))
        stale.append(AuditedOccupancyReport(report.node_id,previous[index].occupancy,replace(report.quality,age_s=2.0,calibration_error=.2)))
    output.update({"false_clean_majority":false_clean,"shifted_majority":shifted,"high_confidence_wrong_majority":wrong,"stale_majority":stale,"key_node_dropout":current[:3]})
    benign=list(current); benign[0]=AuditedOccupancyReport(0,np.clip(benign[0].occupancy+np.asarray([.7,0,0,0,0,0,0,0]),0,1),replace(benign[0].quality,prediction_confidence=.99));output["benign_singleton_false_alarm"]=benign
    return output


def fuse(method:str,reports:list[AuditedOccupancyReport],settings:dict)->tuple[np.ndarray,bool,bool,int]:
    if method in {"mean","median","snr_weighted","stage3_robust_quality"}:
        classic=[NodeOccupancyReport(report.occupancy,report.quality.sensing_snr_db,report.quality.prediction_confidence,report.quality.report_success_probability,report.quality.age_s) for report in reports]
        mode="robust_quality_weighted" if method=="stage3_robust_quality" else method
        return fuse_occupancy(classic,mode),False,False,-1
    if method=="audited_quality_mean":return audited_quality_mean(reports,float(settings["freshness_tau_s"]),float(settings["noise_stability_scale_db"])),False,False,-1
    if method=="reliable_minority":
        result=reliable_minority_fusion(reports,freshness_tau_s=float(settings["freshness_tau_s"]),noise_scale_db=float(settings["noise_stability_scale_db"]),disagreement_threshold=float(settings["node_disagreement_threshold"]),channel_excess_threshold=float(settings["channel_excess_threshold"]),quality_ratio=float(settings["minority_quality_ratio"]),minimum_quality=float(settings["minimum_minority_quality"]),protection_strength=float(settings["protection_strength"]),maximum_protected_channels=int(settings["maximum_protected_channels"]));return result.occupancy,result.protected_node>=0,result.retransmission_triggered,result.protected_node
    confirmed=settings["confirmed_retransmission"];result=confirmed_conflict_retransmission(reports,trigger_threshold=float(confirmed["trigger_threshold"]),maximum_corrected_channels=int(confirmed["maximum_corrected_channels"]),blend_strength=float(confirmed["confirmed_blend_strength"]),freshness_tau_s=float(settings["freshness_tau_s"]),noise_scale_db=float(settings["noise_stability_scale_db"]));return result.occupancy,False,result.retransmission_triggered,result.protected_node


def metrics(predicted:np.ndarray,truth:np.ndarray,demand:int,protected:bool,triggered:bool,extra_bits:float=0)->dict[str,float]:
    kernel=np.ones(demand)/demand; truth_blocks=np.convolve(truth,kernel,mode="valid"); chosen=int(np.argmin(np.convolve(predicted,kernel,mode="valid")))
    return {"regret":discrete_scene_regret(predicted,truth,demand),"clean":float(truth_blocks[chosen]<=.02),"missed_occupancy":float(np.maximum(truth-predicted,0).mean()),"false_occupancy":float(np.maximum(predicted-truth,0).mean()),"brier":float(np.square(predicted-truth).mean()),"protected":float(protected),"retransmission_triggered":float(triggered),"extra_bits":float(extra_bits)}


def summarize(rows:list[dict[str,float]])->dict[str,float]:
    regrets=np.asarray([row["regret"] for row in rows]);tail=max(1,int(np.ceil(.1*len(regrets))));return {"mean_regret":float(regrets.mean()),"cvar_0_9_regret":float(np.sort(regrets)[-tail:].mean()),"clean_resource_rate":float(np.mean([r["clean"] for r in rows])),"missed_occupancy":float(np.mean([r["missed_occupancy"] for r in rows])),"false_occupancy":float(np.mean([r["false_occupancy"] for r in rows])),"brier":float(np.mean([r["brier"] for r in rows])),"protection_rate":float(np.mean([r["protected"] for r in rows])),"retransmission_trigger_rate":float(np.mean([r["retransmission_triggered"] for r in rows])),"mean_extra_bits":float(np.mean([r["extra_bits"] for r in rows]))}


def paired_bootstrap(rows:dict[str,dict[str,list[dict[str,float]]]],repetitions:int,seed:int)->dict:
    rng=np.random.default_rng(seed);output={}
    for scenario,methods in rows.items():
        baseline=methods["stage3_robust_quality"];output[scenario]={}
        for proposed_name in ("reliable_minority","conflict_triggered_G3"):
            proposed=methods[proposed_name];indices=rng.integers(0,len(proposed),size=(repetitions,len(proposed)));output[scenario][proposed_name]={}
            for metric in ("regret","missed_occupancy","false_occupancy"):
                difference=np.asarray([a[metric]-b[metric] for a,b in zip(proposed,baseline)]);samples=difference[indices].mean(axis=1);output[scenario][proposed_name][metric]={"difference":float(difference.mean()),"ci95":np.quantile(samples,[.025,.975]).tolist()}
    return output


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--protocol",type=Path,default=PROJECT_DIR/"configs"/"stage4_protocol.json");parser.add_argument("--stage2-config",type=Path,default=PROJECT_DIR/"configs"/"stage2_digital_link.json");parser.add_argument("--cache-result",type=Path,default=PROJECT_DIR/"results"/"stage4"/"multinode_cache_v1"/"multinode_cache_result.json");parser.add_argument("--out-dir",type=Path,default=PROJECT_DIR/"results"/"stage4"/"reliable_minority_v1");args=parser.parse_args();protocol=json.loads(args.protocol.read_text(encoding="utf-8"));assert_valid_stage4_protocol(protocol);settings=protocol["reliable_minority_protocol"];stage2=json.loads(args.stage2_config.read_text(encoding="utf-8"));meta=json.loads(args.cache_result.read_text(encoding="utf-8"));cache=load(PROJECT_DIR/meta["splits"]["validation"]["cache"]);demand=int(meta["task"]["demand_channels"]);methods=settings["comparison_methods"];links=tuple(build_link_config(stage2,float(value)) for value in protocol["counterfactual_value_protocol"]["reporting_ebn0_db"]);rows={scenario:{method:[] for method in methods} for scenario in settings["controlled_scenarios"]}
    for index in range(len(cache["scene_ids"])):
        current=base_reports(cache["node_occupancy"][index],cache["node_quality"][index]);previous=base_reports(cache["node_occupancy"][index-1],cache["node_quality"][index-1]);truth=cache["truth_occupancy"][index]
        for scenario,reports in scenarios(current,previous).items():
            for method in methods:
                predicted,protected,triggered,selected_node=fuse(method,reports,settings);extra_bits=0.0
                if method=="conflict_triggered_G3" and triggered and selected_node>=0:
                    selected=next(report for report in reports if report.node_id==selected_node);extra_bits=expected_transmitted_bits_awgn(int(settings["confirmed_retransmission"]["feedback_application_bits"]),links[selected_node]);extra_bits+=candidate_message_cost(str(cache["scene_ids"][index]),selected_node,3,selected.occupancy,selected.quality,links[selected_node])[2]
                rows[scenario][method].append(metrics(predicted,truth,demand,protected,triggered,extra_bits))
    summaries={scenario:{method:summarize(values) for method,values in methods_rows.items()} for scenario,methods_rows in rows.items()};pooled={method:summarize([row for scenario in rows.values() for row in scenario[method]]) for method in methods};result={"experiment_id":"stage4_C3_reliable_minority_v1","protocol_sha256":sha256_file(args.protocol),"cache_result_sha256":sha256_file(args.cache_result),"scene_count":len(cache["scene_ids"]),"scenario_metrics":summaries,"pooled_metrics":pooled,"paired_scene_bootstrap":paired_bootstrap(rows,int(settings["bootstrap_repetitions"]),int(protocol["seed"])),"claim_boundary":settings["claim_boundary"],"final_holdout_accessed":False,"environment":environment_snapshot(["numpy"])};args.out_dir.mkdir(parents=True,exist_ok=True);(args.out_dir/"reliable_minority_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# Stage 4 C3 Reliable Minority Validation","","| Method | pooled regret | CVaR | miss | false occupancy | clean | protect | trigger | extra bit |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in methods:
        row=pooled[method];lines.append(f"| {method} | {row['mean_regret']:.6f} | {row['cvar_0_9_regret']:.6f} | {row['missed_occupancy']:.6f} | {row['false_occupancy']:.6f} | {row['clean_resource_rate']:.4f} | {row['protection_rate']:.4f} | {row['retransmission_trigger_rate']:.4f} | {row['mean_extra_bits']:.2f} |")
    lines.extend(["","Controlled failure robustness only; this is not real H3 evidence and no final holdout was accessed."]);(args.out_dir/"reliable_minority_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(args.out_dir/"reliable_minority_report.md")


if __name__=="__main__":main()
