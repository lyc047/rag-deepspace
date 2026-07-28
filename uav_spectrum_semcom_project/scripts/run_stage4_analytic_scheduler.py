"""Fit the train-only analytic value calibrator and evaluate fixed budget curves."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
for path in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_stage2_digital_link import build_link_config
from spectrum_semcom.analytic_scheduler import CandidateAction, analytic_raw_score, exact_multiple_choice_knapsack, fit_monotonic_value_calibrator, greedy_actions, train_margin_temperature
from spectrum_semcom.counterfactual_value import candidate_message_cost, discrete_scene_regret, fuse_state, quantized_node_reports
from spectrum_semcom.multigranular_semantics import SemanticQuality
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file
from spectrum_semcom.stage4_protocol import assert_valid_stage4_protocol


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def quality(row: np.ndarray) -> SemanticQuality:
    return SemanticQuality(*[float(value) for value in row])


def fit_train_calibrator(labels: dict[str, np.ndarray], demand: int, n_bins: int) -> tuple[float, Any, dict[str, Any]]:
    initial = np.all(labels["current_state"] == 1, axis=1)
    indices = np.flatnonzero(initial)
    beliefs = labels["current_belief"][indices]
    _, unique_indices = np.unique(labels["scene_id"][indices], return_index=True)
    temperature = train_margin_temperature(beliefs[np.sort(unique_indices)], demand)
    scores = np.asarray([analytic_raw_score(labels["current_belief"][index], labels["candidate_current_report"][index], labels["candidate_quality"][index], demand, int(labels["target_granularity"][index]), temperature) for index in indices])
    values = labels["marginal_value"][indices].astype(np.float64)
    calibrator = fit_monotonic_value_calibrator(scores, values, n_bins)
    correlation = None
    if float(np.ptp(scores)) > 1e-15:
        ranks_x = np.argsort(np.argsort(scores, kind="stable"), kind="stable")
        ranks_y = np.argsort(np.argsort(values, kind="stable"), kind="stable")
        correlation = float(np.corrcoef(ranks_x, ranks_y)[0, 1])
    audit = {"rows": len(indices), "scene_count": len(unique_indices), "margin_temperature": temperature, "raw_score_unique_count": int(np.unique(scores).size), "raw_score_nonzero_fraction": float(np.mean(np.abs(scores) > 1e-15)), "raw_score_value_rank_correlation": correlation, "score_knots": calibrator.score_knots.tolist(), "value_knots": calibrator.value_knots.tolist(), "positive_calibrated_knot_fraction": float(np.mean(calibrator.value_knots > 0))}
    return temperature, calibrator, audit


def outcome(belief: np.ndarray, truth: np.ndarray, demand: int, bits: float, node: int, target: int) -> dict[str, float]:
    kernel = np.ones(demand) / demand
    truth_blocks = np.convolve(truth, kernel, mode="valid")
    chosen = int(np.argmin(np.convolve(belief, kernel, mode="valid")))
    return {"regret": discrete_scene_regret(belief, truth, demand), "bits": float(bits), "clean": float(truth_blocks[chosen] <= 0.02), "chosen_occupancy": float(truth_blocks[chosen]), "brier": float(np.mean(np.square(belief - truth))), "node": int(node), "target": int(target)}


def scene_actions(scene_id: str, occupancy: np.ndarray, qualities: np.ndarray, truth: np.ndarray, links: tuple, demand: int, temperature: float, calibrator: Any, scheduler_reports: np.ndarray | None = None, scheduler_qualities: np.ndarray | None = None, scheduler_preview_bits: np.ndarray | None = None) -> tuple[dict[str, float], list[dict[str, Any]]]:
    reports = quantized_node_reports(occupancy, 1)
    state = (1,) * occupancy.shape[0]
    belief = fuse_state(reports, state)
    preview_cost = float(np.sum(scheduler_preview_bits)) if scheduler_preview_bits is not None else sum(candidate_message_cost(scene_id, node, 1, reports[1][node], quality(qualities[node]), links[node], probability_bits=1)[2] for node in range(occupancy.shape[0]))
    scheduler_reports = reports[1] if scheduler_reports is None else scheduler_reports
    scheduler_qualities = qualities if scheduler_qualities is None else scheduler_qualities
    baseline = outcome(belief, truth, demand, preview_cost, -1, 0)
    candidates = []
    for node in range(occupancy.shape[0]):
        for target in (2, 3):
            _, _, incremental = candidate_message_cost(scene_id, node, target, reports[target][node], quality(qualities[node]), links[node])
            upgraded = list(state); upgraded[node] = target
            fused = fuse_state(reports, tuple(upgraded))
            raw = analytic_raw_score(belief, scheduler_reports[node], scheduler_qualities[node], demand, target, temperature)
            predicted = float(calibrator.predict(raw))
            row = outcome(fused, truth, demand, preview_cost + incremental, node, target)
            row.update({"incremental_bits": incremental, "predicted_value": predicted, "raw_score": raw, "snr": float(scheduler_qualities[node, 0]), "confidence": float(scheduler_qualities[node, 1]), "reliability": float(scheduler_qualities[node, 8])})
            candidates.append(row)
    return baseline, candidates


def choose_methods(baseline: dict[str, float], rows: list[dict[str, Any]], budget: float, max_actions: int) -> dict[str, dict[str, float]]:
    affordable = [row for row in rows if row["incremental_bits"] <= budget + 1e-12]
    by_key = {(row["node"], row["target"]): row for row in rows}
    def fixed_g3(field: str) -> dict[str, float]:
        node = max(range(4), key=lambda index: by_key[(index, 3)][field])
        row = by_key[(node, 3)]
        return row if row["incremental_bits"] <= budget + 1e-12 else baseline
    prior = max(affordable, key=lambda row: (1.0 / (1.0 + np.exp(-row["snr"] / 4.0))) * row["reliability"] / row["incremental_bits"], default=baseline)
    actions = [CandidateAction(int(row["node"]), int(row["target"]), float(row["predicted_value"]), float(row["incremental_bits"])) for row in rows]
    greedy = greedy_actions(actions, budget, max_actions)
    exact = exact_multiple_choice_knapsack(actions, budget, max_actions)
    greedy_row = by_key[(greedy[0].node, greedy[0].target_granularity)] if greedy else baseline
    exact_row = by_key[(exact[0].node, exact[0].target_granularity)] if exact else baseline
    oracle = min([baseline, *affordable], key=lambda row: (row["regret"], row["bits"]))
    return {"no_upgrade": baseline, "sensing_snr_G3": fixed_g3("snr"), "confidence_G3": fixed_g3("confidence"), "stage3_prior_value_per_bit": prior, "analytic_greedy": greedy_row, "analytic_exact_knapsack": exact_row, "budget_oracle": oracle}


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    regrets = np.asarray([row["regret"] for row in rows])
    tail_count = max(1, int(np.ceil(0.1 * len(regrets))))
    cvar = float(np.sort(regrets)[-tail_count:].mean())
    return {"mean_regret": float(regrets.mean()), "cvar_0_9_regret": cvar, "mean_expected_total_bits": float(np.mean([row["bits"] for row in rows])), "clean_resource_rate": float(np.mean([row["clean"] for row in rows])), "brier": float(np.mean([row["brier"] for row in rows])), "no_upgrade_fraction": float(np.mean([row["node"] < 0 for row in rows]))}


def evaluate(cache: dict[str, np.ndarray], links: tuple, demand: int, temperature: float, calibrator: Any, budgets: list[float], maximum_actions: int, scheduler_cache: dict[str, np.ndarray] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    results = {}; raw_rows = {}
    if scheduler_cache is not None and not np.array_equal(cache["scene_ids"], scheduler_cache["scene_ids"]): raise ValueError("scheduler preview cache scene alignment mismatch")
    scenes = [scene_actions(str(cache["scene_ids"][index]), cache["node_occupancy"][index], cache["node_quality"][index], cache["truth_occupancy"][index], links, demand, temperature, calibrator, None if scheduler_cache is None else scheduler_cache["scheduler_report"][index], None if scheduler_cache is None else scheduler_cache["scheduler_quality"][index], None if scheduler_cache is None else scheduler_cache["expected_transmitted_bits"][index]) for index in range(len(cache["scene_ids"]))]
    for budget in budgets:
        collected: dict[str, list[dict[str, float]]] = defaultdict(list)
        for baseline, candidates in scenes:
            for method, row in choose_methods(baseline, candidates, budget, maximum_actions).items(): collected[method].append(row)
        results[str(int(budget))] = {method: summarize(rows) for method, rows in collected.items()}
        raw_rows[str(int(budget))] = collected
    return results, raw_rows


def bootstrap(raw_by_budget: dict[str, Any], repetitions: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed); output = {}
    for budget, collected in raw_by_budget.items():
        output[budget] = {}
        proposed = collected["analytic_exact_knapsack"]
        for baseline in ("no_upgrade", "sensing_snr_G3", "confidence_G3", "stage3_prior_value_per_bit"):
            regret = np.asarray([a["regret"] - b["regret"] for a, b in zip(proposed, collected[baseline])]); bits = np.asarray([a["bits"] - b["bits"] for a, b in zip(proposed, collected[baseline])]); indices = rng.integers(0, len(regret), size=(repetitions, len(regret)))
            output[budget][baseline] = {"regret_difference": float(regret.mean()), "regret_ci95": np.quantile(regret[indices].mean(axis=1), [.025, .975]).tolist(), "bit_difference": float(bits.mean()), "bit_ci95": np.quantile(bits[indices].mean(axis=1), [.025, .975]).tolist()}
    return output


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--protocol",type=Path,default=PROJECT_DIR/"configs"/"stage4_protocol.json"); parser.add_argument("--stage2-config",type=Path,default=PROJECT_DIR/"configs"/"stage2_digital_link.json"); parser.add_argument("--labels",type=Path,default=PROJECT_DIR/"results"/"stage4"/"counterfactual_labels_v1"/"train_counterfactual_labels.npz"); parser.add_argument("--cache-result",type=Path,default=PROJECT_DIR/"results"/"stage4"/"multinode_cache_v1"/"multinode_cache_result.json"); parser.add_argument("--scheduler-preview-result",type=Path); parser.add_argument("--out-dir",type=Path,default=PROJECT_DIR/"results"/"stage4"/"analytic_scheduler_v1"); args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text(encoding="utf-8")); assert_valid_stage4_protocol(protocol); settings=protocol["analytic_scheduler_protocol"]; stage2=json.loads(args.stage2_config.read_text(encoding="utf-8")); meta=json.loads(args.cache_result.read_text(encoding="utf-8")); demand=int(meta["task"]["demand_channels"]); links=tuple(build_link_config(stage2,float(value)) for value in protocol["counterfactual_value_protocol"]["reporting_ebn0_db"])
    temperature,calibrator,train_audit=fit_train_calibrator(load_npz(args.labels),demand,10); budgets=[float(value) for value in settings["incremental_budget_bits"]]
    split_results={}; validation_raw=None
    preview_meta=json.loads(args.scheduler_preview_result.read_text(encoding="utf-8")) if args.scheduler_preview_result else None
    for split in ("calibration","validation"):
        scheduler_cache=load_npz(PROJECT_DIR/preview_meta["splits"][split]["cache"]) if preview_meta else None
        metrics,raw=evaluate(load_npz(PROJECT_DIR/meta["splits"][split]["cache"]),links,demand,temperature,calibrator,budgets,int(settings["maximum_actions_for_gate_b"]),scheduler_cache); split_results[split]=metrics
        if split=="validation": validation_raw=raw
    result={"experiment_id":"stage4_c2_analytic_scheduler_v1","protocol_sha256":sha256_file(args.protocol),"labels_sha256":sha256_file(args.labels),"cache_result_sha256":sha256_file(args.cache_result),"train_calibration_audit":train_audit,"budget_metrics":split_results,"validation_paired_bootstrap":bootstrap(validation_raw,int(settings["bootstrap_repetitions"]),int(protocol["seed"])),"truth_access_audit":{"train_truth_used_for_monotonic_value_calibration":True,"calibration_validation_labels_saved":False,"final_holdout_accessed":False},"claim_boundary":"Controlled one-step C2/H2 development result; not final H2 or real H3.","environment":environment_snapshot(["numpy"])}; args.out_dir.mkdir(parents=True,exist_ok=True); (args.out_dir/"analytic_scheduler_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    result["experiment_id"]="stage4_c2_analytic_scheduler_sketch_v1" if preview_meta else "stage4_c2_analytic_scheduler_v1"; result["scheduler_preview_used"]=preview_meta is not None; (args.out_dir/"analytic_scheduler_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    correlation="undefined_constant_score" if train_audit["raw_score_value_rank_correlation"] is None else f"{train_audit['raw_score_value_rank_correlation']:.4f}"; lines=["# Stage 4 Analytic Scheduler Validation","",f"Scheduler residual sketch used: {preview_meta is not None}.","",f"Train raw-score/value rank correlation: {correlation}; unique scores: {train_audit['raw_score_unique_count']}; PAV knots: {len(train_audit['score_knots'])}.",""]
    for budget in budgets:
        lines.extend([f"## Incremental budget {budget:.0f} bit","","| Method | regret | CVaR | total bit | clean | no upgrade |","|---|---:|---:|---:|---:|---:|"])
        for method in settings["comparison_methods"]:
            row=split_results["validation"][str(int(budget))][method]; lines.append(f"| {method} | {row['mean_regret']:.6f} | {row['cvar_0_9_regret']:.6f} | {row['mean_expected_total_bits']:.2f} | {row['clean_resource_rate']:.4f} | {row['no_upgrade_fraction']:.4f} |")
        lines.append("")
    lines.append("No calibration/validation labels were serialized and no final holdout was accessed."); (args.out_dir/"analytic_scheduler_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); print(args.out_dir/"analytic_scheduler_report.md")


if __name__=="__main__": main()
