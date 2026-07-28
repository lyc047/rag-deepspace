"""ACK-triggered candidate fallback and risk-adaptive multi-UAV semantic fusion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
for path in [PROJECT_DIR / "src", PROJECT_DIR / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from spectrum_semcom.models import TinyOccupancyCNN
from spectrum_semcom.multi_uav_fusion import NodeOccupancyReport, fuse_occupancy, maximum_pair_disagreement
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file, sha256_strings
from spectrum_semcom.research_scope import assert_valid_experiment_metadata
from run_stage2_digital_link import mean_std_ci, write_csv
from run_stage3_fusion_baselines import evaluate_estimates
from run_stage3_robustness import fit_calibrator, infer_node_views, load_temporal_scenes
from run_stage3_value_bit_robust import (
    evaluate_policy,
    paired_method_difference,
    prepare_cache,
    ranked_nodes,
    validation_node_statistics,
)


def evaluate_adaptive(
    cache,
    truths,
    config,
    priors,
    expected_bits,
    disagreement_threshold: float,
    stale_trigger_s: float,
    target_reports: int,
    method_name: str,
    fusion_mode: str,
) -> list[dict[str, Any]]:
    protocol = config["protocol"]
    adaptive = config["adaptive_fallback_protocol"]
    scenarios = config["robustness_protocol"]["scenarios"]
    repeats = sorted({key[0] for key in cache})
    output = []
    for repeat in repeats:
        for scenario in scenarios:
            ranking = ranked_nodes(scenario, priors, expected_bits, config)
            estimates = []
            total_bits = 0.0
            attempted_counts = []
            ack_triggers = 0
            risk_triggers = 0
            for records in cache[(repeat, str(scenario["id"]))]:
                reports: list[NodeOccupancyReport] = []
                attempted: set[int] = set()

                def attempt(node: int) -> None:
                    nonlocal total_bits
                    attempted.add(node)
                    report, bits = records.get(node, (None, 0.0))
                    total_bits += float(bits)
                    if report is not None:
                        reports.append(report)

                for node in ranking[: int(adaptive["initial_selected_nodes"])]:
                    attempt(node)
                next_nodes = [node for node in ranking if node not in attempted]
                ack_triggered = len(reports) < int(adaptive["minimum_delivered_reports"])
                if ack_triggered:
                    ack_triggers += 1
                while len(reports) < int(adaptive["minimum_delivered_reports"]) and next_nodes:
                    attempt(next_nodes.pop(0))

                stale = any(float(report.age_s) >= float(stale_trigger_s) for report in reports)
                disagree = maximum_pair_disagreement(reports) >= float(disagreement_threshold)
                if (ack_triggered or stale or disagree) and next_nodes:
                    risk_triggers += 1
                    while len(reports) < int(target_reports) and next_nodes:
                        attempt(next_nodes.pop(0))

                attempted_counts.append(len(attempted))
                if not reports:
                    estimates.append(np.zeros(int(protocol["n_channels"]), dtype=np.float32))
                elif fusion_mode != "mean" and len(reports) >= 3:
                    estimates.append(
                        fuse_occupancy(
                            reports,
                            fusion_mode,
                            float(protocol["occupancy_vote_threshold"]),
                            float(protocol["freshness_tau_s"]),
                            float(adaptive["disagreement_scale"]),
                            float(adaptive["max_weight_ratio"]),
                        )
                    )
                else:
                    estimates.append(fuse_occupancy(reports, "mean"))
            output.append({
                "scenario": str(scenario["id"]),
                "method": method_name,
                "repeat": repeat,
                **evaluate_estimates(np.stack(estimates), truths, protocol),
                "mean_transmitted_bits": float(total_bits / len(truths)),
                "mean_attempted_nodes": float(np.mean(attempted_counts)),
                "ack_fallback_rate": float(ack_triggers / len(truths)),
                "risk_expansion_rate": float(risk_triggers / len(truths)),
            })
    return output


def select_policy(cache, truths, config, priors, expected_bits):
    adaptive = config["adaptive_fallback_protocol"]
    baseline = evaluate_policy(cache, truths, config, priors, expected_bits, 4, 0.05, 2.0, "full_mean", "mean", True)
    baseline_average = float(np.mean([row["mean_occupancy_regret"] for row in baseline]))
    baseline_scenarios = {
        scenario: float(np.mean([row["mean_occupancy_regret"] for row in baseline if row["scenario"] == scenario]))
        for scenario in {row["scenario"] for row in baseline}
    }
    audit = []
    for disagreement in adaptive["candidate_pair_disagreement_thresholds"]:
        for stale in adaptive["candidate_stale_trigger_s"]:
            for target_reports in adaptive["candidate_target_reports"]:
                for fusion_mode in adaptive["candidate_fusion_modes"]:
                    rows = evaluate_adaptive(
                        cache, truths, config, priors, expected_bits, float(disagreement), float(stale), int(target_reports), "candidate", str(fusion_mode)
                    )
                    average_regret = float(np.mean([row["mean_occupancy_regret"] for row in rows]))
                    mean_bits = float(np.mean([row["mean_transmitted_bits"] for row in rows]))
                    scenario_regrets = {
                        scenario: float(np.mean([row["mean_occupancy_regret"] for row in rows if row["scenario"] == scenario]))
                        for scenario in baseline_scenarios
                    }
                    feasible = (
                        average_regret <= baseline_average + float(adaptive["average_regret_tolerance"])
                        and all(
                            scenario_regrets[scenario] <= baseline_scenarios[scenario] + float(adaptive["per_scenario_regret_tolerance"])
                            for scenario in baseline_scenarios
                        )
                    )
                    audit.append({
                        "pair_disagreement_threshold": float(disagreement),
                        "stale_trigger_s": float(stale),
                        "target_reports": int(target_reports),
                        "fusion_mode": str(fusion_mode),
                        "average_regret": average_regret,
                        "mean_transmitted_bits": mean_bits,
                        "baseline_average_regret": baseline_average,
                        "feasible": bool(feasible),
                    })
    feasible = [row for row in audit if row["feasible"]]
    selected = min(feasible, key=lambda row: (row["mean_transmitted_bits"], row["average_regret"])) if feasible else min(audit, key=lambda row: row["average_regret"])
    return selected, audit


def aggregate(rows):
    metrics = [
        "clean_resource_rate", "mean_occupancy_regret", "oracle_equivalent_rate", "mean_transmitted_bits",
        "mean_attempted_nodes", "ack_fallback_rate", "risk_expansion_rate",
    ]
    output = []
    for scenario, method in sorted({(row["scenario"], row["method"]) for row in rows}):
        items = [row for row in rows if row["scenario"] == scenario and row["method"] == method]
        merged: dict[str, Any] = {"scenario": scenario, "method": method, "repeat_count": len(items)}
        for metric in metrics:
            values = [float(item.get(metric, 0.0)) for item in items]
            for name, value in mean_std_ci(values).items():
                merged[f"{metric}_{name}"] = value
        output.append(merged)
    return output


def make_report(summary, deltas, fusion_deltas, policy, priors, expected_bits, adaptive) -> str:
    lines = [
        "# Stage 3 Adaptive ACK and Risk Fallback",
        "",
        f"Validation offset {adaptive['validation_source_offset']}; single fresh holdout offset {adaptive['holdout_source_offset']}.",
        "The scheduler starts with two value/bit-ranked nodes, adds candidates after delivery failure, and expands to three reports after disagreement or staleness.",
        "All fallback transmissions are included in the bit count.",
        "",
        "## Frozen policy",
        "",
        f"Pair-disagreement threshold: {policy['pair_disagreement_threshold']}; stale trigger: {policy['stale_trigger_s']} s; target reports: {policy['target_reports']}; fusion: {policy['fusion_mode']}.",
        f"Validation node priors: {np.round(priors, 4).tolist()}; expected bit/node: {np.round(expected_bits, 1).tolist()}.",
        "",
        "## Fresh holdout results",
        "",
        "| Scenario | Method | Clean | Regret | bit/decision | attempted nodes | ACK fallback | risk expansion |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['scenario']} | {row['method']} | {row['clean_resource_rate_mean']:.4f} | "
            f"{row['mean_occupancy_regret_mean']:.6f} | {row['mean_transmitted_bits_mean']:.1f} | "
            f"{row['mean_attempted_nodes_mean']:.2f} | {row['ack_fallback_rate_mean']:.3f} | "
            f"{row['risk_expansion_rate_mean']:.3f} |"
        )
    lines += ["", "## Adaptive proposed minus full mean", "", "| Scenario | Regret delta (95% CI) | bit delta |", "|---|---:|---:|"]
    for row in deltas:
        lines.append(
            f"| {row['scenario']} | {row['regret_delta_mean']:.6f} "
            f"[{row['regret_delta_ci95_low']:.6f}, {row['regret_delta_ci95_high']:.6f}] | {row['bit_delta_mean']:.1f} |"
        )
    lines += ["", "## Robust fusion minus adaptive mean", "", "Both methods use identical adaptive scheduling and bits.", "", "| Scenario | Regret delta (95% CI) |", "|---|---:|"]
    for row in fusion_deltas:
        lines.append(
            f"| {row['scenario']} | {row['regret_delta_mean']:.6f} "
            f"[{row['regret_delta_ci95_low']:.6f}, {row['regret_delta_ci95_high']:.6f}] |"
        )
    lines += ["", "This remains semi-synthetic evidence and does not establish real spatial diversity."]
    return "\n".join(lines) + "\n"


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "stage3_multi_uav_fusion.json")
    parser.add_argument("--stage2-config", type=Path, default=PROJECT_DIR / "configs" / "stage2_digital_link.json")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--mask-result", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask_result.json")
    parser.add_argument("--mask-checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage3" / "adaptive_fallback_holdout_v3")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    assert_valid_experiment_metadata(config)
    stage2 = json.loads(args.stage2_config.read_text(encoding="utf-8"))
    adaptive = config["adaptive_fallback_protocol"]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    model_info = json.loads(args.mask_result.read_text(encoding="utf-8"))
    model = TinyOccupancyCNN(channels=16).to(device)
    model.load_state_dict(torch.load(args.mask_checkpoint, map_location=device))
    model.eval()

    val_scenes, val_truths, val_ids = load_temporal_scenes(
        args.root, str(adaptive["validation_split"]), int(adaptive["validation_source_offset"]),
        int(adaptive["validation_scenes"]), int(adaptive["sources_per_scene"]),
    )
    calibrator, calibration = fit_calibrator(val_scenes, val_truths, config, model, model_info, device)
    validation_repeats = 1 if args.policy is not None else int(adaptive["validation_repeats"])
    val_predictions = []
    val_raw = []
    for repeat in range(validation_repeats):
        predictions, raw = infer_node_views(val_scenes, config, model, model_info, device, repeat, seed_offset=160_000_000)
        val_predictions.append(predictions)
        val_raw.append(raw)
    priors, expected_bits = validation_node_statistics(val_predictions[0], val_truths, config, stage2)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.policy is None:
        val_cache = prepare_cache(val_scenes, val_truths, val_predictions, val_raw, calibrator, config, stage2, 170_000_000)
        policy, audit = select_policy(val_cache, val_truths, config, priors, expected_bits)
        write_csv(args.out_dir / "validation_policy_audit.csv", audit)
        (args.out_dir / "frozen_policy.json").write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    else:
        policy = json.loads(args.policy.read_text(encoding="utf-8"))
    if args.validation_only:
        print(args.out_dir / "frozen_policy.json")
        return

    test_scenes, test_truths, test_ids = load_temporal_scenes(
        args.root, str(adaptive["holdout_split"]), int(adaptive["holdout_source_offset"]),
        int(adaptive["holdout_scenes"]), int(adaptive["sources_per_scene"]),
    )
    test_predictions = []
    test_raw = []
    for repeat in range(int(adaptive["holdout_repeats"])):
        predictions, raw = infer_node_views(test_scenes, config, model, model_info, device, repeat, seed_offset=180_000_000)
        test_predictions.append(predictions)
        test_raw.append(raw)
    test_cache = prepare_cache(test_scenes, test_truths, test_predictions, test_raw, calibrator, config, stage2, 190_000_000)
    disagreement = float(policy["pair_disagreement_threshold"])
    stale = float(policy["stale_trigger_s"])
    target_reports = int(policy["target_reports"])
    rows = []
    full_mean = evaluate_policy(test_cache, test_truths, config, priors, expected_bits, 4, 0.05, 2.0, "full_mean", "mean", True)
    for row in full_mean:
        row.update({"mean_attempted_nodes": float(len(next(s for s in config["robustness_protocol"]["scenarios"] if s["id"] == row["scenario"])["active_nodes"])), "ack_fallback_rate": 0.0, "risk_expansion_rate": 0.0})
    rows += full_mean
    fixed_two = evaluate_policy(test_cache, test_truths, config, priors, expected_bits, 2, 0.05, 2.0, "fixed_two_mean", "mean", False)
    for row in fixed_two:
        row.update({"mean_attempted_nodes": float(row["selected_node_count"]), "ack_fallback_rate": 0.0, "risk_expansion_rate": 0.0})
    rows += fixed_two
    rows += evaluate_adaptive(test_cache, test_truths, config, priors, expected_bits, disagreement, stale, target_reports, "adaptive_mean", "mean")
    rows += evaluate_adaptive(test_cache, test_truths, config, priors, expected_bits, disagreement, stale, target_reports, "proposed_adaptive_robust", str(policy["fusion_mode"]))
    summary = aggregate(rows)
    deltas = paired_method_difference(rows, "proposed_adaptive_robust", "full_mean", "adaptive_proposed_minus_full_mean")
    fusion_deltas = paired_method_difference(rows, "proposed_adaptive_robust", "adaptive_mean", "robust_fusion_minus_adaptive_mean")
    write_csv(args.out_dir / "holdout_trials.csv", rows)
    write_csv(args.out_dir / "holdout_summary.csv", summary)
    write_csv(args.out_dir / "paired_deltas.csv", deltas)
    write_csv(args.out_dir / "fusion_component_deltas.csv", fusion_deltas)
    (args.out_dir / "stage3_adaptive_fallback_report.md").write_text(
        make_report(summary, deltas, fusion_deltas, policy, priors, expected_bits, adaptive), encoding="utf-8"
    )
    result = {
        "config_sha256": sha256_file(args.config),
        "registry_sha256": sha256_file(PROJECT_DIR / "configs" / "stage3_holdout_registry_v3.json"),
        "validation_scene_ids_sha256": sha256_strings(val_ids),
        "holdout_scene_ids_sha256": sha256_strings(test_ids),
        "frozen_policy": policy,
        "validation_node_priors": priors.tolist(),
        "validation_expected_bits": expected_bits.tolist(),
        "calibration": calibration,
        "summary": summary,
        "paired_deltas": deltas,
        "fusion_component_deltas": fusion_deltas,
        "environment": environment_snapshot(["numpy", "torch"]),
        "claim_boundary": config["claim_boundary"],
    }
    (args.out_dir / "stage3_adaptive_fallback_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.out_dir / "stage3_adaptive_fallback_report.md")


if __name__ == "__main__":
    main()
