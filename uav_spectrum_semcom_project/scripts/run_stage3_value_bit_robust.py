"""Validation-selected value/bit node scheduling plus consensus-robust fusion."""

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

from spectrum_semcom.bit_budget import hard_box_packet_bits
from spectrum_semcom.digital_link import transmit_payload_analytic
from spectrum_semcom.models import TinyOccupancyCNN
from spectrum_semcom.multi_uav_fusion import NodeOccupancyReport, fuse_occupancy
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file, sha256_strings
from spectrum_semcom.research_scope import assert_valid_experiment_metadata
from run_stage2_digital_link import build_link_config, mean_std_ci, write_csv
from run_stage3_fusion_baselines import evaluate_estimates
from run_stage3_robustness import (
    fit_calibrator,
    infer_node_views,
    load_temporal_scenes,
    report_quality,
)
from simulate_resource_optimization import frame_channel_occupancy


def validation_node_statistics(predictions, truths, config, stage2) -> tuple[np.ndarray, np.ndarray]:
    protocol = config["protocol"]
    sensing = config["sensing_channel"]
    reporting = config["reporting_channel"]
    nodes = len(sensing["snr_db_per_node"])
    qualities = [[] for _ in range(nodes)]
    bits = [[] for _ in range(nodes)]
    for scene_index, truth in enumerate(truths):
        truth_occ = frame_channel_occupancy(truth, int(protocol["n_channels"]), str(protocol["channel_axis"]))
        for node in range(nodes):
            boxes = predictions[scene_index * nodes + node]
            occupancy = frame_channel_occupancy(boxes, int(protocol["n_channels"]), str(protocol["channel_axis"]))
            qualities[node].append(report_quality(occupancy, truth_occ, int(protocol["demand_channels"])))
            link = build_link_config(stage2, float(reporting["ebn0_db_per_node"][node]))
            result = transmit_payload_analytic(
                hard_box_packet_bits(len(boxes)),
                link,
                int(config["seed"]) + 110_000_000 + scene_index * 100 + node,
            )
            bits[node].append(float(result.transmitted_bits))
    return (
        np.asarray([np.mean(values) for values in qualities], dtype=np.float64),
        np.asarray([max(np.mean(values), 1.0) for values in bits], dtype=np.float64),
    )


def ranked_nodes(scenario, priors, expected_bits, config) -> list[int]:
    reporting = config["reporting_channel"]
    robust = config["robustness_protocol"]
    scores = []
    for node in [int(value) for value in scenario["active_nodes"]]:
        reliability = float(1.0 / (1.0 + np.exp(-(float(reporting["ebn0_db_per_node"][node]) - 3.0) / 2.0)))
        age = int(scenario["lag_steps"][node]) * float(robust["temporal_step_s"])
        freshness = float(np.exp(-age / float(config["protocol"]["freshness_tau_s"])))
        score = float(priors[node]) * reliability * freshness / float(expected_bits[node])
        scores.append((score, node))
    return [node for _, node in sorted(scores, reverse=True)]


def prepare_cache(scenes, truths, predictions_by_repeat, raw_by_repeat, calibrator, config, stage2, seed_offset: int):
    protocol = config["protocol"]
    sensing = config["sensing_channel"]
    reporting = config["reporting_channel"]
    robust = config["robustness_protocol"]
    nodes = len(sensing["snr_db_per_node"])
    cache: dict[tuple[int, str], list[dict[int, tuple[NodeOccupancyReport | None, float]]]] = {}
    for repeat, predictions in enumerate(predictions_by_repeat):
        raw_confidence = raw_by_repeat[repeat]
        for scenario_index, scenario in enumerate(robust["scenarios"]):
            active = set(int(value) for value in scenario["active_nodes"])
            forced = set(int(value) for value in scenario.get("forced_failure_nodes", []))
            anomaly = set(int(value) for value in scenario.get("anomaly_nodes", []))
            per_scene = []
            for scene_index in range(len(scenes)):
                records: dict[int, tuple[NodeOccupancyReport | None, float]] = {}
                for node in active:
                    if node in forced:
                        records[node] = (None, 0.0)
                        continue
                    lag = int(scenario["lag_steps"][node])
                    observation_index = max(0, scene_index - lag)
                    boxes = predictions[observation_index * nodes + node]
                    occupancy = frame_channel_occupancy(boxes, int(protocol["n_channels"]), str(protocol["channel_axis"]))
                    confidence = float(calibrator.predict(raw_confidence[observation_index, node]))
                    if node in anomaly:
                        occupancy = np.roll(occupancy, int(scenario.get("anomaly_channel_shift", 1)))
                        confidence = float(scenario.get("anomaly_confidence", 0.99))
                    link = build_link_config(stage2, float(reporting["ebn0_db_per_node"][node]))
                    result = transmit_payload_analytic(
                        hard_box_packet_bits(len(boxes)),
                        link,
                        int(config["seed"]) + seed_offset + repeat * 1_000_000 + scene_index * 100 + node,
                    )
                    report = None
                    if result.frame_success:
                        reliability = float(1.0 / (1.0 + np.exp(-(float(reporting["ebn0_db_per_node"][node]) - 3.0) / 2.0)))
                        report = NodeOccupancyReport(
                            occupancy,
                            float(sensing["snr_db_per_node"][node]),
                            confidence,
                            reliability,
                            lag * float(robust["temporal_step_s"]),
                        )
                    records[node] = (report, float(result.transmitted_bits))
                per_scene.append(records)
            cache[(repeat, str(scenario["id"]))] = per_scene
    return cache


def evaluate_policy(
    cache,
    truths,
    config,
    priors,
    expected_bits,
    selected_count: int,
    disagreement_scale: float,
    max_weight_ratio: float,
    method_name: str,
    fusion_method: str,
    use_all_nodes: bool,
) -> list[dict[str, Any]]:
    protocol = config["protocol"]
    scenarios = config["robustness_protocol"]["scenarios"]
    repeats = sorted({key[0] for key in cache})
    rows = []
    for repeat in repeats:
        for scenario in scenarios:
            ranking = ranked_nodes(scenario, priors, expected_bits, config)
            scheduled = set(ranking if use_all_nodes else ranking[: int(selected_count)])
            estimates = []
            total_bits = 0.0
            for records in cache[(repeat, str(scenario["id"]))]:
                reports = []
                for node in scheduled:
                    report, bits = records.get(node, (None, 0.0))
                    total_bits += bits
                    if report is not None:
                        reports.append(report)
                if not reports:
                    estimates.append(np.zeros(int(protocol["n_channels"]), dtype=np.float32))
                else:
                    estimates.append(
                        fuse_occupancy(
                            reports,
                            fusion_method,
                            float(protocol["occupancy_vote_threshold"]),
                            float(protocol["freshness_tau_s"]),
                            float(disagreement_scale),
                            float(max_weight_ratio),
                        )
                    )
            rows.append({
                "scenario": str(scenario["id"]),
                "method": method_name,
                "repeat": repeat,
                "selected_node_count": len(scheduled),
                **evaluate_estimates(np.stack(estimates), truths, protocol),
                "mean_transmitted_bits": float(total_bits / len(truths)),
            })
    return rows


def policy_audit(cache, truths, config, priors, expected_bits):
    improve = config["value_bit_robust_protocol"]
    baseline = evaluate_policy(cache, truths, config, priors, expected_bits, 4, 0.05, 2.0, "full_mean", "mean", True)
    baseline_average = float(np.mean([row["mean_occupancy_regret"] for row in baseline]))
    baseline_scenarios = {
        scenario: float(np.mean([row["mean_occupancy_regret"] for row in baseline if row["scenario"] == scenario]))
        for scenario in {row["scenario"] for row in baseline}
    }
    audit = []
    for count in improve["candidate_selected_node_counts"]:
        for scale in improve["candidate_disagreement_scales"]:
            for ratio in improve["candidate_max_weight_ratios"]:
                rows = evaluate_policy(
                    cache, truths, config, priors, expected_bits, int(count), float(scale), float(ratio),
                    "candidate", "robust_quality_weighted", False,
                )
                average_regret = float(np.mean([row["mean_occupancy_regret"] for row in rows]))
                mean_bits = float(np.mean([row["mean_transmitted_bits"] for row in rows]))
                scenario_regrets = {
                    scenario: float(np.mean([row["mean_occupancy_regret"] for row in rows if row["scenario"] == scenario]))
                    for scenario in baseline_scenarios
                }
                feasible = (
                    average_regret <= baseline_average + float(improve["average_regret_tolerance"])
                    and all(
                        scenario_regrets[scenario] <= baseline_scenarios[scenario] + float(improve["per_scenario_regret_tolerance"])
                        for scenario in baseline_scenarios
                    )
                )
                audit.append({
                    "selected_node_count": int(count),
                    "disagreement_scale": float(scale),
                    "max_weight_ratio": float(ratio),
                    "average_regret": average_regret,
                    "mean_transmitted_bits": mean_bits,
                    "baseline_average_regret": baseline_average,
                    "feasible": bool(feasible),
                })
    feasible = [row for row in audit if row["feasible"]]
    selected = min(feasible, key=lambda row: (row["mean_transmitted_bits"], row["average_regret"])) if feasible else min(audit, key=lambda row: row["average_regret"])
    return selected, audit, baseline


def aggregate(rows):
    output = []
    for scenario, method in sorted({(row["scenario"], row["method"]) for row in rows}):
        items = [row for row in rows if row["scenario"] == scenario and row["method"] == method]
        merged: dict[str, Any] = {"scenario": scenario, "method": method, "repeat_count": len(items)}
        for metric in ["clean_resource_rate", "mean_occupancy_regret", "oracle_equivalent_rate", "mean_transmitted_bits"]:
            for name, value in mean_std_ci([float(item[metric]) for item in items]).items():
                merged[f"{metric}_{name}"] = value
        output.append(merged)
    return output


def paired_method_difference(rows, method: str, reference: str, comparison: str):
    output = []
    scenarios = sorted({row["scenario"] for row in rows})
    for scenario in scenarios:
        base = {row["repeat"]: row for row in rows if row["scenario"] == scenario and row["method"] == reference}
        proposed = {row["repeat"]: row for row in rows if row["scenario"] == scenario and row["method"] == method}
        repeats = sorted(set(base) & set(proposed))
        regret = [proposed[r]["mean_occupancy_regret"] - base[r]["mean_occupancy_regret"] for r in repeats]
        bits = [proposed[r]["mean_transmitted_bits"] - base[r]["mean_transmitted_bits"] for r in repeats]
        row: dict[str, Any] = {"scenario": scenario, "comparison": comparison}
        for prefix, values in [("regret_delta", regret), ("bit_delta", bits)]:
            for name, value in mean_std_ci(values).items():
                row[f"{prefix}_{name}"] = value
        output.append(row)
    return output


def make_report(summary, deltas, component_deltas, selected, priors, expected_bits, improve) -> str:
    lines = [
        "# Stage 3 Value/Bit Node Selection and Robust Fusion",
        "",
        f"The policy was selected only on validation sources beginning at offset {improve['validation_source_offset']} and evaluated once on fresh test sources beginning at offset {improve['holdout_source_offset']}.",
        "Scheduling uses frozen validation node priors, declared link reliability and age, and expected transmitted bits. It does not inspect current occupancy or test truth.",
        "",
        "## Frozen policy",
        "",
        f"Selected nodes: {selected['selected_node_count']}; disagreement scale: {selected['disagreement_scale']}; maximum weight ratio: {selected['max_weight_ratio']}.",
        f"Validation node correctness priors: {np.round(priors, 4).tolist()}.",
        f"Validation expected bit/node: {np.round(expected_bits, 1).tolist()}.",
        "",
        "## Fresh holdout results",
        "",
        "| Scenario | Method | Clean rate | Regret | Oracle-equivalent | bit/decision |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['scenario']} | {row['method']} | {row['clean_resource_rate_mean']:.4f} | "
            f"{row['mean_occupancy_regret_mean']:.6f} | {row['oracle_equivalent_rate_mean']:.4f} | "
            f"{row['mean_transmitted_bits_mean']:.1f} |"
        )
    lines += ["", "## Proposed minus full-mean paired differences", "", "| Scenario | Regret delta (95% CI) | bit delta |", "|---|---:|---:|"]
    for row in deltas:
        lines.append(
            f"| {row['scenario']} | {row['regret_delta_mean']:.6f} "
            f"[{row['regret_delta_ci95_low']:.6f}, {row['regret_delta_ci95_high']:.6f}] | "
            f"{row['bit_delta_mean']:.1f} |"
        )
    lines += ["", "## Robust weighting minus selected-mean differences", "", "Both methods use the same selected nodes and identical bits.", "", "| Scenario | Regret delta (95% CI) |", "|---|---:|"]
    for row in component_deltas:
        lines.append(
            f"| {row['scenario']} | {row['regret_delta_mean']:.6f} "
            f"[{row['regret_delta_ci95_low']:.6f}, {row['regret_delta_ci95_high']:.6f}] |"
        )
    lines += [
        "",
        "## Claim boundary",
        "",
        "This remains semi-synthetic same-scene evidence. Lower bits with non-inferior regret supports a scheduling/efficiency claim, not yet a real spatial-diversity claim.",
    ]
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
    parser.add_argument("--diagnostic-scenes", type=int, default=None)
    parser.add_argument("--diagnostic-repeats", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage3" / "value_bit_robust_holdout_v2")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    assert_valid_experiment_metadata(config)
    stage2 = json.loads(args.stage2_config.read_text(encoding="utf-8"))
    improve = config["value_bit_robust_protocol"]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    model_info = json.loads(args.mask_result.read_text(encoding="utf-8"))
    model = TinyOccupancyCNN(channels=16).to(device)
    model.load_state_dict(torch.load(args.mask_checkpoint, map_location=device))
    model.eval()

    validation_count = int(args.diagnostic_scenes or improve["validation_scenes"])
    validation_repeats = int(args.diagnostic_repeats or improve["validation_repeats"])
    val_scenes, val_truths, val_ids = load_temporal_scenes(
        args.root, str(improve["validation_split"]), int(improve["validation_source_offset"]),
        validation_count, int(improve["sources_per_scene"]),
    )
    calibrator, calibration = fit_calibrator(val_scenes, val_truths, config, model, model_info, device)
    val_predictions = []
    val_raw = []
    for repeat in range(validation_repeats):
        predictions, raw = infer_node_views(val_scenes, config, model, model_info, device, repeat, seed_offset=120_000_000)
        val_predictions.append(predictions)
        val_raw.append(raw)
    priors, expected_bits = validation_node_statistics(val_predictions[0], val_truths, config, stage2)
    val_cache = prepare_cache(val_scenes, val_truths, val_predictions, val_raw, calibrator, config, stage2, 130_000_000)
    selected, audit, _ = policy_audit(val_cache, val_truths, config, priors, expected_bits)

    holdout_count = int(args.diagnostic_scenes or improve["holdout_scenes"])
    holdout_repeats = int(args.diagnostic_repeats or improve["holdout_repeats"])
    test_scenes, test_truths, test_ids = load_temporal_scenes(
        args.root, str(improve["holdout_split"]), int(improve["holdout_source_offset"]),
        holdout_count, int(improve["sources_per_scene"]),
    )
    test_predictions = []
    test_raw = []
    for repeat in range(holdout_repeats):
        predictions, raw = infer_node_views(test_scenes, config, model, model_info, device, repeat, seed_offset=140_000_000)
        test_predictions.append(predictions)
        test_raw.append(raw)
    test_cache = prepare_cache(test_scenes, test_truths, test_predictions, test_raw, calibrator, config, stage2, 150_000_000)
    count = int(selected["selected_node_count"])
    scale = float(selected["disagreement_scale"])
    ratio = float(selected["max_weight_ratio"])
    rows = []
    rows += evaluate_policy(test_cache, test_truths, config, priors, expected_bits, 4, scale, ratio, "full_mean", "mean", True)
    rows += evaluate_policy(test_cache, test_truths, config, priors, expected_bits, 4, scale, ratio, "full_quality", "quality_weighted", True)
    rows += evaluate_policy(test_cache, test_truths, config, priors, expected_bits, 4, scale, ratio, "full_robust", "robust_quality_weighted", True)
    rows += evaluate_policy(test_cache, test_truths, config, priors, expected_bits, count, scale, ratio, "selected_mean", "mean", False)
    rows += evaluate_policy(test_cache, test_truths, config, priors, expected_bits, count, scale, ratio, "proposed_value_bit_robust", "robust_quality_weighted", False)
    rows += evaluate_policy(test_cache, test_truths, config, priors, expected_bits, 1, scale, ratio, "best_prior_single", "mean", False)
    summary = aggregate(rows)
    deltas = paired_method_difference(
        rows, "proposed_value_bit_robust", "full_mean", "proposed_minus_full_mean"
    )
    component_deltas = paired_method_difference(
        rows, "proposed_value_bit_robust", "selected_mean", "robust_weighting_minus_selected_mean"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "validation_policy_audit.csv", audit)
    write_csv(args.out_dir / "holdout_trials.csv", rows)
    write_csv(args.out_dir / "holdout_summary.csv", summary)
    write_csv(args.out_dir / "paired_deltas.csv", deltas)
    write_csv(args.out_dir / "component_paired_deltas.csv", component_deltas)
    (args.out_dir / "stage3_value_bit_robust_report.md").write_text(
        make_report(summary, deltas, component_deltas, selected, priors, expected_bits, improve), encoding="utf-8"
    )
    result = {
        "config_sha256": sha256_file(args.config),
        "registry_sha256": sha256_file(PROJECT_DIR / "configs" / "stage3_holdout_registry_v2.json"),
        "validation_scene_ids_sha256": sha256_strings(val_ids),
        "holdout_scene_ids_sha256": sha256_strings(test_ids),
        "selected_policy": selected,
        "validation_node_priors": priors.tolist(),
        "validation_expected_bits": expected_bits.tolist(),
        "calibration": calibration,
        "summary": summary,
        "paired_deltas": deltas,
        "component_paired_deltas": component_deltas,
        "environment": environment_snapshot(["numpy", "torch"]),
        "claim_boundary": config["claim_boundary"],
    }
    (args.out_dir / "stage3_value_bit_robust_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.out_dir / "stage3_value_bit_robust_report.md")


if __name__ == "__main__":
    main()
