"""Train and evaluate the preregistered C2 value-per-bit predictor."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
for path in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_stage2_digital_link import build_link_config
from spectrum_semcom.counterfactual_value import candidate_message_cost, discrete_scene_regret, fuse_state, quantized_node_reports
from spectrum_semcom.multigranular_semantics import SemanticQuality
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file
from spectrum_semcom.resource_losses import empirical_cvar
from spectrum_semcom.stage4_protocol import assert_valid_stage4_protocol
from spectrum_semcom.value_prediction import CounterfactualValueNetwork, Standardization, build_deployable_value_features, fit_standardization, grouped_value_loss


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def build_training_groups(labels: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, Standardization]:
    availability = labels["quality_availability"]
    features = np.stack(
        [
            build_deployable_value_features(
                current_belief=labels["current_belief"][index],
                candidate_current_report=labels["candidate_current_report"][index],
                candidate_quality=labels["candidate_quality"][index],
                quality_availability=availability,
                current_state=labels["current_state"][index],
                candidate_node=int(labels["candidate_node"][index]),
                target_granularity=int(labels["target_granularity"][index]),
                expected_transmitted_bits=float(labels["expected_transmitted_bits"][index]),
            )
            for index in range(len(labels["scene_id"]))
        ]
    )
    targets = labels["value_per_expected_bit"].astype(np.float32)
    stats = fit_standardization(features, targets)
    normalized_features = (features - stats.feature_mean) / stats.feature_scale
    normalized_targets = (targets - stats.target_mean) / stats.target_scale
    groups: dict[tuple[str, tuple[int, ...]], list[int]] = defaultdict(list)
    for index, (scene_id, state) in enumerate(zip(labels["scene_id"], labels["current_state"])):
        groups[(str(scene_id), tuple(int(value) for value in state))].append(index)
    maximum = max(len(indices) for indices in groups.values())
    x = np.zeros((len(groups), maximum, features.shape[1]), dtype=np.float32)
    y = np.zeros((len(groups), maximum), dtype=np.float32)
    valid = np.zeros((len(groups), maximum), dtype=bool)
    for group_index, indices in enumerate(groups.values()):
        count = len(indices)
        x[group_index, :count] = normalized_features[indices]
        y[group_index, :count] = normalized_targets[indices]
        valid[group_index, :count] = True
    return x, y, valid, stats


def train_one_seed(x: np.ndarray, y: np.ndarray, valid: np.ndarray, stats: Standardization, settings: dict[str, Any], seed: int) -> tuple[CounterfactualValueNetwork, list[dict[str, float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    model = CounterfactualValueNetwork(x.shape[-1], tuple(int(value) for value in settings["hidden_dims"]), float(settings["dropout"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    x_tensor = torch.as_tensor(x)
    y_tensor = torch.as_tensor(y)
    valid_tensor = torch.as_tensor(valid)
    no_action = (0.0 - stats.target_mean) / stats.target_scale
    history = []
    for epoch in range(int(settings["epochs"])):
        generator = np.random.default_rng(seed + epoch * 1009)
        order = generator.permutation(len(x))
        totals = defaultdict(float)
        groups_seen = 0
        model.train()
        for start in range(0, len(order), int(settings["group_batch_size"])):
            indices = torch.as_tensor(order[start : start + int(settings["group_batch_size"])])
            prediction = model(x_tensor[indices])
            loss, parts = grouped_value_loss(prediction, y_tensor[indices], valid_tensor[indices], no_action, float(settings["huber_delta_standardized"]), float(settings["listwise_loss_weight"]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip_norm"]))
            optimizer.step()
            count = len(indices)
            groups_seen += count
            totals["loss"] += float(loss.detach()) * count
            totals["regression"] += float(parts["regression"]) * count
            totals["listwise"] += float(parts["listwise"]) * count
        history.append({"epoch": epoch + 1, **{key: value / groups_seen for key, value in totals.items()}})
    return model, history


def quality(row: np.ndarray) -> SemanticQuality:
    return SemanticQuality(*[float(value) for value in row])


def scene_candidates(scene_id: str, occupancy: np.ndarray, qualities: np.ndarray, availability: np.ndarray, links: tuple, truth: np.ndarray, demand: int, preview_bits: int, scheduler_reports: np.ndarray | None = None, scheduler_qualities: np.ndarray | None = None, scheduler_preview_bits: np.ndarray | None = None) -> tuple[dict[str, float], list[dict[str, Any]]]:
    reports = quantized_node_reports(occupancy, preview_bits)
    state = (1,) * occupancy.shape[0]
    belief = fuse_state(reports, state)
    baseline_regret = discrete_scene_regret(belief, truth, demand)
    blocks = np.convolve(truth, np.ones(demand) / demand, mode="valid")
    chosen = int(np.argmin(np.convolve(belief, np.ones(demand) / demand, mode="valid")))
    preview_cost = float(np.sum(scheduler_preview_bits)) if scheduler_preview_bits is not None else sum(candidate_message_cost(scene_id, node, 1, reports[1][node], quality(qualities[node]), links[node], probability_bits=preview_bits)[2] for node in range(occupancy.shape[0]))
    scheduler_reports = reports[1] if scheduler_reports is None else scheduler_reports
    scheduler_qualities = qualities if scheduler_qualities is None else scheduler_qualities
    baseline = {"regret": baseline_regret, "bits": preview_cost, "clean": float(blocks[chosen] <= 0.02), "chosen_occupancy": float(blocks[chosen]), "brier": float(np.mean(np.square(belief - truth))), "node": -1, "target": 0}
    candidates = []
    for node in range(occupancy.shape[0]):
        for target in (2, 3):
            _, _, expected_bits = candidate_message_cost(scene_id, node, target, reports[target][node], quality(qualities[node]), links[node])
            upgraded = list(state)
            upgraded[node] = target
            fused = fuse_state(reports, tuple(upgraded))
            regret = discrete_scene_regret(fused, truth, demand)
            chosen = int(np.argmin(np.convolve(fused, np.ones(demand) / demand, mode="valid")))
            features = build_deployable_value_features(current_belief=belief, candidate_current_report=scheduler_reports[node], candidate_quality=scheduler_qualities[node], quality_availability=availability, current_state=state, candidate_node=node, target_granularity=target, expected_transmitted_bits=expected_bits)
            candidates.append({"regret": regret, "bits": preview_cost + expected_bits, "incremental_bits": expected_bits, "clean": float(blocks[chosen] <= 0.02), "chosen_occupancy": float(blocks[chosen]), "brier": float(np.mean(np.square(fused - truth))), "node": node, "target": target, "features": features, "snr": float(scheduler_qualities[node, 0]), "confidence": float(scheduler_qualities[node, 1]), "reliability": float(scheduler_qualities[node, 8])})
    return baseline, candidates


def select_actions(baseline: dict[str, float], candidates: list[dict[str, Any]], models: list[CounterfactualValueNetwork], stats: Standardization) -> dict[str, dict[str, float]]:
    by_node_target = {(int(row["node"]), int(row["target"])): row for row in candidates}
    snr_node = max(range(4), key=lambda node: candidates[node * 2]["snr"])
    confidence_node = max(range(4), key=lambda node: candidates[node * 2]["confidence"])
    prior = max(candidates, key=lambda row: (1.0 / (1.0 + np.exp(-row["snr"] / 4.0))) * row["reliability"] / row["incremental_bits"])
    oracle = min([baseline, *candidates], key=lambda row: (row["regret"], row["bits"]))
    x = torch.as_tensor(np.stack([(row["features"] - stats.feature_mean) / stats.feature_scale for row in candidates]), dtype=torch.float32)
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            standardized = model(x).numpy()
        predictions.append(standardized * stats.target_scale + stats.target_mean)
    mean_prediction = np.mean(predictions, axis=0)
    learned_index = int(np.argmax(mean_prediction))
    learned = candidates[learned_index] if mean_prediction[learned_index] > 0 else baseline
    result = {"no_upgrade": baseline, "sensing_snr_G3": by_node_target[(snr_node, 3)], "confidence_G3": by_node_target[(confidence_node, 3)], "stage3_prior_value_per_bit": prior, "learned_counterfactual_value_per_bit": learned, "oracle": oracle}
    for seed_index, values in enumerate(predictions):
        index = int(np.argmax(values))
        result[f"learned_seed_{seed_index}"] = candidates[index] if values[index] > 0 else baseline
    return result


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    regrets = np.asarray([row["regret"] for row in rows])
    return {"mean_regret": float(regrets.mean()), "cvar_0_9_regret": float(empirical_cvar(torch.as_tensor(regrets, dtype=torch.float32), 0.9)), "mean_expected_transmitted_bits": float(np.mean([row["bits"] for row in rows])), "clean_resource_rate": float(np.mean([row["clean"] for row in rows])), "mean_chosen_occupancy": float(np.mean([row["chosen_occupancy"] for row in rows])), "brier": float(np.mean([row["brier"] for row in rows])), "no_upgrade_fraction": float(np.mean([row["node"] < 0 for row in rows]))}


def evaluate_split(cache: dict[str, np.ndarray], links: tuple, models: list[CounterfactualValueNetwork], stats: Standardization, demand: int, preview_bits: int, scheduler_cache: dict[str, np.ndarray] | None = None) -> tuple[dict[str, dict[str, float]], dict[str, list[dict[str, float]]]]:
    collected: dict[str, list[dict[str, float]]] = defaultdict(list)
    availability = cache["quality_availability"] if scheduler_cache is None else scheduler_cache["quality_availability"]
    for index, scene_id in enumerate(cache["scene_ids"]):
        if scheduler_cache is not None and not np.array_equal(cache["scene_ids"],scheduler_cache["scene_ids"]): raise ValueError("scheduler preview cache scene alignment mismatch")
        baseline, candidates = scene_candidates(str(scene_id), cache["node_occupancy"][index], cache["node_quality"][index], availability, links, cache["truth_occupancy"][index], demand, preview_bits, None if scheduler_cache is None else scheduler_cache["scheduler_report"][index], None if scheduler_cache is None else scheduler_cache["scheduler_quality"][index], None if scheduler_cache is None else scheduler_cache["expected_transmitted_bits"][index])
        for method, outcome in select_actions(baseline, candidates, models, stats).items():
            collected[method].append(outcome)
    return {method: summarize(rows) for method, rows in collected.items()}, collected


def paired_bootstrap(collected: dict[str, list[dict[str, float]]], repetitions: int, seed: int) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    learned = collected["learned_counterfactual_value_per_bit"]
    output = {}
    for baseline in ("no_upgrade", "sensing_snr_G3", "confidence_G3", "stage3_prior_value_per_bit"):
        regret = np.asarray([a["regret"] - b["regret"] for a, b in zip(learned, collected[baseline])])
        bits = np.asarray([a["bits"] - b["bits"] for a, b in zip(learned, collected[baseline])])
        indices = rng.integers(0, len(regret), size=(repetitions, len(regret)))
        regret_samples = regret[indices].mean(axis=1)
        bits_samples = bits[indices].mean(axis=1)
        output[baseline] = {"mean_regret_difference": float(regret.mean()), "regret_difference_ci95": [float(x) for x in np.quantile(regret_samples, [0.025, 0.975])], "mean_expected_bit_difference": float(bits.mean()), "expected_bit_difference_ci95": [float(x) for x in np.quantile(bits_samples, [0.025, 0.975])]}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROJECT_DIR / "configs" / "stage4_protocol.json")
    parser.add_argument("--stage2-config", type=Path, default=PROJECT_DIR / "configs" / "stage2_digital_link.json")
    parser.add_argument("--labels", type=Path, default=PROJECT_DIR / "results" / "stage4" / "counterfactual_labels_v1" / "train_counterfactual_labels.npz")
    parser.add_argument("--cache-result", type=Path, default=PROJECT_DIR / "results" / "stage4" / "multinode_cache_v1" / "multinode_cache_result.json")
    parser.add_argument("--preview-bits", type=int, choices=(1, 2), default=1)
    parser.add_argument("--scheduler-preview-result", type=Path)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage4" / "value_network_v1")
    args = parser.parse_args()
    if not args.labels.is_absolute(): args.labels = PROJECT_DIR / args.labels
    if not args.out_dir.is_absolute(): args.out_dir = PROJECT_DIR / args.out_dir
    protocol = json.loads(args.protocol.read_text(encoding="utf-8")); assert_valid_stage4_protocol(protocol)
    settings = protocol["value_prediction_protocol"]
    stage2 = json.loads(args.stage2_config.read_text(encoding="utf-8"))
    cache_meta = json.loads(args.cache_result.read_text(encoding="utf-8"))
    links = tuple(build_link_config(stage2, float(value)) for value in protocol["counterfactual_value_protocol"]["reporting_ebn0_db"])
    labels = load_npz(args.labels)
    x, y, valid, stats = build_training_groups(labels)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = []
    seed_records = []
    for seed in settings["training_seeds"]:
        model, history = train_one_seed(x, y, valid, stats, settings, int(seed))
        checkpoint = args.out_dir / f"value_network_seed{seed}.pt"
        torch.save({"state_dict": model.state_dict(), "input_dim": x.shape[-1], "feature_mean": stats.feature_mean, "feature_scale": stats.feature_scale, "target_mean": stats.target_mean, "target_scale": stats.target_scale, "seed": int(seed)}, checkpoint)
        models.append(model)
        seed_records.append({"seed": int(seed), "checkpoint": str(checkpoint.relative_to(PROJECT_DIR)).replace("\\", "/"), "checkpoint_sha256": sha256_file(checkpoint), "final_training": history[-1]})
    demand = int(cache_meta["task"]["demand_channels"])
    calibration_cache = load_npz(PROJECT_DIR / cache_meta["splits"]["calibration"]["cache"])
    validation_cache = load_npz(PROJECT_DIR / cache_meta["splits"]["validation"]["cache"])
    preview_meta=json.loads(args.scheduler_preview_result.read_text(encoding="utf-8")) if args.scheduler_preview_result else None
    calibration_scheduler=load_npz(PROJECT_DIR/preview_meta["splits"]["calibration"]["cache"]) if preview_meta else None
    validation_scheduler=load_npz(PROJECT_DIR/preview_meta["splits"]["validation"]["cache"]) if preview_meta else None
    calibration_metrics, _ = evaluate_split(calibration_cache, links, models, stats, demand, args.preview_bits, calibration_scheduler)
    validation_metrics, validation_rows = evaluate_split(validation_cache, links, models, stats, demand, args.preview_bits, validation_scheduler)
    seed_metrics = {str(seed): validation_metrics.pop(f"learned_seed_{index}") for index, seed in enumerate(settings["training_seeds"])}
    result = {"experiment_id": "stage4_c2_value_network_v1", "protocol_sha256": sha256_file(args.protocol), "label_sha256": sha256_file(args.labels), "multinode_cache_result_sha256": sha256_file(args.cache_result), "training_groups": len(x), "training_rows": int(valid.sum()), "feature_dimension": int(x.shape[-1]), "target_mean": stats.target_mean, "target_scale": stats.target_scale, "seeds": seed_records, "calibration_aggregate_metrics_no_tuning": {key: value for key, value in calibration_metrics.items() if not key.startswith("learned_seed_")}, "validation_metrics": validation_metrics, "individual_seed_validation_metrics": seed_metrics, "paired_scene_bootstrap": paired_bootstrap(validation_rows, int(settings["bootstrap_repetitions"]), int(protocol["seed"])), "truth_access_audit": {"train_labels_loaded": True, "calibration_truth_used_for_aggregate_diagnostic_only": True, "validation_truth_used_for_aggregate_method_evaluation_only": True, "calibration_or_validation_labels_saved": False, "candidate_high_precision_report_used_as_predictor_input": False, "final_holdout_accessed": False}, "claim_boundary": "Controlled same-scene H2/C2 validation; not final H2 and not real H3 evidence.", "environment": environment_snapshot(["numpy", "torch"])}
    result["experiment_id"] = "stage4_c2_value_network_scheduler_sketch_v1" if preview_meta else f"stage4_c2_value_network_G1q{args.preview_bits}"
    result["scheduler_preview_used"] = preview_meta is not None
    result["preview_probability_bits"] = args.preview_bits
    result_path = args.out_dir / "value_network_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    learned = validation_metrics["learned_counterfactual_value_per_bit"]
    lines = ["# Stage 4 C2 Value Network Validation", "", f"G1 preview probability bits: {args.preview_bits}.", "", f"Train groups/rows: {len(x)} / {int(valid.sum())}; features: {x.shape[-1]}; five preregistered seeds, fixed final epoch.", "", "| Method | regret | CVaR | expected bit | clean rate | no upgrade |", "|---|---:|---:|---:|---:|---:|"]
    for method in settings["comparison_methods"]:
        metric = validation_metrics[method]
        lines.append(f"| {method} | {metric['mean_regret']:.6f} | {metric['cvar_0_9_regret']:.6f} | {metric['mean_expected_transmitted_bits']:.2f} | {metric['clean_resource_rate']:.4f} | {metric['no_upgrade_fraction']:.4f} |")
    lines.extend(["", f"Learned ensemble validation regret/bit: {learned['mean_regret']:.6f} / {learned['mean_expected_transmitted_bits']:.2f}.", "", "No validation labels were serialized; no final holdout was created or accessed."])
    (args.out_dir / "value_network_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_dir / "value_network_report.md")


if __name__ == "__main__":
    main()
