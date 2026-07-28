#!/usr/bin/env python
"""Train and evaluate post-Final C1-v2 on development caches only."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage2_digital_link import build_link_config  # noqa: E402
from spectrum_semcom.c1_v2 import calibration_shift_views  # noqa: E402
from spectrum_semcom.digital_link import nominal_transmitted_bits  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.gate_a_model import FixedPrecisionOccupancyHead  # noqa: E402
from spectrum_semcom.gate_a_training import load_gate_a_cache  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402
from spectrum_semcom.resource_losses import GateALossConfig, compute_gate_a_loss, contiguous_block_ranking_loss, discrete_occupancy_regret, empirical_cvar  # noqa: E402


def evaluate(model, base, truth, demand, fixed_cost, clean_threshold=0.02):
    model.eval()
    with torch.inference_mode():
        predicted = model.infer(base)
        regrets = discrete_occupancy_regret(predicted, truth, demand, reduction="none")
        pred_blocks = torch.nn.functional.avg_pool1d(predicted[:, None], demand, stride=1).squeeze(1)
        truth_blocks = torch.nn.functional.avg_pool1d(truth[:, None], demand, stride=1).squeeze(1)
        chosen = torch.argmin(pred_blocks, dim=1)
        chosen_truth = truth_blocks.gather(1, chosen[:, None]).squeeze(1)
    return {
        "mean_discrete_regret": float(regrets.mean()),
        "cvar_0_9_regret": float(empirical_cvar(regrets, 0.9)),
        "clean_resource_rate": float((chosen_truth <= clean_threshold).float().mean()),
        "brier": float(torch.square(predicted - truth).mean()),
        "mean_nominal_transmitted_bits": float(fixed_cost),
    }, regrets.numpy()


def bootstrap_difference(proposed, baseline, repetitions, seed):
    proposed = np.asarray(proposed); baseline = np.asarray(baseline)
    rng = np.random.default_rng(seed); indices = rng.integers(0, len(proposed), size=(repetitions, len(proposed)))
    mean_samples = (proposed[indices] - baseline[indices]).mean(axis=1)
    tail = max(1, int(np.ceil(0.1 * len(proposed))))
    left = np.partition(proposed[indices], len(proposed) - tail, axis=1)[:, -tail:].mean(axis=1)
    right = np.partition(baseline[indices], len(proposed) - tail, axis=1)[:, -tail:].mean(axis=1)
    return {
        "mean_regret_difference": float(np.mean(proposed - baseline)),
        "mean_regret_ci95": np.quantile(mean_samples, [0.025, 0.975]).tolist(),
        "cvar_difference": float(np.mean(np.sort(proposed)[-tail:]) - np.mean(np.sort(baseline)[-tail:])),
        "cvar_ci95": np.quantile(left - right, [0.025, 0.975]).tolist(),
        "nominal_bit_difference": 0.0,
        "bootstrap_repetitions": repetitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/stage4_c1_v2_exploratory.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results/stage4/c1_v2_exploratory_v1")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("final_data_accessed_for_training_or_selection") is not False or config.get("validation_used_for_selection") is not False:
        raise ValueError("C1-v2 governance flags are invalid")
    cache_meta_path = PROJECT_DIR / config["development_cache"]
    if "final_holdout" in cache_meta_path.as_posix().lower() or "aerpaw" in cache_meta_path.as_posix().lower():
        raise ValueError("Final/AERPAW data are forbidden in C1-v2 training")
    meta = json.loads(cache_meta_path.read_text(encoding="utf-8")); task = meta["task"]
    caches = {name: load_gate_a_cache(PROJECT_DIR / row["cache"]) for name, row in meta["splits"].items()}
    if set(caches) != set(config["allowed_splits"]):
        raise ValueError("development cache split set differs from the exploratory protocol")
    _, train_base, train_truth, _ = caches["train"]
    _, calibration_base, calibration_truth, _ = caches["calibration"]
    _, validation_base, validation_truth, _ = caches["validation"]
    stage2 = json.loads((PROJECT_DIR / "configs/stage2_digital_link.json").read_text(encoding="utf-8"))
    link = build_link_config(stage2, 6.0)
    n_channels = int(task["n_channels"]); demand = int(task["demand_channels"])
    app_bits = 151 + n_channels * int(config["fixed_probability_bits"])
    fixed_cost = nominal_transmitted_bits(app_bits, link)
    args.out_dir.mkdir(parents=True, exist_ok=True); (args.out_dir / "checkpoints").mkdir(exist_ok=True)
    run_rows, validation_scene_regrets = [], {method: [] for method in config["methods"]}

    for seed in config["training_seeds"]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        template = FixedPrecisionOccupancyHead(n_channels, int(config["hidden_dim"]), int(config["fixed_probability_bits"]))
        initial = copy.deepcopy(template.state_dict())
        permutation_generator = torch.Generator().manual_seed(seed)
        baseline_method = config["methods"][0]; baseline_best_state = None
        for method in config["methods"]:
            model = FixedPrecisionOccupancyHead(n_channels, int(config["hidden_dim"]), int(config["fixed_probability_bits"]))
            warm_start = method != baseline_method and bool(config.get("warm_start_proposed_from_baseline", False))
            model.load_state_dict(baseline_best_state if warm_start else initial)
            learning_rate = float(config.get("fine_tune_learning_rate", config["learning_rate"])) if warm_start else float(config["learning_rate"])
            epochs = int(config.get("fine_tune_epochs", config["epochs"])) if warm_start else int(config["epochs"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=float(config["weight_decay"]))
            best_score = None; best_state = None; best_epoch = 0
            loss_config = GateALossConfig(
                method,
                resource_weight=float(config["resource_weight"]) if method != baseline_method else 0.0,
                tail_weight=float(config["tail_weight"]) if method != baseline_method else 0.0,
                inverse_temperature=float(config["inverse_temperature"]), cvar_alpha=float(config["cvar_alpha"]),
            )
            for epoch in range(1, epochs + 1):
                permutation_generator.manual_seed(seed + epoch * 1009)
                order = torch.randperm(len(train_base), generator=permutation_generator); model.train()
                for start in range(0, len(order), int(config["batch_size"])):
                    index = order[start:start + int(config["batch_size"])]
                    view_generator = torch.Generator().manual_seed(seed + epoch * 100_003 + start)
                    views = calibration_shift_views(train_base[index], config["view_specs"], generator=view_generator)
                    losses, predictions = [], []
                    for view in views:
                        _, reconstructed = model(view); predictions.append(reconstructed)
                        expected = torch.full((len(index),), float(fixed_cost), dtype=reconstructed.dtype)
                        objective = compute_gate_a_loss(reconstructed, train_truth[index], expected, fixed_cost, loss_config, demand_channels=demand).total
                        if method != baseline_method:
                            objective = objective + float(config.get("ranking_weight", 0.0)) * contiguous_block_ranking_loss(reconstructed, train_truth[index], demand, float(config.get("ranking_margin", 0.02)))
                        losses.append(objective)
                    stacked = torch.stack(losses); robust = stacked.mean() + float(config["worst_view_weight"]) * (stacked.max() - stacked.mean())
                    consistency = torch.stack([torch.square(item - predictions[0]).mean() for item in predictions[1:]]).mean()
                    total = robust + float(config["consistency_weight"]) * consistency
                    optimizer.zero_grad(set_to_none=True); total.backward(); optimizer.step()
                calibration_metrics, _ = evaluate(model, calibration_base, calibration_truth, demand, fixed_cost)
                score = (calibration_metrics["mean_discrete_regret"], calibration_metrics["cvar_0_9_regret"], calibration_metrics["brier"])
                if best_score is None or score < best_score:
                    best_score = score; best_state = copy.deepcopy(model.state_dict()); best_epoch = epoch
            model.load_state_dict(best_state)
            if method == baseline_method:
                baseline_best_state = copy.deepcopy(best_state)
            checkpoint = args.out_dir / "checkpoints" / f"{method}_seed{seed}.pt"; torch.save(best_state, checkpoint)
            for split, (_, base, truth, _) in caches.items():
                metrics, regrets = evaluate(model, base, truth, demand, fixed_cost)
                run_rows.append({"seed": seed, "method": method, "split": split, "best_epoch": best_epoch, **metrics})
                if split == "validation": validation_scene_regrets[method].append(regrets)

    aggregated = {method: np.mean(np.stack(rows), axis=0) for method, rows in validation_scene_regrets.items()}
    comparison = bootstrap_difference(aggregated[config["methods"][1]], aggregated[config["methods"][0]], int(config["bootstrap_repetitions"]), int(config["bootstrap_seed"]))
    comparison["requirements"] = {
        "mean_regret_ci_upper_below_0": comparison["mean_regret_ci95"][1] < 0,
        "cvar_ci_upper_below_0": comparison["cvar_ci95"][1] < 0,
        "nominal_bit_difference_exactly_0": comparison["nominal_bit_difference"] == 0,
    }
    comparison["exploratory_success"] = all(comparison["requirements"].values())
    result = {
        "experiment_id": config["experiment_id"], "config_sha256": sha256_file(args.config), "development_cache_sha256": sha256_file(cache_meta_path),
        "fixed_probability_bits": config["fixed_probability_bits"], "fixed_application_bits": app_bits, "fixed_nominal_transmitted_bits": fixed_cost,
        "run_metrics": run_rows, "paired_validation_bootstrap": comparison,
        "governance": {"final_access_state_observed": json.loads((PROJECT_DIR / "configs/stage4_final_access_state.json").read_text(encoding="utf-8"))["status"], "final_data_loaded": False, "validation_used_for_checkpoint_selection": False},
        "environment": environment_snapshot(["numpy", "torch"]), "claim_boundary": config["claim_boundary"],
    }
    atomic_write_json(args.out_dir / "c1_v2_result.json", result)
    lines = ["# C1-v2 Post-Final Exploratory Development Result", "", f"Fixed precision: {config['fixed_probability_bits']} bit; nominal transmitted bit: {fixed_cost} for both methods.", "", "| Metric | Proposed - baseline | 95% CI | Pass |", "|---|---:|---:|---|", f"| mean regret | {comparison['mean_regret_difference']:.6f} | [{comparison['mean_regret_ci95'][0]:.6f}, {comparison['mean_regret_ci95'][1]:.6f}] | {comparison['requirements']['mean_regret_ci_upper_below_0']} |", f"| CVaR0.9 | {comparison['cvar_difference']:.6f} | [{comparison['cvar_ci95'][0]:.6f}, {comparison['cvar_ci95'][1]:.6f}] | {comparison['requirements']['cvar_ci_upper_below_0']} |", f"| nominal bit | 0 | [0, 0] | True |", "", "This is post-Final exploratory evidence on the original development splits. It does not replace or repair the completed AERPAW Final result."]
    (args.out_dir / "c1_v2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out_dir / 'c1_v2_result.json'), "exploratory_success": comparison["exploratory_success"], "final_data_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
