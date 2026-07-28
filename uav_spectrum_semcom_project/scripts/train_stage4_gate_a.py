"""Controlled five-seed Gate A training on frozen development caches."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
for path in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from spectrum_semcom.gate_a_model import VariableRateOccupancyHead
from spectrum_semcom.gate_a_training import evaluate_gate_a_head, load_gate_a_cache, precision_link_costs
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file
from spectrum_semcom.resource_losses import compute_gate_a_loss, gate_a_loss_configs, update_rate_dual_multiplier
from spectrum_semcom.stage4_protocol import assert_valid_stage4_protocol
from run_stage2_digital_link import build_link_config


def mean_ci(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean()); std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95_low": mean - 1.96 * std / math.sqrt(len(array)), "ci95_high": mean + 1.96 * std / math.sqrt(len(array))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROJECT_DIR / "configs" / "stage4_protocol.json")
    parser.add_argument("--stage2-config", type=Path, default=PROJECT_DIR / "configs" / "stage2_digital_link.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage4" / "gate_a_training_v1")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8")); assert_valid_stage4_protocol(protocol)
    stage2 = json.loads(args.stage2_config.read_text(encoding="utf-8"))
    settings = protocol["gate_a_controlled_training"]; cache_meta_path = PROJECT_DIR / settings["cache_result"]
    cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8")); task = cache_meta["task"]
    caches = {name: load_gate_a_cache(PROJECT_DIR / item["cache"]) for name, item in cache_meta["splits"].items()}
    train_ids, train_base, train_truth, _ = caches["train"]
    _, calibration_base, calibration_truth, _ = caches["calibration"]
    _, validation_base, validation_truth, _ = caches["validation"]
    link = build_link_config(stage2, 6.0); budget = float(settings["nominal_transmitted_bit_budget"])
    defaults = protocol["gate_a"]["loss_defaults"]; methods = list(settings["methods"])
    args.out_dir.mkdir(parents=True, exist_ok=True); (args.out_dir / "checkpoints").mkdir(exist_ok=True)
    run_rows: list[dict] = []; history_rows: list[dict] = []
    for seed in settings["training_seeds"]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        template = VariableRateOccupancyHead(int(task["n_channels"]), int(settings["hidden_dim"])); initial = copy.deepcopy(template.state_dict())
        for method in methods:
            model = VariableRateOccupancyHead(int(task["n_channels"]), int(settings["hidden_dim"])); model.load_state_dict(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
            option_costs = precision_link_costs(model, link); dual = 0.0; best_score = None; best_state = None; best_epoch = 0
            for epoch in range(1, int(settings["epochs"]) + 1):
                fraction = (epoch - 1) / max(1, int(settings["epochs"]) - 1)
                rate_temperature = float(settings["rate_temperature_start"]) * (float(settings["rate_temperature_end"]) / float(settings["rate_temperature_start"])) ** fraction
                generator = torch.Generator().manual_seed(int(seed) + epoch * 1009)
                permutation = torch.randperm(len(train_ids), generator=generator)
                epoch_costs: list[float] = []; model.train()
                for start in range(0, len(permutation), int(settings["batch_size"])):
                    index = permutation[start : start + int(settings["batch_size"])]
                    output = model(train_base[index], rate_temperature)
                    expected_cost = output.precision_probabilities @ option_costs
                    configs = gate_a_loss_configs(rate_weight=dual, resource_weight=float(defaults["resource_weight"]), miss_weight=float(defaults["miss_weight"]), tail_weight=float(defaults["tail_weight"]), calibration_weight=float(defaults["calibration_weight"]), inverse_temperature=float(defaults["inverse_temperature"]), cvar_alpha=float(defaults["cvar_alpha"]))
                    loss = compute_gate_a_loss(output.reconstructed_occupancy, train_truth[index], expected_cost, budget, configs[method], demand_channels=int(task["demand_channels"]))
                    optimizer.zero_grad(set_to_none=True); loss.total.backward(); optimizer.step()
                    epoch_costs.append(float(expected_cost.detach().mean()))
                if method in {"detection_plus_rate", "full_joint_loss"}:
                    dual = update_rate_dual_multiplier(dual, float(np.mean(epoch_costs)), budget, float(settings["dual_step_size"]), float(settings["dual_maximum"]))
                validation = evaluate_gate_a_head(model, validation_base, validation_truth, int(task["demand_channels"]), link, budget)
                score = (validation["budget_violation"], validation["mean_discrete_regret"], validation["cvar_0_9_regret"], validation["brier"])
                history_rows.append({"seed": seed, "method": method, "epoch": epoch, "dual": dual, "rate_temperature": rate_temperature, **{key: value for key, value in validation.items() if key != "precision_counts"}})
                if best_score is None or score < best_score:
                    best_score = score; best_epoch = epoch; best_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state)
            torch.save(best_state, args.out_dir / "checkpoints" / f"{method}_seed{seed}.pt")
            for split, base, truth in (("train", train_base, train_truth), ("calibration", calibration_base, calibration_truth), ("validation", validation_base, validation_truth)):
                metrics = evaluate_gate_a_head(model, base, truth, int(task["demand_channels"]), link, budget)
                run_rows.append({"seed": seed, "method": method, "split": split, "best_epoch": best_epoch, **metrics})
    with (args.out_dir / "run_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        flat = [{**row, "precision_counts": json.dumps(row["precision_counts"], sort_keys=True)} for row in run_rows]
        writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    with (args.out_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0])); writer.writeheader(); writer.writerows(history_rows)
    validation_rows = [row for row in run_rows if row["split"] == "validation"]
    summary = []
    for method in methods:
        items = [row for row in validation_rows if row["method"] == method]; merged = {"method": method}
        for metric in ("mean_discrete_regret", "cvar_0_9_regret", "clean_resource_rate", "brier", "mean_application_bits", "mean_nominal_transmitted_bits", "budget_violation"):
            merged[metric] = mean_ci([float(row[metric]) for row in items])
        summary.append(merged)
    baseline = {row["seed"]: row for row in validation_rows if row["method"] == "detection_only"}; paired = []
    for method in methods[1:]:
        values = {row["seed"]: row for row in validation_rows if row["method"] == method}
        for seed in settings["training_seeds"]:
            paired.append({"seed": seed, "method": method, "regret_delta_vs_detection_only": values[seed]["mean_discrete_regret"] - baseline[seed]["mean_discrete_regret"], "cvar_delta_vs_detection_only": values[seed]["cvar_0_9_regret"] - baseline[seed]["cvar_0_9_regret"], "bit_delta_vs_detection_only": values[seed]["mean_nominal_transmitted_bits"] - baseline[seed]["mean_nominal_transmitted_bits"]})
    result = {"experiment_id": "stage4_gate_a_controlled_training_v1", "protocol_sha256": sha256_file(args.protocol), "stage2_config_sha256": sha256_file(args.stage2_config), "cache_result_sha256": sha256_file(cache_meta_path), "settings": settings, "task": task, "summary": summary, "paired_deltas": paired, "environment": environment_snapshot(["numpy", "torch"]), "final_holdout_accessed": False, "claim_boundary": "Train/calibration/validation development evidence only."}
    (args.out_dir / "gate_a_training_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["# Stage 4 Gate A Controlled Training", "", "Five seeds; identical initialization and epoch permutations across methods. Development validation only.", "", "| Method | Regret mean [95% CI] | CVaR0.9 | Brier | nominal bit |", "|---|---:|---:|---:|---:|"]
    for row in summary:
        r=row["mean_discrete_regret"]; c=row["cvar_0_9_regret"]; lines.append(f"| {row['method']} | {r['mean']:.6f} [{r['ci95_low']:.6f}, {r['ci95_high']:.6f}] | {c['mean']:.6f} | {row['brier']['mean']:.6f} | {row['mean_nominal_transmitted_bits']['mean']:.1f} |")
    lines += ["", "No final holdout was created or accessed; this report selects the next development decision only."]
    (args.out_dir / "gate_a_training_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(args.out_dir / "gate_a_training_report.md")


if __name__ == "__main__":
    main()
