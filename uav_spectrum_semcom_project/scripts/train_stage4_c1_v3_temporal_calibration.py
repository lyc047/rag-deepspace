#!/usr/bin/env python
"""Train the frozen C1-v3 grid and select one candidate on new calibration data."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.aerpaw_final_inference import power_block_regret_db  # noqa: E402
from spectrum_semcom.c1_v2 import calibration_shift_views  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.gate_a_model import FixedPrecisionOccupancyHead  # noqa: E402
from spectrum_semcom.gate_a_training import load_gate_a_cache  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402
from spectrum_semcom.resource_losses import (  # noqa: E402
    GateALossConfig,
    compute_gate_a_loss,
    contiguous_block_ranking_loss,
    discrete_occupancy_regret,
    empirical_cvar,
    tail_contiguous_block_ranking_loss,
)


def dev_metrics(model, base, truth, demand):
    model.eval()
    with torch.inference_mode():
        predicted = model.infer(base)
        regrets = discrete_occupancy_regret(predicted, truth, demand, reduction="none")
    return (
        float(regrets.mean()),
        float(empirical_cvar(regrets, 0.9)),
        float(torch.square(predicted - truth).mean()),
    )


def real_proxy_metrics(model, base: torch.Tensor, power: np.ndarray, demand: int):
    model.eval()
    with torch.inference_mode():
        predicted = model.infer(base).cpu().numpy()
    regrets = np.asarray(
        [power_block_regret_db(predicted[index], power[index], demand) for index in range(len(power))],
        dtype=np.float64,
    )
    return regrets, float(np.mean(np.square(predicted - base.cpu().numpy())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/stage4_c1_v3_temporal_calibration.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results/stage4/c1_v3_temporal_calibration_v1")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol_path = PROJECT_DIR / config["temporal_protocol"]
    inventory_path = PROJECT_DIR / config["prefinal_inventory"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    access = json.loads((PROJECT_DIR / "configs/stage4_c1_temporal_access_state.json").read_text(encoding="utf-8"))
    if access.get("status") != "not_accessed" or access.get("access_count") != 0:
        raise RuntimeError("candidate training is forbidden after temporal confirmation access")
    if inventory.get("confirmation_measurement_values_interpreted") is not False:
        raise ValueError("confirmation holdout is no longer opaque")
    grid = protocol["candidate_grid"]
    if config["tail_ranking_weights"] != grid["tail_ranking_weights"] or config["ranking_margins"] != grid["ranking_margins"]:
        raise ValueError("training grid differs from the frozen temporal protocol")
    if config["fixed_probability_bits"] != grid["fixed_probability_bits"][0]:
        raise ValueError("fixed precision differs from the frozen temporal protocol")

    development_meta_path = PROJECT_DIR / config["development_cache"]
    development = json.loads(development_meta_path.read_text(encoding="utf-8"))
    train = load_gate_a_cache(PROJECT_DIR / development["splits"]["train"]["cache"])
    calibration = load_gate_a_cache(PROJECT_DIR / development["splits"]["calibration"]["cache"])
    _, train_base, train_truth, _ = train
    _, dev_cal_base, dev_cal_truth, _ = calibration
    task = development["task"]
    n_channels = int(task["n_channels"])
    demand = int(task["demand_channels"])
    real_cache_path = Path(inventory["calibration_cache"])
    if sha256_file(real_cache_path) != inventory["calibration_cache_sha256"]:
        raise ValueError("new temporal calibration cache hash mismatch")
    with np.load(real_cache_path, allow_pickle=False) as cache:
        real_scene_ids = cache["scene_ids"].astype(str)
        real_base = torch.as_tensor(cache["base_occupancy"], dtype=torch.float32)
        real_power = np.asarray(cache["channel_power_dbm"], dtype=np.float64)
    if len(real_scene_ids) != inventory["calibration_scene_count"] or len(real_scene_ids) != 60:
        raise ValueError("new calibration scene count mismatch")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    train_base = train_base.to(device); train_truth = train_truth.to(device)
    dev_cal_base = dev_cal_base.to(device); dev_cal_truth = dev_cal_truth.to(device)
    real_base_device = real_base.to(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.out_dir / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    candidates = [
        {"candidate_id": f"tail{weight:g}_margin{margin:g}", "tail_ranking_weight": float(weight), "ranking_margin": float(margin)}
        for weight, margin in itertools.product(config["tail_ranking_weights"], config["ranking_margins"])
    ]
    per_candidate = {row["candidate_id"]: {"regrets": [], "brier": [], "runs": []} for row in candidates}
    baseline_regrets, baseline_brier, baseline_checkpoints = [], [], []

    for seed in config["training_seeds"]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        baseline_path = PROJECT_DIR / config["baseline_checkpoint_pattern"].format(seed=seed)
        baseline = FixedPrecisionOccupancyHead(n_channels, int(config["hidden_dim"]), int(config["fixed_probability_bits"])).to(device)
        baseline.load_state_dict(torch.load(baseline_path, map_location=device, weights_only=True))
        regrets, brier = real_proxy_metrics(baseline, real_base_device, real_power, demand)
        baseline_regrets.append(regrets); baseline_brier.append(brier)
        baseline_checkpoints.append({"seed": seed, "path": str(baseline_path.resolve()), "sha256": sha256_file(baseline_path)})

        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            model = FixedPrecisionOccupancyHead(n_channels, int(config["hidden_dim"]), int(config["fixed_probability_bits"])).to(device)
            model.load_state_dict(copy.deepcopy(baseline.state_dict()))
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(config["fine_tune_learning_rate"]), weight_decay=float(config["weight_decay"])
            )
            loss_config = GateALossConfig(
                "c1_v3_tail_ranking",
                resource_weight=float(config["resource_weight"]),
                tail_weight=float(config["soft_tail_weight"]),
                inverse_temperature=float(config["inverse_temperature"]),
                cvar_alpha=float(config["cvar_alpha"]),
            )
            best_score = None; best_state = None; best_epoch = None
            for epoch in range(1, int(config["fine_tune_epochs"]) + 1):
                order = torch.randperm(len(train_base), generator=torch.Generator().manual_seed(seed + epoch * 1009), device="cpu").to(device)
                model.train()
                for start in range(0, len(order), int(config["batch_size"])):
                    index = order[start : start + int(config["batch_size"])]
                    generator = torch.Generator(device=device).manual_seed(seed + epoch * 100_003 + start)
                    views = calibration_shift_views(train_base[index], config["view_specs"], generator=generator)
                    losses, predictions = [], []
                    for view in views:
                        _, reconstructed = model(view)
                        predictions.append(reconstructed)
                        expected = torch.ones(len(index), dtype=reconstructed.dtype, device=device)
                        objective = compute_gate_a_loss(
                            reconstructed,
                            train_truth[index],
                            expected,
                            1.0,
                            loss_config,
                            demand_channels=demand,
                        ).total
                        objective = objective + float(config["ranking_weight"]) * contiguous_block_ranking_loss(
                            reconstructed, train_truth[index], demand, candidate["ranking_margin"]
                        )
                        objective = objective + candidate["tail_ranking_weight"] * tail_contiguous_block_ranking_loss(
                            reconstructed,
                            train_truth[index],
                            demand,
                            candidate["ranking_margin"],
                            float(config["cvar_alpha"]),
                        )
                        losses.append(objective)
                    stacked = torch.stack(losses)
                    robust = stacked.mean() + float(config["worst_view_weight"]) * (stacked.max() - stacked.mean())
                    consistency = torch.stack([torch.square(item - predictions[0]).mean() for item in predictions[1:]]).mean()
                    total = robust + float(config["consistency_weight"]) * consistency
                    optimizer.zero_grad(set_to_none=True); total.backward(); optimizer.step()
                score = dev_metrics(model, dev_cal_base, dev_cal_truth, demand)
                if best_score is None or score < best_score:
                    best_score = score; best_state = copy.deepcopy(model.state_dict()); best_epoch = epoch
            model.load_state_dict(best_state)
            checkpoint = checkpoint_root / f"{candidate_id}_seed{seed}.pt"
            torch.save(best_state, checkpoint)
            regrets, brier = real_proxy_metrics(model, real_base_device, real_power, demand)
            per_candidate[candidate_id]["regrets"].append(regrets)
            per_candidate[candidate_id]["brier"].append(brier)
            per_candidate[candidate_id]["runs"].append(
                {"seed": seed, "best_epoch": best_epoch, "development_calibration_score": list(best_score), "checkpoint": str(checkpoint.relative_to(PROJECT_DIR)).replace("\\", "/"), "checkpoint_sha256": sha256_file(checkpoint)}
            )

    baseline_scene = np.mean(np.stack(baseline_regrets), axis=0)
    baseline_metrics = {
        "mean_regret_db": float(np.mean(baseline_scene)),
        "cvar_0_9_regret_db": float(np.mean(np.sort(baseline_scene)[-max(1, int(np.ceil(0.1 * len(baseline_scene)))):])),
        "brier_to_input_proxy": float(np.mean(baseline_brier)),
    }
    candidate_rows = []
    for candidate in candidates:
        values = per_candidate[candidate["candidate_id"]]
        scene = np.mean(np.stack(values["regrets"]), axis=0)
        tail_count = max(1, int(np.ceil(0.1 * len(scene))))
        metrics = {
            "mean_regret_db": float(np.mean(scene)),
            "cvar_0_9_regret_db": float(np.mean(np.sort(scene)[-tail_count:])),
            "brier_to_input_proxy": float(np.mean(values["brier"])),
        }
        candidate_rows.append({**candidate, **metrics, "runs": values["runs"]})
    selected = min(candidate_rows, key=lambda row: (row["mean_regret_db"], row["cvar_0_9_regret_db"], row["brier_to_input_proxy"], row["candidate_id"]))
    result = {
        "experiment_id": config["experiment_id"],
        "status": "one_candidate_selected_and_frozen_before_confirmation_access",
        "config_sha256": sha256_file(args.config),
        "protocol_sha256": sha256_file(protocol_path),
        "prefinal_inventory_sha256": sha256_file(inventory_path),
        "development_cache_metadata_sha256": sha256_file(development_meta_path),
        "device": device,
        "baseline_calibration_metrics": baseline_metrics,
        "baseline_checkpoints": baseline_checkpoints,
        "candidate_calibration_results": candidate_rows,
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate": selected,
        "selection_order": config["selection_order"],
        "governance": {
            "original_development_validation_loaded": False,
            "original_three_site_final_loaded": False,
            "new_confirmation_measurement_values_loaded": False,
            "new_confirmation_access_count": 0,
            "all_nine_preregistered_candidates_disclosed": len(candidate_rows) == 9,
        },
        "environment": environment_snapshot(["numpy", "torch"]),
        "claim_boundary": config["claim_boundary"],
    }
    output = args.out_dir / "c1_v3_calibration_result.json"
    atomic_write_json(output, result)
    manifest = {
        "version": "1.0",
        "status": "candidate_frozen_before_confirmation_access",
        "selected_candidate_id": selected["candidate_id"],
        "tail_ranking_weight": selected["tail_ranking_weight"],
        "ranking_margin": selected["ranking_margin"],
        "fixed_probability_bits": int(config["fixed_probability_bits"]),
        "training_seeds": config["training_seeds"],
        "candidate_checkpoints": selected["runs"],
        "baseline_checkpoints": baseline_checkpoints,
        "calibration_result_path": str(output.relative_to(PROJECT_DIR)).replace("\\", "/"),
        "calibration_result_sha256": sha256_file(output),
        "confirmation_accessed": False,
    }
    manifest_path = args.out_dir / "selected_candidate_manifest.json"
    atomic_write_json(manifest_path, manifest)
    print(json.dumps({"output": str(output), "selected_candidate": selected["candidate_id"], "mean_regret_db": selected["mean_regret_db"], "cvar_0_9_regret_db": selected["cvar_0_9_regret_db"], "confirmation_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
