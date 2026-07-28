#!/usr/bin/env python
"""Aggregate validation diagnostics for the frozen C2 value predictor.

No validation candidate labels or per-scene rows are serialized.  Truth is
used only inside this process to calculate aggregate evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# The packaged Windows NumPy and PyTorch builds load distinct Intel OpenMP
# copies.  Keep this single-threaded diagnostic consistent with other project
# scripts that use the same local environment.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
for path in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_stage2_digital_link import build_link_config
from train_stage4_value_network import load_npz, scene_candidates
from spectrum_semcom.ranking_metrics import spearman_correlation, top_k_recall
from spectrum_semcom.value_prediction import CounterfactualValueNetwork, Standardization


def load_models(root: Path, protocol: dict) -> tuple[list[CounterfactualValueNetwork], Standardization]:
    models = []
    stats = None
    settings = protocol["value_prediction_protocol"]
    for seed in settings["training_seeds"]:
        # Checkpoints were generated locally by the frozen training script and
        # contain NumPy standardization arrays in addition to tensor weights.
        record = torch.load(root / f"results/stage4/value_network_v1/value_network_seed{seed}.pt", map_location="cpu", weights_only=False)
        model = CounterfactualValueNetwork(int(record["input_dim"]), tuple(settings["hidden_dims"]), float(settings["dropout"]))
        model.load_state_dict(record["state_dict"])
        model.eval()
        models.append(model)
        current = Standardization(record["feature_mean"].numpy() if isinstance(record["feature_mean"], torch.Tensor) else record["feature_mean"], record["feature_scale"].numpy() if isinstance(record["feature_scale"], torch.Tensor) else record["feature_scale"], float(record["target_mean"]), float(record["target_scale"]))
        if stats is None:
            stats = current
        elif not np.array_equal(stats.feature_mean, current.feature_mean) or stats.target_mean != current.target_mean:
            raise ValueError("checkpoint standardization mismatch")
    assert stats is not None
    return models, stats


def ensemble_predict(features: np.ndarray, models: list[CounterfactualValueNetwork], stats: Standardization) -> np.ndarray:
    x = torch.as_tensor((features - stats.feature_mean) / stats.feature_scale, dtype=torch.float32)
    predictions = []
    with torch.no_grad():
        for model in models:
            predictions.append(model(x).numpy() * stats.target_scale + stats.target_mean)
    return np.mean(predictions, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_DIR)
    parser.add_argument("--output", type=Path, default=Path("results/stage4/value_diagnostics_v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((root / "configs/stage4_protocol.json").read_text(encoding="utf-8"))
    stage2 = json.loads((root / "configs/stage2_digital_link.json").read_text(encoding="utf-8"))
    cache_meta = json.loads((root / "results/stage4/multinode_cache_v1/multinode_cache_result.json").read_text(encoding="utf-8"))
    cache = load_npz(root / cache_meta["splits"]["validation"]["cache"])
    models, stats = load_models(root, protocol)
    links = tuple(build_link_config(stage2, float(value)) for value in protocol["counterfactual_value_protocol"]["reporting_ebn0_db"])
    demand = int(cache_meta["task"]["demand_channels"])

    predictions, targets = [], []
    baseline_regrets, selected_regrets, oracle_regrets = [], [], []
    baseline_bits, selected_bits = [], []
    availability = cache["quality_availability"]
    for index, scene_id in enumerate(cache["scene_ids"]):
        baseline, candidates = scene_candidates(
            str(scene_id), cache["node_occupancy"][index], cache["node_quality"][index],
            availability, links, cache["truth_occupancy"][index], demand, 1,
        )
        feature_matrix = np.stack([row["features"] for row in candidates])
        predicted = ensemble_predict(feature_matrix, models, stats)
        true = np.asarray([(baseline["regret"] - row["regret"]) / row["incremental_bits"] for row in candidates])
        chosen = int(np.argmax(predicted))
        selected = candidates[chosen] if predicted[chosen] > 0 else baseline
        oracle = min([baseline, *candidates], key=lambda row: (row["regret"], row["bits"]))
        predictions.append(predicted)
        targets.append(true)
        baseline_regrets.append(baseline["regret"])
        selected_regrets.append(selected["regret"])
        oracle_regrets.append(oracle["regret"])
        baseline_bits.append(baseline["bits"])
        selected_bits.append(selected["bits"])

    prediction = np.stack(predictions)
    target = np.stack(targets)
    per_scene_spearman = [spearman_correlation(x, y) for x, y in zip(prediction, target)]
    valid_spearman = [x for x in per_scene_spearman if x is not None]
    baseline_regrets = np.asarray(baseline_regrets)
    selected_regrets = np.asarray(selected_regrets)
    oracle_regrets = np.asarray(oracle_regrets)
    result = {
        "experiment_id": "stage4_value_diagnostics_v1",
        "evaluation_split": "validation_aggregate_metrics_only",
        "scene_count": int(prediction.shape[0]),
        "candidate_count_per_scene": int(prediction.shape[1]),
        "regression": {
            "value_per_expected_bit_MAE": float(np.mean(np.abs(prediction - target))),
            "value_per_expected_bit_RMSE": float(np.sqrt(np.mean((prediction - target) ** 2))),
            "target_standard_deviation": float(np.std(target)),
        },
        "ranking": {
            "pooled_spearman": spearman_correlation(prediction.reshape(-1), target.reshape(-1)),
            "mean_scene_spearman": float(np.mean(valid_spearman)),
            "defined_scene_spearman_count": len(valid_spearman),
            "top_1_recall": top_k_recall(prediction, target, 1),
            "top_2_recall": top_k_recall(prediction, target, 2),
            "top_3_recall": top_k_recall(prediction, target, 3),
        },
        "scheduler_effect": {
            "mean_baseline_regret": float(np.mean(baseline_regrets)),
            "mean_selected_regret": float(np.mean(selected_regrets)),
            "mean_oracle_regret": float(np.mean(oracle_regrets)),
            "selected_regret_reduction": float(np.mean(baseline_regrets - selected_regrets)),
            "oracle_regret_reduction": float(np.mean(baseline_regrets - oracle_regrets)),
            "mean_selected_incremental_expected_bits": float(np.mean(np.asarray(selected_bits) - np.asarray(baseline_bits))),
        },
        "truth_access_audit": {
            "validation_truth_used_for_aggregate_metrics_only": True,
            "validation_candidate_labels_saved": False,
            "per_scene_rows_saved": False,
            "final_holdout_accessed": False,
        },
        "acceptance": {"completed": True, "gate_b_decision": "reject_C2_as_headline_algorithm_after_negative_system_level_result"},
        "claim_boundary": "These diagnostics complete the rejected C2 mechanism audit; they do not overturn Gate B or establish final H2/H3.",
    }
    (output / "value_diagnostics_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    r, q, s = result["regression"], result["ranking"], result["scheduler_effect"]
    lines = ["# Stage-4 value predictor diagnostics", "", "Validation aggregates only; no validation labels or per-scene candidate rows were serialized.", "", "| Metric | Value |", "|---|---:|", f"| Value/bit MAE | {r['value_per_expected_bit_MAE']:.8g} |", f"| Pooled Spearman | {q['pooled_spearman']:.4f} |", f"| Mean scene Spearman | {q['mean_scene_spearman']:.4f} |", f"| Top-1 / Top-2 / Top-3 recall | {q['top_1_recall']:.4f} / {q['top_2_recall']:.4f} / {q['top_3_recall']:.4f} |", f"| Selected regret reduction | {s['selected_regret_reduction']:.6f} |", f"| Oracle regret reduction | {s['oracle_regret_reduction']:.6f} |", "", f"> {result['claim_boundary']}", ""]
    (output / "value_diagnostics_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"regression": r, "ranking": q, "scheduler_effect": s}, indent=2))


if __name__ == "__main__":
    main()
