"""Paired G1 1-bit versus 2-bit preview identifiability/cost diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
for path in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_stage2_digital_link import build_link_config
from spectrum_semcom.counterfactual_value import candidate_message_cost, discrete_scene_regret, evaluate_one_step_oracle, fuse_state, quantized_node_reports
from spectrum_semcom.multigranular_semantics import SemanticQuality
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file
from spectrum_semcom.stage4_protocol import assert_valid_stage4_protocol


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def quality(row: np.ndarray) -> SemanticQuality:
    return SemanticQuality(*[float(value) for value in row])


def evaluate(cache: dict[str, np.ndarray], links: tuple, demand: int, preview_bits: int) -> dict[str, np.ndarray]:
    rows = {key: [] for key in ("regret", "bits", "clean", "brier", "oracle_regret", "oracle_reduction", "oracle_improved")}
    kernel = np.ones(demand) / demand
    for index, scene_id in enumerate(cache["scene_ids"]):
        occupancy = cache["node_occupancy"][index]
        truth = cache["truth_occupancy"][index]
        reports = quantized_node_reports(occupancy, preview_bits)
        belief = fuse_state(reports, (1,) * occupancy.shape[0])
        truth_blocks = np.convolve(truth, kernel, mode="valid")
        chosen = int(np.argmin(np.convolve(belief, kernel, mode="valid")))
        rows["regret"].append(discrete_scene_regret(belief, truth, demand))
        rows["bits"].append(sum(candidate_message_cost(str(scene_id), node, 1, reports[1][node], quality(cache["node_quality"][index, node]), links[node], probability_bits=preview_bits)[2] for node in range(occupancy.shape[0])))
        rows["clean"].append(float(truth_blocks[chosen] <= 0.02))
        rows["brier"].append(float(np.mean(np.square(belief - truth))))
        oracle = evaluate_one_step_oracle(occupancy, truth, demand, preview_bits)
        rows["oracle_regret"].append(oracle.best_upgraded_regret)
        rows["oracle_reduction"].append(oracle.regret_reduction)
        rows["oracle_improved"].append(float(oracle.regret_reduction > 1e-12))
    return {key: np.asarray(values, dtype=np.float64) for key, values in rows.items()}


def summarize(rows: dict[str, np.ndarray]) -> dict[str, float]:
    return {"mean_regret": float(rows["regret"].mean()), "mean_expected_preview_bits": float(rows["bits"].mean()), "clean_resource_rate": float(rows["clean"].mean()), "brier": float(rows["brier"].mean()), "mean_one_step_oracle_regret": float(rows["oracle_regret"].mean()), "mean_oracle_reduction": float(rows["oracle_reduction"].mean()), "oracle_improved_scene_fraction": float(rows["oracle_improved"].mean())}


def paired_differences(control: dict[str, np.ndarray], treatment: dict[str, np.ndarray], repetitions: int, seed: int) -> dict[str, dict[str, object]]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(control["regret"]), size=(repetitions, len(control["regret"])))
    output = {}
    for key in ("regret", "bits", "clean", "brier", "oracle_regret", "oracle_reduction"):
        difference = treatment[key] - control[key]
        samples = difference[indices].mean(axis=1)
        output[key] = {"treatment_minus_control_mean": float(difference.mean()), "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])]}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROJECT_DIR / "configs" / "stage4_protocol.json")
    parser.add_argument("--stage2-config", type=Path, default=PROJECT_DIR / "configs" / "stage2_digital_link.json")
    parser.add_argument("--cache-result", type=Path, default=PROJECT_DIR / "results" / "stage4" / "multinode_cache_v1" / "multinode_cache_result.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage4" / "preview_diagnostic_v1")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8")); assert_valid_stage4_protocol(protocol)
    stage2 = json.loads(args.stage2_config.read_text(encoding="utf-8"))
    meta = json.loads(args.cache_result.read_text(encoding="utf-8"))
    links = tuple(build_link_config(stage2, float(value)) for value in protocol["counterfactual_value_protocol"]["reporting_ebn0_db"])
    demand = int(meta["task"]["demand_channels"])
    split_results = {}
    for split in ("calibration", "validation"):
        cache = load_npz(PROJECT_DIR / meta["splits"][split]["cache"])
        control = evaluate(cache, links, demand, 1)
        treatment = evaluate(cache, links, demand, 2)
        split_results[split] = {"G1q1": summarize(control), "G1q2": summarize(treatment), "paired_bootstrap": paired_differences(control, treatment, int(protocol["value_prediction_protocol"]["bootstrap_repetitions"]), int(protocol["seed"]))}
    result = {"experiment_id": "stage4_G1_preview_identifiability_v1", "protocol_sha256": sha256_file(args.protocol), "cache_result_sha256": sha256_file(args.cache_result), "only_material_change": protocol["preview_identifiability_diagnostic"]["only_material_change"], "splits": split_results, "truth_access_audit": {"aggregate_calibration_validation_diagnostic_only": True, "labels_saved": False, "final_holdout_accessed": False}, "decision_boundary": "Development diagnostic only; no final H2/H3 claim.", "environment": environment_snapshot(["numpy"])}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "preview_diagnostic_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    val = split_results["validation"]
    lines = ["# Stage 4 G1 Preview Diagnostic", "", "| G1 | regret | preview bit | clean rate | Brier | oracle regret | upgrade-improved scenes |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("G1q1", "G1q2"):
        row = val[name]
        lines.append(f"| {name} | {row['mean_regret']:.6f} | {row['mean_expected_preview_bits']:.2f} | {row['clean_resource_rate']:.4f} | {row['brier']:.6f} | {row['mean_one_step_oracle_regret']:.6f} | {row['oracle_improved_scene_fraction']:.4f} |")
    difference = val["paired_bootstrap"]
    lines.extend(["", f"G1q2-q1 regret: {difference['regret']['treatment_minus_control_mean']:.6f}, 95% CI {difference['regret']['ci95']}.", f"G1q2-q1 preview bit: {difference['bits']['treatment_minus_control_mean']:.2f}, 95% CI {difference['bits']['ci95']}.", "", "No per-scene labels were saved and no final holdout was accessed."])
    (args.out_dir / "preview_diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_dir / "preview_diagnostic_report.md")


if __name__ == "__main__":
    main()
