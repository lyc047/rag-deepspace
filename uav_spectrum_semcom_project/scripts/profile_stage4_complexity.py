#!/usr/bin/env python
"""Profile frozen stage-4 heads without changing any algorithm setting."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.complexity import benchmark_callable, linear_macs, parameter_summary
from spectrum_semcom.gate_a_model import VariableRateOccupancyHead
from spectrum_semcom.value_prediction import CounterfactualValueNetwork


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_DIR)
    parser.add_argument("--output", type=Path, default=Path("results/stage4/complexity_profile_v1"))
    parser.add_argument("--repetitions", type=int, default=2000)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    protocol = json.loads((root / "configs/stage4_protocol.json").read_text(encoding="utf-8"))

    gate_checkpoint = root / protocol["counterfactual_value_protocol"]["canonical_c1_checkpoint"]
    gate = VariableRateOccupancyHead(8, int(protocol["gate_a_controlled_training"]["hidden_dim"]))
    gate.load_state_dict(torch.load(gate_checkpoint, map_location="cpu", weights_only=True))
    gate.eval()
    gate_input = torch.full((1, 8), 0.25)

    value_models = []
    first_checkpoint = None
    for seed in protocol["value_prediction_protocol"]["training_seeds"]:
        record = torch.load(root / f"results/stage4/value_network_v1/value_network_seed{seed}.pt", map_location="cpu", weights_only=False)
        model = CounterfactualValueNetwork(int(record["input_dim"]), tuple(protocol["value_prediction_protocol"]["hidden_dims"]), float(protocol["value_prediction_protocol"]["dropout"]))
        model.load_state_dict(record["state_dict"])
        model.eval()
        value_models.append(model)
        first_checkpoint = record if first_checkpoint is None else first_checkpoint
    assert first_checkpoint is not None
    value_input = torch.zeros((8, int(first_checkpoint["input_dim"])))

    with torch.inference_mode():
        gate_forward_latency = benchmark_callable(lambda: gate(gate_input), repetitions=args.repetitions)
        gate_discrete_latency = benchmark_callable(lambda: gate.infer(gate_input), repetitions=args.repetitions)
        value_one_latency = benchmark_callable(lambda: value_models[0](value_input), repetitions=args.repetitions)
        value_ensemble_latency = benchmark_callable(lambda: tuple(model(value_input) for model in value_models), repetitions=args.repetitions)

    gate_size = parameter_summary(gate)
    value_size = parameter_summary(value_models[0])
    result = {
        "experiment_id": "stage4_complexity_profile_v1",
        "device": "cpu_single_thread",
        "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(), "torch_num_threads": torch.get_num_threads()},
        "c1_variable_rate_head": {
            **gate_size,
            "linear_MACs_per_scene": linear_macs(gate),
            "checkpoint_bytes": gate_checkpoint.stat().st_size,
            "continuous_forward_latency": gate_forward_latency,
            "discrete_infer_latency": gate_discrete_latency,
            "scope": "trainable stage4 head only; frozen upstream detector excluded and reported separately in prior stages",
        },
        "c2_value_network": {
            **value_size,
            "linear_MACs_per_candidate": linear_macs(value_models[0]),
            "linear_MACs_for_8_candidates_one_seed": linear_macs(value_models[0], 8),
            "linear_MACs_for_8_candidates_5_seed_ensemble": linear_macs(value_models[0], 8) * len(value_models),
            "checkpoint_bytes_per_seed": (root / f"results/stage4/value_network_v1/value_network_seed{protocol['value_prediction_protocol']['training_seeds'][0]}.pt").stat().st_size,
            "eight_candidate_one_seed_latency": value_one_latency,
            "eight_candidate_five_seed_latency": value_ensemble_latency,
            "scope": "rejected Gate-B diagnostic model; profiled for completeness, not a deployment recommendation",
        },
        "mac_definition": "one multiply-accumulate per Linear weight; activations, softmax, quantization, Python control, detector, codec, and link simulation excluded",
        "acceptance": {"completed": True},
        "final_holdout_accessed": False,
    }
    (output / "complexity_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    c1, c2 = result["c1_variable_rate_head"], result["c2_value_network"]
    lines = ["# Stage-4 complexity profile", "", "CPU, one thread; latency is local-machine evidence and not UAV-board latency.", "", "| Component | Params | FP32 parameter bytes | Linear MACs | Median / p95 latency |", "|---|---:|---:|---:|---:|", f"| C1 head, one scene | {c1['total_parameters']} | {c1['parameter_bytes_fp32']} | {c1['linear_MACs_per_scene']} | {c1['discrete_infer_latency']['median_ms']:.4f} / {c1['discrete_infer_latency']['p95_ms']:.4f} ms |", f"| C2 network, 8 candidates, one seed | {c2['total_parameters']} | {c2['parameter_bytes_fp32']} | {c2['linear_MACs_for_8_candidates_one_seed']} | {c2['eight_candidate_one_seed_latency']['median_ms']:.4f} / {c2['eight_candidate_one_seed_latency']['p95_ms']:.4f} ms |", f"| C2 network, 8 candidates, five seeds | {c2['total_parameters'] * 5} | {c2['parameter_bytes_fp32'] * 5} | {c2['linear_MACs_for_8_candidates_5_seed_ensemble']} | {c2['eight_candidate_five_seed_latency']['median_ms']:.4f} / {c2['eight_candidate_five_seed_latency']['p95_ms']:.4f} ms |", "", f"> {result['mac_definition']}", ""]
    (output / "complexity_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"c1": c1, "c2": c2}, indent=2))


if __name__ == "__main__":
    main()
