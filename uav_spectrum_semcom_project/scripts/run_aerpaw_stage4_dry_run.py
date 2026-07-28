#!/usr/bin/env python
"""Exercise AERPAW-power -> C1 -> G2 -> digital link -> resource selection.

The default input is a deterministic synthetic ``rf32_le``-equivalent sweep.
It validates interfaces and profiles execution only; it is not a performance
experiment and cannot consume or stand in for the final holdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.aerpaw_spectrum import (  # noqa: E402
    AerpawPowerSweep,
    channel_occupancy,
    channel_power_dbm,
    cleanest_contiguous_block,
    load_aerpaw_power_sweep,
)
from spectrum_semcom.digital_link import DigitalLinkConfig  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.gate_a_model import VariableRateOccupancyHead  # noqa: E402
from spectrum_semcom.multigranular_semantics import (  # noqa: E402
    SemanticMessage,
    SemanticQuality,
    transmit_semantic_message,
)
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402


def synthetic_sweep() -> AerpawPowerSweep:
    rng = np.random.default_rng(20260721)
    frequencies = np.linspace(87.16000366210938, 6019.18017578125, 98_868, dtype=np.float64)
    powers = rng.normal(-133.0, 1.5, frequencies.size).astype(np.float32)
    fractions = (0.05, 0.55, 0.12, 0.75, 0.30, 0.08, 0.62, 0.18)
    for indices, fraction in zip(np.array_split(np.arange(frequencies.size), 8), fractions):
        active = indices[: int(round(indices.size * fraction))]
        powers[active] = rng.normal(-75.0, 4.0, active.size)
    return AerpawPowerSweep(
        frequencies,
        powers,
        "SYNTHETIC_LW1_SCHEMA",
        "2022-02-01T00:00:00-05:00",
        Path("synthetic.sigmf-meta"),
        Path("synthetic.sigmf-data"),
        {"global": {"core:datatype": "rf32_le", "dataset:num_bins": 98_868}},
    )


def percentile_ms(samples_ns: list[int]) -> dict[str, float]:
    values = np.asarray(samples_ns, dtype=np.float64) / 1e6
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.quantile(values, 0.95))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, help="Optional permanently excluded pilot .sigmf-meta")
    parser.add_argument("--threshold-dbm", type=float, default=-110.0)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "results/stage4/aerpaw_dry_run_v1/dry_run_result.json")
    args = parser.parse_args()
    if args.repeats < 10:
        raise ValueError("repeats must be at least 10")

    sweep = load_aerpaw_power_sweep(args.meta) if args.meta else synthetic_sweep()
    protocol = json.loads((PROJECT_DIR / "configs/stage4_protocol.json").read_text(encoding="utf-8"))
    n_channels = 8
    hidden_dim = int(protocol["gate_a_controlled_training"]["hidden_dim"])
    checkpoints = {
        "detection_only": PROJECT_DIR / "results/stage4/gate_a_training_v1/checkpoints/detection_only_seed20260716.pt",
        "detection_plus_resource": PROJECT_DIR / "results/stage4/gate_a_training_v1/checkpoints/detection_plus_resource_seed20260716.pt",
    }
    models = {}
    for name, checkpoint in checkpoints.items():
        model = VariableRateOccupancyHead(n_channels, hidden_dim)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        models[name] = model

    base = channel_occupancy(sweep, n_channels, args.threshold_dbm)
    costs = channel_power_dbm(sweep, n_channels, reducer="mean")
    input_tensor = torch.as_tensor(base[None], dtype=torch.float32)
    quality = SemanticQuality(
        sensing_snr_db=0.0,
        prediction_confidence=float(np.mean(np.abs(base - 0.5) * 2.0)),
        normalized_entropy=0.5,
        clipping_ratio=0.0,
        out_of_band_leakage_ratio=0.0,
        noise_floor_stability_db=float(np.std(costs)),
        peak_to_background_db=float(np.max(costs) - np.median(costs)),
        age_s=0.0,
        report_success_probability=1.0,
        calibration_error=0.0,
    )
    link = DigitalLinkConfig(channel="awgn", ebn0_db=20.0, max_retransmissions=1)

    timings = {"power_to_8_channel_occupancy": [], "C1_inference_both_methods": [], "G2_codec_and_digital_link": [], "resource_selection": []}
    outputs = {}
    for repeat in range(args.repeats):
        start = perf_counter_ns(); base_repeat = channel_occupancy(sweep, n_channels, args.threshold_dbm); timings["power_to_8_channel_occupancy"].append(perf_counter_ns() - start)
        start = perf_counter_ns()
        inferred = {}
        with torch.inference_mode():
            for name, model in models.items():
                occupancy, probability_bits, application_bits = model.infer(input_tensor)
                inferred[name] = (occupancy[0].numpy(), int(probability_bits[0]), int(application_bits[0]))
        timings["C1_inference_both_methods"].append(perf_counter_ns() - start)
        start = perf_counter_ns()
        transmitted = {}
        for method_index, (name, (occupancy, probability_bits, _)) in enumerate(inferred.items()):
            message = SemanticMessage("G2", 0, "aerpaw-interface-dry-run", occupancy, probability_bits, quality=quality)
            tx = transmit_semantic_message(message, link, 20260721 + repeat * 10 + method_index)
            if tx.decoded is None:
                raise RuntimeError("high-SNR dry-run message unexpectedly failed to decode")
            transmitted[name] = tx
        timings["G2_codec_and_digital_link"].append(perf_counter_ns() - start)
        start = perf_counter_ns()
        selected = {name: cleanest_contiguous_block(tx.decoded.occupancy, 4)[0] for name, tx in transmitted.items()}
        oracle_start, oracle_cost = cleanest_contiguous_block(costs, 4)
        timings["resource_selection"].append(perf_counter_ns() - start)
        outputs = {
            name: {
                "refined_occupancy": tx.decoded.occupancy.tolist(),
                "probability_bits": inferred[name][1],
                "application_bits": tx.encoded.application_bits,
                "actual_transmitted_bits": tx.link.transmitted_bits,
                "selected_block_start": selected[name],
            }
            for name, tx in transmitted.items()
        }

    result = {
        "experiment_id": "aerpaw_stage4_interface_dry_run_v1",
        "role": "interface_and_complexity_dry_run_not_performance_evaluation",
        "input": "permanently_excluded_pilot" if args.meta else "deterministic_synthetic_rf32_power_sweep",
        "input_bins": sweep.n_bins,
        "frequency_span_mhz": [float(sweep.frequencies_mhz[0]), float(sweep.frequencies_mhz[-1])],
        "threshold_dbm": args.threshold_dbm,
        "base_occupancy": base.tolist(),
        "channel_mean_power_dbm": costs.tolist(),
        "oracle_cleanest_block_start": oracle_start,
        "oracle_cleanest_block_mean_dbm": oracle_cost,
        "methods": outputs,
        "timing": {name: percentile_ms(samples) for name, samples in timings.items()},
        "repeats": args.repeats,
        "checkpoint_sha256": {name: sha256_file(path) for name, path in checkpoints.items()},
        "final_access_consumed": False,
        "claim_boundary": "Dry-run validates data shape, frozen C1, G2 codec, digital link, and resource-selection interfaces only.",
    }
    atomic_write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "input": result["input"], "final_access_consumed": False}, indent=2))


if __name__ == "__main__":
    main()
