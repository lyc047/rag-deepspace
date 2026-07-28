#!/usr/bin/env python
"""Exercise the frozen C1 and selective-G2 chain with three synthetic sites.

This is a cardinality/interface test only.  It uses no AERPAW non-pilot
measurement and cannot create or consume a final holdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
for path in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(path))

from run_stage2_digital_link import build_link_config  # noqa: E402
from run_stage4_end_to_end_sweep import run_method  # noqa: E402
from spectrum_semcom.aerpaw_spectrum import (  # noqa: E402
    AerpawPowerSweep,
    channel_occupancy,
    channel_power_dbm,
    cleanest_contiguous_block,
    robust_power_threshold_dbm,
)
from spectrum_semcom.counterfactual_value import discrete_scene_regret  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.gate_a_model import VariableRateOccupancyHead  # noqa: E402
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402


def synthetic_site(site: str, index: int) -> AerpawPowerSweep:
    rng = np.random.default_rng(20260721 + index)
    frequencies = np.linspace(2400.0, 2483.49, 1391, dtype=np.float64)
    powers = rng.normal(-134.0 + 0.8 * index, 0.9, frequencies.size).astype(np.float32)
    for channel, fraction in enumerate((0.05, 0.12, 0.35, 0.08, 0.55, 0.18, 0.03, 0.25)):
        bins = np.array_split(np.arange(frequencies.size), 8)[channel]
        shift = (channel + 2 * index) % len(bins)
        active = np.roll(bins, shift)[: int(round(len(bins) * fraction))]
        powers[active] = rng.normal(-112.0 + index, 2.0, active.size)
    return AerpawPowerSweep(
        frequencies,
        powers,
        site,
        f"2022-02-08T12:00:0{index}-05:00",
        Path(f"synthetic-{site}.sigmf-meta"),
        Path(f"synthetic-{site}.sigmf-data"),
        {"global": {"core:datatype": "rf32_le", "dataset:site": site}},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR / "configs/aerpaw_three_site_prefinal_protocol_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "results/stage4/aerpaw_three_site_dry_run_v1/three_site_dry_run_result.json",
    )
    args = parser.parse_args()
    access = json.loads((PROJECT_DIR / "configs/stage4_final_access_state.json").read_text(encoding="utf-8"))
    if access.get("access_count") != 0:
        raise RuntimeError("pre-final dry-run must occur before final access")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    base = json.loads((PROJECT_DIR / "configs/stage4_protocol.json").read_text(encoding="utf-8"))
    stage2 = json.loads((PROJECT_DIR / "configs/stage2_digital_link.json").read_text(encoding="utf-8"))
    sites = tuple(protocol["sites"])
    proxy = protocol["resource_proxy"]
    rule = proxy["threshold_rule"]

    sweeps = [synthetic_site(site, index) for index, site in enumerate(sites)]
    base_occupancy = []
    costs = []
    thresholds = []
    for sweep in sweeps:
        threshold = robust_power_threshold_dbm(
            sweep,
            sigma_multiplier=float(rule["sigma_multiplier"]),
            gaussian_mad_scale=float(rule["gaussian_mad_scale"]),
            sigma_floor_db=float(rule["sigma_floor_db"]),
        )
        thresholds.append(threshold)
        base_occupancy.append(channel_occupancy(sweep, int(proxy["n_equal_width_channels"]), threshold))
        costs.append(channel_power_dbm(sweep, int(proxy["n_equal_width_channels"]), reducer="mean"))
    base_occupancy_array = np.stack(base_occupancy)

    hidden = int(base["gate_a_controlled_training"]["hidden_dim"])
    checkpoints = {
        "detection_only": PROJECT_DIR / "results/stage4/gate_a_training_v1/checkpoints/detection_only_seed20260716.pt",
        "detection_plus_resource": PROJECT_DIR / "results/stage4/gate_a_training_v1/checkpoints/detection_plus_resource_seed20260716.pt",
    }
    refined = {}
    tensor = torch.as_tensor(base_occupancy_array, dtype=torch.float32)
    with torch.inference_mode():
        for name, checkpoint in checkpoints.items():
            model = VariableRateOccupancyHead(int(proxy["n_equal_width_channels"]), hidden)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            model.eval()
            occupancy, probability_bits, application_bits = model.infer(tensor)
            refined[name] = {
                "occupancy": occupancy.numpy(),
                "probability_bits": probability_bits.numpy().tolist(),
                "application_bits": application_bits.numpy().tolist(),
            }

    node_quality = np.zeros((len(sites), 10), dtype=np.float32)
    node_quality[:, 0] = [-1.0, 2.0, 4.0]
    node_quality[:, 1] = np.mean(np.abs(refined["detection_plus_resource"]["occupancy"] - 0.5) * 2.0, axis=1)
    node_quality[:, 2] = 0.5
    node_quality[:, 6] = 12.0
    node_quality[:, 8] = 0.99
    truth_proxy = np.mean(base_occupancy_array, axis=0)
    link_results = []
    relative_values = tuple(float(value) for value in protocol["three_site_reporting_link"]["relative_offset_values_db"])
    for assignment_index, offsets in enumerate(permutations(relative_values)):
        links = tuple(
            replace(build_link_config(stage2, 9.0 + offset), channel="awgn") for offset in offsets
        )
        for method_index, method in enumerate(("G1_plus_stage3_prior_G2", "all_G2")):
            fused, metrics = run_method(
                method,
                "synthetic-three-site-prefinal",
                refined["detection_plus_resource"]["occupancy"],
                node_quality,
                links,
                20260721 + assignment_index * 1000 + method_index * 100,
            )
            link_results.append(
                {
                    "relative_offset_assignment_db": list(offsets),
                    "method": method,
                    "regret_proxy": discrete_scene_regret(
                        fused, truth_proxy, int(proxy["resource_demand_channels"])
                    ),
                    **metrics,
                }
            )

    c1_interface = {}
    for name, values in refined.items():
        selected = []
        for site_index, occupancy in enumerate(values["occupancy"]):
            start, _ = cleanest_contiguous_block(occupancy, int(proxy["resource_demand_channels"]))
            selected.append({"site": sites[site_index], "selected_block_start": start})
        c1_interface[name] = {
            "shape": list(values["occupancy"].shape),
            "selected_blocks": selected,
            "probability_bits": values["probability_bits"],
            "application_bits": values["application_bits"],
        }

    result = {
        "experiment_id": "aerpaw_three_site_prefinal_dry_run_v1",
        "role": "synthetic_cardinality_interface_test_not_performance_evaluation",
        "protocol_sha256": sha256_file(args.protocol),
        "site_count": len(sites),
        "sites": list(sites),
        "frequency_band_mhz": [proxy["frequency_low_mhz_inclusive"], proxy["frequency_high_mhz_exclusive"]],
        "thresholds_dbm": thresholds,
        "base_occupancy_shape": list(base_occupancy_array.shape),
        "C1_interfaces": c1_interface,
        "selective_G2_interface": {
            "offset_permutation_count": len(set(tuple(row["relative_offset_assignment_db"]) for row in link_results)),
            "all_runs_completed": len(link_results) == 12,
            "rows": link_results,
        },
        "checkpoint_sha256": {name: sha256_file(path) for name, path in checkpoints.items()},
        "nonpilot_real_data_accessed": False,
        "final_catalog_created": False,
        "final_access_consumed": False,
        "claim_boundary": "Only proves that frozen C1 and digital selective-G2 code execute with three sites.",
    }
    atomic_write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "site_count": len(sites), "final_access_consumed": False}, indent=2))


if __name__ == "__main__":
    main()
