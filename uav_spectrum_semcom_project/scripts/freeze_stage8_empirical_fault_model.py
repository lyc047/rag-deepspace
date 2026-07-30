"""Freeze the Stage-8 development-fitted fault model before confirmation access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = {
        name: ROOT / value
        for name, value in config["inputs"].items()
        if name in {
            "stage8_fault_protocol",
            "stage8_split",
            "aadm_development",
            "alfa_development",
            "matched_scan_config",
        }
    }
    protocol = json.loads(paths["stage8_fault_protocol"].read_text(encoding="utf-8"))
    split = json.loads(paths["stage8_split"].read_text(encoding="utf-8"))
    aadm = json.loads(paths["aadm_development"].read_text(encoding="utf-8"))
    alfa = json.loads(paths["alfa_development"].read_text(encoding="utf-8"))
    matched = json.loads(paths["matched_scan_config"].read_text(encoding="utf-8"))
    if aadm["access_log"]["confirmation_flights_opened"] != 0:
        raise ValueError("AADM confirmation values were already opened")
    if alfa["access_log"]["confirmation_raw_flights_opened"] != 0:
        raise ValueError("ALFA confirmation values were already opened")
    if split["status"] != "FROZEN_BEFORE_SIGNAL_ACCESS":
        raise ValueError("unexpected split state")
    if aadm["development_decision"]["homogeneous_iid_loss_supported"]:
        raise ValueError("hierarchical model lacks development justification")
    if not aadm["development_decision"][
        "within_segment_short_range_iid_diagnostic_supported"
    ]:
        raise ValueError("Bernoulli-within-flight model failed short-range diagnostic")

    rates = [
        {
            "flight_id": row["flight_id"],
            "internal_gap_count": row["internal_gap_count"],
            "internal_frame_span": row["internal_frame_span"],
            "loss_probability_lower_bound": row[
                "internal_gap_fraction_lower_bound"
            ],
        }
        for row in aadm["per_flight"]
    ]
    condition = matched["conditions"]["primary_fault"]
    expected_fixed = {
        "installation_loss_probability": float(
            config["fault_model"]["installation_loss_probability"]
        ),
        "ack_and_heartbeat_response_loss_probability": float(
            config["fault_model"]["ack_and_heartbeat_response_loss_probability"]
        ),
        "delayed_duplicate_probability": float(
            config["fault_model"]["delayed_duplicate_probability"]
        ),
        "receiver_context_reset_probability_per_scene": float(
            config["fault_model"]["receiver_context_reset_probability_per_scene"]
        ),
    }
    actual_fixed = {
        "installation_loss_probability": float(
            condition["install_loss_probability"]
        ),
        "ack_and_heartbeat_response_loss_probability": float(
            condition["ack_loss_probability"]
        ),
        "delayed_duplicate_probability": float(
            condition["delayed_duplicate_probability"]
        ),
        "receiver_context_reset_probability_per_scene": float(
            condition["receiver_context_reset_probability"]
        ),
    }
    if expected_fixed != actual_fixed:
        raise ValueError("registered fixed stress parameters differ from Stage-6")

    result = {
        "version": "1.0",
        "status": "EMPIRICAL_FAULT_MODEL_FROZEN_BEFORE_CONFIRMATION_ACCESS",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256(config_path),
        "input_hashes": {name: sha256(path) for name, path in paths.items()},
        "confirmation_access_at_freeze": {
            "aadm_flights_opened": 0,
            "alfa_raw_flights_opened": 0,
        },
        "task_link_model": {
            "family": "flight_hierarchical_bernoulli",
            "independent_sampling_unit": "AADM physical-testbed flight",
            "flight_sampling": "uniform_with_replacement_per_spectrum_session_trajectory",
            "within_trajectory_probability": "sampled flight internal-gap lower-bound",
            "flight_rates": rates,
            "rate_count": len(rates),
            "mean_rate": sum(row["loss_probability_lower_bound"] for row in rates)
            / len(rates),
            "random_seed": int(config["fault_model"]["random_seed"]),
            "posterior_smoothing": config["fault_model"]["posterior_smoothing"],
        },
        "fixed_stress_parameters": actual_fixed,
        "evidence_interpretation": {
            "short_range_burst_supported": aadm["aggregate"][
                "short_range_burst_failure_evidence"
            ],
            "between_flight_heterogeneity_supported": aadm["aggregate"][
                "between_flight_rate_heterogeneity_evidence"
            ],
            "ack_empirically_calibrated": False,
            "semantic_context_reset_empirically_calibrated": False,
            "two_percent_reset_role": "stress_only",
        },
        "replay_candidates": {
            "baseline": config["intervention"]["baseline_heartbeat_scenes"],
            "candidates": config["intervention"]["candidate_heartbeat_scenes"],
            "selection_order": config["intervention"]["selection_order"],
        },
        "claim_boundary": config["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/stage8_empirical_fault_replay_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/stage8/stage8_empirical_fault_model_v1/result.json",
    )
    args = parser.parse_args()
    result = freeze(args.config, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "task_link_model": result["task_link_model"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
