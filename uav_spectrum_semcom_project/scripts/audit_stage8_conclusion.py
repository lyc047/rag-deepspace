"""Build an immutable interpretation and fallacy audit from frozen Stage-8 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def audit(
    confirmation_path: Path,
    replay_path: Path,
    access_state_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    access = json.loads(access_state_path.read_text(encoding="utf-8"))
    if confirmation["status"] != "STAGE8_FAULT_CONFIRMATION_COMPLETE":
        raise ValueError("confirmation is incomplete")
    if replay["selection"]["selected_development_candidate"] != "NO_CANDIDATE":
        raise ValueError("unexpected development selection")
    if any(role["access_count"] != 1 for role in access["roles"].values()):
        raise ValueError("confirmation access ledger is inconsistent")

    aadm = confirmation["aadm_confirmation"]
    alfa = confirmation["alfa_confirmation"]
    threshold = float(aadm["short_range_lag1_threshold"])
    max_lag1 = float(aadm["maximum_absolute_segment_lag1"])
    hierarchy_confirmed = bool(aadm["homogeneous_model_rejected"])
    within_flight_iid_confirmed = max_lag1 < threshold
    full_development_model_confirmed = (
        hierarchy_confirmed and within_flight_iid_confirmed
    )
    result = {
        "version": "1.0",
        "status": "STAGE8_CONCLUSION_AUDIT_COMPLETE",
        "overall_conclusion": "MIXED_MODEL_EVIDENCE_NO_NEW_PROTOCOL_CANDIDATE",
        "evidence_components": {
            "between_flight_loss_heterogeneity_confirmed": hierarchy_confirmed,
            "development_within_flight_bernoulli_assumption_confirmed": (
                within_flight_iid_confirmed
            ),
            "full_flight_hierarchical_bernoulli_model_confirmed": (
                full_development_model_confirmed
            ),
            "exact_10_percent_mean_loss_rejected": not bool(
                aadm["fixed_10_percent_inside_interval"]
            ),
            "semantic_context_reset_probability_calibrated": bool(
                alfa["semantic_receiver_context_reset_probability_calibratable"]
            ),
            "new_heartbeat_candidate_selected": False,
            "fixed10_is_empirically_optimal_on_real_board": False,
            "fixed10_remains_engineering_stress_boundary": True,
        },
        "key_numbers": {
            "aadm_confirmation_flights": aadm["independent_flight_count"],
            "aadm_pooled_internal_gap_fraction_lower_bound": aadm[
                "pooled_internal_gap_fraction_lower_bound"
            ],
            "aadm_cluster_bootstrap_95_ci": aadm["cluster_bootstrap_95_ci"],
            "aadm_flight_rate_range": [
                aadm["flight_gap_fraction_minimum"],
                aadm["flight_gap_fraction_maximum"],
            ],
            "aadm_overdispersion_p_value": aadm[
                "homogeneous_binomial_overdispersion_test"
            ]["monte_carlo_p_value"],
            "aadm_maximum_absolute_segment_lag1": max_lag1,
            "alfa_confirmation_raw_flights": alfa["registered_raw_flight_count"],
            "alfa_processed_coverage_flights": alfa[
                "raw_flights_with_processed_coverage"
            ],
            "alfa_observed_hours": alfa["merged_observed_duration_hours"],
            "alfa_fcu_boot_resets": alfa["fcu_boot_reset_event_count"],
        },
        "eleven_fallacy_audit": [
            {
                "risk": "pseudoreplication",
                "status": "MITIGATED",
                "evidence": "AADM complete flight and ALFA raw flight are the independent units; processed clips inherit their raw-flight split.",
            },
            {
                "risk": "development-confirmation leakage",
                "status": "MITIGATED",
                "evidence": "SHA-256 flight-level splits were frozen before signal access.",
            },
            {
                "risk": "test-set tuning",
                "status": "MITIGATED",
                "evidence": "The development replay selected no candidate; confirmation values were not used to change a heartbeat interval.",
            },
            {
                "risk": "optional stopping or repeated confirmation",
                "status": "MITIGATED",
                "evidence": "Both confirmation roles have access_count=1 and rerun is prohibited.",
            },
            {
                "risk": "selective reporting across heartbeat intervals",
                "status": "MITIGATED",
                "evidence": "All preregistered intervals and all four N values are retained, including negative results.",
            },
            {
                "risk": "aggregation or Simpson effect",
                "status": "DETECTED_AND_CORRECTED",
                "evidence": "The pooled Gilbert-Elliott proxy was not treated as stationary; flight-level heterogeneity and segment-level lag are reported separately.",
            },
            {
                "risk": "mean-only reporting hides heterogeneity",
                "status": "MITIGATED",
                "evidence": "Flight-cluster intervals, 0-to-0.8 rate range, and an overdispersion test accompany the pooled mean.",
            },
            {
                "risk": "causal overclaim from proxy data",
                "status": "MITIGATED",
                "evidence": "AADM uplink loss and ALFA FCU faults are not claimed to cause or equal semantic receiver-context loss.",
            },
            {
                "risk": "absence of evidence treated as zero events",
                "status": "MITIGATED",
                "evidence": "Zero observed ALFA boot resets is accompanied by coverage, duration, and a weak zero-event upper bound.",
            },
            {
                "risk": "metric or construct mismatch",
                "status": "ACTIVE_LIMITATION",
                "evidence": "Internal frame gaps are a lower-bound LoRa uplink proxy and do not calibrate ACK loss or exact semantic packet loss.",
            },
            {
                "risk": "external-validity overclaim",
                "status": "ACTIVE_LIMITATION",
                "evidence": "No target-board process restart, power interruption, persistent-state, or independent spectrum-algorithm Final was performed.",
            },
        ],
        "frozen_decision": {
            "candidate_id": "S6R-FH10-v1",
            "algorithm_or_protocol_change": "NONE",
            "stage8_public_data_work": "COMPLETE",
            "stage8_hardware_validation": "NOT_EXECUTED",
            "next_valid_research_action": (
                "instrument a real receiver process or hardware-in-the-loop target "
                "to observe boot identity, process lifetime, persistent codebook "
                "version, volatile action cache, packet/ACK outcomes, and recovery latency"
            ),
        },
        "claim_boundary": [
            "Stage-8 provides fault-model and protocol-boundary evidence, not a new algorithm Final.",
            "The full development flight-hierarchical Bernoulli model was not confirmed because confirmation showed strong segment-level dependence.",
            "Fixed10 is retained because no alternative passed the frozen development gate, not because it was proven globally optimal.",
            "Stage-6 Final remains failed and Stage-7 remains without an independent external Final.",
        ],
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
        "--confirmation",
        type=Path,
        default=ROOT / "results/stage8/stage8_fault_confirmation_v1/result.json",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=ROOT / "results/stage8/stage8_empirical_fault_replay_v1/result.json",
    )
    parser.add_argument(
        "--access-state",
        type=Path,
        default=ROOT / "configs/stage8_fault_confirmation_access_state_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/stage8/stage8_conclusion_audit_v1/result.json",
    )
    args = parser.parse_args()
    result = audit(
        args.confirmation, args.replay, args.access_state, args.output
    )
    print(json.dumps(result["evidence_components"], ensure_ascii=False))


if __name__ == "__main__":
    main()
