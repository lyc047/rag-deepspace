import json
from pathlib import Path

from spectrum_semcom.reproducibility import sha256_file


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_stage6_task_codebook_protocol_is_development_only() -> None:
    protocol = json.loads(
        (
            PROJECT_DIR / "configs/stage6_task_codebook_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    governance = protocol["governance"]
    assert protocol["task_grid"]["epsilon_db"] == 0.2
    assert governance["stage5_frozen_baseline_files_modified"] is False
    assert governance["external_final_archives_may_be_opened"] is False
    assert governance["external_final_signal_values_may_be_loaded"] is False
    assert governance["external_final_access_may_be_consumed"] is False
    assert governance["output_is_confirmatory_final"] is False


def test_stage6_protocol_uses_only_hash_locked_excluded_pilot() -> None:
    protocol = json.loads(
        (
            PROJECT_DIR / "configs/stage6_task_codebook_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    source = protocol["development_source"]
    assert "permanently_excluded" in source["role"]
    assert sha256_file(PROJECT_DIR / source["cache"]) == source["cache_sha256"]


def test_external_final_access_history_is_immutable_after_consumption() -> None:
    state = json.loads(
        (
            PROJECT_DIR / "configs/stage5_external_final_access_state.json"
        ).read_text(encoding="utf-8")
    )
    stage6 = json.loads(
        (
            PROJECT_DIR / "configs/stage6_external_final_access_state.json"
        ).read_text(encoding="utf-8")
    )
    assert state["access_count"] == stage6["access_count"]
    if state["access_count"] == 0:
        assert state["final_signal_values_accessed"] is False
        assert state["final_method_outputs_accessed"] is False
    else:
        assert state["status"].startswith("superseded_by_stage6")
        assert state["final_signal_values_accessed"] is True
        assert state["reset_permitted"] is False


def test_stage6_task_codec_protocol_accounts_for_identity_overhead() -> None:
    protocol = json.loads(
        (
            PROJECT_DIR / "configs/stage6_task_codec_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    grid = protocol["task_grid"]
    identity = protocol["session_identity"]
    governance = protocol["governance"]
    assert grid["stage6_codec_header_bits"] == 48
    assert grid["codebook_packet_tag_bits"] == 16
    assert identity["full_manifest_sha256_verified_at_installation"] is True
    assert "not_cryptographic" in identity["security_claim"]
    assert governance["external_final_signal_values_may_be_loaded"] is False
    assert governance["external_final_access_may_be_consumed"] is False
    assert governance["output_is_confirmatory_final"] is False


def test_stage6_context_protocol_reports_both_provisioning_views() -> None:
    protocol = json.loads(
        (
            PROJECT_DIR / "configs/stage6_context_event_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["accounting_views"] == [
        "preprovisioned_operational_updates_only",
        "in_band_binary_context_install_plus_operational_updates",
    ]
    comparison = protocol["comparison"]
    governance = protocol["governance"]
    assert comparison["both_methods_evaluate_all_queries_each_scene"] is True
    assert comparison["context_install_count_per_logical_session"] == 1
    assert comparison["ack_faults_injected"] is False
    assert governance["external_final_signal_values_may_be_loaded"] is False
    assert governance["external_final_access_may_be_consumed"] is False
    assert governance["output_is_confirmatory_final"] is False


def test_stage6_context_recovery_protocol_is_controlled_simulation() -> None:
    protocol = json.loads(
        (
            PROJECT_DIR
            / "configs/stage6_context_recovery_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["monte_carlo"]["paired_common_random_numbers"] is True
    assert protocol["task_grid"]["ack_application_bits"] == 24
    assert protocol["task_grid"]["outage_penalty_db"] == 10.0
    assert len(protocol["fault_conditions"]) == 5
    assert protocol["primary_gates"][
        "wrong_codebook_decode_count_must_equal_zero"
    ] is True
    governance = protocol["governance"]
    assert governance["faults_are_controlled_injections_not_measurements"] is True
    assert governance["external_final_signal_values_may_be_loaded"] is False
    assert governance["external_final_access_may_be_consumed"] is False
    assert governance["output_is_confirmatory_final"] is False


def test_stage6_heartbeat_protocol_fixes_real_codec_and_sensitivity() -> None:
    protocol = json.loads(
        (
            PROJECT_DIR
            / "configs/stage6_context_heartbeat_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    grid = protocol["task_grid"]
    assert grid["heartbeat_request_application_bits"] == 32
    assert grid["heartbeat_response_application_bits"] == 32
    assert [
        value["silence_interval_scenes"]
        for value in protocol["heartbeat_policies"]
    ] == [None, 10, 20, 40]
    assert protocol["monte_carlo"]["paired_common_random_numbers"] is True
    assert protocol["predecessor_result"]["sha256"] == (
        "137d5e4ce5b22339aea51f4a6c2bb758"
        "62d846aac7982f0092bb0cc9a36eb8a3"
    )
    governance = protocol["governance"]
    assert governance["external_final_signal_values_may_be_loaded"] is False
    assert governance["external_final_access_may_be_consumed"] is False
    assert governance["output_is_confirmatory_final"] is False


def test_stage6_heartbeat_statistics_are_paired_and_hash_locked() -> None:
    analysis = json.loads(
        (
            PROJECT_DIR
            / "configs/stage6_context_heartbeat_statistics_v1.json"
        ).read_text(encoding="utf-8")
    )
    resampling = analysis["resampling"]
    assert resampling["unit"] == "complete_167_scene_trajectory"
    assert resampling["paired_across_policies"] is True
    assert resampling["bootstrap_replicates"] == 10000
    for specification in (
        analysis["heartbeat_protocol"],
        analysis["frozen_aggregate_result"],
    ):
        assert sha256_file(PROJECT_DIR / specification["path"]) == (
            specification["sha256"]
        )
    governance = analysis["governance"]
    assert governance["external_final_signal_values_may_be_loaded"] is False
    assert governance["external_final_access_may_be_consumed"] is False
    assert governance["output_is_confirmatory_final"] is False


def test_stage6_temporal_hazard_protocol_keeps_hard_safety_boundary() -> None:
    protocol = json.loads(
        (
            PROJECT_DIR
            / "configs/stage6_temporal_hazard_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["split"][
        "one_step_pairs_never_cross_group_boundaries"
    ] is True
    assert protocol["predictors"]["operating_threshold"]["source"] == (
        "training_out_of_fold_predictions_only"
    )
    safety = protocol["safety_boundary"]
    assert safety[
        "prediction_may_only_schedule_early_updates_or_context_probes"
    ] is True
    assert safety["prediction_may_not_suppress_a_hard_regret_trigger"] is True
    assert safety["prediction_may_not_replace_context_belief_recovery"] is True
    governance = protocol["governance"]
    assert governance["external_final_signal_values_may_be_loaded"] is False
    assert governance["external_final_access_may_be_consumed"] is False
    assert governance["output_is_confirmatory_final"] is False


def test_stage6_predictive_repetition_protocol_charges_false_reservations() -> None:
    protocol = json.loads(
        (
            PROJECT_DIR
            / "configs/stage6_predictive_repetition_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    protection = protocol["protection"]
    assert protection[
        "false_positive_reservation_consumes_capacity_even_without_an_update"
    ] is True
    assert protection[
        "reserved_capacity_uses_codebook_known_worst_case_compact_update_bits"
    ] is True
    safety = protocol["safety_boundary"]
    assert safety[
        "exact_current_spectrum_selects_every_transmitted_action"
    ] is True
    assert safety[
        "prediction_may_not_create_or_suppress_a_hard_update"
    ] is True
    governance = protocol["governance"]
    assert governance["external_final_signal_values_may_be_loaded"] is False
    assert governance["external_final_access_may_be_consumed"] is False
    assert governance["output_is_confirmatory_final"] is False


def test_stage6_joint_predictive_recovery_protocol_preserves_safety() -> None:
    protocol = json.loads(
        (
            PROJECT_DIR
            / "configs/stage6_joint_predictive_recovery_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["controller"][
        "false_positive_reservation_consumes_capacity"
    ] is True
    assert protocol["monte_carlo"][
        "paired_common_random_numbers"
    ] is True
    assert protocol["fault_condition"][
        "receiver_context_reset_probability"
    ] == 0.02
    safety = protocol["safety_boundary"]
    assert safety["hard_regret_trigger_cannot_be_suppressed"] is True
    assert safety["prediction_cannot_select_or_modify_spectrum_actions"] is True
    assert safety[
        "context_loss_hypothesis_and_fail_closed_decoding_remain_mandatory"
    ] is True
    governance = protocol["governance"]
    assert governance["external_final_signal_values_may_be_loaded"] is False
    assert governance["external_final_access_may_be_consumed"] is False
    assert governance["output_is_confirmatory_final"] is False
