import json
from pathlib import Path

from scripts.audit_stage5_external_final_registry import audit


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/stage5_external_final_registry_v1.json"


def test_external_registry_keeps_pilot_and_final_campaigns_disjoint() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    pilot = registry["development_adapter_pilot"]
    final = registry["final_candidates"]
    assert pilot["final_eligible"] is False
    assert len(final) == 3
    assert pilot["dataset_id"] not in {row["dataset_id"] for row in final}
    assert len({row["campaign"] for row in final}) == 3
    assert all(
        row["source_values_or_method_outputs_accessed"] is False
        for row in final
    )


def test_external_registry_size_totals_and_governance_are_consistent() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    pilot = registry["development_adapter_pilot"]
    final = registry["final_candidates"]
    sizes = registry["size_accounting"]
    assert sizes["final_archives_total_bytes"] == sum(
        row["archive_size_bytes"] for row in final
    )
    assert sizes["pilot_plus_final_archives_total_bytes"] == (
        pilot["archive_size_bytes"] + sizes["final_archives_total_bytes"]
    )
    assert sizes[
        "recommended_free_space_gib_including_compact_caches_logs_and_margin"
    ] >= 15.0
    governance = registry["governance"]
    assert governance["stage4_consumed_final_may_be_loaded"] is False
    assert governance["final_signal_values_may_be_opened_before_code_freeze"] is False
    assert governance[
        "final_method_outputs_may_be_opened_before_single_access"
    ] is False
    assert governance["ack_faults_are_measured_by_these_datasets"] is False
    assert all("download_bundle" in row for row in [pilot, *final])


def test_registry_audit_does_not_open_archive_contents() -> None:
    result = audit(REGISTRY)
    assert result["registered_archives"] == 4
    assert result["signal_values_loaded"] is False
    assert result["method_outputs_loaded"] is False
    assert all(row["archive_contents_opened"] is False for row in result["rows"])
    assert all(row["signal_values_loaded"] is False for row in result["rows"])


def test_external_final_access_state_is_unused() -> None:
    state = json.loads(
        (
            ROOT / "configs/stage5_external_final_access_state.json"
        ).read_text(encoding="utf-8")
    )
    assert state["status"] == "downloaded_integrity_verified_not_accessed"
    assert state["access_count"] == 0
    assert state["final_signal_values_accessed"] is False
    assert state["final_method_outputs_accessed"] is False
