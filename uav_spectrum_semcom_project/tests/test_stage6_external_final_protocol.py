import json
from pathlib import Path

from spectrum_semcom.reproducibility import sha256_file
from spectrum_semcom.stage6_final_governance import (
    validate_external_final_catalog,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/stage6_external_final_protocol_v1.json"
REGISTRY = ROOT / "configs/stage5_external_final_registry_v1.json"
STATE = ROOT / "configs/stage6_external_final_access_state.json"
DEVELOPMENT_ANALYSIS = (
    ROOT / "results/stage6/task_scale_sensitivity_analysis_v1/analysis.json"
)


def test_stage6_final_primary_hypothesis_is_frozen() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    primary = protocol["decision_rules"]["primary_configuration"]
    assert primary == {
        "n_channels": 8,
        "epsilon_db": 0.2,
        "query_set_id": "multi_025_050_075",
        "target_clean_rate": 0.9,
    }
    assert protocol["resource_task"]["secondary_n_channels"] == [16, 32, 64]
    assert protocol["system_evaluation"][
        "auxiliary_matched_clean_targets"
    ] == [0.85]
    assert "0.95 and 0.97" in protocol["decision_rules"][
        "high_reliability_boundary"
    ]


def test_stage6_final_freeze_is_authorized_by_s6_7c() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    entry = protocol["frozen_development_evidence"]["task_scale_analysis"]
    assert sha256_file(ROOT / entry["path"]) == entry["sha256"]
    analysis = json.loads(DEVELOPMENT_ANALYSIS.read_text(encoding="utf-8"))
    assert analysis["decision_summary"]["broad_rate_risk_support_passed"]
    assert analysis["decision_summary"]["primary_reproduction_passed"]
    assert analysis["checks"]["external_final_access_count"] == 0


def test_stage6_final_access_state_follows_irreversible_lifecycle() -> None:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    assert state["reset_permitted"] is False
    assert state["access_count"] in {0, 1}
    if state["access_count"] == 0:
        assert state["status"] == "downloaded_integrity_verified_not_accessed"
        assert state["final_signal_values_accessed"] is False
        assert state["final_method_outputs_accessed"] is False
    else:
        assert state["status"].startswith("access_consumed")
        assert state["final_signal_values_accessed"] is True
        assert state["receipt_sha256"]


def test_existing_metadata_catalog_satisfies_stage6_sampling_rules() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    catalog = json.loads(
        (
            ROOT / "results/stage5/external_final_catalog_v1/catalog.json"
        ).read_text(encoding="utf-8")
    )
    assert validate_external_final_catalog(catalog, protocol, registry) == []
    assert catalog["signal_values_loaded_for_selection"] is False
    assert catalog["method_outputs_loaded_for_selection"] is False


def test_final_claim_boundary_excludes_unmeasured_link_and_mimo_claims() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    boundary = protocol["claim_boundary"].lower()
    assert "measured backhaul reliability" in boundary
    assert "real ack-fault recovery" in boundary
    assert "mimo" in boundary
    assert protocol["fault_model"]["evidence_boundary"].startswith(
        "Controlled reporting-link injection"
    )
