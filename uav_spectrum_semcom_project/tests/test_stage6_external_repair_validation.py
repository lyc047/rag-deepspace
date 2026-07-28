import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage6_external_repair_validation_v1.json"


def test_repair_validation_is_explicitly_non_confirmatory() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["classification"] == "external_repair_validation_non_confirmatory"
    interpretation = config["interpretation"]
    assert not interpretation["may_be_called_external_confirmatory_final"]
    assert not interpretation["may_be_called_single_access_final"]
    assert interpretation["registered_final_acceptance_rules_are_descriptive_only"]


def test_repair_scope_is_adapter_only_and_catalog_is_complete() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    change = config["permitted_change"]
    assert change["path"] == "src/spectrum_semcom/aerpaw_helikite_repair.py"
    assert change["parent_sha256"] is None
    assert not change["may_change_algorithm"]
    assert not change["may_change_task_thresholds"]
    assert not change["may_change_scene_selection"]
    assert config["parent_final"]["catalog"]["required_scene_count"] == 200
    assert config["execution"]["all_200_scenes_required"]
    assert not config["execution"]["subset_analysis_as_primary_result_permitted"]


def test_parent_final_access_failure_remains_immutable() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    state = json.loads(
        (ROOT / config["parent_final"]["access_state"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert state["status"] == "access_consumed_final_execution_failed"
    assert state["access_count"] == 1
    assert state["final_signal_values_accessed"]
    assert not state["retry_on_this_final_permitted"]
    assert not state["reset_permitted"]
