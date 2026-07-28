import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCOPE_PATH = PROJECT_DIR / "configs" / "research_scope.json"


def load_scope() -> dict:
    with SCOPE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_stage0_scope_has_fixed_hypotheses_and_contributions() -> None:
    scope = load_scope()

    assert scope["stage"] == 0
    assert scope["status"] == "canonical"
    assert [item["id"] for item in scope["hypotheses"]] == ["H1", "H2", "H3", "H4"]
    assert [item["id"] for item in scope["contributions"]] == ["C1", "C2", "C3"]
    assert [item["priority"] for item in scope["contributions"][:2]] == ["core", "core"]


def test_stage0_scope_requires_claim_and_fairness_metadata() -> None:
    scope = load_scope()
    required = set(scope["required_experiment_metadata"])

    assert {
        "hypothesis_ids",
        "split_id",
        "manifest_checksum",
        "seed",
        "fairness_budget_type",
        "fairness_budget_value",
        "sensing_channel",
        "reporting_channel",
        "metrics",
        "repeat_count",
    } <= required


def test_stage0_scope_keeps_visual_work_outside_core_evidence() -> None:
    scope = load_scope()

    assert scope["primary_scope"] == "frequency_semantic_reporting_and_resource_selection"
    assert scope["visual_module_role"] == "optional_system_extension_not_core_evidence"
    assert "new_visual_detector_or_visual_jscc" in scope["non_goals"]
    assert "mean_occupancy_regret" in scope["primary_metrics"]
    assert "total_transmitted_bits_per_decision" in scope["primary_metrics"]
