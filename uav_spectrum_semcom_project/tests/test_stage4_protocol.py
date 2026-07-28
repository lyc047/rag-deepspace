import copy
import json
from pathlib import Path

from spectrum_semcom.stage4_protocol import validate_stage4_protocol, validate_stage4_registry


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((PROJECT_DIR / "configs" / name).read_text(encoding="utf-8"))


def test_frozen_stage4_protocol_is_valid() -> None:
    protocol = _load("stage4_protocol.json")
    assert validate_stage4_protocol(protocol) == []
    assert protocol["final_holdout"]["access_count"] == 0
    assert protocol["claim_boundary"]["image_semantics_core_contribution"] is False


def test_protocol_only_stage4_registry_is_valid_and_honest() -> None:
    registry = _load("stage4_dataset_registry_v1.json")
    assert validate_stage4_registry(registry) == []
    assert registry["status"] == "protocol_only_awaiting_independent_scene_catalog"
    assert all(entry["scene_count"] == 0 for entry in registry["splits"].values())
    assert registry["final_holdout"]["labels_or_model_outputs_accessed"] is False


def test_stage4_registry_rejects_cross_split_scene_leakage() -> None:
    registry = _load("stage4_dataset_registry_v1.json")
    broken = copy.deepcopy(registry)
    for split in ("train", "validation"):
        broken["splits"][split].update(
            {"scene_count": 1, "scene_ids": ["shared-scene"], "scene_ids_sha256": "wrong"}
        )
    errors = validate_stage4_registry(broken)
    assert any("scene leakage" in error for error in errors)


def test_stage4_protocol_rejects_final_holdout_retuning() -> None:
    protocol = _load("stage4_protocol.json")
    broken = copy.deepcopy(protocol)
    broken["final_holdout"]["retuning_after_access"] = True
    assert "retuning_after_access must be false" in validate_stage4_protocol(broken)


def test_stage4_protocol_rejects_counterfactual_feature_leakage() -> None:
    protocol = _load("stage4_protocol.json")
    broken = copy.deepcopy(protocol)
    broken["value_prediction_protocol"]["prohibited_inputs"].remove("candidate_high_precision_report")
    assert any("high-precision" in error for error in validate_stage4_protocol(broken))
