import json
from pathlib import Path

import pytest

from spectrum_semcom.aerpaw_final_statistics import (
    AERPAW_C1_FINAL_METRICS_SCHEMA,
    evaluate_aerpaw_c1_only_statistics,
    validate_aerpaw_c1_metrics,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def protocols() -> tuple[dict, dict]:
    base = json.loads((PROJECT_DIR / "configs/stage4_protocol.json").read_text(encoding="utf-8"))
    aerpaw = json.loads(
        (PROJECT_DIR / "configs/aerpaw_three_site_prefinal_protocol_v1.json").read_text(encoding="utf-8")
    )
    return base, aerpaw


def payload(success: bool = True) -> dict:
    rows = []
    for index in range(200):
        scene = f"aerpaw-scene-{index:03d}"
        baseline = 0.03 + 0.00001 * (index % 7)
        proposed = baseline - 0.01 if success else baseline + 0.005
        rows.extend(
            [
                {"scene_id": scene, "method": "detection_plus_resource", "regret": proposed, "actual_bits": 512.0},
                {"scene_id": scene, "method": "detection_only", "regret": baseline, "actual_bits": 512.0},
            ]
        )
    return {
        "schema_version": AERPAW_C1_FINAL_METRICS_SCHEMA,
        "registry_sha256": "a" * 64,
        "access_receipt_sha256": "b" * 64,
        "families": {"C1_resource_loss": {"rows": rows}},
    }


def test_aerpaw_c1_only_success_branch_uses_one_family_adjustment() -> None:
    base, aerpaw = protocols()
    result = evaluate_aerpaw_c1_only_statistics(payload(True), base, aerpaw)
    assert result["analysis_id"] == "aerpaw_c1_only_final_statistics_v1"
    assert result["M1_status"] == "failed_before_model_execution"
    assert result["family_decision"] is True
    p = result["family_p_values"]["C1_resource_loss"]
    assert p["holm_adjusted"] == p["raw"]
    assert "Selective G2 was not evaluated" in result["claim_boundary"]


def test_aerpaw_c1_only_failure_is_not_rescued() -> None:
    base, aerpaw = protocols()
    result = evaluate_aerpaw_c1_only_statistics(payload(False), base, aerpaw)
    assert result["family_decision"] is False
    assert result["c1"]["requirements"]["regret_ci_upper_below_0"] is False


def test_aerpaw_c1_only_payload_rejects_second_family_and_missing_pair() -> None:
    value = payload()
    value["families"]["selective_G2_digital_reporting"] = {"rows": []}
    assert any("exactly C1_resource_loss" in error for error in validate_aerpaw_c1_metrics(value))
    value = payload()
    value["families"]["C1_resource_loss"]["rows"].pop()
    with pytest.raises(ValueError, match="scene sets do not match"):
        base, aerpaw = protocols()
        evaluate_aerpaw_c1_only_statistics(value, base, aerpaw)
