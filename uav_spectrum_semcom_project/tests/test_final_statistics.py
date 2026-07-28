import copy

import pytest

from spectrum_semcom.final_statistics import FINAL_METRICS_SCHEMA, canonical_sha256, evaluate_final_statistics, holm_adjust, validate_access_binding, validate_final_metrics_payload


def protocol(repetitions: int = 2000) -> dict:
    return {"final_evaluation_protocol": {"minimum_independent_scene_count": 200, "bootstrap_repetitions": repetitions, "bootstrap_seed": 7, "cvar_alpha": .9, "familywise_alpha": .05, "primary_claims": {"selective_G2_digital_reporting": {"regret_noninferiority_margin": .0026813}}}}


def payload(success: bool = True) -> dict:
    c1, selective = [], []
    for index in range(220):
        scene = f"scene-{index:03d}"
        base_regret = .03 + .005 * ((index % 11) / 10)
        c1_delta = -.004 if success else .001
        for method, regret, bits in (("detection_only", base_regret, 1000 + index % 5), ("detection_plus_resource", base_regret + c1_delta, 1000 + (index % 5) + ((index % 3) - 1))):
            c1.append({"scene_id": scene, "method": method, "regret": regret, "actual_bits": bits})
        selective_delta = .0005 if success else .004
        selective.extend([
            {"scene_id": scene, "method": "all_G2", "regret": base_regret, "actual_bits": 2200 + index % 9},
            {"scene_id": scene, "method": "selective_G2", "regret": base_regret + selective_delta, "actual_bits": 1750 + index % 9},
        ])
    return {"schema_version": FINAL_METRICS_SCHEMA, "registry_sha256": "a" * 64, "access_receipt_sha256": "b" * 64, "families": {"C1_resource_loss": {"rows": c1}, "selective_G2_digital_reporting": {"rows": selective}}}


def test_success_and_failure_decisions() -> None:
    passed = evaluate_final_statistics(payload(True), protocol())
    assert passed["family_decisions"] == {"C1_resource_loss": True, "selective_G2_digital_reporting": True}
    failed = evaluate_final_statistics(payload(False), protocol())
    assert failed["family_decisions"] == {"C1_resource_loss": False, "selective_G2_digital_reporting": False}


def test_payload_rejects_missing_pair_and_duplicate() -> None:
    value = payload()
    value["families"]["C1_resource_loss"]["rows"].pop()
    assert any("scene sets do not match" in x for x in validate_final_metrics_payload(value))
    duplicate = payload()
    duplicate["families"]["C1_resource_loss"]["rows"].append(copy.deepcopy(duplicate["families"]["C1_resource_loss"]["rows"][0]))
    assert any("duplicate scene-method" in x for x in validate_final_metrics_payload(duplicate))


def test_access_binding_and_holm() -> None:
    receipt = {"registry_sha256": "a" * 64, "timestamp": "x"}
    value = payload(); value["access_receipt_sha256"] = canonical_sha256(receipt)
    state = {"status": "access_consumed", "access_count": 1, "receipt": receipt}
    assert validate_access_binding(value, state) == []
    assert validate_access_binding(value, {"status": "not_accessed", "access_count": 0})
    adjusted = holm_adjust({"a": .01, "b": .04})
    assert adjusted == pytest.approx({"a": .02, "b": .04})
