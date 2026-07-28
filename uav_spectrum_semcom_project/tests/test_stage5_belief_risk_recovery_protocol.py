import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_belief_risk_protocol_is_exploratory_and_final_closed() -> None:
    protocol = json.loads(
        (
            ROOT
            / "configs/stage5_belief_risk_recovery_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert set(protocol["methods"]) == {
        "ideal_ack",
        "naive_ack",
        "epoch_guard",
        "uncertainty_recovery",
        "belief_risk_recovery",
    }
    assert protocol["methods"]["belief_risk_recovery"][
        "semantic_belief_recovery"
    ]
    assert protocol["fixed_policy"]["parameters_may_be_retuned"] is False
    governance = protocol["governance"]
    assert governance["stage4_final_measurements_may_be_loaded"] is False
    assert governance["stage4_final_metrics_may_be_loaded"] is False
    assert governance["output_is_confirmatory_final"] is False


def test_belief_risk_result_passes_noninferiority_but_keeps_strict_boundary() -> None:
    result = json.loads(
        (
            ROOT
            / "results/stage5/belief_risk_recovery_development_v1"
            / "belief_risk_recovery_result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["status"] == "stage5_belief_risk_recovery_development_complete"
    assert len(result["conditions"]) == 8
    belief_passes = [
        row["comparisons"][
            "belief_risk_recovery_vs_uncertainty_recovery"
        ]["passes_belief_gate"]
        for row in result["conditions"].values()
    ]
    strict_passes = [
        row["comparisons"]["belief_risk_recovery_vs_naive"][
            "passes_recovery_gate"
        ]
        for row in result["conditions"].values()
    ]
    assert sum(belief_passes) == 8
    assert sum(strict_passes) == 3
    assert result["governance"]["confirmatory_final"] is False


def test_belief_risk_result_is_exactly_reproducible() -> None:
    development = (
        ROOT
        / "results/stage5/belief_risk_recovery_development_v1"
        / "belief_risk_recovery_result.json"
    )
    reproduction = (
        ROOT
        / "results/stage5/belief_risk_recovery_reproduction_v1"
        / "belief_risk_recovery_result.json"
    )
    expected = "c59b08afe58c7f12debaa79b9d21b95fff3a68a25e39f672fb9c46754bd7caea"
    assert hashlib.sha256(development.read_bytes()).hexdigest() == expected
    assert hashlib.sha256(reproduction.read_bytes()).hexdigest() == expected
