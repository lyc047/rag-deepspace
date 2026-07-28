import json
from pathlib import Path


def test_stage6_acceptance_authorizes_registration_not_final_execution() -> None:
    project = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (
            project / "configs/stage6_development_acceptance_v1.json"
        ).read_text(encoding="utf-8")
    )
    decision = protocol["decision_rule"]
    assert decision[
        "passing_authorizes_stage6_final_protocol_registration_only"
    ]
    assert not decision["passing_authorizes_external_final_execution"]
    assert decision[
        "caution_flags_must_be_carried_into_final_protocol_and_thesis_claims"
    ]
    assert not protocol["governance"][
        "external_final_access_may_be_consumed"
    ]
