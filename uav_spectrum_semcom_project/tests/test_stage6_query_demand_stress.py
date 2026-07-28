import json
from pathlib import Path


def test_query_demand_protocol_uses_one_joint_frozen_codebook() -> None:
    project = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (
            project / "configs/stage6_query_demand_stress_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["analysis_rules"][
        "fit_one_joint_codebook_for_all_three_demands"
    ]
    assert protocol["analysis_rules"][
        "do_not_refit_a_separate_codebook_per_demand"
    ]
    assert not protocol["analysis_rules"][
        "results_may_modify_frozen_architecture_or_parameters"
    ]
    assert not protocol["governance"][
        "external_final_access_may_be_consumed"
    ]
