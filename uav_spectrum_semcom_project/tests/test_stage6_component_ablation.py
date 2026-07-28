import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "src"))

from run_stage6_component_ablation_development import (  # noqa: E402
    _exact_bundle_bits,
    _simulate_exact_query_bundle,
)
from spectrum_semcom.stage6_task_codebook import (  # noqa: E402
    SpectrumTaskQuery,
    build_task_state,
)


def test_exact_bundle_width_matches_three_query_schema() -> None:
    queries = tuple(
        SpectrumTaskQuery(demand) for demand in (2, 4, 6)
    )
    assert _exact_bundle_bits(8, queries, header_bits=48) == 56


def test_exact_bundle_reference_is_self_contained_and_fail_safe() -> None:
    queries = (SpectrumTaskQuery(2),)
    states = [
        build_task_state(
            np.asarray([-100.0, -90.0, -80.0, -70.0]),
            queries,
            epsilon_db=0.2,
        ),
        build_task_state(
            np.asarray([-70.0, -80.0, -90.0, -100.0]),
            queries,
            epsilon_db=0.2,
        ),
    ]
    random_values = np.ones((2, 9), dtype=np.float64)
    result = _simulate_exact_query_bundle(
        states,
        np.asarray([0, 1]),
        random_values=random_values,
        packet_loss_probability=0.1,
        receiver_reset_probability=0.02,
        packet_bits=50,
        epsilon_db=0.2,
        outage_penalty_db=10.0,
    )
    assert result.total_application_bits == 100
    assert result.ack_application_bits == 0
    assert result.available == [True, True]
    assert result.clean == [True, True]
    assert result.wrong_codebook_decode_count == 0


def test_ablation_protocol_cannot_modify_frozen_architecture() -> None:
    protocol = json.loads(
        (
            PROJECT
            / "configs/stage6_component_ablation_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert not protocol["architecture_freeze"][
        "algorithm_modules_may_be_added"
    ]
    assert not protocol["architecture_freeze"][
        "frozen_parameters_may_be_retuned"
    ]
    assert not protocol["analysis_rules"][
        "ablation_results_may_change_frozen_architecture"
    ]
    assert not protocol["governance"][
        "external_final_access_may_be_consumed"
    ]
    assert (
        protocol["ablation_cells"][
            "minus_context_belief_recovery"
        ]["role"]
        == "dependency_group_ablation"
    )
