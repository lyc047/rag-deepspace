import numpy as np
import pytest

from spectrum_semcom.counterfactual_value import admissible_current_states, candidate_message_cost, evaluate_one_step_oracle, generate_counterfactual_labels, quantized_node_reports
from spectrum_semcom.digital_link import DigitalLinkConfig
from spectrum_semcom.multigranular_semantics import SemanticQuality


def quality() -> SemanticQuality:
    return SemanticQuality(6, 0.8, 0.2, 0, 0.1, 1, 10, 0.1, 0.95, 0.05)


def test_state_enumeration_limits_current_conditional_depth() -> None:
    states = admissible_current_states(2, max_upgraded_nodes=1)
    assert (1, 1) in states and (2, 1) in states and (1, 3) in states
    assert (2, 3) not in states


def test_two_bit_preview_is_supported_without_changing_G2_G3() -> None:
    nodes = np.asarray([[0.1, 0.4, 0.6, 0.9], [0.2, 0.3, 0.7, 0.8]])
    one = quantized_node_reports(nodes, 1)
    two = quantized_node_reports(nodes, 2)
    assert not np.array_equal(one[1], two[1])
    assert np.array_equal(one[2], two[2]) and np.array_equal(one[3], two[3])


def test_counterfactual_labels_are_truth_audited_and_costed_by_digital_codec() -> None:
    nodes = np.asarray([[0.1, 0.9, 0.2, 0.8], [0.2, 0.7, 0.3, 0.6]])
    truth = np.asarray([0.0, 0.8, 0.1, 0.7])
    links = (DigitalLinkConfig(ebn0_db=2), DigitalLinkConfig(ebn0_db=8))
    labels = generate_counterfactual_labels(split_role="train", scene_id="s", node_occupancy=nodes, truth_occupancy=truth, qualities=(quality(), quality()), link_configs=links, demand_channels=1)
    assert labels
    assert all(label.nominal_transmitted_bits > label.application_bits > 0 for label in labels)
    assert all(label.expected_transmitted_bits >= label.nominal_transmitted_bits for label in labels)
    assert all(np.isfinite(label.value_per_expected_bit) for label in labels)
    assert all(len(label.candidate_current_report) == nodes.shape[1] for label in labels)
    assert all(len(label.candidate_quality) == 10 for label in labels)


def test_counterfactual_label_generation_rejects_validation_truth() -> None:
    with pytest.raises(PermissionError, match="train only"):
        generate_counterfactual_labels(split_role="validation", scene_id="s", node_occupancy=np.full((2, 4), 0.5), truth_occupancy=np.zeros(4), qualities=(quality(), quality()), link_configs=(DigitalLinkConfig(), DigitalLinkConfig()), demand_channels=1)


def test_validation_oracle_helper_returns_only_best_aggregate_action() -> None:
    result=evaluate_one_step_oracle(np.asarray([[0.1,0.9,0.2,0.8],[0.2,0.7,0.3,0.6]]),np.asarray([0.0,0.8,0.1,0.7]),1)
    assert result.regret_reduction >= 0


def test_candidate_cost_uses_granularity_specific_codec_fields() -> None:
    occupancy = np.asarray([0.0, 1.0, 0.0, 1.0])
    link = DigitalLinkConfig(ebn0_db=6)
    costs = [candidate_message_cost("s", 0, target, occupancy, quality(), link) for target in (1, 2, 3)]
    assert costs[0][0] < costs[1][0] < costs[2][0]
    assert all(expected >= nominal for _, nominal, expected in costs)
