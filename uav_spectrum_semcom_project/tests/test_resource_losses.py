import torch
import pytest

from spectrum_semcom.resource_losses import (
    compute_gate_a_loss,
    contiguous_block_ranking_loss,
    contiguous_block_occupancy,
    discrete_occupancy_regret,
    empirical_cvar,
    gate_a_loss_configs,
    missed_occupancy_loss,
    resource_task_diagnostics,
    soft_occupancy_regret,
    tail_contiguous_block_ranking_loss,
    update_rate_dual_multiplier,
)


def test_contiguous_block_occupancy_matches_stage2_definition() -> None:
    occupancy = torch.tensor([[0.1, 0.5, 0.9, 0.3]])
    assert torch.allclose(contiguous_block_occupancy(occupancy, 2), torch.tensor([[0.3, 0.7, 0.6]]))


def test_discrete_regret_is_zero_for_oracle_and_positive_for_wrong_choice() -> None:
    truth = torch.tensor([[0.1, 0.8, 0.4, 0.2]])
    assert discrete_occupancy_regret(truth, truth, 1).item() == pytest.approx(0.0)
    wrong = torch.tensor([[0.9, 0.0, 0.4, 0.2]])
    assert discrete_occupancy_regret(wrong, truth, 1).item() == pytest.approx(0.7)


def test_soft_regret_approaches_discrete_choice_and_has_gradient() -> None:
    truth = torch.tensor([[0.1, 0.8, 0.4, 0.2]])
    predicted = torch.tensor([[0.9, 0.0, 0.4, 0.2]], requires_grad=True)
    loss = soft_occupancy_regret(predicted, truth, 1, inverse_temperature=100.0)
    assert loss.item() == pytest.approx(0.7, abs=1e-5)
    loss.backward()
    assert predicted.grad is not None
    assert torch.all(torch.isfinite(predicted.grad))
    assert torch.sum(torch.abs(predicted.grad)).item() > 0


def test_block_ranking_loss_prefers_truth_oracle_and_has_gradient() -> None:
    truth = torch.tensor([[0.0, 0.1, 0.8, 0.9]])
    good = torch.tensor([[0.0, 0.1, 0.8, 0.9]], requires_grad=True)
    wrong = torch.tensor([[0.9, 0.8, 0.1, 0.0]], requires_grad=True)
    assert contiguous_block_ranking_loss(good, truth, 2) < contiguous_block_ranking_loss(wrong, truth, 2)
    loss = contiguous_block_ranking_loss(wrong, truth, 2)
    loss.backward()
    assert wrong.grad is not None and torch.sum(torch.abs(wrong.grad)) > 0


def test_tail_block_ranking_loss_targets_worst_scene_and_has_gradient() -> None:
    truth = torch.tensor([[0.0, 0.1, 0.8, 0.9], [0.0, 0.1, 0.8, 0.9]])
    predicted = torch.tensor(
        [[0.0, 0.1, 0.8, 0.9], [0.9, 0.8, 0.1, 0.0]], requires_grad=True
    )
    per_scene = contiguous_block_ranking_loss(predicted, truth, 2, reduction="none")
    tail = tail_contiguous_block_ranking_loss(predicted, truth, 2, cvar_alpha=0.5)
    assert tail.item() == pytest.approx(float(per_scene.detach().max()))
    tail.backward()
    assert predicted.grad is not None and torch.sum(torch.abs(predicted.grad[1])) > 0


def test_missed_occupancy_loss_penalizes_underestimation() -> None:
    truth = torch.ones((1, 2))
    low = missed_occupancy_loss(torch.full((1, 2), 0.1), truth)
    high = missed_occupancy_loss(torch.full((1, 2), 0.9), truth)
    assert low > high


def test_empirical_cvar_uses_worst_tail() -> None:
    losses = torch.tensor([0.0, 1.0, 2.0, 9.0])
    assert empirical_cvar(losses, alpha=0.75).item() == pytest.approx(9.0)
    assert empirical_cvar(losses, alpha=0.5).item() == pytest.approx(5.5)


def test_gate_a_variants_change_only_preregistered_loss_terms() -> None:
    configs = gate_a_loss_configs()
    assert set(configs) == {"detection_only", "detection_plus_rate", "detection_plus_resource", "full_joint_loss"}
    assert configs["detection_only"].rate_weight == configs["detection_only"].resource_weight == 0.0
    assert configs["detection_plus_rate"].rate_weight > 0 and configs["detection_plus_rate"].resource_weight == 0.0
    assert configs["detection_plus_resource"].resource_weight > 0 and configs["detection_plus_resource"].rate_weight == 0.0
    assert configs["full_joint_loss"].tail_weight > 0 and configs["full_joint_loss"].miss_weight > 0


def test_compute_gate_a_loss_reports_rate_and_task_components() -> None:
    predicted = torch.tensor([[0.2, 0.8, 0.3], [0.7, 0.1, 0.5]], requires_grad=True)
    truth = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.2, 0.4]])
    bits = torch.tensor([1200.0, 800.0])
    output = compute_gate_a_loss(predicted, truth, bits, 1000.0, gate_a_loss_configs()["full_joint_loss"])
    assert output.rate_violation.item() == pytest.approx(0.0)
    assert output.total.item() > 0
    output.total.backward()
    assert predicted.grad is not None and torch.all(torch.isfinite(predicted.grad))


def test_resource_diagnostics_identify_ambiguous_saturation() -> None:
    saturated = resource_task_diagnostics(torch.full((4, 4), 0.01), 1, clean_threshold=0.02)
    varied = resource_task_diagnostics(
        torch.tensor([[0.0, 0.2, 0.6, 0.8], [0.7, 0.1, 0.5, 0.9]]), 1, clean_threshold=0.02
    )
    assert saturated.oracle_clean_rate == 1.0
    assert saturated.mean_random_oracle_gap == pytest.approx(0.0)
    assert saturated.ambiguous_oracle_rate == 1.0
    assert varied.mean_random_oracle_gap > 0.2
    assert varied.ambiguous_oracle_rate == 0.0


def test_rate_dual_update_increases_only_for_budget_pressure_and_is_projected() -> None:
    assert update_rate_dual_multiplier(0.2, 1200.0, 1000.0, 0.5) == pytest.approx(0.3)
    assert update_rate_dual_multiplier(0.2, 500.0, 1000.0, 1.0) == 0.0
    assert update_rate_dual_multiplier(1.9, 2000.0, 1000.0, 1.0, maximum_multiplier=2.0) == 2.0


def test_resource_losses_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match=r"same \[batch, channels\] shape"):
        soft_occupancy_regret(torch.zeros((2, 4)), torch.zeros((2, 3)), 1)
