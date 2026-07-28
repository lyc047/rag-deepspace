import numpy as np
import torch

from spectrum_semcom.value_prediction import CounterfactualValueNetwork, build_deployable_value_features, fit_standardization, grouped_value_loss


def test_deployable_features_mask_unavailable_quality_and_exclude_counterfactual_report() -> None:
    kwargs = dict(current_belief=np.asarray([0.1, 0.5]), candidate_current_report=np.asarray([0.0, 1.0]), candidate_quality=np.asarray([6.0, 0.9, 123.0]), quality_availability=np.asarray([1, 1, 0]), current_state=(1, 1), candidate_node=0, target_granularity=2, expected_transmitted_bits=600.0)
    first = build_deployable_value_features(**kwargs)
    kwargs["candidate_quality"] = np.asarray([6.0, 0.9, -999.0])
    second = build_deployable_value_features(**kwargs)
    assert np.array_equal(first, second)
    assert first.ndim == 1 and np.all(np.isfinite(first))


def test_grouped_loss_and_standardization_are_finite() -> None:
    features = np.arange(24, dtype=np.float32).reshape(6, 4)
    targets = np.asarray([0.0, 1.0, -1.0, 0.5, 0.0, 0.2])
    stats = fit_standardization(features, targets)
    model = CounterfactualValueNetwork(4, (8,), 0.0)
    prediction = model(torch.as_tensor((features - stats.feature_mean) / stats.feature_scale).reshape(2, 3, 4))
    target = torch.as_tensor(((targets - stats.target_mean) / stats.target_scale).reshape(2, 3), dtype=torch.float32)
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[1, 2] = False
    loss, parts = grouped_value_loss(prediction, target, valid, (0.0 - stats.target_mean) / stats.target_scale)
    assert torch.isfinite(loss) and torch.isfinite(parts["regression"]) and torch.isfinite(parts["listwise"])
