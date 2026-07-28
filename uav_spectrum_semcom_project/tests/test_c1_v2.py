import torch

from spectrum_semcom.c1_v2 import calibration_shift_views
from spectrum_semcom.gate_a_model import FixedPrecisionOccupancyHead


def test_fixed_precision_head_guarantees_one_application_length() -> None:
    model = FixedPrecisionOccupancyHead(8, 16, probability_bits=2)
    values = model.infer(torch.tensor([[0.1] * 8, [0.9] * 8]))
    assert values.shape == (2, 8)
    assert model.application_bits == 167
    assert torch.allclose(values * 3, torch.round(values * 3))


def test_calibration_views_are_bounded_and_reproducible() -> None:
    base = torch.tensor([[0.01, 0.1, 0.5, 0.9]])
    specs = [{"temperature": 1.0}, {"temperature": 1.2, "bias": -0.5, "noise_std": 0.02}]
    left = calibration_shift_views(base, specs, generator=torch.Generator().manual_seed(7))
    right = calibration_shift_views(base, specs, generator=torch.Generator().manual_seed(7))
    assert len(left) == 2 and all(torch.all((x > 0) & (x < 1)) for x in left)
    assert all(torch.equal(x, y) for x, y in zip(left, right))
