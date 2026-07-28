import numpy as np
import pytest

from spectrum_semcom.aerpaw_final_inference import occupancy_quality, power_block_regret_db


def test_power_block_regret_uses_dbm_cost_not_proxy_as_truth() -> None:
    power = np.asarray([-100, -100, -100, -100, -90, -90, -90, -90], dtype=float)
    good = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    bad = 1 - good
    assert power_block_regret_db(good, power, 4) == 0.0
    assert power_block_regret_db(bad, power, 4) == pytest.approx(10.0)


def test_frozen_quality_payload_is_finite_and_truth_free() -> None:
    quality = occupancy_quality(np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert 0 <= quality.prediction_confidence <= 1
    assert 0 <= quality.normalized_entropy <= 1
    assert quality.report_success_probability == 1.0


def test_power_block_regret_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        power_block_regret_db(np.asarray([1.2, 0.0]), np.asarray([-100.0, -90.0]), 1)
