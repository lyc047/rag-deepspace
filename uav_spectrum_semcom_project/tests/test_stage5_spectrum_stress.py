import numpy as np
import pytest

from spectrum_semcom.stage5_spectrum_stress import apply_spectrum_regime


def test_stress_regimes_are_deterministic_and_shape_preserving() -> None:
    power = np.arange(48, dtype=float).reshape(6, 8)
    sites = np.asarray(["A", "A", "A", "B", "B", "B"])
    regimes = {
        "observed": {},
        "gradual_drift": {"maximum_edge_offset_db": 3.0},
        "abrupt_band_hop": {"period_scenes": 1, "shift_channels": 3},
        "burst_block_interference": {
            "period_scenes": 3,
            "duration_scenes": 1,
            "block_width_channels": 3,
            "interference_db": 8.0,
            "block_start_stride": 2,
        },
    }
    for name, parameters in regimes.items():
        first = apply_spectrum_regime(
            power, sites, regime=name, parameters=parameters
        )
        second = apply_spectrum_regime(
            power, sites, regime=name, parameters=parameters
        )
        assert first.shape == power.shape
        assert np.array_equal(first, second)
    assert np.array_equal(
        apply_spectrum_regime(power, sites, regime="observed", parameters={}),
        power,
    )


def test_stress_regimes_have_expected_mechanisms() -> None:
    power = np.tile(np.arange(8, dtype=float), (4, 1))
    sites = np.asarray(["A"] * 4)
    hopped = apply_spectrum_regime(
        power,
        sites,
        regime="abrupt_band_hop",
        parameters={"period_scenes": 2, "shift_channels": 3},
    )
    assert np.array_equal(hopped[0], power[0])
    assert np.array_equal(hopped[2], np.roll(power[2], 3))
    burst = apply_spectrum_regime(
        power,
        sites,
        regime="burst_block_interference",
        parameters={
            "period_scenes": 4,
            "duration_scenes": 1,
            "block_width_channels": 3,
            "interference_db": 8.0,
            "block_start_stride": 2,
        },
    )
    assert np.array_equal(burst[0, :3], power[0, :3] + 8.0)
    assert np.array_equal(burst[1], power[1])
    with pytest.raises(ValueError):
        apply_spectrum_regime(power, sites, regime="unknown", parameters={})

