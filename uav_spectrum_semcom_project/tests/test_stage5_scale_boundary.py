import pytest

from spectrum_semcom.stage5_scale_boundary import (
    demand_set_from_fractions,
    index_width,
    representation_plans,
)


def test_fractional_query_families_scale_exactly() -> None:
    assert demand_set_from_fractions(
        8, ["1/4", "1/2", "3/4"]
    ) == (2, 4, 6)
    assert demand_set_from_fractions(
        64, ["1/8", "1/4", "3/8", "1/2", "5/8", "3/4", "7/8"]
    ) == (8, 16, 24, 32, 40, 48, 56)
    with pytest.raises(ValueError):
        demand_set_from_fractions(10, ["1/8"])


def test_representation_plans_have_auditable_payloads() -> None:
    plans = {
        plan.name: plan
        for plan in representation_plans(
            n_channels=8,
            demand_channels=(2, 4, 6),
            application_header_bits=40,
            soft_power_bits_per_channel=4,
        )
    }
    assert index_width(8, 2) == 3
    assert plans["single_index_mean"].application_packet_bits == (43, 43, 42)
    assert plans["bundle_all_queries"].application_packet_bits == (48,)
    assert plans["separate_indices_all_queries"].application_packet_bits == (
        43,
        43,
        42,
    )
    assert plans["occupancy_vector_all_queries"].application_packet_bits == (48,)
    assert plans["soft_power_all_queries"].application_packet_bits == (72,)

