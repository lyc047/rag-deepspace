import numpy as np
import pytest

from spectrum_semcom.stage5_event_semantics import (
    QueryState,
    absolute_index_width,
    block_regret,
    choose_event_update,
    contiguous_block_costs,
    decode_signed_gamma,
    decode_update,
    encode_signed_gamma,
    encode_update,
    next_success_state,
    signed_gamma_width,
)


def test_query_block_costs_and_regret_follow_contiguous_demand() -> None:
    power = np.array([-100.0, -102.0, -90.0, -80.0, -110.0, -108.0])
    costs = contiguous_block_costs(power, 2)
    np.testing.assert_allclose(costs, [-101.0, -96.0, -85.0, -95.0, -109.0])
    assert int(np.argmin(costs)) == 4
    assert block_regret(costs, 4) == 0.0
    assert block_regret(costs, 0) == 8.0


@pytest.mark.parametrize("delta", [-7, -3, -1, 1, 2, 9])
def test_signed_gamma_codec_round_trips(delta: int) -> None:
    encoded = encode_signed_gamma(delta)
    assert encoded.size == signed_gamma_width(delta)
    assert decode_signed_gamma(encoded) == delta


def test_absolute_and_delta_update_codecs_are_real_bitstreams() -> None:
    absolute = encode_update(
        mode="absolute",
        target_start=4,
        cached_start=None,
        candidate_blocks=5,
        node_id=7,
        demand_channels=4,
        epoch=3,
        application_header_bits=40,
    )
    assert absolute.size == 40 + absolute_index_width(5)
    decoded_absolute = decode_update(
        absolute,
        candidate_blocks=5,
        cached_start=None,
        application_header_bits=40,
    )
    assert decoded_absolute.selected_start == 4
    assert decoded_absolute.mode == "absolute"
    assert decoded_absolute.node_id == 7
    assert decoded_absolute.demand_channels == 4

    delta = encode_update(
        mode="delta",
        target_start=3,
        cached_start=2,
        candidate_blocks=5,
        node_id=7,
        demand_channels=4,
        epoch=4,
        application_header_bits=152,
    )
    decoded_delta = decode_update(
        delta,
        candidate_blocks=5,
        cached_start=2,
        application_header_bits=152,
    )
    assert decoded_delta.selected_start == 3
    assert decoded_delta.mode == "delta"
    assert delta.size == 152 + signed_gamma_width(1)


def test_policy_silences_only_when_exact_reuse_regret_is_within_threshold() -> None:
    costs = np.array([-100.0, -101.0, -102.0, -104.0, -103.0])
    state = QueryState(4, "2022-02-10T00:00:00", 2)
    decision = choose_event_update(
        costs,
        state=state,
        timestamp_local="2022-02-10T00:20:00",
        regret_threshold_db=1.0,
        max_age_minutes=60.0,
        demand_channels=4,
    )
    assert decision.mode == "silence"
    assert decision.reuse_regret_db == 1.0
    assert decision.application_bits == 0

    update = choose_event_update(
        costs,
        state=state,
        timestamp_local="2022-02-10T00:20:00",
        regret_threshold_db=0.5,
        max_age_minutes=60.0,
        demand_channels=4,
    )
    assert update.transmits
    assert update.target_start == 3
    assert update.reuse_regret_db > 0.5


def test_missing_or_expired_state_forces_absolute_refresh() -> None:
    costs = np.array([-102.0, -100.0, -99.0])
    missing = choose_event_update(
        costs,
        state=None,
        timestamp_local="2022-02-10T00:00:00",
        regret_threshold_db=0.0,
        max_age_minutes=60.0,
        demand_channels=6,
    )
    assert missing.mode == "absolute"

    stale = choose_event_update(
        costs,
        state=QueryState(0, "2022-02-10T00:00:00", 8),
        timestamp_local="2022-02-10T01:00:01",
        regret_threshold_db=100.0,
        max_age_minutes=60.0,
        demand_channels=6,
    )
    assert stale.mode == "absolute"
    refreshed = next_success_state(
        stale,
        previous=QueryState(0, "2022-02-10T00:00:00", 8),
        timestamp_local="2022-02-10T01:00:01",
    )
    assert refreshed.selected_start == int(np.argmin(costs))
    assert refreshed.epoch == 9


def test_invalid_code_or_time_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        encode_signed_gamma(0)
    with pytest.raises(ValueError):
        decode_signed_gamma(np.array([0, 1, 1], dtype=np.uint8))
    with pytest.raises(ValueError, match="precedes"):
        choose_event_update(
            np.array([-2.0, -1.0]),
            state=QueryState(0, "2022-02-10T01:00:00", 0),
            timestamp_local="2022-02-10T00:00:00",
            regret_threshold_db=0.0,
            max_age_minutes=60.0,
            demand_channels=1,
        )
