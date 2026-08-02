from __future__ import annotations

import numpy as np
import pytest

from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    build_task_state,
    encode_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage9_9_temporal_batching import (
    build_immediate_schedule,
    markov_loss_mask,
    packed_batch_payload_bits,
    replay_frozen_schedule,
)


def _alternating_fixture():
    powers = [
        np.asarray([0.0, 10.0]),
        np.asarray([10.0, 0.0]),
        np.asarray([0.0, 10.0]),
        np.asarray([10.0, 0.0]),
    ]
    query = (SpectrumTaskQuery(1),)
    states = [build_task_state(row, query, epsilon_db=0.0) for row in powers]
    codebook = fit_greedy_task_codebook(states)
    decisions = [encode_task_state(codebook, state) for state in states]
    timestamps = [
        "2026-01-01T00:00:00",
        "2026-01-01T00:00:30",
        "2026-01-01T00:01:00",
        "2026-01-01T00:01:30",
    ]
    initial, events = build_immediate_schedule(states, decisions)
    return states, timestamps, initial, events


def test_registered_batch_widths():
    assert packed_batch_payload_bits(1) == 56
    assert packed_batch_payload_bits(2) == 96
    assert packed_batch_payload_bits(4) == 176


def test_b1_is_clean_and_b2_exposes_waiting_regret():
    states, timestamps, initial, events = _alternating_fixture()
    assert [event.scene_index for event in events] == [1, 2, 3]
    b1 = replay_frozen_schedule(
        states, timestamps, initial, events, batch_size=1
    )
    b2 = replay_frozen_schedule(
        states, timestamps, initial, events, batch_size=2
    )
    assert b1["clean_rate"] == 1.0
    assert b1["h8_update_path_bits"] == 168
    assert b2["clean_rate"] < b1["clean_rate"]
    assert b2["h8_update_path_bits"] == 152
    assert b2["maximum_event_wait_scenes"] == 1
    assert b2["maximum_event_wait_seconds"] == 30.0
    assert b2["group_end_remainder_item_count"] == 1


def test_loss_mask_is_deterministic_and_can_drop_a_full_batch():
    first = markov_loss_mask(
        20,
        seed=7,
        good_to_bad=0.1,
        bad_to_good=0.3,
        drop_good=0.05,
        drop_bad=0.5,
    )
    second = markov_loss_mask(
        20,
        seed=7,
        good_to_bad=0.1,
        bad_to_good=0.3,
        drop_good=0.05,
        drop_bad=0.5,
    )
    assert np.array_equal(first, second)
    states, timestamps, initial, events = _alternating_fixture()
    forced = np.zeros(len(states), dtype=np.bool_)
    forced[2] = True
    replay = replay_frozen_schedule(
        states,
        timestamps,
        initial,
        events,
        batch_size=2,
        loss_mask=forced,
    )
    assert replay["lost_datagram_count"] == 1
    assert replay["clean_rate"] < 1.0


def test_invalid_batch_inputs_fail_closed():
    with pytest.raises(ValueError):
        packed_batch_payload_bits(0)
    states, timestamps, initial, events = _alternating_fixture()
    with pytest.raises(ValueError):
        replay_frozen_schedule(
            states,
            timestamps,
            initial,
            events,
            batch_size=2,
            loss_mask=[False],
        )


def test_no_event_group_has_zero_bits_for_every_batch_size():
    state = build_task_state(
        np.asarray([0.0, 10.0]),
        (SpectrumTaskQuery(1),),
        epsilon_db=0.0,
    )
    codebook = fit_greedy_task_codebook([state])
    decision = encode_task_state(codebook, state)
    initial, events = build_immediate_schedule([state, state], [decision, decision])
    assert events == ()
    replay = replay_frozen_schedule(
        [state, state],
        ["2026-01-01T00:00:00", "2026-01-01T00:00:30"],
        initial,
        events,
        batch_size=4,
    )
    assert replay["h8_update_path_bits"] == 0
    assert replay["clean_rate"] == 1.0
