from __future__ import annotations

import copy

import numpy as np

from spectrum_semcom.stage9_6_process_protocol import (
    REGISTERED_WIDTHS,
    ReceiverEndpoint,
    SenderEndpoint,
    bits_to_wire,
    protocol_catalog,
    protocol_session,
    protocol_state,
    wire_to_bits,
)
from spectrum_semcom.stage6_context_codec import encode_compact_update


def test_registered_widths_and_wire_roundtrip():
    sender = SenderEndpoint()
    receiver = ReceiverEndpoint.__new__(ReceiverEndpoint)
    assert protocol_catalog().catalog_epoch == 7
    bits, decision = encode_compact_update(
        protocol_state(1), protocol_session(), node_id=1, update_epoch=4
    )
    assert not decision.uses_fallback
    assert bits.size == REGISTERED_WIDTHS["update"] == 42
    assert np.array_equal(wire_to_bits(bits_to_wire(bits)), bits)
    bad = copy.deepcopy(bits_to_wire(bits))
    bad["payload_hex"] = bad["payload_hex"][:-2] + "01"
    try:
        wire_to_bits(bad)
    except ValueError as exc:
        assert "padding" in str(exc)
    else:
        raise AssertionError("nonzero wire padding was accepted")
    assert sender.status()["identity"] is None
    assert receiver is not None


def test_context_frame_does_not_restore_action_until_update(tmp_path):
    sender = SenderEndpoint()
    receiver = ReceiverEndpoint(tmp_path / "catalog.json")
    receiver.boot(boot_id=11, cold=True)
    update = sender.make_update(symbol=1, update_epoch=9)["wire"]
    nack_response = receiver.receive_update(wire=update)
    nack = nack_response["feedback_wire"]
    assert nack["bit_length"] == REGISTERED_WIDTHS["reset_nack"]
    sender.receive_reset_nack(wire=nack)
    recovery = sender.make_recovery()
    assert recovery["kind"] == "full_install"
    assert recovery["wire"]["bit_length"] == REGISTERED_WIDTHS["full_install"]
    assert receiver.receive_recovery(
        kind=recovery["kind"], wire=recovery["wire"]
    )["accepted_context"]
    assert not receiver.execute()["available"]
    assert receiver.receive_update(wire=update)["accepted"]
    assert receiver.execute()["available"]
