import numpy as np
import pytest

from spectrum_semcom.stage5_query_bundle import (
    decode_query_bundle,
    encode_query_bundle,
    query_bundle_payload_width,
)


def test_query_bundle_width_and_codec_round_trip() -> None:
    demands = [2, 4, 6]
    assert query_bundle_payload_width(8, demands) == 8
    encoded = encode_query_bundle(
        {2: 6, 4: 3, 6: 2},
        n_channels=8,
        demand_order=demands,
        node_id=5,
        epoch=9,
        application_header_bits=40,
    )
    assert encoded.size == 48
    decoded = decode_query_bundle(
        encoded,
        n_channels=8,
        demand_order=demands,
        application_header_bits=40,
    )
    assert decoded.node_id == 5
    assert decoded.epoch == 9
    assert decoded.target_starts == {2: 6, 4: 3, 6: 2}


def test_query_bundle_fails_closed_on_schema_or_target_error() -> None:
    with pytest.raises(ValueError):
        encode_query_bundle(
            {2: 7, 4: 0, 6: 0},
            n_channels=8,
            demand_order=[2, 4, 6],
            node_id=0,
            epoch=0,
        )
    valid = encode_query_bundle(
        {2: 0, 4: 0, 6: 0},
        n_channels=8,
        demand_order=[2, 4, 6],
        node_id=0,
        epoch=0,
    )
    with pytest.raises(ValueError):
        decode_query_bundle(
            valid,
            n_channels=8,
            demand_order=[2, 4],
        )
