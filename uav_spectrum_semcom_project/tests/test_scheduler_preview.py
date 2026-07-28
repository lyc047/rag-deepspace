import numpy as np
import pytest

from spectrum_semcom.digital_link import DigitalLinkConfig, expected_transmitted_bits_awgn, nominal_transmitted_bits
from spectrum_semcom.scheduler_preview import build_scheduler_preview, decode_scheduler_preview, encode_scheduler_preview


def test_scheduler_preview_round_trip_and_sparse_reconstruction() -> None:
    soft = np.asarray([.1, .49, .8, .55, .05, .9, .3, .7])
    preview = build_scheduler_preview(soft, 3, "scene", 6, .8, .02, .1, 2, 3)
    encoded = encode_scheduler_preview(preview)
    decoded = decode_scheduler_preview(encoded.bits)
    assert encoded.application_bits == 113
    assert decoded.residual_indices == preview.residual_indices
    assert np.allclose(decoded.scheduler_report, preview.scheduler_report)
    assert np.array_equal(decoded.hard_occupancy, np.rint(soft))


def test_scheduler_preview_cost_counts_packet_overhead_and_rejects_trailing_bits() -> None:
    preview = build_scheduler_preview(np.linspace(.1, .9, 8), 0, "s", 0, .5, 0, .1)
    encoded = encode_scheduler_preview(preview); link = DigitalLinkConfig(ebn0_db=4)
    assert expected_transmitted_bits_awgn(encoded.application_bits, link) >= nominal_transmitted_bits(encoded.application_bits, link) > encoded.application_bits
    with pytest.raises(ValueError, match="trailing"):
        decode_scheduler_preview(np.append(encoded.bits, 0))
