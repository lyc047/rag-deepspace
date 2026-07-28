import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from evaluate_stage2_classical_baselines import channel_energy, quantize_unit_interval, transmit_tiled_image
from spectrum_semcom.digital_link import DigitalLinkConfig, PacketConfig


def test_channel_energy_respects_resource_axis() -> None:
    image = np.zeros((8, 8), dtype=np.uint8)
    image[4:, :] = 255

    y_energy = channel_energy(image, n_channels=2, axis="y")
    x_energy = channel_energy(image, n_channels=2, axis="x")

    assert np.allclose(y_energy, [0.0, 1.0])
    assert np.allclose(x_energy, [0.5, 0.5])


def test_quantized_psd_uses_declared_number_of_levels() -> None:
    values = np.asarray([0.0, 0.1, 0.5, 0.9, 1.0], dtype=np.float32)
    quantized = quantize_unit_interval(values, bits=2)
    assert np.allclose(quantized * 3.0, np.rint(quantized * 3.0))
    assert quantized[0] == 0.0
    assert quantized[-1] == 1.0


def test_independent_tiles_reconstruct_exactly_on_reliable_link() -> None:
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    link = DigitalLinkConfig(
        packet=PacketConfig(payload_bits=1024, header_bits=32, crc_bits=16, alignment_bits=8),
        modulation="qpsk",
        fec="hamming74",
        channel="awgn",
        ebn0_db=100.0,
        max_retransmissions=0,
        symbol_rate_baud=1_000_000.0,
        propagation_delay_s=0.0,
        per_attempt_processing_s=0.0,
        latency_budget_s=1.0,
    )

    restored, metrics = transmit_tiled_image(image, link, tile_size=4, tile_header_bits=16, seed=12)

    assert np.array_equal(restored, image)
    assert metrics["payload_delivery_ratio"] == 1.0
    assert metrics["latency_budget_exhausted"] == 0.0
