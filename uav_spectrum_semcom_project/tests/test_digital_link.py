from dataclasses import replace

import numpy as np

from spectrum_semcom.digital_link import (
    DigitalLinkConfig,
    PacketConfig,
    analytic_packet_success_probability,
    append_crc16,
    binomial_wilson_interval,
    check_and_strip_crc16,
    demodulate,
    estimate_waveform_packet_success,
    fec_decode,
    fec_encode,
    modulate,
    packet_plan,
    packetize,
    transmit_payload_analytic,
)


def base_config(**overrides) -> DigitalLinkConfig:
    config = DigitalLinkConfig(
        packet=PacketConfig(payload_bits=128, header_bits=32, crc_bits=16, alignment_bits=8),
        modulation="qpsk",
        fec="hamming74",
        channel="awgn",
        ebn0_db=8.0,
        max_retransmissions=1,
        symbol_rate_baud=1_000_000.0,
        propagation_delay_s=0.0,
        per_attempt_processing_s=0.0,
    )
    return replace(config, **overrides)


def test_packetization_accounts_for_protocol_fec_and_modulation() -> None:
    config = base_config()
    plans = packetize(300, config)

    assert [plan.application_bits for plan in plans] == [128, 128, 44]
    first = plans[0]
    assert first.information_bits == 176
    assert first.aligned_information_bits == 176
    assert first.coded_bits == 308
    assert first.symbols == 154


def test_crc_detects_corruption() -> None:
    bits = np.asarray([0, 1, 1, 0, 1, 0, 0, 1] * 5, dtype=np.uint8)
    protected = append_crc16(bits)
    valid, recovered = check_and_strip_crc16(protected)
    assert valid
    assert np.array_equal(recovered, bits)

    protected[3] ^= 1
    valid, _ = check_and_strip_crc16(protected)
    assert not valid


def test_fec_roundtrip_and_hamming_single_error_correction() -> None:
    rng = np.random.default_rng(9)
    bits = rng.integers(0, 2, size=29, dtype=np.uint8)
    for scheme in ["none", "repetition3", "hamming74"]:
        encoded, padding = fec_encode(bits, scheme)
        if scheme == "hamming74":
            encoded[5] ^= 1
        decoded = fec_decode(encoded, scheme, padding)
        assert np.array_equal(decoded, bits)


def test_modulation_noiseless_roundtrip() -> None:
    rng = np.random.default_rng(11)
    bits = rng.integers(0, 2, size=101, dtype=np.uint8)
    for modulation in ["bpsk", "qpsk", "16qam"]:
        symbols, padding = modulate(bits, modulation)
        decoded = demodulate(symbols, modulation, padding)
        assert np.array_equal(decoded, bits)


def test_latency_budget_is_applied_equally_at_packet_level() -> None:
    config = base_config(latency_budget_s=0.00031, max_retransmissions=0, ebn0_db=100.0)
    plan = packet_plan(128, config)
    assert abs(plan.duration_per_attempt_s - 0.000154) < 1e-12

    result = transmit_payload_analytic(384, config, seed=3)
    assert result.latency_budget_exhausted
    assert result.delivered_packets == 2
    assert result.packet_count == 3
    assert result.transmitted_bits == 2 * plan.coded_bits
    assert result.packet_traces[-1].stop_reason == "latency_budget"


def test_analytic_probability_matches_waveform_within_sampling_error() -> None:
    config = base_config(fec="none", ebn0_db=6.0, max_retransmissions=0)
    plan = packet_plan(64, config)
    predicted = analytic_packet_success_probability(plan, config, np.random.default_rng(1))
    measured = estimate_waveform_packet_success(64, config, trials=1500, seed=2)

    assert measured["ci95_low"] - 0.03 <= predicted <= measured["ci95_high"] + 0.03


def test_high_snr_waveform_chain_delivers_all_supported_modes() -> None:
    for modulation in ["bpsk", "qpsk", "16qam"]:
        for fec in ["none", "repetition3", "hamming74"]:
            config = base_config(modulation=modulation, fec=fec, ebn0_db=30.0)
            result = estimate_waveform_packet_success(73, config, trials=20, seed=27)
            assert result["success_rate"] == 1.0


def test_wilson_interval_is_non_degenerate_at_extreme_counts() -> None:
    zero_low, zero_high = binomial_wilson_interval(0, 50)
    full_low, full_high = binomial_wilson_interval(50, 50)
    assert zero_low == 0.0
    assert 0.0 < zero_high < 0.1
    assert 0.9 < full_low < 1.0
    assert full_high == 1.0
