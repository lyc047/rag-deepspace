from spectrum_semcom.ldpc_link import (
    LdpcCodeConfig,
    make_ldpc_matrices,
    simulate_ldpc_packet_success,
    transmit_payload_empirical_code,
)


def small_code():
    config = LdpcCodeConfig(
        code_length=120,
        variable_node_degree=2,
        check_node_degree=4,
        systematic=True,
        matrix_seed=17,
        max_decoder_iterations=50,
        application_header_bits=8,
        crc_bits=16,
    )
    return config, make_ldpc_matrices(config)


def test_ldpc_matrix_dimensions_and_payload_accounting() -> None:
    config, matrices = small_code()
    assert matrices.code_length == 120
    assert matrices.information_bits == matrices.generator.shape[1]
    assert matrices.application_payload_bits == matrices.information_bits - config.application_header_bits - 16
    assert 0.0 < matrices.code_rate < 1.0


def test_actual_ldpc_waveform_succeeds_at_high_snr() -> None:
    config, matrices = small_code()
    result = simulate_ldpc_packet_success(
        matrices, config, ebn0_db=20.0, trials=20, batch_size=5, seed=8
    )
    assert result["success_rate"] == 1.0
    assert result["bler"] == 0.0


def test_empirical_code_link_counts_codewords_and_retransmissions() -> None:
    result = transmit_payload_empirical_code(
        application_bits=80,
        application_payload_bits_per_codeword=40,
        codeword_bits=120,
        packet_success_probability=1.0,
        symbol_rate_baud=1_000_000.0,
        propagation_delay_s=0.0,
        per_attempt_processing_s=0.0,
        max_retransmissions=1,
        latency_budget_s=1.0,
        seed=4,
    )
    assert result.frame_success
    assert result.packet_count == 2
    assert result.transmitted_bits == 240
    assert result.retransmissions == 0


def test_empirical_code_link_respects_deadline() -> None:
    result = transmit_payload_empirical_code(
        application_bits=80,
        application_payload_bits_per_codeword=40,
        codeword_bits=120,
        packet_success_probability=1.0,
        symbol_rate_baud=1_000_000.0,
        propagation_delay_s=0.0,
        per_attempt_processing_s=0.0,
        max_retransmissions=0,
        latency_budget_s=0.00012,
        seed=4,
    )
    assert result.latency_budget_exhausted
    assert result.delivered_packets == 1
    assert not result.frame_success
