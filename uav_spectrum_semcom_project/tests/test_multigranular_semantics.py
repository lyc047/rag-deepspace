import numpy as np
import pytest

from spectrum_semcom.digital_link import DigitalLinkConfig, PacketConfig
from spectrum_semcom.multigranular_semantics import (
    SemanticEvidence,
    SemanticMessage,
    SemanticQuality,
    decode_semantic_message,
    encode_semantic_message,
    quantize_probabilities,
    scene_tag,
    transmit_semantic_message,
)


def quality() -> SemanticQuality:
    return SemanticQuality(
        sensing_snr_db=6.0,
        prediction_confidence=0.87,
        normalized_entropy=0.24,
        clipping_ratio=0.03,
        out_of_band_leakage_ratio=0.08,
        noise_floor_stability_db=1.2,
        peak_to_background_db=14.5,
        age_s=0.12,
        report_success_probability=0.94,
        calibration_error=0.07,
    )


def test_g0_is_true_silence_with_zero_link_cost() -> None:
    message = SemanticMessage("G0", 2, "scene-a", np.empty(0), probability_bits=0)
    result = transmit_semantic_message(message, DigitalLinkConfig(), seed=1)
    assert result.encoded.application_bits == 0
    assert result.link.transmitted_bits == 0
    assert result.link.packet_count == 0
    assert result.decoded is None


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
def test_probability_quantization_is_bounded_and_exactly_auditable(bits: int) -> None:
    source = np.asarray([0.0, 0.17, 0.51, 0.89, 1.0])
    codes, reconstructed = quantize_probabilities(source, bits)
    assert np.all((codes >= 0) & (codes <= 2**bits - 1))
    assert np.max(np.abs(source - reconstructed)) <= 0.5 / (2**bits - 1) + 1e-12


def test_g1_round_trip_preserves_header_and_quantized_occupancy() -> None:
    source = np.asarray([0.02, 0.31, 0.62, 0.99])
    encoded = encode_semantic_message(SemanticMessage("G1", 7, "scene-preview", source, probability_bits=2))
    decoded = decode_semantic_message(encoded.bits)
    assert decoded.granularity == "G1"
    assert decoded.node_id == 7
    assert decoded.scene_tag == scene_tag("scene-preview")
    assert decoded.probability_bits == 2
    assert np.allclose(decoded.occupancy, np.rint(source * 3) / 3)
    assert decoded.quality is None


def test_g2_and_g3_round_trip_and_application_bits_are_ordered() -> None:
    occupancy = np.linspace(0.0, 1.0, 8)
    g1 = encode_semantic_message(SemanticMessage("G1", 1, "scene", occupancy, probability_bits=2))
    g2 = encode_semantic_message(SemanticMessage("G2", 1, "scene", occupancy, probability_bits=4, quality=quality()))
    evidence = (SemanticEvidence(1, 3, 4, 0.91, 0.12), SemanticEvidence(5, 8, 2, 0.73, 0.31))
    g3 = encode_semantic_message(
        SemanticMessage("G3", 1, "scene", occupancy, probability_bits=4, quality=quality(), evidence=evidence)
    )
    assert g1.application_bits < g2.application_bits < g3.application_bits
    decoded = decode_semantic_message(g3.bits)
    assert decoded.granularity == "G3"
    assert len(decoded.evidence) == 2
    assert decoded.evidence[0].start_channel == 1
    assert decoded.evidence[1].stop_channel == 8
    assert decoded.quality is not None
    assert abs(decoded.quality.sensing_snr_db - 6.0) <= 0.5


def test_stage2_link_audit_counts_overhead_and_decodes_after_full_delivery() -> None:
    message = SemanticMessage("G2", 3, "scene-link", np.linspace(0.1, 0.9, 16), probability_bits=4, quality=quality())
    config = DigitalLinkConfig(
        packet=PacketConfig(payload_bits=128, header_bits=96, crc_bits=16, alignment_bits=8),
        ebn0_db=30.0,
        max_retransmissions=0,
    )
    result = transmit_semantic_message(message, config, seed=2026)
    assert result.link.frame_success
    assert result.decoded is not None
    assert result.link.packet_count >= 2
    assert result.link.transmitted_bits > result.encoded.application_bits


def test_granularity_contract_rejects_hidden_metadata_or_invalid_precision() -> None:
    occupancy = np.asarray([0.1, 0.9])
    with pytest.raises(ValueError, match="G1 carries occupancy preview only"):
        SemanticMessage("G1", 1, "scene", occupancy, probability_bits=2, quality=quality())
    with pytest.raises(ValueError, match="G3 probability_bits"):
        SemanticMessage("G3", 1, "scene", occupancy, probability_bits=1, quality=quality())
    with pytest.raises(ValueError, match="G2 requires quality"):
        SemanticMessage("G2", 1, "scene", occupancy, probability_bits=2)
