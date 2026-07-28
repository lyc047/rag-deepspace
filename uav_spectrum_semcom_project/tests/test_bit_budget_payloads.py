from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.bit_budget import feature_token_packet_bits, hard_box_packet_bits, soft_box_packet_bits


def test_payload_scheme_bits_are_ordered():
    hard = hard_box_packet_bits(3)
    soft = soft_box_packet_bits(3, n_classes=8)
    token = feature_token_packet_bits(3, token_dim=16, bits_per_value=6)
    assert hard > 0
    assert soft > hard
    assert token > hard

