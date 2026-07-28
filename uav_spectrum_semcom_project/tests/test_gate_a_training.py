import torch

from spectrum_semcom.digital_link import DigitalLinkConfig, PacketConfig, packetize
from spectrum_semcom.gate_a_model import VariableRateOccupancyHead
from spectrum_semcom.gate_a_training import evaluate_gate_a_head, expected_transmitted_bits_awgn, nominal_transmitted_bits, precision_link_costs


def link() -> DigitalLinkConfig:
    return DigitalLinkConfig(packet=PacketConfig(1024, 96, 16, 8), fec="hamming74", modulation="qpsk")


def test_nominal_link_cost_counts_crc_header_fec_and_padding() -> None:
    assert nominal_transmitted_bits(159, link()) > 159
    model = VariableRateOccupancyHead(8, hidden_dim=8)
    costs = precision_link_costs(model, link())
    assert torch.all(costs[1:] > costs[:-1])
    assert expected_transmitted_bits_awgn(159, DigitalLinkConfig(ebn0_db=0, max_retransmissions=1)) >= nominal_transmitted_bits(159, DigitalLinkConfig(ebn0_db=0, max_retransmissions=1))


def test_gate_a_evaluation_uses_exact_inference_precision() -> None:
    model = VariableRateOccupancyHead(4, hidden_dim=8)
    with torch.no_grad():
        model.rate_logits.bias[:] = torch.tensor([0.0, 0.0, 4.0, 0.0])
    base = torch.tensor([[0.1, 0.7, 0.4, 0.2], [0.8, 0.2, 0.6, 0.3]])
    metrics = evaluate_gate_a_head(model, base, base, 1, link(), 512)
    assert metrics["mean_discrete_regret"] == 0.0
    assert metrics["precision_counts"]["4"] == 2
    assert metrics["mean_nominal_transmitted_bits"] > metrics["mean_application_bits"]


def test_expected_transmitted_bits_respects_latency_budget() -> None:
    base = DigitalLinkConfig(ebn0_db=0, max_retransmissions=1)
    duration = packetize(159, base)[0].duration_per_attempt_s
    one_attempt = DigitalLinkConfig(ebn0_db=0, max_retransmissions=1, latency_budget_s=duration)
    blocked = DigitalLinkConfig(ebn0_db=0, max_retransmissions=1, latency_budget_s=duration / 2)
    assert expected_transmitted_bits_awgn(159, one_attempt) == nominal_transmitted_bits(159, one_attempt)
    assert expected_transmitted_bits_awgn(159, blocked) == 0.0
