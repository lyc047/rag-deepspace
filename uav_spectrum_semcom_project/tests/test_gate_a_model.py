import torch

from spectrum_semcom.gate_a_model import VariableRateOccupancyHead, ste_quantize_probabilities


def test_ste_quantizer_is_discrete_forward_and_differentiable_backward() -> None:
    values = torch.tensor([0.2, 0.8], requires_grad=True)
    quantized = ste_quantize_probabilities(values, 2)
    assert torch.allclose(quantized, torch.tensor([1 / 3, 2 / 3]))
    quantized.sum().backward()
    assert torch.allclose(values.grad, torch.ones_like(values))


def test_shared_head_starts_as_identity_before_quantization() -> None:
    head = VariableRateOccupancyHead(4, hidden_dim=8)
    base = torch.tensor([[0.1, 0.3, 0.7, 0.9]])
    output = head(base)
    assert torch.allclose(output.refined_occupancy, base, atol=1e-6)
    assert torch.allclose(output.precision_probabilities, torch.full((1, 4), 0.25))
    assert output.expected_application_bits.item() == 151 + 4 * (1 + 2 + 4 + 8) / 4


def test_rate_logits_receive_gradient_through_rate_and_reconstruction() -> None:
    head = VariableRateOccupancyHead(4, hidden_dim=8)
    output = head(torch.tensor([[0.12, 0.36, 0.61, 0.88]]))
    loss = output.reconstructed_occupancy.square().mean() + output.expected_application_bits.mean() / 1000
    loss.backward()
    assert head.rate_logits.bias.grad is not None
    assert torch.sum(torch.abs(head.rate_logits.bias.grad)) > 0


def test_inference_uses_one_exact_precision_and_integer_application_cost() -> None:
    head = VariableRateOccupancyHead(8, hidden_dim=8)
    with torch.no_grad():
        head.rate_logits.bias[:] = torch.tensor([-2.0, -1.0, 3.0, 0.0])
    reconstructed, selected_bits, application_bits = head.infer(torch.full((2, 8), 0.6))
    assert selected_bits.tolist() == [4, 4]
    assert application_bits.tolist() == [183, 183]
    assert torch.allclose(reconstructed, torch.full((2, 8), 9 / 15))
