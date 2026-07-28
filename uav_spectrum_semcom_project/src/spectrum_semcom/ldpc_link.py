from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
import warnings

import numpy as np

from .digital_link import (
    PacketTrace,
    TransmissionResult,
    append_crc16,
    binomial_wilson_interval,
    check_and_strip_crc16,
)


@dataclass(frozen=True)
class LdpcCodeConfig:
    code_length: int = 1200
    variable_node_degree: int = 3
    check_node_degree: int = 6
    systematic: bool = True
    matrix_seed: int = 20260712
    max_decoder_iterations: int = 100
    application_header_bits: int = 96
    crc_bits: int = 16

    def __post_init__(self) -> None:
        if self.code_length <= 0:
            raise ValueError("code_length must be positive")
        if self.check_node_degree < self.variable_node_degree:
            raise ValueError("check_node_degree must be at least variable_node_degree")
        if self.code_length % self.check_node_degree:
            raise ValueError("check_node_degree must divide code_length")
        if self.crc_bits != 16:
            raise ValueError("current LDPC waveform validation requires CRC-16")


@dataclass(frozen=True)
class LdpcMatrices:
    parity_check: np.ndarray
    generator: object
    information_bits: int
    application_payload_bits: int

    @property
    def code_length(self) -> int:
        return int(self.parity_check.shape[1])

    @property
    def code_rate(self) -> float:
        return self.information_bits / self.code_length


def make_ldpc_matrices(config: LdpcCodeConfig) -> LdpcMatrices:
    try:
        from pyldpc import make_ldpc
    except ImportError as exc:  # pragma: no cover - exercised only in environments without the optional extra.
        raise RuntimeError('pyldpc is required; install the project with the "research" extra') from exc

    parity_check, generator = make_ldpc(
        config.code_length,
        config.variable_node_degree,
        config.check_node_degree,
        systematic=config.systematic,
        sparse=True,
        seed=config.matrix_seed,
    )
    information_bits = int(generator.shape[1])
    application_payload_bits = information_bits - config.application_header_bits - config.crc_bits
    if application_payload_bits <= 0:
        raise ValueError("LDPC information block is too small for application header and CRC")
    return LdpcMatrices(parity_check, generator, information_bits, application_payload_bits)


def simulate_ldpc_packet_success(
    matrices: LdpcMatrices,
    config: LdpcCodeConfig,
    ebn0_db: float,
    trials: int,
    batch_size: int,
    seed: int,
) -> dict[str, float | int]:
    """Run actual BPSK/AWGN belief-propagation decoding with CRC checking.

    ``pyldpc`` defines SNR as 10log10(1/noise_variance) for unit-energy
    transmitted BPSK symbols. We therefore interpret the configured value as
    Eb/N0 per transmitted coded bit, matching the existing waveform model.
    """
    try:
        from pyldpc import decode, encode, get_message
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('pyldpc is required; install the project with the "research" extra') from exc

    if trials <= 0 or batch_size <= 0:
        raise ValueError("trials and batch_size must be positive")
    rng = np.random.default_rng(seed)
    successes = 0
    completed = 0
    while completed < trials:
        current = min(batch_size, trials - completed)
        data_length = matrices.information_bits - config.crc_bits
        data = rng.integers(0, 2, size=(data_length, current), dtype=np.uint8)
        messages = np.empty((matrices.information_bits, current), dtype=np.uint8)
        for column in range(current):
            messages[:, column] = append_crc16(data[:, column])
        noise_seed = int(rng.integers(0, 2**31 - 1))
        received = encode(matrices.generator, messages, float(ebn0_db), seed=noise_seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decoded_codewords = decode(
                matrices.parity_check,
                received,
                float(ebn0_db),
                maxiter=config.max_decoder_iterations,
            )
        if current == 1:
            decoded_codewords = decoded_codewords[:, None]
        for column in range(current):
            decoded_message = np.asarray(
                get_message(matrices.generator, decoded_codewords[:, column]), dtype=np.uint8
            )
            valid, decoded_data = check_and_strip_crc16(decoded_message)
            successes += int(valid and np.array_equal(decoded_data, data[:, column]))
        completed += current
    rate = successes / trials
    standard_error = sqrt(max(rate * (1.0 - rate), 0.0) / trials)
    ci95_low, ci95_high = binomial_wilson_interval(successes, trials)
    return {
        "trials": trials,
        "successes": successes,
        "success_rate": rate,
        "bler": 1.0 - rate,
        "standard_error": standard_error,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
    }


def transmit_payload_empirical_code(
    application_bits: int,
    application_payload_bits_per_codeword: int,
    codeword_bits: int,
    packet_success_probability: float,
    symbol_rate_baud: float,
    propagation_delay_s: float,
    per_attempt_processing_s: float,
    max_retransmissions: int,
    latency_budget_s: float | None,
    seed: int,
) -> TransmissionResult:
    """Use a waveform-measured codeword success rate in scalable task runs."""
    if application_bits < 0:
        raise ValueError("application_bits must be non-negative")
    if application_payload_bits_per_codeword <= 0 or codeword_bits <= 0:
        raise ValueError("codeword sizes must be positive")
    probability = float(np.clip(packet_success_probability, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    packet_count = int(ceil(application_bits / application_payload_bits_per_codeword)) if application_bits else 0
    duration_per_attempt = codeword_bits / symbol_rate_baud + propagation_delay_s + per_attempt_processing_s
    traces: list[PacketTrace] = []
    delivered_bits = 0
    transmitted_bits = 0
    attempts = 0
    crc_failures = 0
    elapsed = 0.0
    exhausted = False
    remaining_bits = application_bits
    for index in range(packet_count):
        payload = min(application_payload_bits_per_codeword, remaining_bits)
        remaining_bits -= payload
        packet_attempts = 0
        packet_failures = 0
        packet_tx = 0
        packet_duration = 0.0
        delivered = False
        stop_reason = "retry_limit"
        for _ in range(max_retransmissions + 1):
            if latency_budget_s is not None and elapsed + duration_per_attempt > latency_budget_s + 1e-15:
                exhausted = True
                stop_reason = "latency_budget"
                break
            packet_attempts += 1
            attempts += 1
            packet_tx += codeword_bits
            transmitted_bits += codeword_bits
            packet_duration += duration_per_attempt
            elapsed += duration_per_attempt
            if rng.random() < probability:
                delivered = True
                delivered_bits += payload
                stop_reason = "delivered"
                break
            packet_failures += 1
            crc_failures += 1
        traces.append(
            PacketTrace(
                packet_index=index,
                application_bits=payload,
                delivered=delivered,
                attempts=packet_attempts,
                crc_failures=packet_failures,
                transmitted_bits=packet_tx,
                duration_s=packet_duration,
                success_probability_last_attempt=probability,
                stop_reason=stop_reason,
            )
        )
        if exhausted:
            for remaining_index in range(index + 1, packet_count):
                remaining_payload = min(application_payload_bits_per_codeword, remaining_bits)
                remaining_bits -= remaining_payload
                traces.append(
                    PacketTrace(
                        packet_index=remaining_index,
                        application_bits=remaining_payload,
                        delivered=False,
                        attempts=0,
                        crc_failures=0,
                        transmitted_bits=0,
                        duration_s=0.0,
                        success_probability_last_attempt=probability,
                        stop_reason="latency_budget",
                    )
                )
            break
    attempted_packets = sum(trace.attempts > 0 for trace in traces)
    return TransmissionResult(
        application_bits=application_bits,
        delivered_application_bits=delivered_bits,
        transmitted_bits=transmitted_bits,
        packet_count=packet_count,
        delivered_packets=sum(trace.delivered for trace in traces),
        attempts=attempts,
        retransmissions=max(0, attempts - attempted_packets),
        crc_failures=crc_failures,
        duration_s=elapsed,
        latency_budget_exhausted=exhausted,
        packet_traces=tuple(traces),
    )
