from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, erfc, sqrt
from typing import Literal

import numpy as np


Modulation = Literal["bpsk", "qpsk", "16qam"]
FecScheme = Literal["none", "repetition3", "hamming74"]
ChannelModel = Literal["awgn", "rayleigh", "rician", "burst"]


@dataclass(frozen=True)
class PacketConfig:
    """Packetization overhead shared by every source representation.

    ``payload_bits`` is the maximum application payload in one packet. Header
    and CRC bits are protected by the same FEC as the application payload.
    """

    payload_bits: int = 1024
    header_bits: int = 96
    crc_bits: int = 16
    alignment_bits: int = 8

    def __post_init__(self) -> None:
        if self.payload_bits <= 0:
            raise ValueError("payload_bits must be positive")
        if self.header_bits < 0 or self.crc_bits < 0:
            raise ValueError("header_bits and crc_bits must be non-negative")
        if self.crc_bits not in {0, 16}:
            raise ValueError("waveform validation currently supports crc_bits 0 or 16")
        if self.alignment_bits <= 0:
            raise ValueError("alignment_bits must be positive")


@dataclass(frozen=True)
class DigitalLinkConfig:
    packet: PacketConfig = PacketConfig()
    modulation: Modulation = "qpsk"
    fec: FecScheme = "hamming74"
    channel: ChannelModel = "awgn"
    ebn0_db: float = 6.0
    rician_k_db: float = 6.0
    max_retransmissions: int = 1
    symbol_rate_baud: float = 1_000_000.0
    propagation_delay_s: float = 0.001
    per_attempt_processing_s: float = 0.00005
    latency_budget_s: float | None = None
    # Gilbert--Elliott parameters, active only when channel == "burst".
    burst_good_to_bad: float = 0.02
    burst_bad_to_good: float = 0.25
    burst_bad_ber: float = 0.08

    def __post_init__(self) -> None:
        if self.modulation not in {"bpsk", "qpsk", "16qam"}:
            raise ValueError(f"unsupported modulation: {self.modulation}")
        if self.fec not in {"none", "repetition3", "hamming74"}:
            raise ValueError(f"unsupported FEC: {self.fec}")
        if self.channel not in {"awgn", "rayleigh", "rician", "burst"}:
            raise ValueError(f"unsupported channel: {self.channel}")
        if self.max_retransmissions < 0:
            raise ValueError("max_retransmissions must be non-negative")
        if self.symbol_rate_baud <= 0:
            raise ValueError("symbol_rate_baud must be positive")
        if self.propagation_delay_s < 0 or self.per_attempt_processing_s < 0:
            raise ValueError("delays must be non-negative")
        if self.latency_budget_s is not None and self.latency_budget_s <= 0:
            raise ValueError("latency_budget_s must be positive when set")
        if not 0.0 <= self.burst_good_to_bad <= 1.0 or not 0.0 <= self.burst_bad_to_good <= 1.0:
            raise ValueError("burst transition probabilities must be in [0, 1]")
        if not 0.0 <= self.burst_bad_ber <= 0.5:
            raise ValueError("burst_bad_ber must be in [0, 0.5]")


@dataclass(frozen=True)
class PacketPlan:
    application_bits: int
    information_bits: int
    aligned_information_bits: int
    coded_bits: int
    symbols: int
    padding_bits: int
    duration_per_attempt_s: float


@dataclass(frozen=True)
class PacketTrace:
    packet_index: int
    application_bits: int
    delivered: bool
    attempts: int
    crc_failures: int
    transmitted_bits: int
    duration_s: float
    success_probability_last_attempt: float
    stop_reason: str


@dataclass(frozen=True)
class TransmissionResult:
    application_bits: int
    delivered_application_bits: int
    transmitted_bits: int
    packet_count: int
    delivered_packets: int
    attempts: int
    retransmissions: int
    crc_failures: int
    duration_s: float
    latency_budget_exhausted: bool
    packet_traces: tuple[PacketTrace, ...]

    @property
    def payload_delivery_ratio(self) -> float:
        if self.application_bits == 0:
            return 1.0
        return self.delivered_application_bits / self.application_bits

    @property
    def packet_delivery_ratio(self) -> float:
        if self.packet_count == 0:
            return 1.0
        return self.delivered_packets / self.packet_count

    @property
    def frame_success(self) -> bool:
        return self.delivered_application_bits == self.application_bits

    @property
    def goodput_bps(self) -> float:
        return self.delivered_application_bits / max(self.duration_s, 1e-12)

    @property
    def gross_bitrate_bps(self) -> float:
        return self.transmitted_bits / max(self.duration_s, 1e-12)

    def summary(self) -> dict[str, float | int | bool]:
        result = asdict(self)
        result.pop("packet_traces")
        result.update(
            {
                "payload_delivery_ratio": self.payload_delivery_ratio,
                "packet_delivery_ratio": self.packet_delivery_ratio,
                "frame_success": self.frame_success,
                "goodput_bps": self.goodput_bps,
                "gross_bitrate_bps": self.gross_bitrate_bps,
            }
        )
        return result


def modulation_bits_per_symbol(modulation: Modulation) -> int:
    return {"bpsk": 1, "qpsk": 2, "16qam": 4}[modulation]


def binomial_wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, including meaningful bounds at 0/n and n/n."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = z * sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def fec_coded_length(information_bits: int, fec: FecScheme) -> tuple[int, int]:
    """Return coded length and FEC padding for an information block."""
    if information_bits < 0:
        raise ValueError("information_bits must be non-negative")
    if fec == "none":
        return information_bits, 0
    if fec == "repetition3":
        return information_bits * 3, 0
    groups = int(ceil(information_bits / 4))
    return groups * 7, groups * 4 - information_bits


def packet_plan(application_bits: int, config: DigitalLinkConfig) -> PacketPlan:
    if application_bits < 0 or application_bits > config.packet.payload_bits:
        raise ValueError("application_bits must be within one packet payload")
    information_bits = application_bits + config.packet.header_bits + config.packet.crc_bits
    aligned = int(ceil(information_bits / config.packet.alignment_bits) * config.packet.alignment_bits)
    coded_bits, fec_padding = fec_coded_length(aligned, config.fec)
    bits_per_symbol = modulation_bits_per_symbol(config.modulation)
    symbols = int(ceil(coded_bits / bits_per_symbol))
    modulation_padding = symbols * bits_per_symbol - coded_bits
    duration = (
        symbols / config.symbol_rate_baud
        + config.propagation_delay_s
        + config.per_attempt_processing_s
    )
    return PacketPlan(
        application_bits=application_bits,
        information_bits=information_bits,
        aligned_information_bits=aligned,
        coded_bits=coded_bits,
        symbols=symbols,
        padding_bits=(aligned - information_bits) + fec_padding + modulation_padding,
        duration_per_attempt_s=duration,
    )


def packetize(application_bits: int, config: DigitalLinkConfig) -> list[PacketPlan]:
    if application_bits < 0:
        raise ValueError("application_bits must be non-negative")
    if application_bits == 0:
        return []
    full, remainder = divmod(application_bits, config.packet.payload_bits)
    sizes = [config.packet.payload_bits] * full
    if remainder:
        sizes.append(remainder)
    return [packet_plan(size, config) for size in sizes]


def nominal_transmitted_bits(application_bits: int, config: DigitalLinkConfig) -> int:
    """Channel bits for one attempt of every packet, including all overhead."""
    return int(sum(plan.symbols * modulation_bits_per_symbol(config.modulation) for plan in packetize(int(application_bits), config)))


def expected_transmitted_bits_awgn(application_bits: int, config: DigitalLinkConfig) -> float:
    """Exact expected channel bits under independent AWGN, ARQ, and latency."""
    if config.channel != "awgn":
        raise ValueError("exact expected bits currently requires AWGN")
    rng = np.random.default_rng(0)
    active: dict[float, float] = {0.0: 1.0}
    total = 0.0
    attempts_limit = config.max_retransmissions + 1
    for plan in packetize(int(application_bits), config):
        success = analytic_packet_success_probability(plan, config, rng)
        failure = 1.0 - success
        channel_bits = plan.symbols * modulation_bits_per_symbol(config.modulation)
        next_active: dict[float, float] = {}
        for elapsed, state_probability in active.items():
            allowed = attempts_limit
            if config.latency_budget_s is not None:
                remaining = config.latency_budget_s - elapsed
                allowed = min(allowed, max(0, int(np.floor((remaining + 1e-15) / plan.duration_per_attempt_s))))
            if allowed == 0:
                continue
            total += state_probability * channel_bits * sum(failure**attempt for attempt in range(allowed))
            for attempt in range(1, allowed + 1):
                probability = state_probability * failure ** (attempt - 1) * success
                new_elapsed = round(elapsed + attempt * plan.duration_per_attempt_s, 15)
                next_active[new_elapsed] = next_active.get(new_elapsed, 0.0) + probability
            if allowed == attempts_limit:
                probability = state_probability * failure**allowed
                new_elapsed = round(elapsed + allowed * plan.duration_per_attempt_s, 15)
                next_active[new_elapsed] = next_active.get(new_elapsed, 0.0) + probability
        active = next_active
        if not active:
            break
    return float(total)


def _awgn_ber(ebn0_linear: float, modulation: Modulation) -> float:
    if modulation in {"bpsk", "qpsk"}:
        return 0.5 * erfc(sqrt(max(ebn0_linear, 0.0)))
    # Standard Gray-coded square 16-QAM approximation.
    return 0.375 * erfc(sqrt(max(0.4 * ebn0_linear, 0.0)))


def _sample_attempt_ber(config: DigitalLinkConfig, rng: np.random.Generator) -> float:
    gamma = 10.0 ** (config.ebn0_db / 10.0)
    if config.channel == "awgn":
        instantaneous = gamma
    elif config.channel == "rayleigh":
        instantaneous = gamma * float(rng.exponential())
    else:
        k = 10.0 ** (config.rician_k_db / 10.0)
        los = sqrt(k / (k + 1.0))
        scatter = sqrt(1.0 / (2.0 * (k + 1.0))) * (rng.standard_normal() + 1j * rng.standard_normal())
        instantaneous = gamma * float(abs(los + scatter) ** 2)
    return float(np.clip(_awgn_ber(instantaneous, config.modulation), 0.0, 0.5))


def fec_packet_success_probability(raw_ber: float, information_bits: int, fec: FecScheme) -> float:
    """Ideal-CRC probability that an independently corrupted packet decodes.

    Hamming(7,4) is considered successful when every codeword contains at most
    one channel error. Repetition-3 uses majority decoding per bit. This model
    is deliberately auditable and is validated against the waveform simulator.
    """
    p = float(np.clip(raw_ber, 0.0, 0.5))
    if information_bits <= 0:
        return 1.0
    if fec == "none":
        return float((1.0 - p) ** information_bits)
    if fec == "repetition3":
        residual = 3.0 * p * p * (1.0 - p) + p**3
        return float((1.0 - residual) ** information_bits)
    codewords = int(ceil(information_bits / 4))
    codeword_success = (1.0 - p) ** 7 + 7.0 * p * (1.0 - p) ** 6
    return float(codeword_success**codewords)


def analytic_packet_success_probability(
    plan: PacketPlan,
    config: DigitalLinkConfig,
    rng: np.random.Generator,
) -> float:
    raw_ber = _sample_attempt_ber(config, rng)
    return fec_packet_success_probability(raw_ber, plan.aligned_information_bits, config.fec)


def transmit_payload_analytic(
    application_bits: int,
    config: DigitalLinkConfig,
    seed: int,
) -> TransmissionResult:
    """Packet-level Monte Carlo simulation suitable for very large payloads."""
    rng = np.random.default_rng(seed)
    plans = packetize(application_bits, config)
    traces: list[PacketTrace] = []
    total_time = 0.0
    total_bits = 0
    total_attempts = 0
    total_crc_failures = 0
    delivered_bits = 0
    delivered_packets = 0
    budget_exhausted = False
    burst_bad = False

    for index, plan in enumerate(plans):
        delivered = False
        attempts = 0
        crc_failures = 0
        tx_bits = 0
        duration = 0.0
        last_probability = 0.0
        stop_reason = "retry_limit"
        for _ in range(config.max_retransmissions + 1):
            if (
                config.latency_budget_s is not None
                and total_time + plan.duration_per_attempt_s > config.latency_budget_s + 1e-15
            ):
                budget_exhausted = True
                stop_reason = "latency_budget"
                break
            attempts += 1
            total_attempts += 1
            channel_bits = plan.symbols * modulation_bits_per_symbol(config.modulation)
            tx_bits += channel_bits
            total_bits += channel_bits
            duration += plan.duration_per_attempt_s
            total_time += plan.duration_per_attempt_s
            if config.channel == "burst":
                # The state evolves per packet attempt, so adjacent fragments
                # and retransmissions can share an outage period.
                if burst_bad:
                    burst_bad = not bool(rng.random() < config.burst_bad_to_good)
                else:
                    burst_bad = bool(rng.random() < config.burst_good_to_bad)
                raw_ber = config.burst_bad_ber if burst_bad else _awgn_ber(
                    10.0 ** (config.ebn0_db / 10.0), config.modulation
                )
                last_probability = fec_packet_success_probability(
                    raw_ber, plan.aligned_information_bits, config.fec
                )
            else:
                last_probability = analytic_packet_success_probability(plan, config, rng)
            if rng.random() < last_probability:
                delivered = True
                delivered_bits += plan.application_bits
                delivered_packets += 1
                stop_reason = "delivered"
                break
            crc_failures += 1
            total_crc_failures += 1
        traces.append(
            PacketTrace(
                packet_index=index,
                application_bits=plan.application_bits,
                delivered=delivered,
                attempts=attempts,
                crc_failures=crc_failures,
                transmitted_bits=tx_bits,
                duration_s=duration,
                success_probability_last_attempt=last_probability,
                stop_reason=stop_reason,
            )
        )
        if budget_exhausted:
            for remaining_index, remaining_plan in enumerate(plans[index + 1 :], start=index + 1):
                traces.append(
                    PacketTrace(
                        packet_index=remaining_index,
                        application_bits=remaining_plan.application_bits,
                        delivered=False,
                        attempts=0,
                        crc_failures=0,
                        transmitted_bits=0,
                        duration_s=0.0,
                        success_probability_last_attempt=0.0,
                        stop_reason="latency_budget",
                    )
                )
            break

    return TransmissionResult(
        application_bits=application_bits,
        delivered_application_bits=delivered_bits,
        transmitted_bits=total_bits,
        packet_count=len(plans),
        delivered_packets=delivered_packets,
        attempts=total_attempts,
        retransmissions=max(0, total_attempts - len([trace for trace in traces if trace.attempts > 0])),
        crc_failures=total_crc_failures,
        duration_s=total_time,
        latency_budget_exhausted=budget_exhausted,
        packet_traces=tuple(traces),
    )


def crc16_ccitt(bits: np.ndarray, initial: int = 0xFFFF) -> int:
    """Compute CRC-16/CCITT-FALSE over an arbitrary bit vector."""
    register = int(initial) & 0xFFFF
    for bit in np.asarray(bits, dtype=np.uint8).reshape(-1):
        feedback = ((register >> 15) & 1) ^ int(bit)
        register = (register << 1) & 0xFFFF
        if feedback:
            register ^= 0x1021
    return register


def int_to_bits(value: int, width: int) -> np.ndarray:
    return np.asarray([(value >> shift) & 1 for shift in range(width - 1, -1, -1)], dtype=np.uint8)


def bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in np.asarray(bits, dtype=np.uint8).reshape(-1):
        value = (value << 1) | int(bit)
    return value


def append_crc16(bits: np.ndarray) -> np.ndarray:
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    return np.concatenate([data, int_to_bits(crc16_ccitt(data), 16)])


def check_and_strip_crc16(bits: np.ndarray) -> tuple[bool, np.ndarray]:
    received = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if len(received) < 16:
        return False, np.empty(0, dtype=np.uint8)
    data, checksum = received[:-16], received[-16:]
    return crc16_ccitt(data) == bits_to_int(checksum), data


def fec_encode(bits: np.ndarray, fec: FecScheme) -> tuple[np.ndarray, int]:
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if fec == "none":
        return data.copy(), 0
    if fec == "repetition3":
        return np.repeat(data, 3), 0
    padding = (-len(data)) % 4
    padded = np.pad(data, (0, padding))
    d = padded.reshape(-1, 4)
    encoded = np.empty((len(d), 7), dtype=np.uint8)
    encoded[:, 2] = d[:, 0]
    encoded[:, 4] = d[:, 1]
    encoded[:, 5] = d[:, 2]
    encoded[:, 6] = d[:, 3]
    encoded[:, 0] = encoded[:, 2] ^ encoded[:, 4] ^ encoded[:, 6]
    encoded[:, 1] = encoded[:, 2] ^ encoded[:, 5] ^ encoded[:, 6]
    encoded[:, 3] = encoded[:, 4] ^ encoded[:, 5] ^ encoded[:, 6]
    return encoded.reshape(-1), padding


def fec_decode(bits: np.ndarray, fec: FecScheme, padding: int = 0) -> np.ndarray:
    received = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if fec == "none":
        decoded = received.copy()
    elif fec == "repetition3":
        if len(received) % 3:
            raise ValueError("repetition3 codeword length must be divisible by 3")
        decoded = (received.reshape(-1, 3).sum(axis=1) >= 2).astype(np.uint8)
    else:
        if len(received) % 7:
            raise ValueError("hamming74 codeword length must be divisible by 7")
        words = received.reshape(-1, 7).copy()
        s1 = words[:, 0] ^ words[:, 2] ^ words[:, 4] ^ words[:, 6]
        s2 = words[:, 1] ^ words[:, 2] ^ words[:, 5] ^ words[:, 6]
        s4 = words[:, 3] ^ words[:, 4] ^ words[:, 5] ^ words[:, 6]
        positions = s1 + 2 * s2 + 4 * s4
        rows = np.flatnonzero(positions)
        if len(rows):
            words[rows, positions[rows] - 1] ^= 1
        decoded = words[:, [2, 4, 5, 6]].reshape(-1)
    if padding:
        if padding > len(decoded):
            raise ValueError("padding exceeds decoded length")
        decoded = decoded[:-padding]
    return decoded


def modulate(bits: np.ndarray, modulation: Modulation) -> tuple[np.ndarray, int]:
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    k = modulation_bits_per_symbol(modulation)
    padding = (-len(data)) % k
    padded = np.pad(data, (0, padding))
    if modulation == "bpsk":
        return (1.0 - 2.0 * padded).astype(np.complex128), padding
    if modulation == "qpsk":
        pairs = padded.reshape(-1, 2)
        symbols = ((1.0 - 2.0 * pairs[:, 0]) + 1j * (1.0 - 2.0 * pairs[:, 1])) / sqrt(2.0)
        return symbols.astype(np.complex128), padding
    groups = padded.reshape(-1, 4)
    levels = {(0, 0): 3.0, (0, 1): 1.0, (1, 1): -1.0, (1, 0): -3.0}
    i_values = np.asarray([levels[tuple(row[:2])] for row in groups])
    q_values = np.asarray([levels[tuple(row[2:])] for row in groups])
    return ((i_values + 1j * q_values) / sqrt(10.0)).astype(np.complex128), padding


def _level_to_gray_bits(value: float) -> tuple[int, int]:
    if value >= 2.0:
        return 0, 0
    if value >= 0.0:
        return 0, 1
    if value >= -2.0:
        return 1, 1
    return 1, 0


def demodulate(symbols: np.ndarray, modulation: Modulation, padding: int = 0) -> np.ndarray:
    values = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    if modulation == "bpsk":
        decoded = (values.real < 0).astype(np.uint8)
    elif modulation == "qpsk":
        decoded = np.column_stack((values.real < 0, values.imag < 0)).astype(np.uint8).reshape(-1)
    else:
        scaled = values * sqrt(10.0)
        decoded = np.asarray(
            [bit for symbol in scaled for bit in (*_level_to_gray_bits(symbol.real), *_level_to_gray_bits(symbol.imag))],
            dtype=np.uint8,
        )
    return decoded[:-padding] if padding else decoded


def apply_waveform_channel(
    symbols: np.ndarray,
    config: DigitalLinkConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply block fading plus AWGN with perfect receiver-side equalization."""
    x = np.asarray(symbols, dtype=np.complex128)
    k_mod = modulation_bits_per_symbol(config.modulation)
    gamma_b = 10.0 ** (config.ebn0_db / 10.0)
    noise_variance = 1.0 / (k_mod * gamma_b)
    noise = sqrt(noise_variance / 2.0) * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    if config.channel == "awgn":
        h = 1.0 + 0.0j
    elif config.channel == "rayleigh":
        h = (rng.standard_normal() + 1j * rng.standard_normal()) / sqrt(2.0)
    else:
        k = 10.0 ** (config.rician_k_db / 10.0)
        h = sqrt(k / (k + 1.0)) + sqrt(1.0 / (2.0 * (k + 1.0))) * (
            rng.standard_normal() + 1j * rng.standard_normal()
        )
    y = h * x + noise
    return y / h if abs(h) > 1e-12 else y


def waveform_packet_attempt(
    application_bits: int,
    config: DigitalLinkConfig,
    rng: np.random.Generator,
) -> bool:
    """Transmit one generated packet through the actual bit/waveform chain."""
    plan = packet_plan(application_bits, config)
    app = rng.integers(0, 2, size=application_bits, dtype=np.uint8)
    header = rng.integers(0, 2, size=config.packet.header_bits, dtype=np.uint8)
    information = np.concatenate([header, app])
    if config.packet.crc_bits == 16:
        information = append_crc16(information)
    alignment_padding = plan.aligned_information_bits - len(information)
    aligned = np.pad(information, (0, alignment_padding))
    coded, fec_padding = fec_encode(aligned, config.fec)
    symbols, modulation_padding = modulate(coded, config.modulation)
    received_symbols = apply_waveform_channel(symbols, config, rng)
    received_coded = demodulate(received_symbols, config.modulation, modulation_padding)
    received_aligned = fec_decode(received_coded, config.fec, fec_padding)
    received_information = received_aligned[: len(information)]
    if config.packet.crc_bits == 16:
        valid, data = check_and_strip_crc16(received_information)
    else:
        valid, data = True, received_information
    return bool(valid and np.array_equal(data, np.concatenate([header, app])))


def estimate_waveform_packet_success(
    application_bits: int,
    config: DigitalLinkConfig,
    trials: int,
    seed: int,
) -> dict[str, float | int]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    rng = np.random.default_rng(seed)
    successes = sum(waveform_packet_attempt(application_bits, config, rng) for _ in range(trials))
    proportion = successes / trials
    standard_error = sqrt(max(proportion * (1.0 - proportion), 0.0) / trials)
    ci95_low, ci95_high = binomial_wilson_interval(successes, trials)
    return {
        "trials": trials,
        "successes": successes,
        "success_rate": proportion,
        "standard_error": standard_error,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
    }
