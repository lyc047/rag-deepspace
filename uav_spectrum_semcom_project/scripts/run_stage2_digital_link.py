from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectrum_semcom.digital_link import (
    DigitalLinkConfig,
    PacketConfig,
    analytic_packet_success_probability,
    estimate_waveform_packet_success,
    packet_plan,
    transmit_payload_analytic,
)
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file
from spectrum_semcom.research_scope import assert_valid_experiment_metadata


RAW_METRICS = [
    "payload_delivery_ratio",
    "packet_delivery_ratio",
    "frame_success",
    "transmitted_bits",
    "goodput_bps",
    "duration_s",
    "attempts",
    "retransmissions",
    "crc_failures",
    "latency_budget_exhausted",
]


def build_link_config(config: dict[str, Any], ebn0_db: float) -> DigitalLinkConfig:
    packet = PacketConfig(**config["packet"])
    physical = config["physical_link"]
    reporting = config["reporting_channel"]
    return DigitalLinkConfig(
        packet=packet,
        modulation=physical["modulation"],
        fec=reporting["fec"],
        channel=physical["channel"],
        ebn0_db=float(ebn0_db),
        rician_k_db=float(physical["rician_k_db"]),
        max_retransmissions=int(reporting["max_retransmissions"]),
        symbol_rate_baud=float(physical["symbol_rate_baud"]),
        propagation_delay_s=float(physical["propagation_delay_s"]),
        per_attempt_processing_s=float(physical["per_attempt_processing_s"]),
        latency_budget_s=float(physical["latency_budget_s"]),
    )


def run_packet_benchmark(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_seed = int(config["seed"])
    repeats = int(config["repeat_count"])
    ebn0_values = [float(value) for value in config["reporting_channel"]["ebn0_db_values"]]
    for source_index, source in enumerate(config["source_representations"]):
        application_bits = int(source["application_bits"])
        for ebn0_index, ebn0_db in enumerate(ebn0_values):
            link = build_link_config(config, ebn0_db)
            for repeat in range(repeats):
                seed = base_seed + source_index * 1_000_000 + ebn0_index * 10_000 + repeat
                result = transmit_payload_analytic(application_bits, link, seed)
                summary = result.summary()
                rows.append(
                    {
                        "source": source["name"],
                        "application_bits": application_bits,
                        "ebn0_db": ebn0_db,
                        "repeat": repeat,
                        "seed": seed,
                        **summary,
                    }
                )
    return rows


def mean_std_ci(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if len(array) <= 1:
        return {"mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean}
    std = float(np.std(array, ddof=1))
    half_width = 1.96 * std / np.sqrt(len(array))
    return {"mean": mean, "std": std, "ci95_low": mean - half_width, "ci95_high": mean + half_width}


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), float(row["ebn0_db"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for (source, ebn0_db), items in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        merged: dict[str, Any] = {
            "source": source,
            "application_bits": int(items[0]["application_bits"]),
            "ebn0_db": ebn0_db,
            "repeat_count": len(items),
        }
        for metric in RAW_METRICS:
            values = [float(item[metric]) for item in items]
            stats = mean_std_ci(values)
            for stat_name, value in stats.items():
                merged[f"{metric}_{stat_name}"] = value
        merged["mean_tx_over_application"] = merged["transmitted_bits_mean"] / max(merged["application_bits"], 1)
        summaries.append(merged)
    return summaries


def run_waveform_validation(config: dict[str, Any]) -> list[dict[str, Any]]:
    validation = config["waveform_validation"]
    application_bits = int(validation["application_bits"])
    trials = int(validation["trials"])
    rows: list[dict[str, Any]] = []
    for index, ebn0_db in enumerate(validation["ebn0_db_values"]):
        link = build_link_config(config, float(ebn0_db))
        # One attempt isolates the physical/FEC abstraction from ARQ and latency.
        link = DigitalLinkConfig(**{**asdict(link), "packet": link.packet, "max_retransmissions": 0, "latency_budget_s": None})
        plan = packet_plan(application_bits, link)
        predicted = analytic_packet_success_probability(plan, link, np.random.default_rng(config["seed"] + index))
        measured = estimate_waveform_packet_success(
            application_bits,
            link,
            trials=trials,
            seed=int(config["seed"]) + 500_000 + index,
        )
        rows.append(
            {
                "ebn0_db": float(ebn0_db),
                "application_bits": application_bits,
                "analytic_success_probability": predicted,
                **measured,
                "absolute_error": abs(float(measured["success_rate"]) - predicted),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_markdown(
    config: dict[str, Any],
    summary: list[dict[str, Any]],
    waveform: list[dict[str, Any]],
) -> str:
    selected_ebn0 = [float(value) for value in config["reporting_channel"]["ebn0_db_values"]]
    lines = [
        "# Stage 2 Fair Digital Reporting Link",
        "",
        "## 1. Scope and claim boundary",
        "",
        "This stage replaces the earlier asymmetric gate model with one packetization and digital-link rule shared by hard semantics, quantized soft semantics, compressed spectrograms, uint8 spectrograms, and the theoretical IQ capacity reference.",
        "",
        "The current report establishes link-level fairness and validates the scalable abstraction against a waveform chain. Payload delivery ratio is not treated as downstream sensing accuracy. Full H4 evidence requires the next resource-selection adapter to decode received fragments and evaluate occupancy regret/clean rate.",
        "",
        "## 2. Frozen fairness protocol",
        "",
        f"- Modulation: `{config['physical_link']['modulation']}`.",
        f"- Channel: `{config['physical_link']['channel']}`; Eb/N0 sweep: {selected_ebn0} dB.",
        f"- FEC: `{config['reporting_channel']['fec']}`; maximum retransmissions: {config['reporting_channel']['max_retransmissions']}.",
        f"- Packet application payload: {config['packet']['payload_bits']} bit; link header: {config['packet']['header_bits']} bit; CRC: {config['packet']['crc_bits']} bit.",
        f"- Symbol rate: {config['physical_link']['symbol_rate_baud']:.0f} baud; stop-and-wait latency budget: {config['physical_link']['latency_budget_s']:.3f} s/decision.",
        f"- Repeats: {config['repeat_count']}; uncertainty: sample standard deviation and normal-approximation 95% CI.",
        "",
        "All representations use the same packet, CRC, FEC, modulation, ARQ, channel realization policy, and latency budget. Total transmitted bits include link header, CRC, FEC padding, modulation padding, and retransmissions.",
        "",
        "## 3. Main link results",
        "",
        "| Eb/N0 | Representation | App bits | Delivery ratio (95% CI) | Frame success | Tx bits | Latency (ms) | Goodput (kbit/s) | Budget exhausted |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary, key=lambda value: (float(value["ebn0_db"]), int(value["application_bits"]))):
        lines.append(
            f"| {row['ebn0_db']:.1f} | {row['source']} | {row['application_bits']} | "
            f"{row['payload_delivery_ratio_mean']:.4f} "
            f"[{max(0.0, row['payload_delivery_ratio_ci95_low']):.4f}, {min(1.0, row['payload_delivery_ratio_ci95_high']):.4f}] | "
            f"{row['frame_success_mean']:.3f} | {row['transmitted_bits_mean']:.1f} | "
            f"{1000.0 * row['duration_s_mean']:.2f} | {row['goodput_bps_mean'] / 1000.0:.2f} | "
            f"{row['latency_budget_exhausted_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 4. Analytic abstraction versus waveform simulation",
            "",
            "The validation chain generates random packet bits, appends CRC-16/CCITT, applies the configured FEC, performs modulation, block fading/AWGN, hard demodulation, FEC decoding, and CRC checking. Eb/N0 is defined per transmitted coded bit; FEC therefore improves reliability by spending extra channel bits/energy rather than by receiving a free coding gain.",
            "",
            "| Eb/N0 | Analytic packet success | Waveform success (95% CI) | Absolute error | Trials |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in waveform:
        lines.append(
            f"| {row['ebn0_db']:.1f} | {row['analytic_success_probability']:.4f} | "
            f"{row['success_rate']:.4f} [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
            f"{row['absolute_error']:.4f} | {row['trials']} |"
        )
    lines.extend(
        [
            "",
            "## 5. Interpretation rules",
            "",
            "1. A small representation completing within the deadline demonstrates latency feasibility, not superior sensing accuracy.",
            "2. Partial PNG/spectrogram/IQ delivery is counted only as recovered link payload; it is not assumed to be directly usable by a detector until a fragment-aware decoder is evaluated.",
            "3. The IQ value is a theoretical 1,000,000-sample, 12+12-bit capacity reference because the local RadDet copy contains spectrograms and labels rather than raw IQ.",
            "4. Hamming(7,4) is an auditable research baseline, not a claim of 5G-grade coding. A later hardware/standard-aligned stage should add LDPC or Polar coding.",
            "5. AWGN results are the controlled first step. Rician/Rayleigh, burst errors, HARQ feedback delay, and task-level resource selection are required before the final thesis claim.",
            "",
            "## 6. Reproducibility",
            "",
            "- Configuration: `configs/stage2_digital_link.json`.",
            "- Raw repeated trials: `digital_link_trials.csv`.",
            "- Aggregated statistics: `digital_link_summary.csv`.",
            "- Waveform validation: `waveform_validation.csv`.",
            "- Machine-readable record: `stage2_digital_link_result.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stage-2 fair packetized digital-link benchmark.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "stage2_digital_link.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage2" / "digital_link")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    assert_valid_experiment_metadata(config)
    rows = run_packet_benchmark(config)
    summary = aggregate_rows(rows)
    waveform = run_waveform_validation(config)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.out_dir / "digital_link_trials.csv"
    summary_csv = args.out_dir / "digital_link_summary.csv"
    waveform_csv = args.out_dir / "waveform_validation.csv"
    report_path = args.out_dir / "stage2_digital_link_report.md"
    result_path = args.out_dir / "stage2_digital_link_result.json"
    write_csv(raw_csv, rows)
    write_csv(summary_csv, summary)
    write_csv(waveform_csv, waveform)
    report_path.write_text(report_markdown(config, summary, waveform), encoding="utf-8")

    record = {
        "config": config,
        "config_sha256": sha256_file(args.config),
        "environment": environment_snapshot(["numpy", "scipy", "matplotlib", "torch", "pytest"]),
        "summary": summary,
        "waveform_validation": waveform,
        "artifacts": {
            "trials_csv": raw_csv.relative_to(PROJECT_DIR).as_posix(),
            "summary_csv": summary_csv.relative_to(PROJECT_DIR).as_posix(),
            "waveform_csv": waveform_csv.relative_to(PROJECT_DIR).as_posix(),
            "report": report_path.relative_to(PROJECT_DIR).as_posix(),
        },
    }
    result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
