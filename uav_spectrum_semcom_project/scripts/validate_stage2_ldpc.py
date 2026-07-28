from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
for path in [PROJECT_DIR / "src", PROJECT_DIR / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from spectrum_semcom.bit_budget import hard_box_packet_bits, packet_header_bits
from spectrum_semcom.digital_link import (
    DigitalLinkConfig,
    PacketConfig,
    binomial_wilson_interval,
    estimate_waveform_packet_success,
)
from spectrum_semcom.ldpc_link import (
    LdpcCodeConfig,
    make_ldpc_matrices,
    simulate_ldpc_packet_success,
    transmit_payload_empirical_code,
)
from spectrum_semcom.raddet import iter_raddet_frames
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file, sha256_strings
from spectrum_semcom.research_scope import assert_valid_experiment_metadata
from evaluate_stage2_classical_baselines import evaluate_resource_estimates, quantize_unit_interval
from evaluate_stage2_task_link import recover_semantic_boxes, task_metrics, truth_boxes
from run_stage2_digital_link import mean_std_ci, write_csv


def load_full_cache(cache_path: Path, frames):
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    expected = sha256_strings(frame.stem for frame in frames)
    if data.get("frame_ids_sha256") != expected:
        raise ValueError("full-test feature cache frame hash does not match the frozen split")
    predictions = [[np.asarray(box, dtype=np.float32) for box in item] for item in data["predictions"]]
    energies = np.asarray(data["channel_energies"], dtype=np.float32)
    return predictions, energies


def empirical_transmission(application_bits: int, matrices, settings: dict[str, Any], success: float, seed: int):
    return transmit_payload_empirical_code(
        application_bits=application_bits,
        application_payload_bits_per_codeword=matrices.application_payload_bits,
        codeword_bits=matrices.code_length,
        packet_success_probability=success,
        symbol_rate_baud=float(settings["symbol_rate_baud"]),
        propagation_delay_s=float(settings["propagation_delay_s"]),
        per_attempt_processing_s=float(settings["per_attempt_processing_s"]),
        max_retransmissions=int(settings["max_retransmissions"]),
        latency_budget_s=float(settings["latency_budget_s"]),
        seed=seed,
    )


def run_ldpc_hard(
    predictions,
    truths,
    task: dict[str, Any],
    matrices,
    settings: dict[str, Any],
    packet_success: float,
    seed: int,
    iou_threshold: float,
) -> dict[str, Any]:
    received = []
    bits = []
    durations = []
    ratios = []
    exhausted = []
    for index, boxes in enumerate(predictions):
        result = empirical_transmission(hard_box_packet_bits(len(boxes)), matrices, settings, packet_success, seed + index)
        delivered_packets = {trace.packet_index for trace in result.packet_traces if trace.delivered}
        received.append(
            recover_semantic_boxes(boxes, delivered_packets, matrices.application_payload_bits, 8 + 4 * 16)
        )
        bits.append(result.transmitted_bits)
        durations.append(result.duration_s)
        ratios.append(result.payload_delivery_ratio)
        exhausted.append(float(result.latency_budget_exhausted))
    metrics = task_metrics(received, truths, task, iou_threshold)
    pressure = int(task["observations_per_pressure_decision"])
    usable = len(predictions) - len(predictions) % pressure
    decisions = usable // pressure
    decision_latency = [max(durations[start : start + pressure]) for start in range(0, usable, pressure)]
    return {
        "f1": metrics["f1"],
        **{key: metrics[key] for key in ["clean_resource_rate", "mean_occupancy_regret", "mean_selected_occupancy", "oracle_equivalent_rate"]},
        "mean_transmitted_bits": float(np.sum(bits[:usable]) / decisions),
        "mean_link_latency_s": float(np.mean(decision_latency)),
        "mean_payload_delivery_ratio": float(np.mean(ratios[:usable])),
        "latency_budget_exhaustion_rate": float(np.mean(exhausted[:usable])),
    }


def run_ldpc_psd(
    energies: np.ndarray,
    truths,
    task: dict[str, Any],
    selected_bits: int,
    matrices,
    settings: dict[str, Any],
    packet_success: float,
    seed: int,
) -> dict[str, Any]:
    pressure = int(task["observations_per_pressure_decision"])
    usable = len(energies) - len(energies) % pressure
    application_bits = packet_header_bits() + int(task["n_channels"]) * selected_bits
    quantized = [quantize_unit_interval(row, selected_bits) for row in energies[:usable]]
    received = []
    bits = []
    durations = []
    ratios = []
    exhausted = []
    for index, value in enumerate(quantized):
        result = empirical_transmission(application_bits, matrices, settings, packet_success, seed + index)
        received.append(value if result.frame_success else None)
        bits.append(result.transmitted_bits)
        durations.append(result.duration_s)
        ratios.append(result.payload_delivery_ratio)
        exhausted.append(float(result.latency_budget_exhausted))
    estimates = []
    decision_latency = []
    for start in range(0, usable, pressure):
        available = [value for value in received[start : start + pressure] if value is not None]
        estimates.append(np.mean(available, axis=0) if available else np.zeros(int(task["n_channels"]), dtype=np.float32))
        decision_latency.append(max(durations[start : start + pressure]))
    metrics = evaluate_resource_estimates(np.stack(estimates), truths[:usable], task)
    decisions = usable // pressure
    return {
        "f1": None,
        **metrics,
        "mean_transmitted_bits": float(np.sum(bits) / decisions),
        "mean_link_latency_s": float(np.mean(decision_latency)),
        "mean_payload_delivery_ratio": float(np.mean(ratios)),
        "latency_budget_exhaustion_rate": float(np.mean(exhausted)),
    }


def aggregate_task(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "clean_resource_rate", "mean_occupancy_regret", "mean_transmitted_bits",
        "mean_link_latency_s", "mean_payload_delivery_ratio",
    ]
    output = []
    for ebn0_db in sorted({float(row["ebn0_db"]) for row in rows}):
        for scheme in sorted({str(row["scheme"]) for row in rows}):
            items = [row for row in rows if row["ebn0_db"] == ebn0_db and row["scheme"] == scheme]
            merged: dict[str, Any] = {"ebn0_db": ebn0_db, "scheme": scheme, "repeat_count": len(items)}
            f1_values = [float(item["f1"]) for item in items if item["f1"] is not None]
            merged["f1_mean"] = float(np.mean(f1_values)) if f1_values else None
            for metric in metrics:
                for stat, value in mean_std_ci([float(item[metric]) for item in items]).items():
                    merged[f"{metric}_{stat}"] = value
            output.append(merged)
    return output


def make_report(settings, matrices, waveform_rows, task_summary, frame_count, selected_bits) -> str:
    lines = [
        "# Stage 2 Actual LDPC Waveform and Full-Task Validation",
        "",
        "## Code and energy protocol",
        "",
        f"A deterministic regular LDPC code is generated with pyldpc {settings['library_version']}: n={matrices.code_length}, k={matrices.information_bits}, rate={matrices.code_rate:.4f}, dv={settings['variable_node_degree']}, dc={settings['check_node_degree']}. Each information block contains a {settings['application_header_bits']}-bit application header and CRC-16, leaving {matrices.application_payload_bits} application bits per codeword.",
        "",
        "Both LDPC and Hamming waveform baselines use unit-energy BPSK over AWGN. Eb/N0 is defined per transmitted coded bit; the lower-rate code spends more transmitted bits/energy rather than receiving an artificial SNR shift. Packet success requires CRC validity and exact application/header recovery.",
        "",
        "## Actual waveform packet success",
        "",
        "| Eb/N0 | LDPC success (95% CI) | Hamming success (95% CI) |",
        "|---:|---:|---:|",
    ]
    for row in waveform_rows:
        lines.append(
            f"| {row['ebn0_db']:.1f} | {row['ldpc_success_rate']:.4f} [{row['ldpc_ci95_low']:.4f}, {row['ldpc_ci95_high']:.4f}] | "
            f"{row['hamming_success_rate']:.4f} [{row['hamming_ci95_low']:.4f}, {row['hamming_ci95_high']:.4f}] |"
        )
    lines.extend(
        [
            "",
            f"## Full cached task validation ({frame_count} frames, frozen PSD={selected_bits} bit/value)",
            "",
            "The measured LDPC packet-success probability at each Eb/N0 is used for scalable packet/ARQ Monte Carlo on the frozen full-test features. Both hard semantic and PSD use the same LDPC codeword, CRC, retransmission limit, symbol rate, and deadline.",
            "",
            "| Eb/N0 | Scheme | F1 | Clean rate (95% CI) | Regret (95% CI) | bit/decision | Parallel latency |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in task_summary:
        f1 = "n/a" if row["f1_mean"] is None else f"{row['f1_mean']:.4f}"
        lines.append(
            f"| {row['ebn0_db']:.1f} | {row['scheme']} | {f1} | "
            f"{row['clean_resource_rate_mean']:.4f} [{max(0.0,row['clean_resource_rate_ci95_low']):.4f}, {min(1.0,row['clean_resource_rate_ci95_high']):.4f}] | "
            f"{row['mean_occupancy_regret_mean']:.6f} [{max(0.0,row['mean_occupancy_regret_ci95_low']):.6f}, {row['mean_occupancy_regret_ci95_high']:.6f}] | "
            f"{row['mean_transmitted_bits_mean']:.1f} | {1000*row['mean_link_latency_s_mean']:.2f} ms |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "- This is an actual regular LDPC/BP implementation, but it is not a 3GPP NR base graph, rate-matching, or HARQ implementation.",
            "- The full task run reuses frozen detector/PSD features; waveform decoding is measured separately and transferred only as an empirical packet-success curve.",
            "- AWGN assumes perfect synchronization and channel knowledge. Burst errors, fading, finite-length interleaving, and decoder complexity/energy remain future work.",
            "- The RadDet and unrelated-pressure limitations from the full-test report remain unchanged.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an actual LDPC waveform and apply its measured BLER to the full task.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "stage2_digital_link.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage2" / "ldpc_validation")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--mask-result", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask_result.json")
    parser.add_argument("--cache", type=Path, default=PROJECT_DIR / "results" / "stage2" / "full_test" / "frozen_feature_cache.json")
    parser.add_argument(
        "--refresh-intervals-only",
        action="store_true",
        help="Recompute Wilson intervals and reports from existing deterministic trial counts without rerunning BP decoding.",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    assert_valid_experiment_metadata(config)
    settings = config["ldpc_validation"]
    code_config = LdpcCodeConfig(
        code_length=int(settings["code_length"]),
        variable_node_degree=int(settings["variable_node_degree"]),
        check_node_degree=int(settings["check_node_degree"]),
        systematic=bool(settings["systematic"]),
        matrix_seed=int(settings["matrix_seed"]),
        max_decoder_iterations=int(settings["max_decoder_iterations"]),
        application_header_bits=int(settings["application_header_bits"]),
        crc_bits=int(settings["crc_bits"]),
    )
    matrices = make_ldpc_matrices(code_config)
    if args.refresh_intervals_only:
        result_path = args.out_dir / "stage2_ldpc_validation_result.json"
        if not result_path.exists():
            raise FileNotFoundError("existing LDPC result is required for interval refresh")
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        waveform_rows = existing["waveform"]
        trials = int(settings["waveform_trials"])
        for row in waveform_rows:
            for prefix in ["ldpc", "hamming"]:
                successes = int(round(float(row[f"{prefix}_success_rate"]) * trials))
                low, high = binomial_wilson_interval(successes, trials)
                row[f"{prefix}_ci95_low"] = low
                row[f"{prefix}_ci95_high"] = high
        existing["waveform"] = waveform_rows
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.out_dir / "ldpc_waveform.csv", waveform_rows)
        selected_bits = int(
            json.loads(
                (PROJECT_DIR / config["full_test_validation"]["psd_selection_result"]).read_text(encoding="utf-8")
            )["selected_bits"]
        )
        (args.out_dir / "stage2_ldpc_validation_report.md").write_text(
            make_report(
                settings,
                matrices,
                waveform_rows,
                existing["task_summary"],
                int(config["full_test_validation"]["expected_frame_count"]),
                selected_bits,
            ),
            encoding="utf-8",
        )
        result_path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print("refreshed Wilson intervals without rerunning waveform decoding", flush=True)
        return
    waveform_rows = []
    measured_success = {}
    for index, ebn0_db in enumerate(settings["ebn0_db_values"]):
        print(f"waveform Eb/N0={ebn0_db} dB", flush=True)
        ldpc = simulate_ldpc_packet_success(
            matrices, code_config, float(ebn0_db), int(settings["waveform_trials"]),
            int(settings["batch_size"]), int(config["seed"]) + index * 10_000
        )
        hamming_link = DigitalLinkConfig(
            packet=PacketConfig(
                payload_bits=matrices.application_payload_bits,
                header_bits=int(settings["application_header_bits"]),
                crc_bits=int(settings["crc_bits"]),
                alignment_bits=1,
            ),
            modulation="bpsk", fec="hamming74", channel="awgn", ebn0_db=float(ebn0_db),
            max_retransmissions=0, symbol_rate_baud=float(settings["symbol_rate_baud"]),
            propagation_delay_s=0.0, per_attempt_processing_s=0.0, latency_budget_s=None,
        )
        hamming = estimate_waveform_packet_success(
            matrices.application_payload_bits, hamming_link, int(settings["waveform_trials"]),
            int(config["seed"]) + 500_000 + index
        )
        measured_success[float(ebn0_db)] = float(ldpc["success_rate"])
        waveform_rows.append(
            {
                "ebn0_db": float(ebn0_db),
                "ldpc_success_rate": ldpc["success_rate"],
                "ldpc_bler": ldpc["bler"],
                "ldpc_ci95_low": ldpc["ci95_low"],
                "ldpc_ci95_high": ldpc["ci95_high"],
                "hamming_success_rate": hamming["success_rate"],
                "hamming_ci95_low": hamming["ci95_low"],
                "hamming_ci95_high": hamming["ci95_high"],
            }
        )

    full_settings = config["full_test_validation"]
    frames = iter_raddet_frames(
        args.root, full_settings["split"], read_metadata=False, default_sequence_length=1_000_000,
        max_frames=None
    )
    truths = [truth_boxes(frame) for frame in frames]
    predictions, energies = load_full_cache(args.cache, frames)
    model_config = json.loads(args.mask_result.read_text(encoding="utf-8"))
    task = {
        **config["task_validation"],
        "observations_per_pressure_decision": full_settings["observations_per_pressure_decision"],
    }
    selected_bits = int(
        json.loads((PROJECT_DIR / full_settings["psd_selection_result"]).read_text(encoding="utf-8"))["selected_bits"]
    )
    task_rows = []
    for ebn0_index, ebn0_db in enumerate(settings["task_ebn0_db_values"]):
        success = measured_success[float(ebn0_db)]
        for scheme_index, scheme in enumerate(["hard_semantic_ldpc", "frozen_psd_ldpc"]):
            for repeat_index in range(int(settings["task_repeat_count"])):
                seed = int(config["seed"]) + 30_000_000 + ebn0_index * 100_000 + scheme_index * 10_000 + repeat_index * 1_000
                print(f"task {scheme}, Eb/N0={ebn0_db}, repeat={repeat_index + 1}/{settings['task_repeat_count']}", flush=True)
                if scheme == "hard_semantic_ldpc":
                    row = run_ldpc_hard(
                        predictions, truths, task, matrices, settings, success, seed,
                        float(model_config["iou_threshold"])
                    )
                else:
                    row = run_ldpc_psd(
                        energies, truths, task, selected_bits, matrices, settings, success, seed
                    )
                row.update({"scheme": scheme, "ebn0_db": float(ebn0_db), "seed": seed})
                task_rows.append(row)
    task_summary = aggregate_task(task_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "ldpc_waveform.csv", waveform_rows)
    write_csv(args.out_dir / "ldpc_task_trials.csv", task_rows)
    write_csv(args.out_dir / "ldpc_task_summary.csv", task_summary)
    report_path = args.out_dir / "stage2_ldpc_validation_report.md"
    report_path.write_text(
        make_report(settings, matrices, waveform_rows, task_summary, len(frames), selected_bits), encoding="utf-8"
    )
    result = {
        "config_id": config["config_id"],
        "config_sha256": sha256_file(args.config),
        "pyldpc_version": settings["library_version"],
        "code": {
            "n": matrices.code_length,
            "k": matrices.information_bits,
            "rate": matrices.code_rate,
            "application_payload_bits": matrices.application_payload_bits,
        },
        "waveform": waveform_rows,
        "task_summary": task_summary,
        "data_frame_ids_sha256": sha256_strings(frame.stem for frame in frames),
        "feature_cache_sha256": sha256_file(args.cache),
        "environment": environment_snapshot(["numpy", "scipy", "numba", "llvmlite", "pyldpc"]),
        "claim_boundary": {
            "actual_ldpc_waveform": True,
            "three_gpp_nr_ldpc": False,
            "empirical_bler_transfer_to_task": True,
            "fading_or_burst_errors": False,
        },
    }
    (args.out_dir / "stage2_ldpc_validation_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
