#!/usr/bin/env python
"""Run the frozen Stage-5 scale, query-set, header, and FEC boundary matrix."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage2_digital_link import build_link_config  # noqa: E402
from spectrum_semcom.digital_link import (  # noqa: E402
    nominal_transmitted_bits,
    transmit_payload_analytic,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402
from spectrum_semcom.stage5_query_bundle import (  # noqa: E402
    decode_query_bundle,
    encode_query_bundle,
    query_bundle_payload_width,
)
from spectrum_semcom.stage5_scale_boundary import (  # noqa: E402
    demand_set_from_fractions,
    representation_plans,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_protocol(protocol: dict) -> dict:
    stage2_path = PROJECT_DIR / protocol["link"]["stage2_config"]
    predecessor_protocol = PROJECT_DIR / protocol["predecessor"]["protocol"]
    predecessor_result = PROJECT_DIR / protocol["predecessor"]["result"]
    checks = (
        (stage2_path, protocol["link"]["stage2_config_sha256"]),
        (predecessor_protocol, protocol["predecessor"]["protocol_sha256"]),
        (predecessor_result, protocol["predecessor"]["result_sha256"]),
    )
    for path, expected in checks:
        if sha256_file(path) != expected:
            raise ValueError(f"frozen scale-boundary input hash mismatch: {path}")
    governance = protocol["governance"]
    if (
        governance["stage4_final_measurements_may_be_loaded"]
        or governance["stage4_final_metrics_may_be_loaded"]
        or governance["output_is_confirmatory_final"]
    ):
        raise ValueError("scale-boundary governance is invalid")
    return load_json(stage2_path)


def verify_bundle_codec(
    n_channels: int,
    demands: tuple[int, ...],
    header_bits: int,
) -> None:
    targets = {demand: n_channels - demand for demand in demands}
    encoded = encode_query_bundle(
        targets,
        n_channels=n_channels,
        demand_order=demands,
        node_id=17,
        epoch=23,
        application_header_bits=header_bits,
    )
    expected = header_bits + query_bundle_payload_width(n_channels, demands)
    if encoded.size != expected:
        raise AssertionError("encoded bundle length does not match declaration")
    decoded = decode_query_bundle(
        encoded,
        n_channels=n_channels,
        demand_order=demands,
        application_header_bits=header_bits,
    )
    if decoded.target_starts != targets:
        raise AssertionError("scale-boundary codec round trip failed")


def nominal_cost(plan, link) -> float:
    values = [
        nominal_transmitted_bits(bits, link)
        for bits in plan.application_packet_bits
    ]
    if plan.aggregation == "mean_over_query_alternatives":
        return float(np.mean(values))
    return float(np.sum(values))


def simulate_cost(plan, link, *, repeats: int, seed_base: int) -> dict:
    actual_bits = []
    success = []
    attempts = []
    for repeat in range(repeats):
        packet_bits = []
        packet_success = []
        packet_attempts = []
        for packet_index, application_bits in enumerate(
            plan.application_packet_bits
        ):
            result = transmit_payload_analytic(
                application_bits,
                link,
                seed_base + repeat * 100 + packet_index,
            )
            packet_bits.append(float(result.transmitted_bits))
            packet_success.append(float(result.frame_success))
            packet_attempts.append(float(result.attempts))
        if plan.aggregation == "mean_over_query_alternatives":
            actual_bits.extend(packet_bits)
            success.extend(packet_success)
            attempts.extend(packet_attempts)
        else:
            actual_bits.append(float(np.sum(packet_bits)))
            success.append(float(np.all(np.asarray(packet_success) > 0.5)))
            attempts.append(float(np.sum(packet_attempts)))
    return {
        "monte_carlo_units": len(actual_bits),
        "mean_actual_bits": float(np.mean(actual_bits)),
        "std_actual_bits": float(np.std(actual_bits, ddof=1))
        if len(actual_bits) > 1
        else 0.0,
        "operation_success_rate": float(np.mean(success)),
        "mean_attempts": float(np.mean(attempts)),
    }


def add_comparisons(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (
            row["n_channels"],
            row["query_family"],
            row["application_header_bits"],
            row["fec"],
            row["channel"],
            row["ebn0_db"],
        )
        groups.setdefault(key, {})[row["representation"]] = row
    comparisons = []
    for key, values in groups.items():
        bundle = values["bundle_all_queries"]
        for baseline_name in (
            "single_index_mean",
            "separate_indices_all_queries",
            "occupancy_vector_all_queries",
            "soft_power_all_queries",
        ):
            baseline = values[baseline_name]
            comparisons.append(
                {
                    "n_channels": key[0],
                    "query_family": key[1],
                    "application_header_bits": key[2],
                    "fec": key[3],
                    "channel": key[4],
                    "ebn0_db": key[5],
                    "proposed": "bundle_all_queries",
                    "baseline": baseline_name,
                    "nominal_bit_change_pct": float(
                        100.0
                        * (
                            bundle["nominal_transmitted_bits"]
                            - baseline["nominal_transmitted_bits"]
                        )
                        / baseline["nominal_transmitted_bits"]
                    ),
                    "actual_bit_change_pct": float(
                        100.0
                        * (
                            bundle["mean_actual_bits"]
                            - baseline["mean_actual_bits"]
                        )
                        / baseline["mean_actual_bits"]
                    ),
                    "operation_success_rate_difference": float(
                        bundle["operation_success_rate"]
                        - baseline["operation_success_rate"]
                    ),
                    "event_rate_break_even_vs_always_baseline": float(
                        baseline["mean_actual_bits"]
                        / bundle["mean_actual_bits"]
                    ),
                }
            )
    return comparisons


def summarize_primary(protocol: dict, comparisons: list[dict]) -> dict:
    primary = protocol["primary_slice"]
    selected = [
        row
        for row in comparisons
        if row["application_header_bits"]
        == int(primary["application_header_bits"])
        and row["fec"] == primary["fec"]
    ]
    output = {}
    for n_channels in protocol["scale_grid"]["n_channels"]:
        output[str(n_channels)] = {}
        for family in protocol["scale_grid"]["query_families"]:
            cell = [
                row
                for row in selected
                if row["n_channels"] == n_channels
                and row["query_family"] == family
            ]
            output[str(n_channels)][family] = {}
            for baseline in (
                "single_index_mean",
                "separate_indices_all_queries",
                "occupancy_vector_all_queries",
                "soft_power_all_queries",
            ):
                rows = [row for row in cell if row["baseline"] == baseline]
                output[str(n_channels)][family][baseline] = {
                    "mean_actual_bit_change_pct": float(
                        np.mean([row["actual_bit_change_pct"] for row in rows])
                    ),
                    "worst_operation_success_rate_difference": float(
                        np.min(
                            [
                                row["operation_success_rate_difference"]
                                for row in rows
                            ]
                        )
                    ),
                    "mean_event_rate_break_even": float(
                        np.mean(
                            [
                                row["event_rate_break_even_vs_always_baseline"]
                                for row in rows
                            ]
                        )
                    ),
                }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR / "configs/stage5_scale_codec_boundary_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "results/stage5/scale_codec_boundary_v1",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("refusing to overwrite scale-boundary output")
    protocol = load_json(args.protocol)
    stage2 = validate_protocol(protocol)
    grid = protocol["scale_grid"]
    link_grid = protocol["link"]
    rows = []
    for n_index, n_channels in enumerate(grid["n_channels"]):
        for family_index, (family, fractions) in enumerate(
            grid["query_families"].items()
        ):
            demands = demand_set_from_fractions(n_channels, fractions)
            for header_index, header_bits in enumerate(
                grid["application_header_bits"]
            ):
                verify_bundle_codec(n_channels, demands, header_bits)
                plans = representation_plans(
                    n_channels=n_channels,
                    demand_channels=demands,
                    application_header_bits=header_bits,
                    soft_power_bits_per_channel=int(
                        grid["soft_power_bits_per_channel"]
                    ),
                )
                for fec_index, fec in enumerate(link_grid["fec_schemes"]):
                    for channel_index, channel in enumerate(
                        link_grid["channel_models"]
                    ):
                        for ebn0_index, ebn0 in enumerate(link_grid["ebn0_db"]):
                            base = build_link_config(stage2, float(ebn0))
                            link = replace(base, fec=fec, channel=channel)
                            for representation_index, plan in enumerate(plans):
                                seed = (
                                    int(link_grid["seed"])
                                    + n_index * 10_000_000
                                    + family_index * 1_000_000
                                    + header_index * 100_000
                                    + fec_index * 10_000
                                    + channel_index * 1000
                                    + ebn0_index * 100
                                )
                                simulation = simulate_cost(
                                    plan,
                                    link,
                                    repeats=int(
                                        link_grid["monte_carlo_repeats"]
                                    ),
                                    seed_base=seed,
                                )
                                rows.append(
                                    {
                                        "n_channels": n_channels,
                                        "query_family": family,
                                        "demand_channels": list(demands),
                                        "query_count": len(demands),
                                        "application_header_bits": header_bits,
                                        "representation": plan.name,
                                        "aggregation": plan.aggregation,
                                        "application_packet_bits": list(
                                            plan.application_packet_bits
                                        ),
                                        "fec": fec,
                                        "channel": channel,
                                        "ebn0_db": float(ebn0),
                                        "nominal_transmitted_bits": nominal_cost(
                                            plan, link
                                        ),
                                        **simulation,
                                    }
                                )
            print(f"N={n_channels} {family} complete", flush=True)
    comparisons = add_comparisons(rows)
    result = {
        "version": "1.0",
        "status": "stage5_scale_codec_boundary_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "inputs": {
            "stage2_config_sha256": protocol["link"][
                "stage2_config_sha256"
            ],
            "predecessor_protocol_sha256": protocol["predecessor"][
                "protocol_sha256"
            ],
            "predecessor_result_sha256": protocol["predecessor"][
                "result_sha256"
            ],
        },
        "rows": rows,
        "comparisons": comparisons,
        "primary_slice_summary": summarize_primary(protocol, comparisons),
        "governance": {
            "stage4_final_measurements_loaded": False,
            "stage4_final_metrics_loaded": False,
            "confirmatory_final": False,
            "measured_scaled_spectra_loaded": False,
        },
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": protocol["claim_boundary"],
    }
    args.out_dir.mkdir(parents=True)
    output = args.out_dir / "scale_codec_boundary_result.json"
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": len(rows),
                "comparisons": len(comparisons),
                "final_loaded": False,
                "measured_scaled_spectra_loaded": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

