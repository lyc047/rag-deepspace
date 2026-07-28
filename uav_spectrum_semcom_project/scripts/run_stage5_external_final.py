#!/usr/bin/env python
"""Run the frozen Stage-5 cross-campaign Final after atomic access consumption."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage2_digital_link import build_link_config  # noqa: E402
from spectrum_semcom.aerpaw_helikite import (  # noqa: E402
    load_helikite_zip_power_sweep,
)
from spectrum_semcom.c1_temporal_statistics import (  # noqa: E402
    empirical_cvar_numpy,
    paired_cluster_bootstrap,
)
from spectrum_semcom.c1_v4_block_semantics import (  # noqa: E402
    V4SpectrumArrays,
    build_block_semantic_scene,
    quantize_unit_interval,
    selected_block_start,
)
from spectrum_semcom.digital_link import transmit_payload_analytic  # noqa: E402
from spectrum_semcom.final_holdout import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
)
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)
from spectrum_semcom.stage5_event_semantics import (  # noqa: E402
    block_regret,
    contiguous_block_costs,
)
from spectrum_semcom.stage5_final_governance import (  # noqa: E402
    validate_external_final_catalog,
    verify_code_snapshot,
)
from spectrum_semcom.stage5_query_bundle import query_bundle_payload_width  # noqa: E402
from spectrum_semcom.stage5_scale_boundary import index_width  # noqa: E402


METHODS = (
    "occupancy_fixed4_always",
    "soft_power_fixed4_always",
    "absolute_index_current_query_always",
    "query_bundle_always",
    "query_event_bundle_fixed",
)


def verify_inputs(
    protocol: dict,
    registry: dict,
    state: dict,
    snapshot: dict,
    catalog: dict,
    *,
    preflight: bool,
) -> list[str]:
    errors = verify_code_snapshot(PROJECT_DIR, snapshot)
    errors.extend(validate_external_final_catalog(catalog, protocol, registry))
    if catalog.get("code_snapshot_sha256") != snapshot.get(
        "executable_snapshot_sha256"
    ):
        errors.append("catalog and snapshot are not bound")
    if preflight:
        if (
            state.get("status") != "downloaded_integrity_verified_not_accessed"
            or state.get("access_count") != 0
            or state.get("final_signal_values_accessed") is not False
        ):
            errors.append("preflight requires pristine access state")
    else:
        if (
            state.get("status") != "access_consumed"
            or state.get("access_count") != 1
            or state.get("reset_permitted") is not False
            or state.get("final_signal_values_accessed") is not True
        ):
            errors.append("normal Final requires atomic single-access receipt")
        receipt = state.get("receipt", {})
        if receipt.get("code_snapshot_sha256") != snapshot.get(
            "executable_snapshot_sha256"
        ):
            errors.append("access receipt is not bound to snapshot")
        if receipt.get("catalog_sha256") != canonical_json_sha256(catalog):
            errors.append("access receipt is not bound to catalog")
    for entry in registry["final_candidates"]:
        archive = Path(registry["local_root"]) / entry["archive"]
        if (
            not archive.is_file()
            or archive.stat().st_size != int(entry["archive_size_bytes"])
            or sha256_file(archive) != entry["archive_sha256"]
        ):
            errors.append(f"archive integrity failed: {archive}")
    return errors


def load_scene_arrays(
    catalog: dict,
    protocol: dict,
) -> list[dict]:
    band_low, band_high = map(
        float, protocol["resource_task"]["analysis_band_mhz"]
    )
    n_values = [
        int(protocol["resource_task"]["primary_n_channels"]),
        *map(int, protocol["resource_task"]["secondary_n_channels"]),
    ]
    ratios = tuple(
        float(value) for value in protocol["resource_task"]["demand_ratios"]
    )
    handles: dict[Path, zipfile.ZipFile] = {}
    records = []
    campaign_positions: dict[str, int] = {}
    try:
        for scene in catalog["scenes"]:
            archive_path = Path(scene["archive_path"])
            handle = handles.setdefault(
                archive_path, zipfile.ZipFile(archive_path)
            )
            for member_name, size_key, crc_key in (
                ("meta_member", "meta_size_bytes", "meta_crc32"),
                ("data_member", "data_size_bytes", "data_crc32"),
            ):
                info = handle.getinfo(scene[member_name])
                if (
                    info.file_size != int(scene[size_key])
                    or f"{info.CRC:08x}" != scene[crc_key]
                ):
                    raise ValueError(
                        f"catalog member integrity changed: {scene['scene_id']}"
                    )
            sweep = load_helikite_zip_power_sweep(
                handle,
                meta_member=scene["meta_member"],
                data_member=scene["data_member"],
                timestamp_local=scene["timestamp_local"],
                site=scene["campaign_id"],
                local_timezone=protocol["sampling"]["local_timezone"],
            )
            spectrum = V4SpectrumArrays(
                scene["campaign_id"],
                scene["timestamp_local"],
                sweep.frequencies_mhz,
                sweep.powers_dbm,
            )
            campaign = scene["campaign_id"]
            position = campaign_positions.get(campaign, 0)
            campaign_positions[campaign] = position + 1
            n_data = {}
            for n_channels in n_values:
                demands = tuple(
                    int(round(n_channels * ratio)) for ratio in ratios
                )
                demand = demands[position % len(demands)]
                semantic = build_block_semantic_scene(
                    spectrum,
                    frequency_low_mhz=band_low,
                    frequency_high_mhz=band_high,
                    n_channels=n_channels,
                    demand_channels=demand,
                )
                target_by_demand = {}
                for candidate_demand in demands:
                    target_by_demand[candidate_demand] = int(
                        np.argmin(
                            contiguous_block_costs(
                                semantic.channel_power_dbm,
                                candidate_demand,
                            )
                        )
                    )
                n_data[str(n_channels)] = {
                    "demand": demand,
                    "demands": demands,
                    "channel_power_dbm": semantic.channel_power_dbm,
                    "occupancy": semantic.occupancy,
                    "normalized_channel_power": semantic.normalized_channel_power,
                    "current_costs_dbm": semantic.block_cost_dbm,
                    "target_by_demand": target_by_demand,
                }
            records.append({**scene, "n_data": n_data})
            if len(records) % 25 == 0:
                print(f"loaded Final sweeps: {len(records)}/200", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    return records


def application_bits(
    method: str,
    *,
    n_channels: int,
    demand: int,
    demands: tuple[int, ...],
    header: int,
    value_bits: int,
) -> int:
    if method in {"occupancy_fixed4_always", "soft_power_fixed4_always"}:
        return header + n_channels * value_bits
    if method == "absolute_index_current_query_always":
        return header + index_width(n_channels, demand)
    if method in {"query_bundle_always", "query_event_bundle_fixed"}:
        return header + query_bundle_payload_width(n_channels, demands)
    raise KeyError(method)


def successful_selection(
    method: str, data: dict
) -> int:
    demand = int(data["demand"])
    if method == "occupancy_fixed4_always":
        values = quantize_unit_interval(data["occupancy"], 4)
        return selected_block_start(values, "channel", demand)
    if method == "soft_power_fixed4_always":
        values = quantize_unit_interval(
            data["normalized_channel_power"], 4
        )
        return selected_block_start(values, "channel", demand)
    return int(data["target_by_demand"][demand])


def evaluate_n(
    records: list[dict],
    protocol: dict,
    stage2: dict,
    n_channels: int,
    *,
    seed_offset: int,
) -> dict:
    header = int(protocol["resource_task"]["application_header_bits"])
    value_bits = int(protocol["resource_task"]["soft_power_bits_per_channel"])
    threshold = float(protocol["resource_task"]["regret_threshold_db"])
    max_age = float(protocol["resource_task"]["max_state_age_minutes"])
    link_grid = protocol["reporting_link"]
    accumulators = {
        method: {
            name: np.zeros(len(records), dtype=np.float64)
            for name in ("regret", "bits", "sent", "success")
        }
        for method in METHODS
    }
    trajectories = (
        len(link_grid["channel_models"])
        * len(link_grid["ebn0_db"])
        * int(link_grid["repeats"])
    )
    for channel_index, channel in enumerate(link_grid["channel_models"]):
        for ebn0_index, ebn0 in enumerate(link_grid["ebn0_db"]):
            link = replace(
                build_link_config(stage2, float(ebn0)),
                channel=channel,
                fec=link_grid["fec"],
            )
            for repeat in range(int(link_grid["repeats"])):
                states = {
                    method: {} for method in METHODS
                }
                for position, record in enumerate(records):
                    data = record["n_data"][str(n_channels)]
                    campaign = record["campaign_id"]
                    demand = int(data["demand"])
                    demands = tuple(int(value) for value in data["demands"])
                    costs = np.asarray(data["current_costs_dbm"])
                    timestamp = datetime.fromisoformat(record["timestamp_local"])
                    for method_index, method in enumerate(METHODS):
                        method_states = states[method].setdefault(
                            campaign, {"by_demand": {}, "last_bundle": None}
                        )
                        prior_row = method_states["by_demand"].get(demand)
                        prior_start = None
                        if prior_row is not None:
                            age = (
                                timestamp
                                - datetime.fromisoformat(prior_row["timestamp"])
                            ).total_seconds() / 60.0
                            if age <= max_age:
                                prior_start = int(prior_row["start"])
                        transmits = True
                        if method == "query_event_bundle_fixed":
                            reuse = (
                                float("inf")
                                if prior_start is None
                                else block_regret(costs, prior_start)
                            )
                            transmits = prior_start is None or reuse > threshold
                            last_bundle = method_states["last_bundle"]
                            if last_bundle is not None:
                                bundle_age = (
                                    timestamp
                                    - datetime.fromisoformat(last_bundle)
                                ).total_seconds() / 60.0
                                transmits = transmits or bundle_age > max_age
                        selected = prior_start if prior_start is not None else 0
                        values = accumulators[method]
                        if transmits:
                            app_bits = application_bits(
                                method,
                                n_channels=n_channels,
                                demand=demand,
                                demands=demands,
                                header=header,
                                value_bits=value_bits,
                            )
                            seed = (
                                int(protocol["statistics"]["bootstrap_seed"])
                                + seed_offset
                                + position * 1_000_000
                                + method_index * 100_000
                                + channel_index * 10_000
                                + ebn0_index * 1000
                                + repeat
                            )
                            tx = transmit_payload_analytic(app_bits, link, seed)
                            values["sent"][position] += 1.0
                            values["bits"][position] += float(
                                tx.transmitted_bits
                            )
                            if tx.frame_success:
                                values["success"][position] += 1.0
                                selected = successful_selection(method, data)
                                if method in {
                                    "query_bundle_always",
                                    "query_event_bundle_fixed",
                                }:
                                    for update_demand, start in data[
                                        "target_by_demand"
                                    ].items():
                                        method_states["by_demand"][
                                            int(update_demand)
                                        ] = {
                                            "start": int(start),
                                            "timestamp": record[
                                                "timestamp_local"
                                            ],
                                        }
                                    method_states["last_bundle"] = record[
                                        "timestamp_local"
                                    ]
                                else:
                                    method_states["by_demand"][demand] = {
                                        "start": selected,
                                        "timestamp": record["timestamp_local"],
                                    }
                        values["regret"][position] += block_regret(costs, selected)
            print(
                f"N={n_channels} {channel} {float(ebn0):g} dB complete",
                flush=True,
            )
    scene_rows = []
    summaries = {}
    for method in METHODS:
        values = accumulators[method]
        rows = []
        for position, record in enumerate(records):
            row = {
                "scene_id": record["scene_id"],
                "campaign_id": record["campaign_id"],
                "cluster_id": record["cluster_id"],
                "timestamp_local": record["timestamp_local"],
                "n_channels": n_channels,
                "demand_channels": int(
                    record["n_data"][str(n_channels)]["demand"]
                ),
                "method": method,
                "mean_regret_db": float(
                    values["regret"][position] / trajectories
                ),
                "mean_actual_bits": float(
                    values["bits"][position] / trajectories
                ),
                "transmission_rate": float(
                    values["sent"][position] / trajectories
                ),
                "conditional_success_rate": float(
                    values["success"][position] / values["sent"][position]
                )
                if values["sent"][position] > 0
                else 1.0,
            }
            rows.append(row)
            scene_rows.append(row)
        regret = np.asarray([row["mean_regret_db"] for row in rows])
        summaries[method] = {
            "scene_count": len(rows),
            "mean_regret_db": float(np.mean(regret)),
            "cvar_0_9_regret_db": empirical_cvar_numpy(
                regret, float(protocol["statistics"]["cvar_alpha"])
            ),
            "mean_actual_bits": float(
                np.mean([row["mean_actual_bits"] for row in rows])
            ),
            "mean_transmission_rate": float(
                np.mean([row["transmission_rate"] for row in rows])
            ),
            "campaign_strata": {
                campaign: {
                    "scene_count": len(selected),
                    "mean_regret_db": float(
                        np.mean([row["mean_regret_db"] for row in selected])
                    ),
                    "mean_actual_bits": float(
                        np.mean([row["mean_actual_bits"] for row in selected])
                    ),
                }
                for campaign in sorted(
                    {row["campaign_id"] for row in rows}
                )
                for selected in [
                    [
                        row
                        for row in rows
                        if row["campaign_id"] == campaign
                    ]
                ]
            },
        }
    comparisons = {}
    proposed_rows = [
        row
        for row in scene_rows
        if row["method"] == "query_event_bundle_fixed"
    ]
    for index, baseline in enumerate(METHODS[:-1]):
        baseline_rows = [
            row for row in scene_rows if row["method"] == baseline
        ]
        inference = paired_cluster_bootstrap(
            np.asarray([row["mean_regret_db"] for row in proposed_rows]),
            np.asarray([row["mean_regret_db"] for row in baseline_rows]),
            np.asarray([row["mean_actual_bits"] for row in proposed_rows]),
            np.asarray([row["mean_actual_bits"] for row in baseline_rows]),
            [row["cluster_id"] for row in baseline_rows],
            repetitions=int(
                protocol["statistics"][
                    "paired_campaign_x_15_minute_cluster_bootstrap_repetitions"
                ]
            ),
            seed=int(protocol["statistics"]["bootstrap_seed"])
            + seed_offset
            + index,
            cvar_alpha=float(protocol["statistics"]["cvar_alpha"]),
            one_sided_confidence=float(
                protocol["statistics"]["one_sided_confidence"]
            ),
        )
        baseline_bits = summaries[baseline]["mean_actual_bits"]
        proposed_bits = summaries["query_event_bundle_fixed"][
            "mean_actual_bits"
        ]
        inference["actual_bit_reduction_pct"] = float(
            100.0 * (baseline_bits - proposed_bits) / baseline_bits
        )
        comparisons[baseline] = inference
    return {
        "n_channels": n_channels,
        "trajectory_count_per_scene": trajectories,
        "summaries": summaries,
        "comparisons_proposed_vs_baselines": comparisons,
        "scene_rows": scene_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage5_unified_external_final_protocol_v1.json",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage5_external_final_registry_v1.json",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage5_external_final_access_state.json",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=PROJECT_DIR
        / "results/stage5/external_final_freeze_v1/code_snapshot.json",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_DIR
        / "results/stage5/external_final_catalog_v1/catalog.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR
        / "results/stage5/external_final_v1/final_metrics.json",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    state = json.loads(args.state.read_text(encoding="utf-8"))
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    errors = verify_inputs(
        protocol,
        registry,
        state,
        snapshot,
        catalog,
        preflight=args.preflight,
    )
    if errors:
        raise ValueError("Stage-5 Final input audit failed: " + "; ".join(errors))
    if args.preflight:
        print(
            json.dumps(
                {
                    "status": "ready_for_atomic_single_access",
                    "scene_count": len(catalog["scenes"]),
                    "snapshot_sha256": snapshot["executable_snapshot_sha256"],
                    "catalog_sha256": catalog["catalog_sha256"],
                    "signal_values_loaded": False,
                    "final_access_consumed": False,
                },
                indent=2,
            )
        )
        return
    if args.output.exists():
        raise FileExistsError("refusing to overwrite external Final output")
    records = load_scene_arrays(catalog, protocol)
    stage2_path = PROJECT_DIR / protocol["reporting_link"]["stage2_config"]
    stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
    n_values = [
        int(protocol["resource_task"]["primary_n_channels"]),
        *map(int, protocol["resource_task"]["secondary_n_channels"]),
    ]
    n_results = {}
    for index, n_channels in enumerate(n_values):
        n_results[str(n_channels)] = evaluate_n(
            records,
            protocol,
            stage2,
            n_channels,
            seed_offset=(index + 1) * 10_000_000,
        )
    ack_path = PROJECT_DIR / protocol["secondary_ack_evidence"]["result"]
    if sha256_file(ack_path) != protocol["secondary_ack_evidence"][
        "result_sha256"
    ]:
        raise ValueError("frozen secondary ACK evidence changed")
    result = {
        "version": "1.0",
        "status": "stage5_unified_external_final_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "registry_sha256": sha256_file(args.registry),
        "catalog_sha256": canonical_json_sha256(catalog),
        "access_receipt_sha256": state["receipt_sha256"],
        "code_snapshot_sha256": snapshot["executable_snapshot_sha256"],
        "scene_count": len(records),
        "n_results": n_results,
        "secondary_ack_evidence": {
            **protocol["secondary_ack_evidence"],
            "external_dataset_measures_ack_faults": False,
        },
        "governance": {
            "access_count": 1,
            "post_final_parameter_tuning_permitted": False,
            "all_methods_used_identical_catalog": True,
            "campaign_stratified_results_reported": True,
            "ack_faults_are_measured_claims": False,
            "multi_node_or_mimo_claim": False,
        },
        "environment": environment_snapshot(["numpy", "scipy"]),
        "claim_boundary": protocol["claim_boundary"],
    }
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scene_count": len(records),
                "access_count": 1,
                "post_final_tuning_permitted": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
