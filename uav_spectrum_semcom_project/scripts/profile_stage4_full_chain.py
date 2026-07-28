#!/usr/bin/env python
"""Profile the frozen development chain from spectrogram files to selection.

The profile uses one registered development scene and reports CPU timing only.
It does not read, register, or evaluate any final-holdout scene.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.aerpaw_spectrum import cleanest_contiguous_block  # noqa: E402
from spectrum_semcom.correlated_scenes import compose_max_hold_scene  # noqa: E402
from spectrum_semcom.digital_link import DigitalLinkConfig  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.gate_a_data import pool_probability_mask, resolve_raddet_paths  # noqa: E402
from spectrum_semcom.gate_a_model import VariableRateOccupancyHead  # noqa: E402
from spectrum_semcom.models import TinyOccupancyCNN  # noqa: E402
from spectrum_semcom.multigranular_semantics import (  # noqa: E402
    SemanticMessage,
    SemanticQuality,
    transmit_semantic_message,
)
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402


def normalize_image(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32) / 255.0
    return (value - value.mean()) / (value.std() + 1e-6)


def timing_summary(samples: list[int]) -> dict[str, float]:
    milliseconds = np.asarray(samples, dtype=np.float64) / 1e6
    return {
        "median_ms": float(np.median(milliseconds)),
        "p95_ms": float(np.quantile(milliseconds, 0.95)),
        "mean_ms": float(np.mean(milliseconds)),
    }


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "results/stage4/full_chain_complexity_v1/full_chain_complexity_result.json")
    args = parser.parse_args()
    if args.repeats < 20 or args.warmup < 1:
        raise ValueError("use at least 20 measured repeats and one warmup")

    torch.set_num_threads(1)
    protocol_path = PROJECT_DIR / "configs/stage4_protocol.json"
    registry_path = PROJECT_DIR / "configs/stage4_development_registry_v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    scene = registry["splits"]["validation"]["scenes"][0]
    task = registry["selected_task"]
    n_channels = int(task["n_channels"])
    axis = str(task["channel_axis"])
    demand = int(task["demand_channels"])
    data_root = PROJECT_DIR / "data/raw/raddet/RadDet40k128HW001Tv2"
    image_paths = [resolve_raddet_paths(data_root, frame_id)[0] for frame_id in scene["source_frame_ids"]]
    if not all(path.is_file() for path in image_paths):
        raise FileNotFoundError("registered development source image is missing")

    detector_checkpoint = PROJECT_DIR / "results/phase1/raddet_occupancy_mask.pt"
    c1_checkpoint = PROJECT_DIR / protocol["counterfactual_value_protocol"]["canonical_c1_checkpoint"]
    detector = TinyOccupancyCNN(channels=16)
    detector.load_state_dict(torch.load(detector_checkpoint, map_location="cpu", weights_only=True))
    detector.eval()
    c1 = VariableRateOccupancyHead(n_channels, int(protocol["gate_a_controlled_training"]["hidden_dim"]))
    c1.load_state_dict(torch.load(c1_checkpoint, map_location="cpu", weights_only=True))
    c1.eval()
    link = DigitalLinkConfig(channel="awgn", ebn0_db=20.0, max_retransmissions=1)
    quality = SemanticQuality(0.0, 0.8, 0.2, 0.0, 0.0, 1.0, 12.0, 0.0, 1.0, 0.0)

    timings: dict[str, list[int]] = {
        "load_8_images_and_max_hold": [],
        "normalize_and_tensorize": [],
        "frozen_detector": [],
        "pool_and_C1": [],
        "selective_G2_codec_and_link_5_messages": [],
        "fusion_and_contiguous_resource_selection": [],
        "complete_pipeline": [],
    }
    final_trace: dict[str, object] = {}
    total_iterations = args.warmup + args.repeats
    for iteration in range(total_iterations):
        measured = iteration >= args.warmup
        total_start = perf_counter_ns()
        start = perf_counter_ns()
        images = [np.asarray(Image.open(path).convert("L"), dtype=np.uint8) for path in image_paths]
        composed = compose_max_hold_scene(images)
        load_elapsed = perf_counter_ns() - start
        start = perf_counter_ns()
        tensor = torch.as_tensor(normalize_image(composed)[None, None], dtype=torch.float32)
        tensor_elapsed = perf_counter_ns() - start
        start = perf_counter_ns()
        with torch.inference_mode():
            mask = torch.sigmoid(detector(tensor))[0, 0].numpy()
        detector_elapsed = perf_counter_ns() - start
        start = perf_counter_ns()
        base = pool_probability_mask(mask, n_channels, axis)
        with torch.inference_mode():
            refined, precision_bits, _ = c1.infer(torch.as_tensor(base[None], dtype=torch.float32))
        refined = refined[0].numpy()
        bits = int(precision_bits[0])
        c1_elapsed = perf_counter_ns() - start
        start = perf_counter_ns()
        latest = {}
        total_bits = 0
        application_bits = 0
        plan = [(node, "G1", 1) for node in range(4)] + [(3, "G2", bits)]
        for message_index, (node, granularity, probability_bits) in enumerate(plan):
            occupancy = np.clip(refined + (node - 1.5) * 0.005, 0.0, 1.0)
            message = SemanticMessage(
                granularity,
                node,
                "stage4-development-complexity-profile",
                occupancy,
                probability_bits,
                quality=quality if granularity == "G2" else None,
            )
            transmission = transmit_semantic_message(message, link, 20260721 + iteration * 10 + message_index)
            application_bits += transmission.encoded.application_bits
            total_bits += transmission.link.transmitted_bits
            if transmission.decoded is not None:
                latest[node] = transmission.decoded.occupancy
        codec_elapsed = perf_counter_ns() - start
        start = perf_counter_ns()
        fused = np.mean(np.stack(list(latest.values())), axis=0) if latest else np.full(n_channels, 0.5)
        selected_start, selected_cost = cleanest_contiguous_block(fused, demand)
        selection_elapsed = perf_counter_ns() - start
        total_elapsed = perf_counter_ns() - total_start
        if measured:
            for name, value in (
                ("load_8_images_and_max_hold", load_elapsed),
                ("normalize_and_tensorize", tensor_elapsed),
                ("frozen_detector", detector_elapsed),
                ("pool_and_C1", c1_elapsed),
                ("selective_G2_codec_and_link_5_messages", codec_elapsed),
                ("fusion_and_contiguous_resource_selection", selection_elapsed),
                ("complete_pipeline", total_elapsed),
            ):
                timings[name].append(value)
        final_trace = {
            "base_occupancy": base.tolist(),
            "refined_occupancy": refined.tolist(),
            "C1_probability_bits": bits,
            "attempted_messages": len(plan),
            "decoded_latest_reports": len(latest),
            "application_bits_per_decision": application_bits,
            "actual_transmitted_bits_per_decision": total_bits,
            "selected_block_start": selected_start,
            "selected_block_mean_predicted_occupancy": selected_cost,
        }

    result = {
        "experiment_id": "stage4_frozen_full_chain_complexity_v1",
        "scope": "registered_development_scene_CPU_single_thread",
        "scene_id": scene["scene_id"],
        "sources_per_scene": len(image_paths),
        "timing": {name: timing_summary(samples) for name, samples in timings.items()},
        "parameters": {"frozen_detector": parameter_count(detector), "C1_head": parameter_count(c1)},
        "checkpoint_bytes": {"frozen_detector": detector_checkpoint.stat().st_size, "C1_head": c1_checkpoint.stat().st_size},
        "checkpoint_sha256": {"frozen_detector": sha256_file(detector_checkpoint), "C1_head": sha256_file(c1_checkpoint)},
        "trace": final_trace,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "environment": environment_snapshot(["numpy", "torch", "pillow"]),
        "final_access_consumed": False,
        "claim_boundary": "CPU software profile on one frozen development scene; excludes RF front-end capture, SDR transfer, and embedded-board energy.",
    }
    atomic_write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "complete_pipeline_median_ms": result["timing"]["complete_pipeline"]["median_ms"], "final_access_consumed": False}, indent=2))


if __name__ == "__main__":
    main()
