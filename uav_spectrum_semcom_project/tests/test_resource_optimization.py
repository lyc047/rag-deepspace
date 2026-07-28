from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from simulate_resource_optimization import (
    aggregate_group_boxes,
    block_occupancy,
    frame_channel_occupancy,
    regroup_flat_boxes,
    transmit_large_payload_partial,
)


def test_frame_channel_occupancy_splits_y_axis():
    boxes = [np.array([0.0, 0.0, 1.0, 0.5], dtype=np.float32)]
    occ = frame_channel_occupancy(boxes, n_channels=2, axis="y")
    assert np.allclose(occ, [1.0, 0.0])


def test_block_occupancy_uses_contiguous_average():
    occ = np.array([[0.1, 0.5, 0.9, 0.3]], dtype=np.float32)
    blocks = block_occupancy(occ, demand_channels=2)
    assert np.allclose(blocks, [[0.3, 0.7, 0.6]])


def test_multi_uav_grouping_round_trip():
    frames = [
        [np.array([0.0, 0.0, 0.1, 0.1], dtype=np.float32)],
        [np.array([0.2, 0.2, 0.3, 0.3], dtype=np.float32)],
        [np.array([0.4, 0.4, 0.5, 0.5], dtype=np.float32)],
    ]
    groups = [[0, 2], [1]]
    aggregated = aggregate_group_boxes(frames, groups)
    regrouped = regroup_flat_boxes([frames[0], frames[2], frames[1]], groups)
    assert len(aggregated[0]) == 2
    assert len(aggregated[1]) == 1
    assert all(np.allclose(a, b) for a, b in zip(aggregated[0], regrouped[0]))


def test_large_payload_partial_keeps_boxes_on_clean_link():
    frames = [[np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float32)]]
    delivered, bits = transmit_large_payload_partial(
        frames,
        packet_loss=0.0,
        ber=0.0,
        payload_bits_per_frame=2048,
        packet_bits=1024,
        rng=np.random.default_rng(1),
    )
    assert bits == 2048
    assert len(delivered[0]) == 1
