import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from spectrum_semcom.aerpaw_spectrum import AerpawZipPair
from spectrum_semcom.digital_link import DigitalLinkConfig
from spectrum_semcom.c1_v4_block_semantics import (
    block_regret_db,
    build_block_semantic_scene,
    canonical_v4_payloads,
    decode_resource_semantic_payload,
    encode_resource_semantic_payload,
    guarded_fallback_start,
    load_aerpaw_zip_power_sweep,
    normalize_cost,
    quantize_unit_interval,
    selected_block_start,
    transmit_resource_payload,
)


def test_guarded_fallback_expires_without_adding_message_state() -> None:
    assert guarded_fallback_start(None, None, "2022-02-14T00:00:00", max_age_minutes=60.0) == 0
    assert guarded_fallback_start(3, "2022-02-14T00:00:00", "2022-02-14T01:00:00", max_age_minutes=60.0) == 3
    assert guarded_fallback_start(3, "2022-02-14T00:00:00", "2022-02-14T01:00:01", max_age_minutes=60.0) == 0
    with pytest.raises(ValueError, match="precedes"):
        guarded_fallback_start(3, "2022-02-14T01:00:00", "2022-02-14T00:00:00", max_age_minutes=60.0)


def write_zip_pair(path: Path) -> AerpawZipPair:
    frequency = np.linspace(2400.0, 2483.4, 80)
    power = np.concatenate([np.full(20, -120.0), np.full(20, -100.0), np.full(20, -90.0), np.full(20, -80.0)]).astype("<f4")
    meta = {
        "global": {
            "core:datatype": "rf32_le",
            "dataset:site": "LW1",
            "dataset:frequency_axis_MHz": frequency.tolist(),
            "dataset:num_bins": len(frequency),
        }
    }
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("root/results_20220208_120000.sigmf-meta", json.dumps(meta))
        handle.writestr("root/results_20220208_120000.sigmf-data", power.tobytes())
    return AerpawZipPair(
        "results_20220208_120000",
        path,
        "root/results_20220208_120000.sigmf-meta",
        "root/results_20220208_120000.sigmf-data",
        "2022-02-08T12:00:00",
    )


def test_zip_sweep_to_task_aligned_block_semantics(tmp_path: Path) -> None:
    pair = write_zip_pair(tmp_path / "sweep.zip")
    sweep = load_aerpaw_zip_power_sweep(pair)
    scene = build_block_semantic_scene(
        sweep,
        frequency_low_mhz=2400.0,
        frequency_high_mhz=2483.5,
        n_channels=8,
        demand_channels=4,
    )
    assert scene.features.shape == (8, 4)
    assert scene.block_cost_dbm.shape == (5,)
    assert np.argmin(scene.best_block_indicator) == np.argmin(scene.block_cost_dbm)
    assert selected_block_start(scene.normalized_block_cost, "block", 4) == int(np.argmin(scene.block_cost_dbm))


def test_v4_payloads_count_real_application_bits_and_clean_regret() -> None:
    class Scene:
        occupancy = np.array([0.0, 0.2, 0.8, 0.9, 0.7, 0.2, 0.1, 0.0])
        normalized_channel_power = np.array([0.0, 0.1, 0.8, 1.0, 0.9, 0.4, 0.2, 0.1])
        normalized_block_cost = np.array([0.0, 0.4, 1.0, 0.8, 0.2])
        best_block_indicator = np.array([0.0, 1.0, 1.0, 1.0, 1.0])

    payloads = {item.name: item for item in canonical_v4_payloads(Scene())}
    assert payloads["occupancy_fixed4"].application_bits == 184
    assert payloads["block_score_fixed4"].application_bits == 172
    assert payloads["block_score_fixed6"].application_bits == 182
    assert payloads["best_block_indicator1"].application_bits == 157
    true_cost = np.array([-110.0, -105.0, -100.0, -102.0, -108.0])
    for name in ("block_score_fixed4", "block_score_fixed6", "best_block_indicator1"):
        payload = payloads[name]
        start = selected_block_start(payload.quantized_values, payload.representation_kind, 4)
        assert block_regret_db(start, true_cost) == 0.0


def test_normalization_and_quantization_validate_inputs() -> None:
    np.testing.assert_allclose(normalize_cost(np.array([2.0, 4.0, 6.0])), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(quantize_unit_interval(np.array([0.0, 0.5, 1.0]), 2), [0.0, 2 / 3, 1.0])
    with pytest.raises(ValueError):
        quantize_unit_interval(np.array([-0.1]), 4)


def test_resource_payload_codec_round_trips_and_matches_declared_bits() -> None:
    class Scene:
        occupancy = np.linspace(0.0, 1.0, 8)
        normalized_channel_power = np.linspace(1.0, 0.0, 8)
        normalized_block_cost = np.linspace(0.0, 1.0, 5)
        best_block_indicator = np.array([0.0, 1.0, 1.0, 1.0, 1.0])

    for payload in canonical_v4_payloads(Scene()):
        encoded = encode_resource_semantic_payload(payload, node_id=7, scene_id="scene-a")
        decoded = decode_resource_semantic_payload(encoded)
        assert encoded.size == payload.application_bits
        assert decoded.name == payload.name and decoded.node_id == 7
        np.testing.assert_allclose(decoded.values, payload.quantized_values)


def test_successful_task_payload_selects_true_block_and_counts_link_bits() -> None:
    class Scene:
        occupancy = np.array([0.0] * 4 + [1.0] * 4)
        normalized_channel_power = np.array([0.0] * 4 + [1.0] * 4)
        normalized_block_cost = np.linspace(0.0, 1.0, 5)
        best_block_indicator = np.array([0.0, 1.0, 1.0, 1.0, 1.0])

    payload = {item.name: item for item in canonical_v4_payloads(Scene())}["block_score_fixed4"]
    outcome = transmit_resource_payload(
        payload,
        scene_id="scene-a",
        block_cost_dbm=np.array([-110.0, -105.0, -100.0, -95.0, -90.0]),
        demand_channels=4,
        link=DigitalLinkConfig(ebn0_db=100.0),
        seed=3,
    )
    assert outcome.frame_success and outcome.selected_start == 0 and outcome.regret_db == 0
    assert outcome.transmitted_bits > payload.application_bits
