import json
import zipfile
from pathlib import Path

from spectrum_semcom.stage5_final_governance import (
    discover_timestamped_sigmf_pairs,
    expected_cluster_id,
    select_spaced_evenly,
    validate_external_final_catalog,
    verify_code_snapshot,
)


def test_member_discovery_and_spaced_selection_do_not_read_values(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "campaign.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for minute in range(0, 60, 5):
            stem = f"spec_results_20240101_12{minute:02d}00"
            handle.writestr(f"pow/{stem}.sigmf-meta", b"{}")
            handle.writestr(f"pow/{stem}.sigmf-data", b"signal")
    rows, errors = discover_timestamped_sigmf_pairs(archive)
    assert not errors and len(rows) == 12
    selected = select_spaced_evenly(rows, count=6, minimum_spacing_s=300)
    assert len(selected) == 6
    assert selected[0]["timestamp_local"] == "2024-01-01T12:00:00"


def test_catalog_validation_allows_repeated_clusters_but_not_bad_spacing() -> None:
    allocations = {"a": 2, "b": 1}
    protocol = {
        "sampling": {
            "target_scene_count": 3,
            "campaign_allocations": allocations,
            "minimum_spacing_minutes": 5,
            "cluster_minutes": 15,
        }
    }
    registry = {"final_candidates": [{"dataset_id": "a"}, {"dataset_id": "b"}]}
    scenes = []
    for campaign, timestamp in (
        ("a", "2024-01-01T12:00:00"),
        ("a", "2024-01-01T12:05:00"),
        ("b", "2024-01-01T12:00:00"),
    ):
        stem = f"{campaign}-{timestamp}"
        scenes.append(
            {
                "scene_id": stem,
                "campaign_id": campaign,
                "timestamp_local": timestamp,
                "cluster_id": expected_cluster_id(campaign, timestamp, 15),
                "archive_path": f"{campaign}.zip",
                "archive_sha256": campaign * 64,
                "meta_member": stem + ".sigmf-meta",
                "data_member": stem + ".sigmf-data",
                "meta_crc32": "00000000",
                "data_crc32": "11111111",
                "selection_used_signal_values": False,
            }
        )
    catalog = {
        "status": "candidate_external_final_catalog",
        "scenes": scenes,
        "signal_values_loaded_for_selection": False,
        "method_outputs_loaded_for_selection": False,
    }
    assert validate_external_final_catalog(catalog, protocol, registry) == []
    catalog["scenes"][1]["timestamp_local"] = "2024-01-01T12:04:59"
    assert "a violates minimum spacing" in validate_external_final_catalog(
        catalog, protocol, registry
    )


def test_real_access_state_remains_zero() -> None:
    root = Path(__file__).resolve().parents[1]
    state = json.loads(
        (
            root / "configs/stage5_external_final_access_state.json"
        ).read_text(encoding="utf-8")
    )
    assert state["access_count"] == 0
    assert state["final_signal_values_accessed"] is False


def test_real_snapshot_catalog_and_prefinal_state_are_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (
            root / "configs/stage5_unified_external_final_protocol_v1.json"
        ).read_text(encoding="utf-8")
    )
    registry = json.loads(
        (
            root / "configs/stage5_external_final_registry_v1.json"
        ).read_text(encoding="utf-8")
    )
    snapshot = json.loads(
        (
            root
            / "results/stage5/external_final_freeze_v1"
            / "code_snapshot.json"
        ).read_text(encoding="utf-8")
    )
    catalog = json.loads(
        (
            root
            / "results/stage5/external_final_catalog_v1"
            / "catalog.json"
        ).read_text(encoding="utf-8")
    )
    assert verify_code_snapshot(root, snapshot) == []
    assert validate_external_final_catalog(catalog, protocol, registry) == []
    assert catalog["scene_count"] == 200
    assert len({scene["cluster_id"] for scene in catalog["scenes"]}) == 79
    assert catalog["signal_values_loaded_for_selection"] is False
    assert catalog["method_outputs_loaded_for_selection"] is False
