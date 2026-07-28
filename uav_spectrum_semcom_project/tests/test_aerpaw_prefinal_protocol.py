import json
from itertools import permutations
from pathlib import Path

from spectrum_semcom.aerpaw_spectrum import sha256_file


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return json.loads((PROJECT_DIR / relative).read_text(encoding="utf-8"))


def test_three_site_prefinal_protocol_is_frozen_and_claim_limited() -> None:
    protocol = load("configs/aerpaw_three_site_prefinal_protocol_v1.json")
    assert protocol["sites"] == ["LW1", "CC1", "CC2"]
    assert protocol["resource_proxy"]["frequency_low_mhz_inclusive"] == 2400.0
    assert protocol["resource_proxy"]["frequency_high_mhz_exclusive"] == 2483.5
    assert protocol["resource_proxy"]["n_equal_width_channels"] == 8
    assert protocol["resource_proxy"]["channel_width_mhz"] == (2483.5 - 2400.0) / 8
    threshold = protocol["resource_proxy"]["threshold_rule"]
    assert threshold["estimator"] == "median_plus_scaled_MAD"
    assert threshold["uses_only_current_sweep"] is True
    assert threshold["uses_labels_or_model_outputs"] is False
    assert protocol["scene_independence"]["minimum_utc_separation_s"] >= 1800
    assert protocol["scene_independence"]["minimum_scene_count"] >= 200
    assert protocol["pre_final_rules"]["C1_C2_C3_weight_or_checkpoint_changes_allowed"] is False
    assert "not proof of H3" in protocol["claim_families"]["selective_G2_digital_reporting"]["claim_limit"]


def test_official_archive_contract_and_three_site_link_are_complete() -> None:
    protocol = load("configs/aerpaw_three_site_prefinal_protocol_v1.json")
    for site in protocol["sites"]:
        archive = protocol["official_archives"][site]
        assert archive["size_bytes"] > 10_000_000_000
        assert len(archive["sha256"]) == 64
        int(archive["sha256"], 16)
    link = protocol["three_site_reporting_link"]
    offsets = tuple(link["relative_offset_values_db"])
    assert sorted(offsets) == [-1.0, 0.0, 1.0]
    assert len(set(permutations(offsets))) == 6
    assert link["site_offset_assignment"] == "all_six_permutations_nested_within_scene"


def test_pilot_and_synthetic_dry_run_match_current_protocol_without_final_access() -> None:
    protocol_path = PROJECT_DIR / "configs/aerpaw_three_site_prefinal_protocol_v1.json"
    expected = sha256_file(protocol_path)
    pilot = load("results/stage4/aerpaw_pilot_resource_proxy_v1/pilot_resource_proxy_result.json")
    dry_run = load("results/stage4/aerpaw_three_site_dry_run_v1/three_site_dry_run_result.json")
    access = load("configs/stage4_final_access_state.json")
    assert pilot["protocol_sha256"] == expected
    assert pilot["pilot_scene_count"] == 20
    assert pilot["sanity_checks"]["model_imported_or_executed"] is False
    assert pilot["final_access_consumed"] is False
    assert dry_run["protocol_sha256"] == expected
    assert dry_run["site_count"] == 3
    assert dry_run["base_occupancy_shape"] == [3, 8]
    assert dry_run["selective_G2_interface"]["offset_permutation_count"] == 6
    assert dry_run["selective_G2_interface"]["all_runs_completed"] is True
    assert dry_run["nonpilot_real_data_accessed"] is False
    assert dry_run["final_access_consumed"] is False
    assert access["version"] == "1.0"
    assert access["access_count"] in {0, 1}
    if access["access_count"] == 0:
        assert access["status"] == "not_accessed" and access["receipt"] is None
    else:
        assert access["status"] == "access_consumed"
        assert access["receipt"]["access_number"] == 1
        assert len(access["receipt"]["registry_sha256"]) == 64
        assert len(access["receipt"]["code_snapshot_sha256"]) == 64
