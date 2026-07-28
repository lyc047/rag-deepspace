import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_c1_v4_final_protocol_is_time_separated_and_single_use() -> None:
    protocol = json.loads((ROOT / "configs/stage4_c1_v4_final_protocol.json").read_text(encoding="utf-8"))
    state = json.loads((ROOT / "configs/stage4_c1_v4_final_access_state.json").read_text(encoding="utf-8"))
    assert protocol["final_partition"]["start_local_inclusive"] == "2022-02-14T00:00:00"
    assert protocol["final_partition"]["selected_per_site"] == 140
    assert set(protocol["archives"]) == {"CC1", "CC2"}
    assert protocol["frozen_methods"]["indicator_guarded_last_success"]["max_state_age_minutes"] == 60.0
    assert protocol["frozen_methods"]["indicator_guarded_last_success"]["additional_transmitted_bits"] == 0
    assert state["access_count"] in {0, 1}
    if state["access_count"] == 0:
        assert state["status"] == "not_accessed" and state["receipt"] is None
    else:
        assert state["status"] == "access_consumed" and len(state["receipt_sha256"]) == 64


def test_final_window_is_absent_from_every_declared_prior_provenance() -> None:
    protocol = json.loads((ROOT / "configs/stage4_c1_v4_final_protocol.json").read_text(encoding="utf-8"))
    for value in protocol["permanent_exclusion_artifacts"]:
        path = Path(value)
        path = path if path.is_absolute() else ROOT / path
        text = path.read_text(encoding="utf-8")
        assert "results_20220214_" not in text
        assert "results_20220215_" not in text
