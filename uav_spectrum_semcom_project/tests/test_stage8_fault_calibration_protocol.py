import json
from pathlib import Path
from zipfile import ZipFile

from scripts.prepare_stage8_fault_archives import prepare


ROOT = Path(__file__).resolve().parents[1]


def test_stage8_fault_protocol_freezes_twenty_confirmation_units() -> None:
    config = json.loads(
        (ROOT / "configs/stage8_fault_calibration_protocol_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["sources"]["aadm"]["confirmation_flights"] == 20
    assert config["sources"]["alfa"]["raw_flights_expected"] == 39
    assert config["sources"]["alfa"]["development_raw_flights"] == 19
    assert config["sources"]["alfa"]["confirmation_raw_flights"] == 20
    assert (
        config["alfa_primary_metrics"][
            "processed_segments_from_same_raw_flight_are_not_independent"
        ]
        is True
    )
    assert (
        config["access_governance"][
            "confirmation_values_may_be_opened_before_model_freeze"
        ]
        is False
    )
    assert config["access_governance"]["post_confirmation_retuning"] is False
    assert config["access_governance"]["algorithm_external_final_claim_allowed"] is False


def test_prepare_stage8_archives_copies_nested_bytes_only(tmp_path: Path) -> None:
    source = tmp_path / "outer.zip"
    nested_bytes = b"PK\x05\x06" + b"\x00" * 18
    with ZipFile(source, "w") as archive:
        archive.writestr("nested.zip", nested_bytes)
        archive.writestr("README.md", "metadata")
    config = {
        "protocol_id": "test",
        "sources": {
            "demo": {
                "outer_archive": str(source),
                "outer_bytes": source.stat().st_size,
                "inner_archive_member": "nested.zip",
            }
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result = prepare(config_path, tmp_path / "data", tmp_path / "result.json")
    assert result["signal_values_accessed"] is False
    assert result["prepared"]["demo"]["nested_archives"][0]["bytes"] == len(
        nested_bytes
    )
