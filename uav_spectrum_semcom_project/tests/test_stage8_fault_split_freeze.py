import json
from pathlib import Path
from zipfile import ZipFile

from scripts.freeze_stage8_fault_splits import freeze, stable_split


def test_stable_split_is_deterministic_and_disjoint() -> None:
    units = [f"flight-{index}" for index in range(33)]
    development, confirmation = stable_split(units, "salt", 13)
    repeated = stable_split(list(reversed(units)), "salt", 13)
    assert (development, confirmation) == repeated
    assert len(development) == 13
    assert len(confirmation) == 20
    assert not (set(development) & set(confirmation))


def test_freeze_groups_processed_segments_by_raw_flight(tmp_path: Path) -> None:
    aadm = tmp_path / "aadm.zip"
    with ZipFile(aadm, "w") as archive:
        for index in range(2):
            flight = f"f{index}"
            archive.writestr(
                f"AADM2025Dryad/USRP/Testbed 33 flights Sep/{flight}/power.txt",
                "unread",
            )
            archive.writestr(
                f"AADM2025Dryad/LORA/{flight}/LORAlog.csv",
                "unread",
            )

    raw_ids = ["carbonZ_2018-01-01-00-00-01", "carbonZ_2018-01-01-00-00-02"]
    archives = {}
    for archive_name in ("raw.zip", "processed.zip", "telemetry.zip", "dataflash.zip"):
        path = tmp_path / archive_name
        archives[archive_name] = path
        with ZipFile(path, "w") as archive:
            if archive_name == "raw.zip":
                for raw_id in raw_ids:
                    archive.writestr(f"raw/{raw_id}.bag", "unread")
            elif archive_name == "processed.zip":
                for raw_id in raw_ids:
                    for suffix in ("engine", "elevator"):
                        sequence = f"{raw_id}_{suffix}"
                        archive.writestr(
                            f"processed/{sequence}/{sequence}-mavros-state.csv",
                            "unread",
                        )
            else:
                archive.writestr(f"{archive_name}/metadata.bin", "unread")

    def prepared_item(member: str, path: Path) -> dict[str, object]:
        return {
            "member": member,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": "fixture",
        }

    preparation = {
        "prepared": {
            "aadm": {"nested_archives": [prepared_item("AADM2025Dryad.zip", aadm)]},
            "alfa": {
                "nested_archives": [
                    prepared_item(name, path) for name, path in archives.items()
                ]
            },
        }
    }
    preparation_path = tmp_path / "preparation.json"
    preparation_path.write_text(json.dumps(preparation), encoding="utf-8")
    config = {
        "protocol_id": "fixture",
        "sources": {
            "aadm": {
                "official_testbed_flights_expected": 2,
                "development_flights": 1,
            },
            "alfa": {
                "raw_flights_expected": 2,
                "development_raw_flights": 1,
            },
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = freeze(config_path, preparation_path, tmp_path / "result.json")
    assert result["status"] == "FROZEN_BEFORE_SIGNAL_ACCESS"
    assert result["signal_access"]["confirmation_value_access_count"] == 0
    assert result["integrity_checks"]["confirmation_payload_unread"] is True
    assert {
        len(item["processed_sequence_directories"])
        for item in result["alfa"]["raw_flight_inventory"].values()
    } == {2}
