"""Freeze flight-level Stage-8 development/confirmation splits before signal access."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
AADM_USRP_PREFIX = "AADM2025Dryad/USRP/Testbed 33 flights Sep/"
AADM_LORA_PREFIX = "AADM2025Dryad/LORA/"
ALFA_RAW_PATTERN = re.compile(r"(carbonZ_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_split(
    unit_ids: list[str], salt: str, development_count: int
) -> tuple[list[str], list[str]]:
    ranked = sorted(
        unit_ids,
        key=lambda unit_id: (
            hashlib.sha256(f"{salt}|{unit_id.lower()}".encode("utf-8")).hexdigest(),
            unit_id,
        ),
    )
    return ranked[:development_count], ranked[development_count:]


def directory_units(names: list[str], prefix: str) -> dict[str, list[str]]:
    units: dict[str, list[str]] = defaultdict(list)
    for name in names:
        if not name.startswith(prefix):
            continue
        relative = name[len(prefix) :]
        if not relative or "/" not in relative:
            continue
        unit_id = relative.split("/", 1)[0]
        units[unit_id].append(name)
    return dict(units)


def freeze(config_path: Path, preparation_path: Path, output_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    prepared = preparation["prepared"]

    aadm_nested = Path(prepared["aadm"]["nested_archives"][0]["path"])
    with ZipFile(aadm_nested) as archive:
        aadm_names = [item.filename for item in archive.infolist()]
    usrp_units = directory_units(aadm_names, AADM_USRP_PREFIX)
    lora_inventory = directory_units(aadm_names, AADM_LORA_PREFIX)
    aadm_ids = sorted(usrp_units)
    expected_aadm = int(config["sources"]["aadm"]["official_testbed_flights_expected"])
    if len(aadm_ids) != expected_aadm:
        raise ValueError(f"expected {expected_aadm} AADM flights, found {len(aadm_ids)}")
    missing_lora = sorted(set(aadm_ids) - set(lora_inventory))
    if missing_lora:
        raise ValueError(f"AADM flights missing matching LoRa directories: {missing_lora}")
    aadm_dev, aadm_confirm = stable_split(
        aadm_ids,
        "stage8-aadm-v1",
        int(config["sources"]["aadm"]["development_flights"]),
    )

    alfa_archives = {
        item["member"]: Path(item["path"])
        for item in prepared["alfa"]["nested_archives"]
    }
    with ZipFile(alfa_archives["raw.zip"]) as archive:
        raw_names = [item.filename for item in archive.infolist() if not item.is_dir()]
    alfa_raw_ids = []
    raw_members: dict[str, str] = {}
    for name in raw_names:
        match = ALFA_RAW_PATTERN.search(name)
        if not match:
            continue
        raw_id = match.group(1)
        alfa_raw_ids.append(raw_id)
        raw_members[raw_id] = name
    alfa_raw_ids = sorted(set(alfa_raw_ids))
    expected_alfa = int(config["sources"]["alfa"]["raw_flights_expected"])
    if len(alfa_raw_ids) != expected_alfa:
        raise ValueError(f"expected {expected_alfa} ALFA raw flights, found {len(alfa_raw_ids)}")

    with ZipFile(alfa_archives["processed.zip"]) as archive:
        processed_names = [
            item.filename for item in archive.infolist() if not item.is_dir()
        ]
    processed_dirs: dict[str, set[str]] = defaultdict(set)
    processed_member_counts: Counter[str] = Counter()
    unmapped_processed = []
    for name in processed_names:
        parts = name.split("/")
        sequence_dir = parts[1] if len(parts) > 2 else ""
        match = ALFA_RAW_PATTERN.search(sequence_dir)
        if not match:
            unmapped_processed.append(name)
            continue
        raw_id = match.group(1)
        processed_dirs[raw_id].add(sequence_dir)
        processed_member_counts[sequence_dir] += 1

    unknown_processed_raw_ids = sorted(set(processed_dirs) - set(alfa_raw_ids))
    if unknown_processed_raw_ids:
        raise ValueError(
            "processed sequences lack matching raw flights: "
            f"{unknown_processed_raw_ids}"
        )
    alfa_dev, alfa_confirm = stable_split(
        alfa_raw_ids,
        "stage8-alfa-v1",
        int(config["sources"]["alfa"]["development_raw_flights"]),
    )

    auxiliary_inventory: dict[str, Any] = {}
    for archive_name in ("telemetry.zip", "dataflash.zip"):
        with ZipFile(alfa_archives[archive_name]) as archive:
            members = [item.filename for item in archive.infolist() if not item.is_dir()]
        top_groups = Counter(
            "/".join(name.split("/")[:2]) if "/" in name else name for name in members
        )
        auxiliary_inventory[archive_name] = {
            "member_count": len(members),
            "top_group_counts": dict(sorted(top_groups.items())),
            "raw_flight_mapping_status": "UNRESOLVED_METADATA_ONLY",
            "access_policy": (
                "No payload may be opened until an auditable raw-flight mapping or "
                "a development-only exploratory role is frozen."
            ),
        }

    result = {
        "version": "1.0",
        "status": "FROZEN_BEFORE_SIGNAL_ACCESS",
        "protocol_id": config["protocol_id"],
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "preparation_result_path": str(preparation_path),
        "preparation_result_sha256": sha256(preparation_path),
        "signal_access": {
            "development_value_access_count": 0,
            "confirmation_value_access_count": 0,
            "archive_member_names_accessed": True,
        },
        "aadm": {
            "independent_unit": "official physical-testbed flight",
            "population_count": len(aadm_ids),
            "development_count": len(aadm_dev),
            "confirmation_count": len(aadm_confirm),
            "development_flight_ids": aadm_dev,
            "confirmation_flight_ids": aadm_confirm,
            "flight_inventory": {
                flight_id: {
                    "usrp_member_count": len(usrp_units[flight_id]),
                    "lora_member_count": len(lora_inventory[flight_id]),
                }
                for flight_id in aadm_ids
            },
            "archive_sha256": prepared["aadm"]["nested_archives"][0]["sha256"],
        },
        "alfa": {
            "independent_unit": "raw flight",
            "population_count": len(alfa_raw_ids),
            "development_count": len(alfa_dev),
            "confirmation_count": len(alfa_confirm),
            "development_raw_flight_ids": alfa_dev,
            "confirmation_raw_flight_ids": alfa_confirm,
            "raw_flight_inventory": {
                raw_id: {
                    "raw_member": raw_members[raw_id],
                    "processed_sequence_directories": sorted(processed_dirs.get(raw_id, set())),
                    "processed_sequence_member_counts": {
                        sequence: processed_member_counts[sequence]
                        for sequence in sorted(processed_dirs.get(raw_id, set()))
                    },
                }
                for raw_id in alfa_raw_ids
            },
            "unmapped_processed_member_count": len(unmapped_processed),
            "auxiliary_archives": auxiliary_inventory,
            "archive_sha256": {
                item["member"]: item["sha256"]
                for item in prepared["alfa"]["nested_archives"]
            },
        },
        "integrity_checks": {
            "aadm_development_confirmation_disjoint": not (
                set(aadm_dev) & set(aadm_confirm)
            ),
            "alfa_development_confirmation_disjoint": not (
                set(alfa_dev) & set(alfa_confirm)
            ),
            "all_processed_segments_inherit_raw_flight_split": True,
            "confirmation_payload_unread": True,
        },
        "claim_boundary": [
            "The split was derived from archive member names only.",
            "No AADM or ALFA signal value was opened while freezing this manifest.",
            "ALFA processed segments are grouped by their raw-flight timestamp and are not independent units.",
            "Telemetry and dataflash files remain mapping-unresolved and payload-unread.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/stage8_fault_calibration_protocol_v1.json",
    )
    parser.add_argument(
        "--preparation",
        type=Path,
        default=ROOT
        / "results/stage8/stage8_fault_archive_preparation_v1/result.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/stage8/stage8_fault_split_freeze_v1/result.json",
    )
    args = parser.parse_args()
    result = freeze(args.config, args.preparation, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "aadm": [
                    result["aadm"]["development_count"],
                    result["aadm"]["confirmation_count"],
                ],
                "alfa": [
                    result["alfa"]["development_count"],
                    result["alfa"]["confirmation_count"],
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
