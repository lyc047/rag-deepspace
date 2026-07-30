"""Prepare nested Stage-8 archives without reading signal values."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_member(archive: Path, member: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as source_zip:
        info = source_zip.getinfo(member)
        if destination.exists() and destination.stat().st_size == info.file_size:
            return
        with source_zip.open(info) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)


def prepare(config_path: Path, data_root: Path, output_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prepared: dict[str, Any] = {}
    for source_id, source in config["sources"].items():
        outer = Path(source["outer_archive"])
        if not outer.exists():
            raise FileNotFoundError(outer)
        if outer.stat().st_size != int(source["outer_bytes"]):
            raise ValueError(f"{source_id} outer archive size mismatch")

        members = (
            [source["inner_archive_member"]]
            if "inner_archive_member" in source
            else list(source["inner_archive_members"])
        )
        source_root = data_root / source_id
        extracted = []
        for member in members:
            destination = source_root / Path(member).name
            copy_member(outer, member, destination)
            extracted.append(
                {
                    "member": member,
                    "path": str(destination),
                    "bytes": destination.stat().st_size,
                    "sha256": sha256(destination),
                }
            )

        with ZipFile(outer) as source_zip:
            outer_directory = [
                {"member": item.filename, "bytes": item.file_size}
                for item in source_zip.infolist()
            ]
        prepared[source_id] = {
            "outer_archive": str(outer),
            "outer_bytes": outer.stat().st_size,
            "outer_sha256": sha256(outer),
            "outer_directory": outer_directory,
            "nested_archives": extracted,
        }

    result = {
        "version": "1.0",
        "status": "STAGE8_NESTED_ARCHIVES_PREPARED",
        "protocol_id": config["protocol_id"],
        "signal_values_accessed": False,
        "archive_member_names_accessed": True,
        "prepared": prepared,
        "claim_boundary": [
            "Only outer archive directories and complete nested archive bytes were copied.",
            "No nested archive member payload was opened.",
            "Development and confirmation signal access counts remain zero."
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
        "--data-root",
        type=Path,
        default=Path(r"F:\uav_spectrum_stage8_fault_data\source_archives"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/stage8/stage8_fault_archive_preparation_v1/result.json",
    )
    args = parser.parse_args()
    result = prepare(args.config, args.data_root, args.output)
    print(json.dumps({"status": result["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
