"""Safe metadata registration and role-isolated loading for ElectroSense PSD data."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable

import numpy as np

from spectrum_semcom.final_holdout import atomic_write_json


SPLIT_SALT = "stage6r-electrosense-site-split-v1"
DATASET_MARKER = "spectrum_bands_2"
FILENAME_PATTERN = re.compile(
    r"^(?:.*)?SpectrumBands_(?P<low>\d+)_(?P<high>\d+)_"
    r"(?P<technology>[A-Za-z0-9-]+)_.+\.npy$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class NpyMemberMetadata:
    member: str
    site: str
    date_group: str
    technology: str
    frequency_low_mhz: int
    frequency_high_mhz: int
    shape: tuple[int, ...]
    dtype: str
    fortran_order: bool
    member_size_bytes: int

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        return payload


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_member_path(member_name: str) -> tuple[str, str, int, int, str]:
    path = PurePosixPath(member_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe tar member path: {member_name}")
    try:
        marker_index = path.parts.index(DATASET_MARKER)
    except ValueError as exc:
        raise ValueError(f"dataset marker missing: {member_name}") from exc
    suffix = path.parts[marker_index + 1 :]
    if len(suffix) != 3:
        raise ValueError(f"unexpected dataset member layout: {member_name}")
    site, date_group, filename = suffix
    match = FILENAME_PATTERN.match(filename)
    if match is None:
        raise ValueError(f"unrecognized spectrum-band filename: {filename}")
    return (
        site,
        date_group,
        int(match.group("low")),
        int(match.group("high")),
        match.group("technology").lower(),
    )


def read_npy_header(handle: BinaryIO) -> tuple[tuple[int, ...], str, bool]:
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
            handle
        )
    elif version in {(2, 0), (3, 0)}:
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
            handle
        )
    else:
        raise ValueError(f"unsupported NPY version: {version}")
    if dtype.hasobject:
        raise ValueError("object arrays are not allowed")
    return tuple(int(value) for value in shape), dtype.str, bool(fortran_order)


def inventory_archive(archive_path: Path) -> list[NpyMemberMetadata]:
    rows: list[NpyMemberMetadata] = []
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".npy"):
                continue
            site, date_group, low, high, technology = parse_member_path(
                member.name
            )
            extracted = archive.extractfile(member)
            if extracted is None:
                raise OSError(f"cannot read NPY header: {member.name}")
            shape, dtype, fortran_order = read_npy_header(extracted)
            rows.append(
                NpyMemberMetadata(
                    member=member.name,
                    site=site,
                    date_group=date_group,
                    technology=technology,
                    frequency_low_mhz=low,
                    frequency_high_mhz=high,
                    shape=shape,
                    dtype=dtype,
                    fortran_order=fortran_order,
                    member_size_bytes=int(member.size),
                )
            )
    return sorted(rows, key=lambda row: row.member)


def is_valid_fm_candidate(
    row: NpyMemberMetadata,
    *,
    minimum_rows: int = 200,
    minimum_columns: int = 64,
) -> bool:
    return (
        row.technology == "fm"
        and len(row.shape) == 2
        and row.shape[0] >= minimum_rows
        and row.shape[1] >= minimum_columns
        and np.dtype(row.dtype).kind in {"f", "i", "u"}
    )


def select_site_members(
    inventory: Iterable[NpyMemberMetadata],
    *,
    technology: str = "fm",
    minimum_rows: int = 200,
    minimum_columns: int = 64,
) -> dict[str, NpyMemberMetadata]:
    candidates: dict[str, list[NpyMemberMetadata]] = {}
    for row in inventory:
        if technology != "fm":
            raise ValueError("Stage-6R v1 registry freezes technology='fm'")
        if is_valid_fm_candidate(
            row,
            minimum_rows=minimum_rows,
            minimum_columns=minimum_columns,
        ):
            candidates.setdefault(row.site, []).append(row)
    return {
        site: sorted(rows, key=lambda row: row.member)[0]
        for site, rows in sorted(candidates.items())
    }


def deterministic_site_split(
    sites: Iterable[str],
    *,
    split_salt: str = SPLIT_SALT,
    pilot_count: int = 4,
    final_count: int = 24,
    confirmation_count: int = 6,
) -> dict[str, list[str]]:
    unique_sites = sorted(set(sites))
    required = pilot_count + final_count + confirmation_count
    if len(unique_sites) < required:
        raise ValueError(
            f"need at least {required} eligible sites, got {len(unique_sites)}"
        )
    ordered = sorted(
        unique_sites,
        key=lambda site: hashlib.sha256(
            f"{split_salt}|{site}".encode("utf-8")
        ).hexdigest(),
    )
    pilot_stop = pilot_count
    final_stop = pilot_stop + final_count
    confirmation_stop = final_stop + confirmation_count
    return {
        "pilot": ordered[:pilot_stop],
        "stage6_final": ordered[pilot_stop:final_stop],
        "confirmation_lockbox": ordered[final_stop:confirmation_stop],
        "reserve": ordered[confirmation_stop:],
    }


def load_npy_member(archive_path: Path, member_name: str) -> np.ndarray:
    parse_member_path(member_name)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        try:
            member = archive.getmember(member_name)
        except KeyError as exc:
            raise FileNotFoundError(member_name) from exc
        extracted = archive.extractfile(member)
        if extracted is None:
            raise OSError(f"cannot extract member: {member_name}")
        payload = extracted.read()
    array = np.load(io.BytesIO(payload), allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(f"expected 2-D time-frequency array: {member_name}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"expected numeric PSD array: {member_name}")
    return np.asarray(array, dtype=np.float64)


def load_npy_members(
    archive_path: Path, member_names: Iterable[str]
) -> dict[str, np.ndarray]:
    requested = tuple(member_names)
    if len(requested) != len(set(requested)) or not requested:
        raise ValueError("unique member names are required")
    for member_name in requested:
        parse_member_path(member_name)
    pending = set(requested)
    loaded: dict[str, np.ndarray] = {}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if member.name not in pending:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise OSError(f"cannot extract member: {member.name}")
            array = np.load(
                io.BytesIO(extracted.read()), allow_pickle=False
            )
            if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
                raise ValueError(
                    f"expected numeric 2-D PSD array: {member.name}"
                )
            loaded[member.name] = np.asarray(array, dtype=np.float64)
            pending.remove(member.name)
            if not pending:
                break
    if pending:
        raise FileNotFoundError(
            "archive members missing: " + ", ".join(sorted(pending))
        )
    return {member: loaded[member] for member in requested}


def aggregate_frequency_bins(power: np.ndarray, n_channels: int) -> np.ndarray:
    values = np.asarray(power, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("power must have shape [scene, frequency_bin]")
    if not np.all(np.isfinite(values)):
        raise ValueError("power contains non-finite values")
    if n_channels < 1 or n_channels > values.shape[1]:
        raise ValueError("n_channels must be within the input bin count")
    edges = np.linspace(0, values.shape[1], n_channels + 1, dtype=np.int64)
    if np.any(np.diff(edges) < 1):
        raise ValueError("every aggregate channel must contain at least one bin")
    return np.column_stack(
        [
            np.mean(values[:, edges[index] : edges[index + 1]], axis=1)
            for index in range(n_channels)
        ]
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def role_members(registry: dict, role: str) -> list[dict]:
    roles = registry["split"]["roles"]
    if role not in roles:
        raise KeyError(f"unknown registry role: {role}")
    selected = registry["selected_site_members"]
    return [selected[site] for site in roles[role]]


def claim_role_access(
    *,
    registry_path: Path,
    access_state_path: Path,
    role: str,
    actor: str,
    purpose: str,
    evidence: dict | None = None,
) -> dict:
    registry = read_json(registry_path)
    state = read_json(access_state_path)
    if state.get("registry_sha256") != sha256_file(registry_path):
        raise ValueError("registry hash does not match the access ledger")
    roles = state.get("roles", {})
    if role not in roles:
        raise KeyError(f"unknown access role: {role}")
    role_state = roles[role]
    if role_state.get("access_count") != 0:
        raise ValueError(f"signal access for role '{role}' is already consumed")
    timestamp = datetime.now(timezone.utc).isoformat()
    role_state.update(
        {
            "access_count": 1,
            "signal_values_accessed": True,
            "first_access_at_utc": timestamp,
            "actor": actor,
            "declared_purpose": purpose,
            "member_count": len(role_members(registry, role)),
            "evidence": {} if evidence is None else evidence,
        }
    )
    atomic_write_json(access_state_path, state)
    return {
        "role": role,
        "access_number": 1,
        "first_access_at_utc": timestamp,
        "actor": actor,
        "purpose": purpose,
        "registry_sha256": state["registry_sha256"],
    }
