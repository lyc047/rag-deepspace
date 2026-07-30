import hashlib
import io
import tarfile
from pathlib import Path

import numpy as np
import pytest

from spectrum_semcom.electrosense_psd import (
    NpyMemberMetadata,
    aggregate_frequency_bins,
    claim_role_access,
    deterministic_site_split,
    inventory_archive,
    load_npy_member,
    load_npy_members,
    parse_member_path,
    select_site_members,
)
from spectrum_semcom.final_holdout import atomic_write_json


def _member(site: str, shape: tuple[int, int], *, date: str = "Aug_1"):
    return NpyMemberMetadata(
        member=(
            "root/spectrum_bands_2/"
            f"{site}/{date}/SpectrumBands_87_108_fm_X_87_108.npy"
        ),
        site=site,
        date_group=date,
        technology="fm",
        frequency_low_mhz=87,
        frequency_high_mhz=108,
        shape=shape,
        dtype="<f8",
        fortran_order=False,
        member_size_bytes=100,
    )


def _write_test_archive(path: Path, array: np.ndarray) -> str:
    member_name = (
        "root/spectrum_bands_2/site_a/Aug_1/"
        "SpectrumBands_87_108_fm_X_87_108.npy"
    )
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    payload = buffer.getvalue()
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(member_name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return member_name


def test_parse_member_path_and_reject_traversal():
    parsed = parse_member_path(
        "root/spectrum_bands_2/site/Aug_1/"
        "SpectrumBands_80_110_fm_Swis_80_110.npy"
    )
    assert parsed == ("site", "Aug_1", 80, 110, "fm")
    assert parse_member_path(
        "root/spectrum_bands_2/site/Feb_3/"
        "site_Feb_3_21SpectrumBands_791_821_lte_X_791_821.npy"
    ) == ("site", "Feb_3", 791, 821, "lte")
    with pytest.raises(ValueError):
        parse_member_path(
            "root/spectrum_bands_2/../Aug_1/"
            "SpectrumBands_80_110_fm_X_80_110.npy"
        )


def test_inventory_reads_header_and_loader_reads_values(tmp_path):
    archive = tmp_path / "test.tar.gz"
    values = np.arange(24, dtype=np.float64).reshape(4, 6)
    member_name = _write_test_archive(archive, values)
    rows = inventory_archive(archive)
    assert len(rows) == 1
    assert rows[0].shape == (4, 6)
    assert rows[0].dtype == "<f8"
    assert np.array_equal(load_npy_member(archive, member_name), values)
    assert np.array_equal(
        load_npy_members(archive, [member_name])[member_name], values
    )


def test_site_selection_is_lexicographic_and_shape_filtered():
    selected = select_site_members(
        [
            _member("a", (199, 100)),
            _member("b", (200, 64), date="Sep_1"),
            _member("b", (201, 64), date="Aug_1"),
        ]
    )
    assert list(selected) == ["b"]
    assert "/Aug_1/" in selected["b"].member


def test_deterministic_split_matches_frozen_hash_order():
    sites = [f"site_{index}" for index in range(40)]
    split = deterministic_site_split(sites)
    ordered = sorted(
        sites,
        key=lambda site: hashlib.sha256(
            f"stage6r-electrosense-site-split-v1|{site}".encode()
        ).hexdigest(),
    )
    assert split["pilot"] == ordered[:4]
    assert split["stage6_final"] == ordered[4:28]
    assert split["confirmation_lockbox"] == ordered[28:34]
    assert split["reserve"] == ordered[34:]
    assert len({site for values in split.values() for site in values}) == 40


def test_frequency_aggregation_preserves_scene_order():
    values = np.asarray([[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]])
    aggregated = aggregate_frequency_bins(values, 2)
    assert np.array_equal(aggregated, np.asarray([[2.0, 6.0], [3.0, 7.0]]))
    with pytest.raises(ValueError):
        aggregate_frequency_bins(np.asarray([[1.0, np.nan]]), 1)


def test_role_access_claim_is_single_use(tmp_path):
    registry_path = tmp_path / "registry.json"
    state_path = tmp_path / "state.json"
    registry = {
        "split": {"roles": {"pilot": ["site_a"]}},
        "selected_site_members": {"site_a": {"site": "site_a"}},
    }
    atomic_write_json(registry_path, registry)
    from spectrum_semcom.electrosense_psd import sha256_file

    atomic_write_json(
        state_path,
        {
            "registry_sha256": sha256_file(registry_path),
            "roles": {
                "pilot": {
                    "access_count": 0,
                    "signal_values_accessed": False,
                }
            },
        },
    )
    receipt = claim_role_access(
        registry_path=registry_path,
        access_state_path=state_path,
        role="pilot",
        actor="test",
        purpose="test",
    )
    assert receipt["access_number"] == 1
    with pytest.raises(ValueError):
        claim_role_access(
            registry_path=registry_path,
            access_state_path=state_path,
            role="pilot",
            actor="test",
            purpose="test",
        )
