import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from spectrum_semcom.aerpaw_spectrum import (
    AerpawPair,
    align_aerpaw_pairs_by_timestamp,
    build_permanently_excluded_pilot_manifest,
    channel_occupancy,
    channel_power_dbm,
    cleanest_contiguous_block,
    discover_aerpaw_pairs,
    discover_aerpaw_zip_pairs,
    extract_aerpaw_zip_pairs,
    extract_aligned_archive_scenes,
    filter_aerpaw_pairs_by_local_window,
    load_aerpaw_power_sweep,
    pair_datetime_utc,
    robust_power_threshold_dbm,
    select_frequency_band,
    thin_aligned_scenes_by_time,
    thin_aerpaw_pairs_by_time,
    validate_aligned_power_scene,
)


def write_pair(root: Path, timestamp: str, powers: np.ndarray | None = None, site: str = "LW1") -> Path:
    frequencies = np.asarray([100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7], dtype=float)
    powers = np.asarray(powers if powers is not None else [-120, -119, -80, -79, -110, -109, -70, -69], dtype="<f4")
    stem = f"results_{timestamp}"
    meta = root / f"{stem}.sigmf-meta"
    data = root / f"{stem}.sigmf-data"
    metadata = {
        "global": {
            "core:datatype": "rf32_le",
            "core:version": "1.2.5",
            "dataset:site": site,
            "dataset:frequency_axis_MHz": frequencies.tolist(),
            "dataset:frequency_span_MHz": [float(frequencies[0]), float(frequencies[-1])],
            "dataset:num_bins": int(frequencies.size),
        },
        "captures": [{"core:sample_start": 0, "core:datetime": "2022-02-01T00:00:00-05:00"}],
        "annotations": [{"core:sample_start": 0, "core:sample_count": int(frequencies.size)}],
    }
    meta.write_text(json.dumps(metadata), encoding="utf-8")
    powers.tofile(data)
    return meta


def test_load_real_power_sweep_without_iq_interpretation(tmp_path: Path) -> None:
    meta = write_pair(tmp_path, "20220201_000000")
    sweep = load_aerpaw_power_sweep(meta)
    assert sweep.site == "LW1"
    assert sweep.n_bins == 8
    assert sweep.powers_dbm.dtype == np.float32
    np.testing.assert_allclose(channel_power_dbm(sweep, 4), [-119.5, -79.5, -109.5, -69.5])
    np.testing.assert_allclose(channel_occupancy(sweep, 4, -100.0), [0.0, 1.0, 0.0, 1.0])
    assert cleanest_contiguous_block(channel_power_dbm(sweep, 4), 2) == (0, -99.5)


def test_loader_rejects_wrong_datatype_and_truncated_power_file(tmp_path: Path) -> None:
    meta = write_pair(tmp_path, "20220201_000000")
    metadata = json.loads(meta.read_text(encoding="utf-8"))
    metadata["global"]["core:datatype"] = "cf32_le"
    meta.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="requires rf32_le"):
        load_aerpaw_power_sweep(meta)
    metadata["global"]["core:datatype"] = "rf32_le"
    meta.write_text(json.dumps(metadata), encoding="utf-8")
    meta.with_name(meta.name.replace(".sigmf-meta", ".sigmf-data")).write_bytes(b"bad")
    with pytest.raises(ValueError, match="file size mismatch"):
        load_aerpaw_power_sweep(meta)


def test_discovery_and_pilot_manifest_are_leakage_safe(tmp_path: Path) -> None:
    for timestamp in ("20220201_000000", "20220201_010000", "20220201_020000"):
        write_pair(tmp_path, timestamp)
    mac = tmp_path / "__MACOSX"
    mac.mkdir()
    (mac / "._results_20220201_000000.sigmf-meta").write_text("{}", encoding="utf-8")
    pairs, errors = discover_aerpaw_pairs(tmp_path)
    assert errors == []
    assert len(pairs) == 3
    manifest = build_permanently_excluded_pilot_manifest(
        pairs, dataset_root=tmp_path, count=2, dataset_id="aerpaw-lw1-feb2022"
    )
    assert manifest["status"] == "pilot_only_permanently_excluded_from_final"
    assert manifest["labels_or_model_outputs_accessed"] is False
    assert manifest["final_access_consumed"] is False
    assert [row["scene_id"] for row in manifest["scenes"]] == [
        "aerpaw-lw1-feb2022:pilot:results_20220201_000000",
        "aerpaw-lw1-feb2022:pilot:results_20220201_020000",
    ]


def test_discovery_reports_missing_pair(tmp_path: Path) -> None:
    meta = write_pair(tmp_path, "20220201_000000")
    meta.with_name(meta.name.replace(".sigmf-meta", ".sigmf-data")).unlink()
    pairs, errors = discover_aerpaw_pairs(tmp_path)
    assert pairs == []
    assert any("missing data pair" in error for error in errors)


def test_band_selection_and_robust_threshold_are_deterministic(tmp_path: Path) -> None:
    meta = write_pair(tmp_path, "20220201_000000")
    sweep = load_aerpaw_power_sweep(meta)
    band = select_frequency_band(sweep, low_mhz=100.1, high_mhz=100.7)
    np.testing.assert_allclose(band.frequencies_mhz, [100.1, 100.2, 100.3, 100.4, 100.5, 100.6])
    expected_median = float(np.median(band.powers_dbm))
    expected_mad = float(np.median(np.abs(band.powers_dbm.astype(float) - expected_median)))
    assert robust_power_threshold_dbm(band) == pytest.approx(expected_median + 3.0 * 1.4826 * expected_mad)
    with pytest.raises(ValueError, match="fewer than two bins"):
        select_frequency_band(sweep, low_mhz=200.0, high_mhz=201.0)


def _pair(root: Path, timestamp: str, site: str) -> AerpawPair:
    meta = write_pair(root, timestamp, site=site)
    data = meta.with_name(meta.name.replace(".sigmf-meta", ".sigmf-data"))
    return AerpawPair(meta.stem.removesuffix(".sigmf"), meta, data, timestamp.replace("_", "T"))


def test_three_site_alignment_is_one_to_one_and_timezone_aware(tmp_path: Path) -> None:
    roots = {site: tmp_path / site for site in ("LW1", "CC1", "CC2")}
    for root in roots.values():
        root.mkdir()
    pairs = {
        "LW1": [_pair(roots["LW1"], "20220201_000000", "LW1"), _pair(roots["LW1"], "20220201_000030", "LW1")],
        "CC1": [_pair(roots["CC1"], "20220201_000002", "CC1"), _pair(roots["CC1"], "20220201_000031", "CC1")],
        "CC2": [_pair(roots["CC2"], "20220201_000004", "CC2"), _pair(roots["CC2"], "20220201_000029", "CC2")],
    }
    # Helper input uses the same naive ISO representation emitted by discovery.
    for site in pairs:
        pairs[site] = [
            AerpawPair(p.stem, p.meta_path, p.data_path, p.timestamp_local[:8] + "T" + p.timestamp_local[9:])
            for p in pairs[site]
        ]
    scenes, report = align_aerpaw_pairs_by_timestamp(
        pairs,
        expected_sites=("LW1", "CC1", "CC2"),
        anchor_site="LW1",
        max_offset_s=5.0,
    )
    assert len(scenes) == 2
    assert scenes[0].anchor_datetime_utc == "2022-02-01T05:00:00Z"
    assert scenes[0].offsets_from_anchor_s == {"LW1": 0.0, "CC1": 2.0, "CC2": 4.0}
    assert report["pair_reuse_used"] is False
    assert report["measurement_interpolation_used"] is False
    assert pair_datetime_utc(pairs["LW1"][0]).isoformat() == "2022-02-01T05:00:00+00:00"


def test_alignment_rejects_missing_site_and_thins_by_time(tmp_path: Path) -> None:
    roots = {site: tmp_path / site for site in ("LW1", "CC1", "CC2")}
    for root in roots.values():
        root.mkdir()
    pairs = {}
    for site in roots:
        rows = []
        for timestamp in ("20220201_000000", "20220201_000030", "20220201_010000"):
            p = _pair(roots[site], timestamp, site)
            rows.append(AerpawPair(p.stem, p.meta_path, p.data_path, timestamp[:8] + "T" + timestamp[9:]))
        pairs[site] = rows
    pairs["CC2"] = pairs["CC2"][1:]
    scenes, report = align_aerpaw_pairs_by_timestamp(
        pairs,
        expected_sites=("LW1", "CC1", "CC2"),
        anchor_site="LW1",
        max_offset_s=5.0,
    )
    assert report["rejected_anchor_count"] == 1
    assert len(scenes) == 2
    thinned = thin_aligned_scenes_by_time(scenes, min_separation_s=1800.0)
    assert len(thinned) == 2
    sweeps, errors = validate_aligned_power_scene(
        scenes[0], expected_sites=("LW1", "CC1", "CC2"), max_offset_s=5.0
    )
    assert errors == []
    assert set(sweeps) == {"LW1", "CC1", "CC2"}


def test_aligned_scene_validation_rejects_wrong_site_metadata(tmp_path: Path) -> None:
    roots = {site: tmp_path / site for site in ("LW1", "CC1", "CC2")}
    for root in roots.values():
        root.mkdir()
    pairs = {}
    for site in roots:
        # Deliberately write the CC2 file with the wrong site identifier.
        written_site = "CC1" if site == "CC2" else site
        p = _pair(roots[site], "20220201_000000", written_site)
        pairs[site] = [AerpawPair(p.stem, p.meta_path, p.data_path, "20220201T000000")]
    scenes, _ = align_aerpaw_pairs_by_timestamp(
        pairs,
        expected_sites=("LW1", "CC1", "CC2"),
        anchor_site="LW1",
        max_offset_s=0.0,
    )
    _, errors = validate_aligned_power_scene(
        scenes[0], expected_sites=("LW1", "CC1", "CC2"), max_offset_s=0.0
    )
    assert any("CC2 metadata reports site" in error for error in errors)


def test_zip_discovery_alignment_and_compact_extraction(tmp_path: Path) -> None:
    archives = {}
    zip_pairs = {}
    for site, offset in (("LW1", 0), ("CC1", 2), ("CC2", 4)):
        source = tmp_path / f"source-{site}"
        source.mkdir()
        timestamp = f"20220201_00000{offset}"
        meta = write_pair(source, timestamp, site=site)
        archive = tmp_path / f"{site}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            handle.write(meta, f"Results{site}/{meta.name}")
            data = meta.with_name(meta.name.replace(".sigmf-meta", ".sigmf-data"))
            handle.write(data, f"Results{site}/{data.name}")
            handle.writestr("__MACOSX/._ignored.sigmf-meta", b"ignored")
        archives[site] = archive
        zip_pairs[site], errors = discover_aerpaw_zip_pairs(archive)
        assert errors == []
        assert len(zip_pairs[site]) == 1
    scenes, report = align_aerpaw_pairs_by_timestamp(
        zip_pairs,
        expected_sites=("LW1", "CC1", "CC2"),
        anchor_site="LW1",
        max_offset_s=5.0,
    )
    assert report["aligned_scene_count"] == 1
    compact = extract_aligned_archive_scenes(
        scenes,
        destination=tmp_path / "compact",
        expected_sites=("LW1", "CC1", "CC2"),
    )
    sweeps, errors = validate_aligned_power_scene(
        compact[0], expected_sites=("LW1", "CC1", "CC2"), max_offset_s=5.0
    )
    assert errors == []
    assert set(sweeps) == {"LW1", "CC1", "CC2"}
    # Idempotent reuse of byte-identical compact files must also pass.
    extract_aligned_archive_scenes(
        scenes,
        destination=tmp_path / "compact",
        expected_sites=("LW1", "CC1", "CC2"),
    )


def test_zip_discovery_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../results_20220201_000000.sigmf-meta", b"{}")
    pairs, errors = discover_aerpaw_zip_pairs(archive)
    assert pairs == []
    assert any("unsafe ZIP member path" in error for error in errors)


def test_single_site_window_thinning_and_extraction(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    archive = tmp_path / "LW1.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for timestamp in ("20220220_140000", "20220220_140300", "20220220_140600"):
            meta = write_pair(source, timestamp, site="LW1")
            data = meta.with_name(meta.name.replace(".sigmf-meta", ".sigmf-data"))
            handle.write(meta, f"ResultsLW1/{meta.name}")
            handle.write(data, f"ResultsLW1/{data.name}")
    pairs, errors = discover_aerpaw_zip_pairs(archive)
    assert errors == []
    window = filter_aerpaw_pairs_by_local_window(
        pairs,
        start_local_inclusive="2022-02-20T14:01:00",
        end_local_exclusive="2022-02-20T14:07:00",
    )
    assert [pair.timestamp_local for pair in window] == ["2022-02-20T14:03:00", "2022-02-20T14:06:00"]
    thinned = thin_aerpaw_pairs_by_time(window, min_separation_s=180.0)
    extracted = extract_aerpaw_zip_pairs(thinned, destination=tmp_path / "compact", site="LW1")
    assert len(extracted) == 2
    assert all(pair.meta_path.is_file() and pair.data_path.is_file() for pair in extracted)
    # Reusing byte-identical extracted files is deterministic.
    assert len(extract_aerpaw_zip_pairs(thinned, destination=tmp_path / "compact", site="LW1")) == 2
