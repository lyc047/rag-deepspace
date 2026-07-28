import hashlib
import json
from pathlib import Path

import numpy as np

from spectrum_semcom.aerpaw_helikite import (
    crop_helikite_sweep,
    discover_helikite_power_pairs,
    infer_integer_hour_clock_correction,
    load_helikite_position_log,
    load_helikite_power_sweep,
    nearest_helikite_position,
)


def write_sigmf(
    root: Path,
    name: str,
    values: np.ndarray,
    datatype: str,
    annotations: list[dict],
) -> Path:
    meta = root / f"{name}.sigmf-meta"
    data = root / f"{name}.sigmf-data"
    data.write_bytes(values.tobytes())
    metadata = {
        "global": {
            "core:datatype": datatype,
            "core:sha512": hashlib.sha512(data.read_bytes()).hexdigest(),
        },
        "captures": [{"core:sample_start": 0}],
        "annotations": annotations,
    }
    meta.write_text(json.dumps(metadata), encoding="utf-8")
    return meta


def test_helikite_power_adapter_and_crop(tmp_path: Path) -> None:
    frequencies = np.linspace(3500.0, 3800.0, 16, dtype="<f4")
    powers = np.linspace(-130.0, -90.0, 16, dtype="<f4")
    meta = write_sigmf(
        tmp_path,
        "spec_results_20230826_120152",
        np.concatenate([frequencies, powers]),
        "rf32_le",
        [
            {"core:comment": "freqs", "core:sample_start": 0, "core:sample_count": 16},
            {"core:comment": "powers", "core:sample_start": 16, "core:sample_count": 16},
        ],
    )
    pairs, errors = discover_helikite_power_pairs(tmp_path)
    assert not errors and len(pairs) == 1
    sweep = load_helikite_power_sweep(pairs[0], site="pilot")
    assert sweep.n_bins == 16
    cropped = crop_helikite_sweep(sweep, low_mhz=3550.0, high_mhz=3700.0)
    assert cropped.n_bins > 2
    assert cropped.frequencies_mhz[0] >= 3550.0
    assert meta == pairs[0].meta_path


def test_helikite_position_adapter_and_nearest(tmp_path: Path) -> None:
    n = 3
    values = np.concatenate(
        [
            np.asarray([-78.0, -78.1, -78.2], dtype="<f8"),
            np.asarray([35.0, 35.1, 35.2], dtype="<f8"),
            np.asarray([10.0, 20.0, 30.0], dtype="<f8"),
            np.asarray([100.0, 101.0, 102.0], dtype="<f8"),
        ]
    )
    meta = write_sigmf(
        tmp_path,
        "position",
        values,
        "rf64_le",
        [
            {"core:comment": name, "core:sample_start": index * n, "core:sample_count": n}
            for index, name in enumerate(("longitude", "latitude", "altitude", "timestamp"))
        ],
    )
    log = load_helikite_position_log(meta)
    selected = nearest_helikite_position(log, 100.6)
    assert selected["altitude_m"] == 20.0
    assert abs(selected["offset_s"] - 0.4) < 1e-12
    inferred = infer_integer_hour_clock_correction(
        log, np.asarray([100.0 + 5 * 3600, 102.0 + 5 * 3600])
    )
    assert inferred["correction_hours"] == 5.0
    assert inferred["maximum_absolute_residual_s"] == 0.0
