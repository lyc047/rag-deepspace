from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .types import IQFrame, SignalBox


def read_sigmf_meta(meta_path: str | Path) -> Dict[str, Any]:
    """Read a SigMF metadata JSON file."""

    path = Path(meta_path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sigmf_data_path(meta_path: Path) -> Path:
    if meta_path.name.endswith(".sigmf-meta"):
        return meta_path.with_name(meta_path.name.replace(".sigmf-meta", ".sigmf-data"))
    return meta_path.with_suffix(".sigmf-data")


def _dtype_and_converter(datatype: str) -> Tuple[np.dtype, int, Any]:
    """Return raw dtype, scalars per complex sample, and converter function."""

    datatype = datatype.lower()
    if datatype == "ci32_le":
        return np.dtype("<i4"), 2, lambda x: (x[:, 0].astype(np.float32) + 1j * x[:, 1].astype(np.float32))
    if datatype == "ci16_le":
        return np.dtype("<i2"), 2, lambda x: (x[:, 0].astype(np.float32) + 1j * x[:, 1].astype(np.float32))
    if datatype == "cf32_le":
        return np.dtype("<f4"), 2, lambda x: (x[:, 0].astype(np.float32) + 1j * x[:, 1].astype(np.float32))
    if datatype == "cf32":
        return np.dtype("float32"), 2, lambda x: (x[:, 0].astype(np.float32) + 1j * x[:, 1].astype(np.float32))
    raise ValueError(f"Unsupported SigMF datatype: {datatype}")


def _load_sigmf_iq(
    data_path: Path,
    datatype: str,
    start_sample: int = 0,
    max_samples: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    raw_dtype, scalars_per_complex, converter = _dtype_and_converter(datatype)
    bytes_per_complex = raw_dtype.itemsize * scalars_per_complex
    file_size = data_path.stat().st_size
    total_samples = file_size // bytes_per_complex

    if file_size % bytes_per_complex != 0:
        raise ValueError(f"SigMF data file size is not divisible by sample size: {file_size} bytes")
    if start_sample < 0 or start_sample >= total_samples:
        raise ValueError(f"start_sample={start_sample} outside available samples={total_samples}")

    if max_samples is None:
        n_samples = total_samples - start_sample
    else:
        n_samples = min(int(max_samples), total_samples - start_sample)

    offset = start_sample * bytes_per_complex
    count = n_samples * scalars_per_complex
    raw = np.fromfile(data_path, dtype=raw_dtype, count=count, offset=offset)
    raw = raw.reshape(-1, scalars_per_complex)
    iq = converter(raw).astype(np.complex64)

    if normalize and iq.size:
        scale = float(np.max(np.abs(iq)))
        if scale > 0:
            iq = (iq / scale).astype(np.complex64)
    return iq


def _annotations_to_boxes(
    meta: Dict[str, Any],
    start_sample: int,
    n_samples: int,
    sample_rate_hz: float,
    center_freq_hz: float,
) -> List[SignalBox]:
    boxes: List[SignalBox] = []
    slice_start = start_sample
    slice_end = start_sample + n_samples

    for ann in meta.get("annotations", []):
        ann_start = int(ann.get("core:sample_start", 0))
        ann_count = int(ann.get("core:sample_count", 0))
        ann_end = ann_start + ann_count
        if ann_count <= 0 or ann_end <= slice_start or ann_start >= slice_end:
            continue

        clipped_start = max(ann_start, slice_start)
        clipped_end = min(ann_end, slice_end)
        label = str(ann.get("core:label", "unknown") or "unknown")

        lower_abs = ann.get("core:freq_lower_edge")
        upper_abs = ann.get("core:freq_upper_edge")
        if lower_abs is None or upper_abs is None:
            f_low = -sample_rate_hz / 2.0
            f_high = sample_rate_hz / 2.0
        else:
            f_low = float(lower_abs) - center_freq_hz
            f_high = float(upper_abs) - center_freq_hz

        boxes.append(
            SignalBox(
                label=label,
                t_start_s=(clipped_start - slice_start) / sample_rate_hz,
                t_end_s=(clipped_end - slice_start) / sample_rate_hz,
                f_low_hz=min(f_low, f_high),
                f_high_hz=max(f_low, f_high),
                confidence=1.0,
            )
        )
    return boxes


def load_sigmf_frame(
    meta_path: str | Path,
    start_sample: int = 0,
    max_samples: int | None = None,
    normalize: bool = True,
    frame_id: str | None = None,
) -> IQFrame:
    """Load a SigMF recording or slice as an IQFrame.

    The returned annotation times are relative to the loaded slice. Frequency
    edges are converted to offsets relative to the center frequency.
    """

    meta_path = Path(meta_path)
    data_path = _sigmf_data_path(meta_path)
    meta = read_sigmf_meta(meta_path)
    global_meta = meta.get("global", {})
    captures = meta.get("captures", [{}])
    first_capture = captures[0] if captures else {}

    datatype = str(global_meta.get("core:datatype", ""))
    sample_rate_hz = float(global_meta.get("core:sample_rate", first_capture.get("core:sample_rate", 0.0)))
    center_freq_hz = float(first_capture.get("core:frequency", 0.0))
    if sample_rate_hz <= 0:
        raise ValueError("SigMF sample rate is missing or invalid")

    iq = _load_sigmf_iq(
        data_path=data_path,
        datatype=datatype,
        start_sample=start_sample,
        max_samples=max_samples,
        normalize=normalize,
    )
    boxes = _annotations_to_boxes(meta, start_sample, iq.size, sample_rate_hz, center_freq_hz)

    return IQFrame(
        iq=iq,
        sample_rate_hz=sample_rate_hz,
        center_freq_hz=center_freq_hz,
        boxes=boxes,
        frame_id=frame_id or meta_path.stem.replace(".sigmf", ""),
        metadata={
            "source_format": "sigmf",
            "meta_path": str(meta_path),
            "data_path": str(data_path),
            "datatype": datatype,
            "author": global_meta.get("core:author"),
            "description": global_meta.get("core:description"),
            "hardware": global_meta.get("core:hw"),
            "start_sample": start_sample,
            "loaded_samples": int(iq.size),
        },
    )


def save_npz_frame(frame: IQFrame, path: str | Path) -> Path:
    """Save an IQFrame to a lightweight project-local NPZ format."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    boxes_json = json.dumps([box.__dict__ for box in frame.boxes], ensure_ascii=False)
    metadata_json = json.dumps(frame.metadata, ensure_ascii=False, default=str)
    np.savez_compressed(
        path,
        iq=frame.iq.astype(np.complex64),
        sample_rate_hz=np.array(frame.sample_rate_hz, dtype=np.float64),
        center_freq_hz=np.array(frame.center_freq_hz, dtype=np.float64),
        frame_id=np.array(frame.frame_id),
        boxes_json=np.array(boxes_json),
        metadata_json=np.array(metadata_json),
    )
    return path


def load_npz_frame(path: str | Path) -> IQFrame:
    """Load an IQFrame saved by save_npz_frame."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        boxes = [SignalBox(**item) for item in json.loads(str(data["boxes_json"]))]
        metadata = json.loads(str(data["metadata_json"]))
        metadata.setdefault("source_format", "npz_iqframe")
        metadata.setdefault("path", str(path))
        return IQFrame(
            iq=data["iq"].astype(np.complex64),
            sample_rate_hz=float(data["sample_rate_hz"]),
            center_freq_hz=float(data["center_freq_hz"]),
            boxes=boxes,
            frame_id=str(data["frame_id"]),
            metadata=metadata,
        )
