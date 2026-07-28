from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from .preprocessing import STFTResult
from .types import SignalBox


def _draw_box(ax, box: SignalBox, color: str, linewidth: float, label_prefix: str) -> None:
    x0 = box.t_start_s
    width = max(0.0, box.t_end_s - box.t_start_s)
    y0 = box.f_low_hz / 1e6
    height = max(0.0, (box.f_high_hz - box.f_low_hz) / 1e6)
    rect = Rectangle((x0, y0), width, height, fill=False, edgecolor=color, linewidth=linewidth)
    ax.add_patch(rect)
    if width > 0 and height > 0:
        ax.text(
            x0,
            y0 + height,
            f"{label_prefix}{box.label}",
            color=color,
            fontsize=7,
            va="bottom",
            ha="left",
            clip_on=True,
        )


def plot_stft_with_boxes(
    stft: STFTResult,
    truth: list[SignalBox],
    pred: list[SignalBox],
    output_path: str | Path,
    title: str = "STFT with annotations and predictions",
    dynamic_range_db: float = 80.0,
) -> Path:
    """Save a qualitative STFT plot with ground truth and predicted boxes."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    power = stft.power_db
    vmax = float(np.percentile(power, 99.5))
    vmin = vmax - dynamic_range_db
    extent = [
        float(stft.times_s[0]) if len(stft.times_s) else 0.0,
        float(stft.times_s[-1]) if len(stft.times_s) else 0.0,
        float(stft.freqs_hz[0] / 1e6) if len(stft.freqs_hz) else 0.0,
        float(stft.freqs_hz[-1] / 1e6) if len(stft.freqs_hz) else 0.0,
    ]

    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    im = ax.imshow(
        power,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    for box in truth:
        _draw_box(ax, box, color="#66ff66", linewidth=1.0, label_prefix="T:")
    for box in pred:
        _draw_box(ax, box, color="#38bdf8", linewidth=1.4, label_prefix="P:")

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency offset (MHz)")
    fig.colorbar(im, ax=ax, label="Power (dB)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

