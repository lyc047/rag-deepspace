from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.fiesta import contiguous_true_regions, extract_fiesta_events, fiesta_semantic_event_bits


def test_contiguous_true_regions_filters_short_runs():
    mask = np.array([False, True, True, False, True, False, True, True, True])
    assert contiguous_true_regions(mask, min_bins=2) == [(1, 3), (6, 9)]


def test_extract_fiesta_events_detects_high_power_band():
    psd = np.array([-100, -101, -99, -80, -79, -81, -100, -102], dtype=np.float32)
    events = extract_fiesta_events(psd, center_hz=100e6, bandwidth_hz=8e6, threshold_margin_db=8, min_bins=2)
    assert len(events) == 1
    assert events[0].start_bin == 3
    assert events[0].end_bin == 6
    assert events[0].peak_dbm == -79.0


def test_fiesta_semantic_bits_grow_with_events():
    assert fiesta_semantic_event_bits(2) > fiesta_semantic_event_bits(1)
