# tests/test_reconstruction.py
import numpy as np
import sys
sys.path.insert(0, 'src')
from reconstruction import reconstruct_from_template, compute_residual
from knowledge import KnowledgeBase, TemplateRecord


def test_reconstruct_returns_template():
    kb = KnowledgeBase()
    y_template = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    physics = {'distance_au': 1.5, 'sun_earth_probe_angle': 30.0,
               'snr_db': -150, 'doppler_shift_hz': 0,
               'scintillation_index': 0.1, 'mode': 'cruise'}
    record = TemplateRecord(raw_data=y_template, physics=physics)
    kb.insert(record)
    y_query = np.array([1.1, 2.1, 3.1], dtype=np.float32)
    y_recon = reconstruct_from_template(y_query, [(1, 0.1)], kb)
    np.testing.assert_array_equal(y_recon, y_template)


def test_reconstruct_empty_results():
    kb = KnowledgeBase()
    y_query = np.ones(100, dtype=np.float32)
    y_recon = reconstruct_from_template(y_query, [], kb)
    assert np.allclose(y_recon, np.zeros(100))


def test_compute_residual():
    y_orig = np.array([5.0, 6.0, 7.0])
    y_temp = np.array([5.0, 5.0, 5.0])
    residual = compute_residual(y_orig, y_temp)
    np.testing.assert_array_equal(residual, np.array([0., 1., 2.]))
