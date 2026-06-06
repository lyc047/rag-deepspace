# tests/test_reconstruction.py
import numpy as np
import sys
sys.path.insert(0, 'src')
from reconstruction import reconstruct_from_template, compute_residual, quantize_uniform
from knowledge import KnowledgeBase, TemplateRecord


def test_reconstruct_with_original():
    """有y_original时: 模板+残差过信道重建."""
    kb = KnowledgeBase()
    y_template = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    physics = {'distance_au': 1.5, 'sun_earth_probe_angle': 30.0,
               'snr_db': -150, 'doppler_shift_hz': 0,
               'scintillation_index': 0.1, 'mode': 'cruise'}
    record = TemplateRecord(raw_data=y_template, physics=physics)
    kb.insert(record)

    y_original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y_received = np.array([1.1, 2.1, 3.1], dtype=np.float32)

    # 无channel: 完美残差 → y_recon = y_template + (y_original - y_template) = y_original
    y_recon = reconstruct_from_template(y_received, [(1, 0.1)], kb, y_original=y_original)
    np.testing.assert_array_equal(y_recon, y_original)

    # 无y_original: 回退到直接传
    y_recon = reconstruct_from_template(y_received, [(1, 0.1)], kb)
    np.testing.assert_array_equal(y_recon, y_received)


def test_reconstruct_empty_results():
    """空结果回退到直接传."""
    kb = KnowledgeBase()
    y_received = np.ones(100, dtype=np.float32)
    y_recon = reconstruct_from_template(y_received, [], kb)
    assert np.allclose(y_recon, y_received)


def test_compute_residual():
    y_orig = np.array([5.0, 6.0, 7.0])
    y_temp = np.array([5.0, 5.0, 5.0])
    residual = compute_residual(y_orig, y_temp)
    np.testing.assert_array_equal(residual, np.array([0., 1., 2.]))


def test_quantize_uniform():
    x = np.array([-0.3, 0.0, 0.5, 1.0])
    xq, step = quantize_uniform(x, 2)
    assert step > 0
    assert len(xq) == 4
