# tests/test_reconstruction.py
import numpy as np
import sys
sys.path.insert(0, 'src')
from reconstruction import (reconstruct_from_template, compute_residual,
                            quantize_uniform, bandwidth_bits, crc8)
from knowledge import KnowledgeBase, TemplateRecord


def test_reconstruct_with_original():
    """有y_original时: 模板+残差过信道重建."""
    kb = KnowledgeBase()
    y_t = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    physics = {'distance_au': 1.5, 'sun_earth_probe_angle': 30.0,
               'snr_db': -150, 'doppler_shift_hz': 0,
               'scintillation_index': 0.1, 'mode': 'cruise'}
    record = TemplateRecord(raw_data=y_t, physics=physics)
    kb.insert(record)
    y_orig = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y_recv = np.array([1.1, 2.1, 3.1], dtype=np.float32)

    # 无channel: 完美残差 → y_recon = template + (orig-template) = orig
    info = {}
    y_recon = reconstruct_from_template(y_recv, [(1, 0.1)], kb, y_original=y_orig)
    np.testing.assert_array_equal(y_recon, y_orig)

    # 无y_original: 退回直接传
    y_recon = reconstruct_from_template(y_recv, [(1, 0.1)], kb)
    np.testing.assert_array_equal(y_recon, y_recv)

    # 有相位对齐: 测试output_info
    y_recon = reconstruct_from_template(y_recv, [(1, 0.1)], kb,
        y_original=y_orig, phase_align=True, output_info=info)
    assert 'lag' in info
    assert 'overhead_bits' in info


def test_reconstruct_empty_results():
    """空结果退回直接传."""
    kb = KnowledgeBase()
    y_recv = np.ones(100, dtype=np.float32)
    y_recon = reconstruct_from_template(y_recv, [], kb)
    assert np.allclose(y_recon, y_recv)


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


def test_bandwidth_bits_with_protection():
    """带宽计算含CRC+lag+mode_flag."""
    bw = bandwidth_bits(600, 2000, 4, phase_align=True)
    assert bw['template_id_bits'] == 11 + 8  # 11 raw + 8 CRC
    assert bw['lag_bits'] == 10
    assert bw['mode_flag_bits'] == 1
    assert bw['overhead_bits'] == 19 + 10 + 1  # 30 bits total overhead
    assert bw['bits_per_sample'] > 0


def test_crc8():
    """CRC-8 基本测试."""
    c1 = crc8(42)
    c2 = crc8(42)
    assert c1 == c2  # 确定性
    c3 = crc8(43)
    assert c1 != c3  # 不同输入不同输出
    assert 0 <= c1 <= 255
