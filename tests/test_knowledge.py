# tests/test_knowledge.py
import numpy as np
import sys, tempfile, os
sys.path.insert(0, 'src')
from knowledge import TemplateRecord, KnowledgeBase, KnowledgeBaseBuilder
from telemetry import generate_physical_metadata


def test_template_record_creation():
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    physics = {'distance_au': 2.0, 'sun_earth_probe_angle': 45.0,
               'snr_db': -150, 'doppler_shift_hz': 10000,
               'scintillation_index': 0.1, 'mode': 'cruise'}
    record = TemplateRecord(raw_data=y, physics=physics, signal_type='slow_varying', channel_name='temp_bus')
    assert record.raw_data.shape == (3,)
    assert record.physics['mode'] == 'cruise'
    assert record.stat_features is not None
    assert len(record.stat_features) == 4


def test_template_record_stat_features():
    rng = np.random.RandomState(42)
    y = rng.randn(1000).astype(np.float32)
    physics = {'distance_au': 2.0, 'sun_earth_probe_angle': 30.0,
               'snr_db': -150, 'doppler_shift_hz': 0,
               'scintillation_index': 0.1, 'mode': 'cruise'}
    record = TemplateRecord(raw_data=y, physics=physics, signal_type='periodic')
    assert abs(record.stat_features[0]) < 0.1  # mean ~0
    assert 0.8 < record.stat_features[1] < 1.2  # std ~1


def test_knowledge_base_insert_and_query():
    kb = KnowledgeBase()  # 内存数据库，避免Windows文件锁问题
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    physics = {'distance_au': 2.0, 'sun_earth_probe_angle': 45.0,
               'snr_db': -150, 'doppler_shift_hz': 10000,
               'scintillation_index': 0.1, 'mode': 'cruise'}
    record = TemplateRecord(raw_data=y, physics=physics, signal_type='slow_varying')
    kb.insert(record)
    candidates = kb.coarse_query(distance_au=2.0, sun_angle=45.0,
                                 distance_window=0.5, angle_window=10.0)
    assert len(candidates) >= 1
    assert candidates[0]['distance_au'] == 2.0


def test_knowledge_base_coarse_query_empty():
    kb = KnowledgeBase()  # 内存数据库
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    physics = {'distance_au': 2.0, 'sun_earth_probe_angle': 45.0,
               'snr_db': -150, 'doppler_shift_hz': 10000,
               'scintillation_index': 0.1, 'mode': 'cruise'}
    record = TemplateRecord(raw_data=y, physics=physics, signal_type='slow_varying')
    kb.insert(record)
    candidates = kb.coarse_query(distance_au=0.5, sun_angle=10.0,
                                 distance_window=0.1, angle_window=5.0)
    assert len(candidates) == 0


def test_knowledge_base_builder():
    builder = KnowledgeBaseBuilder(n_templates=50)
    assert len(builder.physics_grid) == 50
    for physics in builder.physics_grid:
        assert 0.5 <= physics['distance_au'] <= 3.0


def test_knowledge_base_stat_index():
    kb = KnowledgeBase()  # 内存数据库
    for i in range(20):
        rng = np.random.RandomState(i)
        y = rng.randn(100).astype(np.float32) + i * 0.1
        physics = generate_physical_metadata(distance_au=1.0 + i * 0.1)
        record = TemplateRecord(raw_data=y, physics=physics, signal_type='slow_varying')
        kb.insert(record)
    kb.build_faiss_index()
    query_features = np.array([0.5, 1.0, 0.0, 3.0], dtype=np.float32)
    indices, distances = kb.search_by_features(query_features, k=3)
    assert len(indices) == 3
