# tests/test_retrieval.py
import numpy as np
import sys
sys.path.insert(0, 'src')
from retrieval import (
    CoarseRetriever, FineRetriever, HierarchicalRetriever,
    RandomRetriever, PhysicsOnlyRetriever, SignalOnlyRetriever,
)
from knowledge import KnowledgeBase, KnowledgeBaseBuilder


def _setup_kb(n=50):
    kb = KnowledgeBase()
    builder = KnowledgeBaseBuilder(n_templates=n, signal_types=['slow_varying'])
    builder.build(kb, seed=42)
    return kb


def test_coarse_retriever_returns_candidates():
    kb = _setup_kb(50)
    retriever = CoarseRetriever(kb)
    results = retriever.retrieve(distance_au=1.5, sun_angle=45.0, k=10,
                                 distance_window=1.0, angle_window=45.0)
    assert len(results) <= 10
    assert len(results) > 0


def test_fine_retriever_ranks_by_similarity():
    kb = _setup_kb(20)
    coarse = CoarseRetriever(kb)
    candidates = coarse.retrieve(distance_au=1.5, sun_angle=45.0,
                                 k=10, distance_window=1.0, angle_window=90.0)
    y_query = np.sin(np.linspace(0, 6*np.pi, 600)).astype(np.float32)
    retriever = FineRetriever(kb, metric='euclidean')
    results = retriever.retrieve(y_query, candidates, k=3)
    assert len(results) == 3
    for i in range(len(results) - 1):
        assert results[i][1] <= results[i + 1][1]


def test_hierarchical_retriever():
    kb = _setup_kb(50)
    retriever = HierarchicalRetriever(kb, coarse_k=20, fine_k=3, metric='euclidean')
    y_query = np.random.randn(600).astype(np.float32)
    results = retriever.retrieve(y_query, distance_au=1.5, sun_angle=45.0)
    assert 1 <= len(results) <= 3


def test_random_retriever():
    kb = _setup_kb(20)
    retriever = RandomRetriever(kb)
    y = np.zeros(100, dtype=np.float32)
    results = retriever.retrieve(y)
    assert len(results) == 1


def test_physics_only_retriever():
    kb = _setup_kb(30)
    retriever = PhysicsOnlyRetriever(kb)
    y = np.zeros(100, dtype=np.float32)
    results = retriever.retrieve(y, distance_au=1.5, sun_angle=45.0)
    assert len(results) >= 1


def test_signal_only_retriever():
    kb = _setup_kb(30)
    retriever = SignalOnlyRetriever(kb, metric='euclidean')
    y = np.random.randn(600).astype(np.float32)
    results = retriever.retrieve(y, k=3)
    assert len(results) == 3
    assert len(results[0]) == 2
