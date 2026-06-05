"""Phase 1 核心实验：MVP基线对比.

5种方法 × SNR扫描(-170dB ~ -130dB) → 第一张论文图.
方法：无检索、随机检索、纯物理检索、纯信号检索、分层检索(Ours)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from tqdm import tqdm

from telemetry import generate_telemetry_segment, generate_physical_metadata
from channel import DeepSpaceChannel
from knowledge import KnowledgeBase, KnowledgeBaseBuilder
from retrieval import (
    HierarchicalRetriever, RandomRetriever,
    PhysicsOnlyRetriever, SignalOnlyRetriever,
)
from reconstruction import reconstruct_from_template
from metrics import mse
from visualization import plot_snr_vs_mse


def run_experiment(
    n_templates: int = 1000,
    n_test_samples: int = 100,
    snr_range: tuple = (-170, -130),
    n_snr_steps: int = 9,
    signal_type: str = 'slow_varying',
    metric: str = 'euclidean',
):
    """运行MVP基线对比实验."""
    print(f"=== Exp01: MVP Baseline Comparison ===")
    print(f"Templates: {n_templates}, Test samples/SNR: {n_test_samples}")
    print(f"SNR range: {snr_range}, Steps: {n_snr_steps}")
    print(f"Signal: {signal_type}, Metric: {metric}")

    # 1. 构建知识库
    print("\n[1/4] Building knowledge base...")
    kb = KnowledgeBase()
    builder = KnowledgeBaseBuilder(n_templates=n_templates, signal_types=[signal_type])
    builder.build(kb, seed=42)
    print(f"  -> {kb.count} templates in knowledge base")

    # 2. 初始化检索器
    print("\n[2/4] Initializing retrievers...")
    # 自适应窗口函数（与exp03一致）
    def _adaptive_dw(sun_angle, snr_db):
        snr_norm = (snr_db + 170) / 40.0
        return min(1.0 + 0.5*(sun_angle/90) + 0.5*(1 - snr_norm), 2.0)

    retrievers = {
        'No Retrieval': None,
        'Random': RandomRetriever(kb),
        'Physics-Only': PhysicsOnlyRetriever(kb),
        'Hier-dw=0.5 (narrow)': HierarchicalRetriever(
            kb, coarse_k=50, fine_k=3, metric=metric, normalize=False,
            distance_window=0.5, angle_window=15.0,
        ),
        'Hier-adaptive (Ours)': 'adaptive',  # 每次查询时动态计算窗口
        'Signal-Only (upper)': SignalOnlyRetriever(kb, metric=metric, normalize=False),
    }

    # 3. SNR扫描
    snr_values = np.linspace(snr_range[0], snr_range[1], n_snr_steps)
    results = {name: [] for name in retrievers.keys()}

    print(f"\n[3/4] Running SNR scan...")
    for snr_db in tqdm(snr_values, desc="SNR scan"):
        mse_per_method = {name: [] for name in retrievers.keys()}

        for _ in range(n_test_samples):
            physics = generate_physical_metadata(snr_db=snr_db)
            sample = generate_telemetry_segment(
                signal_type=signal_type, physics=physics,
                seed=np.random.randint(0, 2**31 - 1),
            )
            y_original = sample.signal

            # 过信道
            channel = DeepSpaceChannel(
                distance_au=physics['distance_au'], snr_db=snr_db,
            )
            y_received = channel.forward(y_original)

            # 各方法重建
            for method_name, retriever in retrievers.items():
                if method_name == 'No Retrieval':
                    y_recon = y_received
                elif method_name == 'Hier-adaptive (Ours)':
                    dw = _adaptive_dw(physics['sun_earth_probe_angle'], physics['snr_db'])
                    da = min(90, dw * 30)
                    hier = HierarchicalRetriever(
                        kb, coarse_k=200, fine_k=3, metric=metric, normalize=False,
                        distance_window=dw, angle_window=da,
                    )
                    rr = hier.retrieve(y_received, distance_au=physics['distance_au'],
                                       sun_angle=physics['sun_earth_probe_angle'])
                    y_recon = reconstruct_from_template(y_received, rr, kb)
                else:
                    retrieval_results = retriever.retrieve(
                        y_received,
                        distance_au=physics['distance_au'],
                        sun_angle=physics['sun_earth_probe_angle'],
                    )
                    y_recon = reconstruct_from_template(
                        y_received, retrieval_results, kb,
                    )
                mse_per_method[method_name].append(mse(y_original, y_recon))

        # 取平均
        for name in retrievers.keys():
            results[name].append(np.mean(mse_per_method[name]))

    # 4. 画图
    print("\n[4/4] Plotting results...")
    plot_snr_vs_mse(
        snr_values=snr_values.tolist(),
        mse_results=results,
        save_path='results/exp01_snr_vs_mse.png',
        title=f'SNR vs MSE ({signal_type}, {n_templates} templates)',
    )

    # 打印关键数据
    print("\n=== Results at SNR=-150dB ===")
    snr_idx = len(snr_values) // 2
    for name in retrievers.keys():
        print(f"  {name:25s}: MSE = {results[name][snr_idx]:.6f}")

    # 计算自适应窗口相对于无检索和窄窗口的增益
    no_retrieval_mse = results['No Retrieval'][snr_idx]
    adaptive_mse = results['Hier-adaptive (Ours)'][snr_idx]
    narrow_mse = results['Hier-dw=0.5 (narrow)'][snr_idx]
    gain_no = (no_retrieval_mse - adaptive_mse) / no_retrieval_mse * 100
    gain_narrow = (narrow_mse - adaptive_mse) / narrow_mse * 100
    print(f"\n  Adaptive gain over No Retrieval: {gain_no:.1f}%")
    print(f"  Adaptive gain over narrow (dw=0.5): {gain_narrow:.1f}%")

    return snr_values, results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--templates', type=int, default=200,
                        help='知识库模板数 (默认200，快速测试)')
    parser.add_argument('--test-samples', type=int, default=30,
                        help='每个SNR点的测试样本数')
    parser.add_argument('--signal-type', default='slow_varying',
                        choices=['slow_varying', 'periodic', 'transient', 'science_data'])
    parser.add_argument('--metric', default='euclidean',
                        choices=['euclidean', 'dtw'])
    args = parser.parse_args()

    run_experiment(
        n_templates=args.templates,
        n_test_samples=args.test_samples,
        signal_type=args.signal_type,
        metric=args.metric,
    )
