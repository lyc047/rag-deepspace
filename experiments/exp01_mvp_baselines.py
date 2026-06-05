"""Phase 1 核心实验：MVP基线对比（增强版）.

改进：
- 样本量提升至200，消除小样本噪声
- 使用中位数MSE绘图（抗离群值），同时报告均值
- SNR步数13，曲线更平滑

5种方法 × SNR扫描(-170dB ~ -130dB) → 第一张论文图.
方法：无检索、随机检索、纯物理检索、纯信号检索、分层检索(Ours)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.stats import trim_mean
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


def _adaptive_dw(sun_angle, snr_db):
    """自适应窗口（与exp03 v2一致）."""
    snr_norm = (snr_db + 170) / 40.0
    return min(1.0 + 0.5*(sun_angle/90) + 0.5*(1 - snr_norm), 2.0)


def run_experiment(
    n_templates: int = 200,
    n_test_samples: int = 200,
    snr_range: tuple = (-170, -130),
    n_snr_steps: int = 13,
    signal_type: str = 'slow_varying',
    metric: str = 'euclidean',
):
    """运行MVP基线对比实验."""
    print(f"=== Exp01: MVP Baseline Comparison (Enhanced) ===")
    print(f"Templates: {n_templates}, Test samples/SNR: {n_test_samples}")
    print(f"SNR steps: {n_snr_steps}, Aggregation: 5%-trimmed mean (plot) + mean/median (report)")
    print(f"Signal: {signal_type}, Metric: {metric}")

    # 1. 构建知识库
    print("\n[1/4] Building knowledge base...")
    kb = KnowledgeBase()
    builder = KnowledgeBaseBuilder(n_templates=n_templates, signal_types=[signal_type])
    builder.build(kb, seed=42)
    print(f"  -> {kb.count} templates in knowledge base")

    # 2. 初始化检索器
    print("\n[2/4] Initializing retrievers...")
    retrievers_cache = {
        'Random': RandomRetriever(kb),
        'Physics-Only': PhysicsOnlyRetriever(kb),
        'Hier-dw=0.5 (narrow)': HierarchicalRetriever(
            kb, coarse_k=50, fine_k=3, metric=metric, normalize=False,
            distance_window=0.5, angle_window=15.0,
        ),
        'Signal-Only': SignalOnlyRetriever(kb, metric=metric, normalize=False),
    }

    method_names = [
        'No Retrieval', 'Random', 'Physics-Only',
        'Hier-dw=0.5 (narrow)', 'Hier-adaptive (Ours)', 'Signal-Only',
    ]

    # 3. SNR扫描
    snr_values = np.linspace(snr_range[0], snr_range[1], n_snr_steps)
    raw_mse = {name: {snr: [] for snr in snr_values} for name in method_names}

    print(f"\n[3/4] Running SNR scan ({n_snr_steps} points × {n_test_samples} samples)...")
    for snr_db in tqdm(snr_values, desc="SNR scan"):
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
            for method_name in method_names:
                if method_name == 'No Retrieval':
                    y_recon = y_received
                elif method_name == 'Random':
                    rr = retrievers_cache['Random'].retrieve(y_received)
                    y_recon = reconstruct_from_template(y_received, rr, kb)
                elif method_name == 'Physics-Only':
                    rr = retrievers_cache['Physics-Only'].retrieve(
                        y_received,
                        distance_au=physics['distance_au'],
                        sun_angle=physics['sun_earth_probe_angle'],
                    )
                    y_recon = reconstruct_from_template(y_received, rr, kb)
                elif method_name == 'Hier-dw=0.5 (narrow)':
                    rr = retrievers_cache['Hier-dw=0.5 (narrow)'].retrieve(
                        y_received,
                        distance_au=physics['distance_au'],
                        sun_angle=physics['sun_earth_probe_angle'],
                    )
                    y_recon = reconstruct_from_template(y_received, rr, kb)
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
                elif method_name == 'Signal-Only':
                    rr = retrievers_cache['Signal-Only'].retrieve(y_received)
                    y_recon = reconstruct_from_template(y_received, rr, kb)

                raw_mse[method_name][snr_db].append(mse(y_original, y_recon))

    # 聚合
    results_plot = {}     # 5% trimmed mean for plotting
    results_median = {}
    results_mean = {}
    for name in method_names:
        results_plot[name] = [trim_mean(raw_mse[name][s], 0.05) for s in snr_values]
        results_median[name] = [np.median(raw_mse[name][s]) for s in snr_values]
        results_mean[name] = [np.mean(raw_mse[name][s]) for s in snr_values]

    # 4. 画图
    print("\n[4/4] Plotting results...")
    plot_snr_vs_mse(
        snr_values=snr_values.tolist(),
        mse_results=results_plot,
        save_path='results/exp01_snr_vs_mse.png',
        title=f'SNR vs MSE ({signal_type}, {n_templates} templates)',
        y_label='5%-Trimmed Mean MSE (log scale)',
    )

    # 打印关键数据
    print("\n" + "=" * 70)
    print("=== Results at SNR=-150dB (trimmed-mean | median | mean) ===")
    print("=" * 70)
    snr_idx = len(snr_values) // 2
    for name in method_names:
        marker = ' ★' if 'Ours' in name else ''
        print(f"  {name:25s}: TrimMean = {results_plot[name][snr_idx]:.6f}  "
              f"| Med = {results_median[name][snr_idx]:.6f}  "
              f"| Mean = {results_mean[name][snr_idx]:.6f}{marker}")

    # 计算自适应窗口相对于无检索和窄窗口的增益
    no_retrieval_tm = results_plot['No Retrieval'][snr_idx]
    adaptive_tm = results_plot['Hier-adaptive (Ours)'][snr_idx]
    narrow_tm = results_plot['Hier-dw=0.5 (narrow)'][snr_idx]
    gain_no = (no_retrieval_tm - adaptive_tm) / no_retrieval_tm * 100
    gain_narrow = (narrow_tm - adaptive_tm) / narrow_tm * 100
    print(f"\n  Adaptive gain over No Retrieval: {gain_no:.1f}%")
    print(f"  Adaptive gain over narrow (dw=0.5): {gain_narrow:.1f}%")

    return snr_values, results_plot, results_mean


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--templates', type=int, default=200,
                        help='知识库模板数 (默认200，快速测试)')
    parser.add_argument('--test-samples', type=int, default=200,
                        help='每个SNR点的测试样本数（默认200）')
    parser.add_argument('--snr-steps', type=int, default=13,
                        help='SNR扫描步数（默认13，曲线更平滑）')
    parser.add_argument('--signal-type', default='slow_varying',
                        choices=['slow_varying', 'periodic', 'transient', 'science_data'])
    parser.add_argument('--metric', default='euclidean',
                        choices=['euclidean', 'dtw'])
    args = parser.parse_args()

    run_experiment(
        n_templates=args.templates,
        n_test_samples=args.test_samples,
        n_snr_steps=args.snr_steps,
        signal_type=args.signal_type,
        metric=args.metric,
    )
