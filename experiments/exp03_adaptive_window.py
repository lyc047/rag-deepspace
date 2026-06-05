"""Phase 3: 自适应窗口实验（增强版）.

改进：
- 样本量 50→200，消除小样本噪声
- 使用 中位数 MSE 绘图（抗离群值），同时报告均值
- 追踪粗筛候选数 → 效率对比图
- SNR 步数 9→13，曲线更平滑

对比7种方案 × SNR扫描：
- No Retrieval, Random, Physics-Only, Signal-Only (4基线)
- Hierarchical dw=0.5 (窄窗口)
- Hierarchical dw=1.5 (宽窗口)
- Hierarchical adaptive v1/v2 (自适应窗口)
"""
import sys, os
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
from visualization import plot_snr_vs_mse, plot_efficiency_comparison


def adaptive_window_v1(sun_angle: float, w_min: float = 0.5, w_max: float = 1.5) -> float:
    """自适应距离窗口 v1：线性映射（仅太阳角）."""
    return w_min + (w_max - w_min) * (sun_angle / 90.0)


def adaptive_window_v2(sun_angle: float, snr_db: float = -150,
                       w_base: float = 1.0, w_max: float = 2.0) -> float:
    """自适应距离窗口 v2：太阳角 + SNR联合调节.

    物理直觉：
    - 太阳角大 → 信号形态与物理参数解耦 → 放宽窗口
    - SNR低   → 信号被噪声污染严重 → 更难匹配 → 放宽窗口
    """
    angle_factor = 0.5 * (sun_angle / 90.0)
    # SNR: 越差(-170dB)窗口越大, 越好(-130dB)窗口越小
    snr_norm = (snr_db + 170) / 40.0   # -170→0, -130→1
    snr_factor = 0.5 * (1 - snr_norm)   # 低SNR→大, 高SNR→小
    dw = w_base + angle_factor + snr_factor
    return min(dw, w_max)


def run_experiment(
    n_templates: int = 500,
    n_test_samples: int = 200,
    snr_range: tuple = (-170, -130),
    n_snr_steps: int = 13,
    signal_type: str = 'slow_varying',
):
    """自适应窗口对比实验."""
    print(f"=== Exp03: Adaptive Window Comparison (Enhanced) ===")
    print(f"Templates: {n_templates}, Test/SNR: {n_test_samples}")
    print(f"SNR steps: {n_snr_steps}, Aggregation: 5%-trimmed mean (plot) + mean/median (report)")

    # 1. 知识库
    print("\n[1/5] Building knowledge base...")
    kb = KnowledgeBase()
    KnowledgeBaseBuilder(n_templates=n_templates, signal_types=[signal_type]).build(kb, seed=42)
    print(f"  -> {kb.count} templates")

    # 2. 方法定义
    print("[2/5] Initializing methods...")
    methods_meta = {
        'No Retrieval':         {'type': 'none'},
        'Random':               {'type': 'random'},
        'Physics-Only':         {'type': 'physics'},
        'Signal-Only':          {'type': 'signal'},
        'Hier-dw=0.5 (narrow)': {'type': 'fixed', 'dw': 0.5,  'da': 15.0},
        'Hier-dw=1.5 (wide)':   {'type': 'fixed', 'dw': 1.5,  'da': 45.0},
        'Hier-adapt-v1 (Ours)': {'type': 'adapt_v1'},
        'Hier-adapt-v2 (Ours)': {'type': 'adapt_v2'},
    }

    # 固定检索器（非自适应方法复用同一个实例）
    retrievers_cache = {
        'Random': RandomRetriever(kb),
        'Physics-Only': PhysicsOnlyRetriever(kb),
        'Signal-Only': SignalOnlyRetriever(kb, normalize=False),
        'Hier-dw=0.5 (narrow)': HierarchicalRetriever(
            kb, coarse_k=200, fine_k=3, normalize=False,
            distance_window=0.5, angle_window=15.0,
        ),
        'Hier-dw=1.5 (wide)': HierarchicalRetriever(
            kb, coarse_k=200, fine_k=3, normalize=False,
            distance_window=1.5, angle_window=45.0,
        ),
    }

    # 3. SNR扫描
    snr_values = np.linspace(snr_range[0], snr_range[1], n_snr_steps)

    # 存储所有原始MSE值（用于计算median和mean）
    raw_mse = {name: {snr: [] for snr in snr_values} for name in methods_meta.keys()}
    raw_coarse = {name: {snr: [] for snr in snr_values}
                  for name in ['Hier-dw=0.5 (narrow)', 'Hier-dw=1.5 (wide)',
                               'Hier-adapt-v1 (Ours)', 'Hier-adapt-v2 (Ours)']}

    print(f"[3/5] Running SNR scan ({n_snr_steps} points × {n_test_samples} samples)...")
    for snr_db in tqdm(snr_values, desc="SNR scan"):
        for _ in range(n_test_samples):
            physics = generate_physical_metadata(snr_db=snr_db)
            sample = generate_telemetry_segment(
                signal_type=signal_type, physics=physics,
                seed=np.random.randint(0, 2**31 - 1),
            )
            y_original = sample.signal
            channel = DeepSpaceChannel(
                distance_au=physics['distance_au'], snr_db=snr_db,
            )
            y_received = channel.forward(y_original)

            qd = physics['distance_au']
            qa = physics['sun_earth_probe_angle']

            for method_name, meta in methods_meta.items():
                if method_name == 'No Retrieval':
                    y_recon = y_received
                elif method_name == 'Random':
                    rr = retrievers_cache['Random'].retrieve(y_received)
                    y_recon = reconstruct_from_template(y_received, rr, kb)
                elif method_name == 'Physics-Only':
                    rr = retrievers_cache['Physics-Only'].retrieve(
                        y_received, distance_au=qd, sun_angle=qa)
                    y_recon = reconstruct_from_template(y_received, rr, kb)
                elif method_name == 'Signal-Only':
                    rr = retrievers_cache['Signal-Only'].retrieve(y_received)
                    y_recon = reconstruct_from_template(y_received, rr, kb)
                elif meta['type'] == 'fixed':
                    hier = retrievers_cache[method_name]
                    rr = hier.retrieve(y_received, distance_au=qd, sun_angle=qa)
                    y_recon = reconstruct_from_template(y_received, rr, kb)
                    raw_coarse[method_name][snr_db].append(hier.last_coarse_count)
                elif meta['type'] == 'adapt_v1':
                    dw = adaptive_window_v1(qa)
                    da = min(90, dw * 30)
                    hier = HierarchicalRetriever(
                        kb, coarse_k=200, fine_k=3, normalize=False,
                        distance_window=dw, angle_window=da,
                    )
                    rr = hier.retrieve(y_received, distance_au=qd, sun_angle=qa)
                    y_recon = reconstruct_from_template(y_received, rr, kb)
                    raw_coarse[method_name][snr_db].append(hier.last_coarse_count)
                elif meta['type'] == 'adapt_v2':
                    dw = adaptive_window_v2(qa, physics['snr_db'])
                    da = min(90, dw * 30)
                    hier = HierarchicalRetriever(
                        kb, coarse_k=200, fine_k=3, normalize=False,
                        distance_window=dw, angle_window=da,
                    )
                    rr = hier.retrieve(y_received, distance_au=qd, sun_angle=qa)
                    y_recon = reconstruct_from_template(y_received, rr, kb)
                    raw_coarse[method_name][snr_db].append(hier.last_coarse_count)

                raw_mse[method_name][snr_db].append(mse(y_original, y_recon))

    # 4. 聚合：修剪均值（绘图，抗离群值+保留系统性差异）+ 中位数/均值（报告）
    print("\n[4/5] Aggregating results...")
    results_plot = {}     # 5% trimmed mean for plotting
    results_median = {}
    results_mean = {}
    for name in methods_meta.keys():
        results_plot[name] = [trim_mean(raw_mse[name][s], 0.05) for s in snr_values]
        results_median[name] = [np.median(raw_mse[name][s]) for s in snr_values]
        results_mean[name] = [np.mean(raw_mse[name][s]) for s in snr_values]

    # 候选数聚合
    candidate_counts = {}
    for name in raw_coarse.keys():
        candidate_counts[name] = [np.mean(raw_coarse[name][s]) for s in snr_values]

    # 5. 画图
    print("[5/5] Plotting...")
    # 5a. MSE图（修剪均值）
    plot_methods = [
        'No Retrieval', 'Physics-Only', 'Signal-Only',
        'Hier-dw=0.5 (narrow)', 'Hier-dw=1.5 (wide)',
        'Hier-adapt-v2 (Ours)',
    ]
    plot_results = {k: results_plot[k] for k in plot_methods}
    plot_snr_vs_mse(
        snr_values=snr_values.tolist(),
        mse_results=plot_results,
        save_path='results/exp03_adaptive_window.png',
        title=f'Adaptive Window: Hierarchical Retrieval ({signal_type}, {n_templates} templates)',
        y_label='5%-Trimmed Mean MSE (log scale)',
    )

    # 5b. 效率对比图
    plot_efficiency_comparison(
        snr_values=snr_values.tolist(),
        candidate_counts=candidate_counts,
        save_path='results/exp03_efficiency.png',
        title=f'检索效率对比：粗筛候选数 vs SNR ({signal_type})',
    )

    # 打印结果
    print("\n" + "=" * 70)
    print("=== Results at SNR=-150dB (trimmed-mean | median | mean) ===")
    print("=" * 70)
    mid = len(snr_values) // 2
    for name in methods_meta.keys():
        marker = ' ★' if 'Ours' in name else ''
        print(f"  {name:30s}: TrimMean = {results_plot[name][mid]:.6f}  "
              f"| Med = {results_median[name][mid]:.6f}  "
              f"| Mean = {results_mean[name][mid]:.6f}{marker}")

    # 效率统计
    print("\n=== Efficiency Summary (avg coarse candidates across all SNR) ===")
    for name in candidate_counts.keys():
        avg_c = np.mean(candidate_counts[name])
        marker = ' ★' if 'Ours' in name else ''
        print(f"  {name:30s}: {avg_c:.1f} candidates{marker}")

    if 'Hier-dw=1.5 (wide)' in candidate_counts and 'Hier-adapt-v2 (Ours)' in candidate_counts:
        wide_avg = np.mean(candidate_counts['Hier-dw=1.5 (wide)'])
        v2_avg = np.mean(candidate_counts['Hier-adapt-v2 (Ours)'])
        saving = (1 - v2_avg / wide_avg) * 100
        print(f"\n  → v2 相比 dw=1.5 平均节省 {saving:.1f}% 粗筛候选数")

    # 各方案增益（修剪均值版）
    print("\n=== Gain over narrow window (dw=0.5) at SNR=-150dB ===")
    baseline = results_plot['Hier-dw=0.5 (narrow)'][mid]
    for name in ['Hier-dw=1.5 (wide)', 'Hier-adapt-v1 (Ours)', 'Hier-adapt-v2 (Ours)']:
        m = results_plot[name][mid]
        gain = (baseline - m) / baseline * 100
        print(f"  {name} gain: {gain:.1f}%")

    # 自适应窗口示例
    print("\n  Adaptive Window v2 examples:")
    for a in [0, 20, 45, 70, 90]:
        for s in [-170, -150, -130]:
            print(f"    angle={a:3}° SNR={s:4}dB → dw={adaptive_window_v2(a,s):.3f}AU")

    return results_median, results_mean, candidate_counts


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--templates', type=int, default=500)
    p.add_argument('--test-samples', type=int, default=200,
                   help='每个SNR点的测试样本数（默认200）')
    p.add_argument('--snr-steps', type=int, default=13,
                   help='SNR扫描步数（默认13，曲线更平滑）')
    args = p.parse_args()
    run_experiment(n_templates=args.templates, n_test_samples=args.test_samples,
                   n_snr_steps=args.snr_steps)
